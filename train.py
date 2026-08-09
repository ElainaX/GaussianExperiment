#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import *
from scene import GaussianModel, Scene
from utils.fast_utils import (  # [FASTGS / GLOSSY PRIOR]
    compute_gaussian_glossy_score,
    compute_gaussian_score_rtsplat,
    edge_aware_loss,
)
from utils.general_utils import GaussianTracker, safe_state
from utils.image_utils import apply_colormap, local_variance, log_normalize, psnr
from utils.loss_utils import binary_cross_entropy, l1_loss, lpips, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    # =========================================================================
    # 阶段①：初始化
    # =========================================================================
    first_iter = 0
    tb_writer, tb_executor = prepare_output_and_logger(dataset)

    # 创建高斯模型（包含位置/颜色/不透明度/协方差等所有可学习参数）
    gaussians = GaussianModel(dataset.sh_degree, dataset)

    # 加载场景：读取相机位姿 + 初始点云（来自 COLMAP 或合成数据集）
    scene = Scene(dataset, gaussians, resolution_scales=[1.0])

    # 给高斯模型设置 Adam 优化器，每个属性（xyz/色彩/不透明度等）有独立学习率
    gaussians.training_setup(opt)

    # 如果指定了 checkpoint，从中恢复模型参数和迭代起点
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # CUDA Event 用于精确测量每次迭代的 GPU 耗时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 训练指标追踪器：每 100 iter 记录高斯数量和显存压力，自动写 JSON（崩溃安全）
    tracker = GaussianTracker(scene.model_path, interval=100)

    # =========================================================================
    # 阶段②：训练前准备
    # =========================================================================
    # 拷贝训练相机列表，每次迭代随机从中取一个视角
    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    print('Training set length', len(viewpoint_stack))
    if opt.glossy_prior_on:
        prior_views = sum(camera.has_image_priors for camera in viewpoint_stack)
        if prior_views == 0:
            raise ValueError(
                '--glossy_prior_on requires complete depth/normal/roughness maps; '
                'set --prior_path to the mapped prior directory.'
            )
        print(f'[GLOSSY-PRIOR] complete prior views: {prior_views}/{len(viewpoint_stack)}')

    ema_loss_dict = {}  # 指数移动平均 loss，用于进度条显示
    progress_bar = tqdm(range(first_iter, opt.iterations), desc='Training progress')
    first_iter += 1
    last_reset_iter = -100000  # 记录上次 reset occupancy 的迭代，用于 prune 阈值判断
    pipe.init_stage = True     # init_stage=True 时渲染管线走简化路径（不含完整 PBR 分解）

    # =========================================================================
    # 阶段③：主训练循环
    # =========================================================================
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # 按迭代数衰减各属性的学习率（位置学习率按指数衰减，其他固定）
        gaussians.update_learning_rate(iteration)

        # Match goodV1_rtsplat-baseline: progressively enable the global SH degree.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ------------------------------------------------------------------
        # 取一个随机训练视角及其 GT 图像
        # ------------------------------------------------------------------
        data_idx = np.random.randint(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack[data_idx]
        gt_image = viewpoint_cam.original_image.cuda()
        gt_transparent_mask = viewpoint_cam.gt_transparent_mask.cuda()  # True=透明区域

        # 若 GT 有 alpha 通道（RGBA），用随机背景色合成，增强对透明区域的泛化
        if gt_image.shape[0] == 4:
            bg = torch.rand((3), device='cuda')
            gt_image = gt_image[:3, ...] * gt_image[3:, ...] + (1 - gt_image[3:, ...]) * bg[:, None, None]
        else:
            bg = torch.zeros((3), device='cuda')

        # ------------------------------------------------------------------
        # 前向渲染：输出完整的渲染分量（这是本项目与标准 3DGS 最大的不同）
        # ------------------------------------------------------------------
        if iteration >= opt.init_until_iter:
            pipe.init_stage = False  # 超过初始化阶段后切换到完整 PBR 渲染路径
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        # 用于 densification 的梯度信号：哪些高斯可见、它们的屏幕空间半径
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']

        # 最终合成图像 = scatter 分量 + transmitted 分量 + specular 分量
        final_rendering = render_pkg['final_rendering']
        final_tran = render_pkg['final_tran']    # 透射：穿过透明物体的背景光
        final_scat = render_pkg['final_scat']    # 散射：物体内部次表面散射的漫反射
        final_spec = render_pkg['final_spec']    # 镜面：菲涅耳反射的高光
        render_tran = render_pkg['render_tran']  # 不含 PBR 分解的原始透射渲染（init 阶段用）
        render_scat = render_pkg['render_scat']  # 不含 PBR 分解的原始散射渲染
        transmissivity = render_pkg['transmissivity']  # 每像素的透明度（标量图）
        foreground = render_pkg['foreground']          # 前景 mask（有高斯覆盖的区域）

        # 法线相关：surface_normal 来自高斯朝向，surface_depth_normal 来自深度梯度
        surface_normal = render_pkg['surface_normal']
        surface_depth_normal = render_pkg['surface_depth_normal']
        surface_opacity = render_pkg['surface_opacity']  # 不透明物体的 alpha 图

        volume_dist = render_pkg['volume_dist']  # 体积内高斯沿射线的分散程度（用于正则化）

        # ------------------------------------------------------------------
        # 计算 Loss（课程式：不同迭代阶段开启不同 loss 组合）
        # ------------------------------------------------------------------
        loss = 0.0
        loss_dict = {}

        # === 阶段 A：早期 init（< norm_loss_from_iter）只用原始透射图对 GT 做监督 ===
        # 此时网络还没稳定，用简单的 render_tran 而非完整 PBR 合成图
        if iteration < opt.norm_loss_from_iter:
            loss_diff = (1.0 - opt.lambda_dssim) * l1_loss(render_tran, gt_image) + opt.lambda_dssim * (1.0 - ssim(render_tran, gt_image))
            loss += loss_diff
            loss_dict['diff'] = loss_diff.item()
        # === 阶段 B：中期 init（norm_loss_from_iter ~ init_until_iter）用完整合成图但不用 PBR 分解 ===
        elif iteration < opt.init_until_iter:
            loss_diff = (1.0 - opt.lambda_dssim) * l1_loss(final_rendering, gt_image) + opt.lambda_dssim * (1.0 - ssim(final_rendering, gt_image))
            loss += loss_diff
            loss_dict['diff'] = loss_diff.item()
        # === 阶段 C：完整 PBR 阶段（>= init_until_iter）===
        else:
            if iteration < opt.mask_loss_from_iter:
                # mask loss 还没开启，直接用合成图
                detached_rendering = final_rendering
            else:
                # 镜面反射区域（spec_complexity 高的地方）用 detach 过的 tran，
                # 防止高光区域的梯度干扰透射分量的学习
                spec_variance = local_variance(final_spec, weights=1 - surface_opacity.detach())
                spec_complexity = 1 - torch.exp(-opt.local_var_scale * spec_variance.detach())
                detached_tran = final_tran.detach() * spec_complexity + final_tran * (1 - spec_complexity)

                detached_rendering = final_scat + detached_tran + final_spec

            loss_pbr = (1.0 - opt.lambda_dssim) * l1_loss(detached_rendering, gt_image) + opt.lambda_dssim * (1.0 - ssim(detached_rendering, gt_image))
            loss += loss_pbr
            loss_dict['pbr'] = loss_pbr.item()

            # LPIPS 感知损失（可选），在 lpips_loss_from_iter 后开启
            if opt.lambda_lpips > 0 and iteration >= opt.lpips_loss_from_iter:
                loss_lpips = opt.lambda_lpips * lpips(detached_rendering, gt_image)
                loss += loss_lpips
                loss_dict['lpips'] = loss_lpips.item()

        # === Occupancy 衰减 loss：惩罚可见高斯的不透明度，鼓励稀疏表示 ===
        if opt.occupancy_decay_weight > 0 and iteration >= opt.init_until_iter:
            occupancy_loss = opt.occupancy_decay_weight * (gaussians.get_occupancy[visibility_filter]).mean()
            loss += occupancy_loss
            loss_dict['occupancy'] = occupancy_loss.item()

        # === 法线一致性 loss：surface_normal（高斯朝向）与 surface_depth_normal（深度梯度法线）对齐 ===
        # 让高斯的朝向与几何表面对齐，改善法线质量
        if iteration >= opt.norm_loss_from_iter and opt.norm_loss_weight > 0:
            error = 1 - (surface_normal * surface_depth_normal).sum(dim=0, keepdim=True)
            error = error * foreground  # 只在有几何的区域算
            norm_loss = opt.norm_loss_weight * error.mean()
            loss += norm_loss
            loss_dict['norm'] = norm_loss.item()

        # === 体积分散 loss：惩罚同一射线上的高斯过于分散，鼓励高斯贴合表面 ===
        if iteration >= opt.dist_loss_from_iter and opt.dist_loss_weight > 0:
            dist_loss = opt.dist_loss_weight * volume_dist.mean()
            loss += dist_loss
            loss_dict['dist'] = dist_loss.item()

        # === Mask loss：监督不透明度与 GT 透明 mask 的一致性（BCE） ===
        if opt.mask_loss_from_iter == -1:
            opt.mask_loss_from_iter = opt.init_until_iter
        if opt.mask_loss_weight > 0 and iteration >= opt.mask_loss_from_iter:
            mask_loss = opt.mask_loss_weight * binary_cross_entropy(1 - surface_opacity, gt_transparent_mask.squeeze(0) * 1.0)
            loss += mask_loss
            loss_dict['mask'] = mask_loss.item()

        # === Transmissivity loss：不透明区域的透明度应接近 0（BCE 强制为二值） ===
        if iteration >= opt.init_until_iter:
            transmissivity_loss = opt.transmissivity_loss_weight * binary_cross_entropy(transmissivity[~gt_transparent_mask], 0)
            loss += transmissivity_loss
            loss_dict['transmissivity'] = transmissivity_loss.item()

        # === Consistency loss：透明区域内的 transmissivity 和 scatter 应当均匀，减少噪声 ===
        if opt.consistency_loss_weight > 0 and iteration >= opt.init_until_iter:
            consistency_loss = 0
            tran_vals = transmissivity[gt_transparent_mask]
            tran_mean = tran_vals.mean()
            consistency_loss += opt.consistency_loss_weight * ((tran_vals - tran_mean) ** 2).sum()

            scatter_vals = render_scat[:, gt_transparent_mask.squeeze(0)]
            scatter_mean = scatter_vals.mean(dim=-1, keepdim=True)
            consistency_loss += opt.consistency_loss_weight * ((scatter_vals - scatter_mean) ** 2).sum(dim=-1).mean()

            loss += consistency_loss
            loss_dict['consistency'] = consistency_loss.item()

        # === 边缘感知 loss：对齐渲染图与 GT 的「边缘×深度」分布 ===
        if opt.lambda_edge_aware > 0 and iteration >= opt.edge_aware_from_iter:
            ea_loss = opt.lambda_edge_aware * edge_aware_loss(
                final_rendering, gt_image, render_pkg['surface_depth']
            )
            loss += ea_loss
            loss_dict['edge_aware'] = ea_loss.item()

        # ------------------------------------------------------------------
        # 反向传播
        # ------------------------------------------------------------------
        total_loss = loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # 用 EMA 平滑各 loss，显示在进度条上
            for key, value in loss_dict.items():
                ema_loss_dict[key] = 0.4 * value + 0.6 * ema_loss_dict.get(key, 0.0)

            if iteration % 10 == 0:
                monitor_dict = {**ema_loss_dict, 'Points': f'{len(gaussians.get_xyz)}'}
                progress_bar.set_postfix(monitor_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_end.synchronize()
            training_report(tb_writer, tb_executor, opt, iteration, loss, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, partial(render, pipe=pipe, bg_color=bg))
            if iteration in saving_iterations:
                print('\n[ITER {}] Saving Gaussians'.format(iteration))
                scene.save(iteration)

            # =========================================================
            # 阶段④：Densification（高斯的自适应增删）
            # =========================================================
            # densify_until_iter 之前持续对高斯数量进行自适应调整
            if iteration < opt.densify_until_iter:
                # 记录每个高斯在屏幕上的最大半径（用于判断是否需要分裂）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 累积梯度统计，用于判断哪些高斯需要分裂（梯度大=重建不足）或克隆
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold：超过 occupancy_reset_interval 后才剔除屏幕过大的高斯
                    size_threshold = 20 if iteration > opt.occupancy_reset_interval else None
                    if opt.fastgs_on:
                        # [FASTGS BEGIN] 多视角一致性引导的 densification + pruning
                        # 只有至少一个子模块需要分数时才跑多相机渲染
                        need_scores = opt.fastgs_densify or opt.fastgs_prune
                        importance_score = pruning_score = protection_score = None
                        if need_scores:
                            importance_score, pruning_score, protection_score = \
                                compute_gaussian_score_rtsplat(
                                    scene.getTrainCameras(), gaussians, pipe, bg, opt
                                )
                        gaussians.densify_and_prune_fastgs(
                            opt, importance_score, pruning_score, protection_score,
                            scene.cameras_extent, size_threshold, last_reset_iter,
                            do_densify=opt.fastgs_densify,
                            do_prune=opt.fastgs_prune,
                            do_protect_prune=opt.fastgs_protect_prune,
                            do_protect_densify=opt.fastgs_protect_densify,
                        )
                        # [FASTGS END]
                    else:
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, opt.occupancy_cull,
                            scene.cameras_extent, size_threshold, last_reset_iter
                        )

                # 定期将所有高斯的 occupancy（不透明度）重置为低值，
                # 让无用高斯在下一轮被 prune 淘汰
                if iteration % opt.occupancy_reset_interval == 0:
                    gaussians.reset_occupancy()
                    last_reset_iter = iteration

                # 在 reset_occupancy 的中间点 reset opacity（标准 3DGS 的做法，与 occupancy 机制配合）
                if iteration >= opt.occupancy_reset_interval and iteration % opt.occupancy_reset_interval == opt.occupancy_reset_interval // 2:
                    gaussians.reset_opacity()

            # =========================================================
            # 阶段⑤：参数更新
            # =========================================================
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Periodically project the 2D material-prior score back to visible
            # Gaussians and fuse it with their cross-view GT color variation.
            if (
                opt.glossy_prior_on
                and iteration >= opt.glossy_from_iter
                and iteration % opt.glossy_interval == 0
            ):
                glossy_stats = compute_gaussian_glossy_score(
                    scene.getTrainCameras(), gaussians, pipe, bg, opt
                )
                if glossy_stats is not None:
                    glossy_count = gaussians.update_glossy_prior(
                        glossy_stats['fused'],
                        ema=opt.glossy_ema,
                        threshold=opt.glossy_threshold,
                        target_roughness=opt.glossy_target_roughness,
                        target_reflectance=opt.glossy_target_reflectance,
                    )
                    visible = glossy_stats['view_count'] > 0
                    mean_score = glossy_stats['fused'][visible].mean().item() if visible.any() else 0.0
                    mean_color = glossy_stats['color_variation'][visible].mean().item() if visible.any() else 0.0
                    print(
                        f"[GLOSSY-PRIOR] cameras={glossy_stats['used_cameras']} "
                        f"mean={mean_score:.4f} color-var={mean_color:.4f} "
                        f"selected={glossy_count}/{gaussians.get_xyz.shape[0]}"
                    )
                    if tb_writer:
                        tb_writer.add_scalar('glossy_prior/mean_fused_score', mean_score, iteration)
                        tb_writer.add_scalar('glossy_prior/mean_color_variation', mean_color, iteration)
                        tb_writer.add_scalar('glossy_prior/selected_gaussians', glossy_count, iteration)

            if iteration in checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + '/chkpnt' + str(iteration) + '.pth')

            tracker.record(iteration, gaussians)

    tracker.draw()
    if tb_executor is not None:
        tb_executor.shutdown()


def prepare_output_and_logger(args):
    scene_name = os.path.basename(args.source_path)
    if not args.model_path:
        args.model_path = os.path.join('./output', scene_name)

    # Set up output folder
    print('Output folder: {}'.format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    tb_executor = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        tb_executor = ThreadPoolExecutor(max_workers=2)
    else:
        print('Tensorboard not available: not logging progress')
    return tb_writer, tb_executor


@torch.no_grad()
def training_report(tb_writer, tb_executor, opt, iteration, loss, loss_dict, elapsed, testing_iterations, scene: Scene, renderFunc):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        for key, value in loss_dict.items():
            tb_writer.add_scalar(f'train_loss_patches/{key}', value, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                window_psnr_test = 0.0
                opaque_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians)
                    image = torch.clamp(render_pkg['final_rendering'], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to('cuda'), 0.0, 1.0)
                    gt_transparent_mask = viewpoint.gt_transparent_mask.cuda()
                    if tb_writer and (idx < 5):
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/render', image, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/final_specular', render_pkg['final_spec'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/final_scatter', render_pkg['final_scat'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/final_transmitted', render_pkg['final_tran'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/render_scatter', render_pkg['render_scat'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/render_specular', render_pkg['render_spec'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/render_transmitted', render_pkg['render_tran'].clip(0, 1), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/reflectance', render_pkg['reflectance'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/transmissivity', render_pkg['transmissivity'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/roughness', render_pkg['roughness'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/glossy_score', render_pkg['glossy_score'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/attenuation', render_pkg['attenuation'], global_step=iteration)

                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/surface_alpha', render_pkg['surface_alpha'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/surface_depth', apply_colormap(log_normalize(render_pkg['surface_depth'])), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/surface_normal', render_pkg['surface_normal'] * 0.5 + 0.5, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/surface_depth_normal', render_pkg['surface_depth_normal'] * 0.5 + 0.5, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/surface_opacity', render_pkg['surface_opacity'], global_step=iteration)

                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/volume_alpha', render_pkg['volume_alpha'], global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/volume_depth', apply_colormap(log_normalize(render_pkg['volume_depth'])), global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/volume_normal', render_pkg['volume_normal'] * 0.5 + 0.5, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/volume_depth_normal', render_pkg['volume_depth_normal'] * 0.5 + 0.5, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/volume_opacity', render_pkg['volume_opacity'], global_step=iteration)

                        spec_variance = local_variance(render_pkg['final_spec'], weights=1 - render_pkg['surface_opacity'].detach())
                        spec_complexity = 1 - torch.exp(-opt.local_var_scale * spec_variance.detach())
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/spec_variance', opt.local_var_scale * spec_variance, global_step=iteration)
                        tb_executor.submit(tb_writer.add_image, config['name'] + f'_view_{viewpoint.image_name}/spec_complexity', spec_complexity, global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_executor.submit(tb_writer.add_image, config['name'] + '_view_{}/ground_truth'.format(viewpoint.image_name), gt_image, global_step=iteration)

                    l1_value = l1_loss(image, gt_image).mean().double()
                    psnr_value = psnr(image, gt_image).mean().double()
                    ssim_value = ssim(image, gt_image).mean().double()
                    lpips_value = lpips(image, gt_image).mean().double()
                    window_psnr_value = psnr(image * gt_transparent_mask, gt_image * gt_transparent_mask).mean().double() if gt_transparent_mask.any() else 0.0
                    opaque_psnr_value = psnr(image * ~gt_transparent_mask, gt_image * ~gt_transparent_mask).mean().double() if (~gt_transparent_mask).any() else 0.0
                    if tb_writer:
                        tb_writer.add_scalar(f'per_view_{config["name"]}/l1_loss - {viewpoint.image_name}', l1_value, iteration)
                        tb_writer.add_scalar(f'per_view_{config["name"]}/psnr - {viewpoint.image_name}', psnr_value, iteration)
                        tb_writer.add_scalar(f'per_view_{config["name"]}/ssim - {viewpoint.image_name}', ssim_value, iteration)
                        tb_writer.add_scalar(f'per_view_{config["name"]}/lpips - {viewpoint.image_name}', lpips_value, iteration)
                        tb_writer.add_scalar(f'per_view_{config["name"]}/window_psnr - {viewpoint.image_name}', window_psnr_value, iteration)
                        tb_writer.add_scalar(f'per_view_{config["name"]}/opaque_psnr - {viewpoint.image_name}', opaque_psnr_value, iteration)
                    l1_test += l1_value
                    psnr_test += psnr_value
                    ssim_test += ssim_value
                    lpips_test += lpips_value
                    window_psnr_test += window_psnr_value
                    opaque_psnr_test += opaque_psnr_value

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                window_psnr_test /= len(config['cameras'])
                opaque_psnr_test /= len(config['cameras'])
                print(f'\n[ITER {iteration}] Evaluating {config["name"]}: PSNR {psnr_test}, SSIM {ssim_test}, LPIPS {lpips_test}, Window PSNR {window_psnr_test}, Opaque PSNR {opaque_psnr_test}')
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - window_psnr', window_psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - opaque_psnr', opaque_psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description='Training script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=range(1_000, 300_001, 1_000))
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print('Optimizing ' + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print('\nTraining complete.')
