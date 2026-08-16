#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith('_'):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, action='store_true')
                elif t == list:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, nargs='+')
                else:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument('--' + key, default=value, action='store_true')
                elif t == list:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, nargs='+')
                else:
                    group.add_argument('--' + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ('_' + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ''
        self._model_path = ''
        self._images = 'images'
        self.prior_path = ''
        self._resolution = -1
        self._white_background = False
        self.data_device = 'cuda'
        self.eval = False

        self.run_dim = 256
        self.rand_init = False
        # Fixed positive-radiance cubemaps captured around glossy regions.
        self.local_probe_on = False
        self.local_probe_count = 4
        self.local_probe_resolution = 64
        self.local_probe_strength = 0.8
        self.local_probe_radiance_max = 4.0
        self.local_probe_glossy_low = 0.10
        self.local_probe_glossy_high = 0.20
        self.local_probe_scope_radius = 15.0
        self.local_probe_surface_offset = 0.25
        self.local_probe_query_radius = 5.0

        self.env_scope_center = [0.0, 0.0, 0.0]
        self.env_scope_radius = 0.0
        self.xyz_axis = [0.0, 0.0, 0.0]

        super().__init__(parser, 'Loading Parameters', sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if g.prior_path:
            g.prior_path = os.path.abspath(os.path.expanduser(g.prior_path))
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self.init_stage = False
        super().__init__(parser, 'Pipeline Parameters')


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 61000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.occupancy_lr = 0.05
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.reflectance_lr = 0.005
        self.roughness_lr = 0.002
        self.transmissivity_lr = 0.01
        self.feature_lr = 0.002
        self.encoding_lr = 0.002
        self.mlp_lr = 0.0005

        self.percent_dense = 0.01
        self.occupancy_cull = 0.05
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0.01
        self.lpips_loss_from_iter = 15000
        self.dist_loss_weight = 0
        self.dist_loss_from_iter = 0
        self.norm_loss_weight = 0.05
        self.norm_loss_from_iter = 0
        # Optional supervision from generated camera-space normal priors.
        self.lambda_normal_prior = 0.0
        self.normal_prior_from_iter = 1000
        self.normal_prior_warmup_iters = 2000
        self.normal_prior_axis_sign = [-1.0, 1.0, 1.0]
        self.normal_prior_edge_suppression = 2.0
        self.normal_prior_min_alpha = 0.05
        self.normal_prior_pool_size = 3
        self.occupancy_decay_weight = 0.001
        self.mask_loss_weight = 0.01
        self.mask_loss_from_iter = -1
        self.transmissivity_loss_weight = 0.01
        self.consistency_loss_weight = 0.000002
        self.local_var_scale = 4

        self.densification_interval = 100
        self.occupancy_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # [FASTGS BEGIN] 多视角一致性 densification 参数
        self.fastgs_on = False                # 总开关：False=原始 RT-Splatting densification，True=FastGS
        self.fastgs_densify = False           # 子开关：是否用 FastGS 多视角门控的 clone/split（False=退回标准梯度增殖）
        self.fastgs_prune = False             # 子开关：是否用 FastGS 加权预算 prune（False=退回标准 occupancy 阈值剔除）
        self.fastgs_protect_prune = False     # 保护开关：prune 时高边缘保护分的高斯降低被采样概率（传此 flag 才开启）
        self.fastgs_protect_densify = False   # 保护开关：densify 时高边缘保护分的高斯降低重要性阈值（传此 flag 才开启）
        self.fastgs_grad_thresh = 0.0002      # clone 判断：位置梯度阈值
        self.fastgs_grad_abs_thresh = 0.0002  # split 判断：尺寸梯度（scaling.grad）阈值
        self.fastgs_dense = 0.01              # 尺寸阈值因子（对应 percent_dense）
        self.fastgs_loss_thresh = 0.5         # metric_map：L1 归一化误差高于此值标记为高误差像素
        self.fastgs_min_importance = 5        # 多视角投票门控：至少被这么多视角认为重建差才参与增殖
        self.fastgs_num_cams = 10             # 每次 score 计算采样的相机数量
        # [FASTGS END]

        # [EDGE AWARE LOSS]
        self.lambda_edge_aware = 0.0          # 边缘感知 loss 权重（0=关闭）
        self.edge_aware_from_iter = 10000     # 从第几个 iter 开始施加

        # [GLOSSY PRIOR] 2D material priors -> alpha-weighted Gaussian scores.
        self.glossy_prior_on = False
        self.glossy_from_iter = 5000
        self.glossy_interval = 500
        self.glossy_num_cams = 8
        self.glossy_min_views = 3
        self.glossy_min_accum_weight = 1e-3
        self.glossy_ema = 0.8
        self.glossy_threshold = 0.15
        self.glossy_roughness_power = 2.0
        self.glossy_wavelet_levels = 2
        # Use a larger image-space wavelet footprint for relatively far pixels.
        # Depth only interpolates the observation scale; it never boosts the
        # final material score directly.
        self.glossy_wavelet_far_levels = 4
        self.glossy_depth_scale_start = 0.35
        self.glossy_depth_scale_end = 0.85
        self.glossy_wavelet_power = 1.0
        self.glossy_highlight_power = 1.0
        self.glossy_geometry_suppression = 2.0
        # Joint depth/normal/roughness filter for surface-consistent 2D scores.
        self.glossy_guided_filter_radius = 1
        self.glossy_guided_filter_iterations = 2
        self.glossy_guided_depth_sigma = 0.03
        self.glossy_guided_normal_sigma = 0.15
        self.glossy_guided_roughness_sigma = 0.08
        # Repair reflection-texture holes using the majority score on a local
        # normal/depth-consistent plane. Roughness is excluded from membership.
        # This threshold operates on the pre-color-gate 2D prior and is
        # therefore intentionally higher than the final Gaussian threshold.
        self.glossy_plane_consensus_threshold = 0.30
        self.glossy_plane_consensus_downsample = 8
        self.glossy_plane_consensus_radius = 5
        self.glossy_plane_consensus_iterations = 1
        self.glossy_plane_consensus_majority = 0.60
        self.glossy_plane_consensus_blend = 0.75
        self.glossy_plane_consensus_normal_sigma = 0.06
        self.glossy_plane_consensus_depth_sigma = 0.05
        self.glossy_plane_consensus_depth_floor = 0.50
        self.glossy_plane_consensus_min_support = 0.35
        self.glossy_color_var_scale = 20.0
        self.glossy_color_floor = 0.35
        # Correct multi-view RGB variance when the available camera rays span
        # only a small angle (common for far surfaces), with a strict cap.
        self.glossy_angle_reference_spread = 0.01
        self.glossy_angle_max_compensation = 4.0
        # Capture local probes once the Gaussian topology, global radiance and
        # glossy marker are stable. Captured cubemaps remain fixed afterwards.
        self.local_probe_from_iter = 30000

        self.gsrgb_loss = False
        self.init_until_iter = 0
        self.alpha_until_iter = -1
        super().__init__(parser, 'Optimization Parameters')


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = 'Namespace()'
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, 'cfg_args')
        print('Looking for config file in', cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print('Config file found: {}'.format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print('Config file not found at')
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
