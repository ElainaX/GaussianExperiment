import random
import torch
import torch.nn.functional as F
from gaussian_renderer import render


def _sample_cameras(viewpoint_stack, num_cams):
    """从 viewpoint_stack 中随机采样 num_cams 个相机（不放回）"""
    num_cams = min(num_cams, len(viewpoint_stack))
    indices = random.sample(range(len(viewpoint_stack)), num_cams)
    return [viewpoint_stack[i] for i in indices]


def _sobel_edge(img_gray_hw):
    """输入 [H,W] float tensor，输出归一化边缘强度 [H,W]，值域 [0,1]。"""
    k_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=img_gray_hw.device).view(1,1,3,3)
    k_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=img_gray_hw.device).view(1,1,3,3)
    g = img_gray_hw.view(1,1,*img_gray_hw.shape)
    ex = F.conv2d(g, k_x, padding=1)[0,0]
    ey = F.conv2d(g, k_y, padding=1)[0,0]
    edge = torch.sqrt(ex**2 + ey**2)
    return edge / (edge.max() + 1e-6)


def compute_gaussian_score_rtsplat(viewpoint_stack, gaussians, pipe, bg, opt, skip_importance=False):
    """计算每个高斯的多视角重建质量分数，用于指导 densification 和 pruning。

    对 opt.fastgs_num_cams 个随机相机视角渲染场景：
    - 先渲染得到图像，计算像素级 L1 误差并归一化，标出高误差像素 metric_map
    - 再用 metric_map 渲染，让 CUDA 统计每个高斯覆盖了多少个高误差像素（accum_counts）
    - 同时记录当前视角的光度误差 E_photo = L1 均值

    三个分数物理含义不同：
      importance_score = floor(Σ accum_counts / |cams|)
        → 纯计数，平均有多少视角认为该高斯重建差，用于 densification 门控（默认常驻）
      pruning_score = normalize(Σ E_photo × accum_counts)
        → E_photo 加权计数，重建误差大的视角权重更高，用于 pruning 采样权重
      protection_score = normalize(Σ accum_protection)
        → alpha加权的边缘×深度保护分，用于 prune 时降低背景轮廓高斯的被删概率

    Args:
        viewpoint_stack: 训练相机列表
        gaussians: GaussianModel
        pipe: 渲染 pipeline 参数
        bg: 背景颜色 tensor
        opt: OptimizationParams，使用 fastgs_num_cams / fastgs_loss_thresh 字段
        skip_importance (bool): True 时跳过 importance_score 的累积与计算，返回 None；
                                默认 False，即 importance_score 常驻计算

    Returns:
        importance_score (Tensor | None): [N] per-Gaussian 整数计数，skip_importance=True 时为 None
        pruning_score (Tensor): [N] 归一化到 [0,1] 的 E_photo 加权分数
        protection_score (Tensor): [N] 归一化到 [0,1] 的边缘×深度保护分
    """
    camlist = _sample_cameras(viewpoint_stack, opt.fastgs_num_cams)

    full_metric_counts = None   # Σ accum_counts（给 importance_score 用）
    full_metric_score  = None   # Σ E_photo × accum_counts（给 pruning_score 用）
    full_protection    = None   # Σ accum_protection（给 protection_score 用）

    with torch.no_grad():
        for cam in camlist:
            # ── 第一次渲染：得到图像、深度，计算高误差像素和保护权重 ───────────
            pkg = render(cam, gaussians, pipe, bg)
            rendered = pkg['final_rendering']
            gt = cam.original_image.cuda()

            # 像素级 L1
            l1_per_pixel = torch.mean(torch.abs(rendered - gt), dim=0)  # [H, W]
            l1_mean = l1_per_pixel.mean()
            if l1_mean < 1e-6:
                continue  # 该视角几乎完美，跳过

            e_photo = l1_mean  # 视角权重

            # 高误差像素标记（超过全图均值的才标记）
            metric_map = (l1_per_pixel > l1_mean).int().flatten()  # [H*W] int

            # 保护图：Sobel边缘强度 × 归一化深度
            # 高边缘+远深度 → 背景轮廓区域 → 减少被 prune 的概率
            depth = pkg['surface_depth'].squeeze()           # [H, W]
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            edge = _sobel_edge(gt.mean(dim=0))               # [H, W]
            protection_map = (edge * depth_norm).flatten()   # [H*W] float

            # ── 第二次渲染：传入 metric_map 和 protection_map ─────────────────
            pkg2 = render(cam, gaussians, pipe, bg,
                          metric_map=metric_map,
                          protection_map=protection_map)
            accum_counts      = pkg2['accum_metric_counts']  # [N] int
            accum_protect     = pkg2['accum_protection']     # [N] float

            # ── 累积各分数 ────────────────────────────────────────────────────
            if not skip_importance:
                full_metric_counts = accum_counts.float() if full_metric_counts is None \
                    else full_metric_counts + accum_counts.float()

            weighted = e_photo.item() * accum_counts.float()
            full_metric_score = weighted if full_metric_score is None \
                else full_metric_score + weighted

            full_protection = accum_protect.clone() if full_protection is None \
                else full_protection + accum_protect

    if full_metric_score is None:
        N = gaussians.get_xyz.shape[0]
        z = torch.zeros(N, device='cuda')
        return (None if skip_importance else z), z, z

    def _norm01(t):
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo + 1e-6)

    # pruning_score：E_photo 加权，归一化到 [0,1]
    pruning_score = _norm01(full_metric_score)

    # protection_score：边缘×深度保护分，归一化到 [0,1]
    protection_score = _norm01(full_protection)

    # importance_score：纯计数，按视角数取整均值（skip_importance=True 时跳过）
    if not skip_importance:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None

    return importance_score, pruning_score, protection_score


def _specular_score(img_chw):
    """GT 图高亮+低饱和+高局部对比 → 高光区域分数 [H,W]，值域 [0,1]。"""
    R, G, B = img_chw[0], img_chw[1], img_chw[2]
    L = 0.299 * R + 0.587 * G + 0.114 * B                          # 亮度
    max_c = img_chw.max(dim=0).values
    min_c = img_chw.min(dim=0).values
    sat = (max_c - min_c) / (max_c + 1e-6)                         # 饱和度
    L4 = L[None, None]
    local_mean = F.avg_pool2d(L4, 7, stride=1, padding=3)[0, 0]
    local_sq   = F.avg_pool2d(L4 ** 2, 7, stride=1, padding=3)[0, 0]
    contrast   = (local_sq - local_mean ** 2).clamp(min=0).sqrt()  # 局部对比度
    score = L * (1 - sat) * contrast
    return score / (score.max() + 1e-6)


def compute_reflection_score(viewpoint_stack, gaussians, pipe, bg, opt):
    """计算每个高斯的反射潜力分数，用于动态扩展 inside_mask。

    R(p) = w_e*E(p) + w_d*D(p) + w_s*S(p) + w_m*M(p)
      E(p): GT 图边缘强度（Sobel）
      D(p): 归一化深度（远处背景轮廓权重高）
      S(p): GT 图高亮+低饱和+高局部对比（高光区域）
      M(p): 光度误差 × GT 亮度（模型未能捕捉的高光残差）

    R(p) 通过 surface pass 的 σ 加权 atomicAdd 聚合到每个高斯上，
    最终归一化到 [0,1]。

    Returns:
        refl_score: [N] float tensor，值域 [0,1]
    """
    camlist = _sample_cameras(viewpoint_stack, opt.fastgs_num_cams)
    full_refl = None

    with torch.no_grad():
        for cam in camlist:
            pkg    = render(cam, gaussians, pipe, bg)
            rendered = pkg['final_rendering']
            depth    = pkg['surface_depth'].squeeze()
            gt       = cam.original_image.cuda()

            E = _sobel_edge(gt.mean(dim=0))                                   # [H,W]
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            D = depth_norm                                                     # 远=1
            S = _specular_score(gt)                                            # [H,W]
            L_gt = gt.mean(dim=0)
            err  = torch.abs(rendered - gt).mean(dim=0)
            M    = err * L_gt
            M    = M / (M.max() + 1e-6)

            R = (opt.refl_weight_e * E + opt.refl_weight_d * D +
                 opt.refl_weight_s * S + opt.refl_weight_m * M)
            R = R / (R.max() + 1e-6)

            # 第二次渲染：用 σ 加权 atomicAdd 将 R(p) 聚合到每个高斯
            pkg2      = render(cam, gaussians, pipe, bg, protection_map=R.flatten())
            accum_r   = pkg2['accum_protection']  # [N]
            full_refl = accum_r.clone() if full_refl is None else full_refl + accum_r

    if full_refl is None:
        return torch.zeros(gaussians.get_xyz.shape[0], device='cuda')

    lo, hi = full_refl.min(), full_refl.max()
    return (full_refl - lo) / (hi - lo + 1e-6)


def edge_aware_loss(rendered, gt, depth):
    """边缘感知 loss：对齐渲染图与 GT 图的「边缘强度 × 归一化深度」分布。

    边缘强度由 Sobel 算子计算，深度作为空间权重让损失聚焦于有几何意义的边缘区域
    （远景背景轮廓权重高，近景内部纹理权重低）。

    Args:
        rendered : [3, H, W] float，渲染图（需要梯度）
        gt       : [3, H, W] float，GT 原图（no grad）
        depth    : [1, H, W] 或 [H, W] float，渲染深度（detach 后用作权重）

    Returns:
        scalar loss（MSE）
    """
    depth = depth.squeeze().detach()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    rendered_edge = _sobel_edge(rendered.mean(dim=0))        # [H, W]，有梯度
    gt_edge       = _sobel_edge(gt.mean(dim=0).detach())     # [H, W]，无梯度

    rendered_map = rendered_edge * depth_norm
    gt_map       = gt_edge * depth_norm

    return F.mse_loss(rendered_map, gt_map)
