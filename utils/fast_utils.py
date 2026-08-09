import random
import torch
import torch.nn.functional as F
from gaussian_renderer import accumulate_gaussian_map, render


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
    edge = torch.sqrt(ex**2 + ey**2 + 1e-8)  # epsilon inside sqrt prevents NaN gradient at flat regions
    return edge / (edge.max() + 1e-6)


def _normalize01(tensor):
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = tensor.amin(), tensor.amax()
    return (tensor - lo) / (hi - lo + 1e-6)


def _haar_detail(gray_hw, levels=2):
    """Return a normalized multi-level Haar detail-energy map in [0, 1]."""
    height, width = gray_hw.shape
    current = gray_hw[None, None]
    detail_full = torch.zeros_like(current)
    for level in range(max(1, int(levels))):
        if min(current.shape[-2:]) < 2:
            break
        pad_h = current.shape[-2] % 2
        pad_w = current.shape[-1] % 2
        padded = F.pad(current, (0, pad_w, 0, pad_h), mode='replicate')
        a = padded[..., 0::2, 0::2]
        b = padded[..., 0::2, 1::2]
        c = padded[..., 1::2, 0::2]
        d = padded[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5
        detail = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-8)
        detail_full += F.interpolate(
            detail, size=(height, width), mode='bilinear', align_corners=False
        ) / float(2**level)
        current = ll
    return _normalize01(detail_full[0, 0])


@torch.no_grad()
def build_prior_glossy_map(camera, opt, device='cuda'):
    """Build a per-view 2D glossy probability from RGB/material priors.

    Low roughness and Haar/RGB highlight evidence increase the score. Depth and
    normal discontinuities suppress ordinary geometry edges that would otherwise
    be mistaken for specular high frequencies.
    """
    if not camera.has_image_priors:
        return None
    priors = camera.load_image_priors(device=device)
    if any(priors[name] is None for name in ('depth', 'normal', 'roughness')):
        return None

    rgb = camera.original_image[:3].to(device=device, dtype=torch.float32)
    gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(dim=0)
    roughness = priors['roughness'][0].clamp(0.0, 1.0)
    depth = _normalize01(priors['depth'][0])
    normal = priors['normal']

    roughness_score = (1.0 - roughness).pow(float(opt.glossy_roughness_power))
    wavelet_score = _haar_detail(gray, opt.glossy_wavelet_levels).pow(
        float(opt.glossy_wavelet_power)
    )
    highlight_score = _normalize01(gray).pow(float(opt.glossy_highlight_power))

    depth_edge = _sobel_edge(depth)
    normal_edge = torch.stack([_sobel_edge(normal[channel]) for channel in range(3)]).mean(dim=0)
    geometry_edge = (depth_edge + normal_edge).clamp(0.0, 1.0)
    geometry_confidence = torch.exp(
        -float(opt.glossy_geometry_suppression) * geometry_edge
    )

    # Low roughness remains the main gate. Haar detail and brightness strengthen
    # the evidence without suppressing broad, smooth reflections entirely.
    appearance_evidence = 0.35 + 0.35 * wavelet_score + 0.30 * highlight_score
    score = roughness_score * appearance_evidence * geometry_confidence
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


@torch.no_grad()
def compute_gaussian_glossy_score(viewpoint_stack, gaussians, pipe, bg, opt):
    """Fuse 2D glossy priors and cross-view RGB variation per Gaussian.

    Each scalar image is accumulated by the CUDA rasterizer with the same
    alpha*transmittance contribution used for surface compositing. RGB is first
    reduced to one observation per Gaussian per view, then its cross-view
    variance is combined with the material prior.
    """
    eligible = [camera for camera in viewpoint_stack if camera.has_image_priors]
    if not eligible:
        return None
    camlist = _sample_cameras(eligible, opt.glossy_num_cams)
    num_gaussians = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    prior_numerator = torch.zeros(num_gaussians, device=device)
    total_weight = torch.zeros(num_gaussians, device=device)
    color_sum = torch.zeros(num_gaussians, 3, device=device)
    color_sq_sum = torch.zeros(num_gaussians, 3, device=device)
    view_count = torch.zeros(num_gaussians, device=device)
    used_cameras = 0

    for camera in camlist:
        glossy_map = build_prior_glossy_map(camera, opt, device=device)
        if glossy_map is None:
            continue
        rgb = camera.original_image[:3].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        weight, _ = accumulate_gaussian_map(camera, gaussians, bg, torch.ones_like(glossy_map))
        glossy_accum, _ = accumulate_gaussian_map(camera, gaussians, bg, glossy_map)
        rgb_accum = torch.stack(
            [accumulate_gaussian_map(camera, gaussians, bg, rgb[channel])[0] for channel in range(3)],
            dim=-1,
        )

        valid = weight > float(opt.glossy_min_accum_weight)
        if not valid.any():
            continue
        observation = rgb_accum / weight.clamp_min(1e-8)[:, None]
        prior_numerator += glossy_accum
        total_weight += weight
        color_sum[valid] += observation[valid]
        color_sq_sum[valid] += observation[valid].square()
        view_count[valid] += 1
        used_cameras += 1

    if used_cameras == 0:
        return None

    prior_score = (prior_numerator / total_weight.clamp_min(1e-8)).clamp(0.0, 1.0)
    mean_color = color_sum / view_count.clamp_min(1.0)[:, None]
    color_variance = color_sq_sum / view_count.clamp_min(1.0)[:, None] - mean_color.square()
    color_variance = color_variance.clamp_min(0.0).mean(dim=-1)
    color_score = 1.0 - torch.exp(-float(opt.glossy_color_var_scale) * color_variance)
    confidence = (view_count / max(1, int(opt.glossy_min_views))).clamp(0.0, 1.0)
    color_gate = float(opt.glossy_color_floor) + (1.0 - float(opt.glossy_color_floor)) * color_score
    fused_score = (prior_score * color_gate * confidence).clamp(0.0, 1.0)

    return {
        'fused': fused_score,
        'prior': prior_score,
        'color_variation': color_score,
        'confidence': confidence,
        'view_count': view_count,
        'used_cameras': used_cameras,
    }


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
    depth = torch.nan_to_num(depth.squeeze().detach(), nan=0.0, posinf=0.0, neginf=0.0)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    rendered_edge = _sobel_edge(rendered.mean(dim=0))        # [H, W]，有梯度
    gt_edge       = _sobel_edge(gt.mean(dim=0).detach())     # [H, W]，无梯度

    rendered_map = rendered_edge * depth_norm
    gt_map       = gt_edge * depth_norm

    return F.mse_loss(rendered_map, gt_map)
