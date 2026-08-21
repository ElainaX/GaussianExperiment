import math
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


def _smoothstep(tensor, edge0, edge1):
    """Smoothly map ``tensor`` from [edge0, edge1] to [0, 1]."""
    edge0 = float(edge0)
    edge1 = max(float(edge1), edge0 + 1e-6)
    value = ((tensor - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


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


def _surface_guided_filter(
    score,
    depth,
    normal,
    roughness,
    radius=1,
    iterations=1,
    depth_sigma=0.03,
    normal_sigma=0.15,
    roughness_sigma=0.08,
):
    """Denoise a score without crossing depth/material boundaries.

    Absolute depth is deliberately not part of the weight.  Only local depth
    continuity, normal agreement and roughness agreement determine whether two
    neighboring pixels may exchange evidence.  Consequently a continuous
    plane is treated consistently even when its camera-space depth changes.
    """
    radius = max(0, int(radius))
    iterations = max(0, int(iterations))
    if radius == 0 or iterations == 0:
        return score

    height, width = score.shape
    depth_sigma = max(float(depth_sigma), 1e-6)
    normal_sigma = max(float(normal_sigma), 1e-6)
    roughness_sigma = max(float(roughness_sigma), 1e-6)

    depth_4d = depth[None, None]
    normal_4d = normal[None]
    roughness_4d = roughness[None, None]
    padded_depth = F.pad(depth_4d, (radius,) * 4, mode='replicate')
    padded_normal = F.pad(normal_4d, (radius,) * 4, mode='replicate')
    padded_roughness = F.pad(roughness_4d, (radius,) * 4, mode='replicate')

    filtered = score
    spatial_sigma_sq = max(float(radius * radius), 1.0)
    for _ in range(iterations):
        padded_score = F.pad(filtered[None, None], (radius,) * 4, mode='replicate')
        numerator = torch.zeros_like(filtered)
        denominator = torch.zeros_like(filtered)
        for offset_y in range(-radius, radius + 1):
            y0 = radius + offset_y
            for offset_x in range(-radius, radius + 1):
                x0 = radius + offset_x
                neighbor_score = padded_score[0, 0, y0:y0 + height, x0:x0 + width]
                neighbor_depth = padded_depth[0, 0, y0:y0 + height, x0:x0 + width]
                neighbor_normal = padded_normal[0, :, y0:y0 + height, x0:x0 + width]
                neighbor_roughness = padded_roughness[0, 0, y0:y0 + height, x0:x0 + width]

                depth_delta = (depth - neighbor_depth).abs()
                normal_delta = 1.0 - (normal * neighbor_normal).sum(dim=0).clamp(-1.0, 1.0)
                roughness_delta = (roughness - neighbor_roughness).abs()
                spatial = math.exp(
                    -(offset_x * offset_x + offset_y * offset_y) /
                    (2.0 * spatial_sigma_sq)
                )
                weight = spatial * torch.exp(
                    -depth_delta / depth_sigma
                    -normal_delta / normal_sigma
                    -roughness_delta / roughness_sigma
                )
                numerator += weight * neighbor_score
                denominator += weight
        filtered = numerator / denominator.clamp_min(1e-8)
    return filtered.clamp(0.0, 1.0)


def _plane_majority_consensus(
    score,
    depth,
    normal,
    threshold=0.30,
    downsample=8,
    radius=5,
    iterations=1,
    majority=0.60,
    blend=0.75,
    normal_sigma=0.06,
    depth_sigma=0.05,
    depth_weight_floor=0.50,
    min_support=0.35,
):
    """Repair low-score holes when a planar neighborhood votes glossy.

    Reflected content may make an image prior predict a tree or building rather
    than the physical glass surface.  Roughness is intentionally excluded from
    plane membership here because it is precisely the potentially corrupted
    cue.  Normal agreement is the main guide, while depth is only a soft
    boundary so an erroneous reflected depth cannot completely isolate a hole.

    The operation is asymmetric: it only raises low outliers when a clear
    high-score majority exists.  This avoids erasing small true mirrors merely
    because a larger diffuse surface lies nearby.
    """
    downsample = max(1, int(downsample))
    radius = max(0, int(radius))
    iterations = max(0, int(iterations))
    if radius == 0 or iterations == 0:
        zeros = torch.zeros_like(score)
        return score, zeros, zeros

    majority = min(max(float(majority), 0.5), 0.999)
    blend = min(max(float(blend), 0.0), 1.0)
    normal_sigma = max(float(normal_sigma), 1e-6)
    depth_sigma = max(float(depth_sigma), 1e-6)
    depth_weight_floor = min(max(float(depth_weight_floor), 0.0), 1.0)
    min_support = min(max(float(min_support), 0.0), 0.999)
    threshold = min(max(float(threshold), 0.0), 1.0)

    height, width = score.shape
    small_size = (
        max(1, (height + downsample - 1) // downsample),
        max(1, (width + downsample - 1) // downsample),
    )
    small_score = F.interpolate(
        score[None, None], size=small_size, mode='area'
    )[0, 0]
    small_depth = F.interpolate(
        depth[None, None], size=small_size, mode='area'
    )[0, 0]
    small_normal = F.interpolate(
        normal[None], size=small_size, mode='area'
    )[0]
    small_normal = F.normalize(small_normal, dim=0, eps=1e-6)

    small_height, small_width = small_score.shape
    padded_depth = F.pad(
        small_depth[None, None], (radius,) * 4, mode='replicate'
    )
    padded_normal = F.pad(
        small_normal[None], (radius,) * 4, mode='replicate'
    )
    spatial_sigma_sq = max(float(radius * radius), 1.0)
    possible_support = 0.0
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            possible_support += math.exp(
                -(offset_x * offset_x + offset_y * offset_y) /
                (2.0 * spatial_sigma_sq)
            )

    original_small_score = small_score
    consensus_confidence = torch.zeros_like(small_score)
    for _ in range(iterations):
        padded_score = F.pad(
            small_score[None, None], (radius,) * 4, mode='replicate'
        )
        support_weight = torch.zeros_like(small_score)
        high_weight = torch.zeros_like(small_score)
        high_score_sum = torch.zeros_like(small_score)

        for offset_y in range(-radius, radius + 1):
            y0 = radius + offset_y
            for offset_x in range(-radius, radius + 1):
                x0 = radius + offset_x
                neighbor_score = padded_score[
                    0, 0, y0:y0 + small_height, x0:x0 + small_width
                ]
                neighbor_depth = padded_depth[
                    0, 0, y0:y0 + small_height, x0:x0 + small_width
                ]
                neighbor_normal = padded_normal[
                    0, :, y0:y0 + small_height, x0:x0 + small_width
                ]

                normal_delta = 1.0 - (
                    small_normal * neighbor_normal
                ).sum(dim=0).clamp(-1.0, 1.0)
                depth_delta = (small_depth - neighbor_depth).abs()
                spatial = math.exp(
                    -(offset_x * offset_x + offset_y * offset_y) /
                    (2.0 * spatial_sigma_sq)
                )
                normal_weight = torch.exp(-normal_delta / normal_sigma)
                depth_weight = depth_weight_floor + (1.0 - depth_weight_floor) * torch.exp(
                    -depth_delta / depth_sigma
                )
                plane_weight = spatial * normal_weight * depth_weight
                is_high = (neighbor_score >= threshold).to(neighbor_score.dtype)

                support_weight += plane_weight
                high_weight += plane_weight * is_high
                high_score_sum += plane_weight * is_high * neighbor_score

        high_fraction = high_weight / support_weight.clamp_min(1e-8)
        support_fraction = support_weight / max(possible_support, 1e-8)
        has_consensus = (
            (high_fraction >= majority) &
            (support_fraction >= min_support)
        )
        high_target = high_score_sum / high_weight.clamp_min(1e-8)
        repair = has_consensus & (small_score < high_target)
        consensus_confidence = torch.maximum(
            consensus_confidence,
            repair.to(small_score.dtype) * high_fraction,
        )
        small_score = torch.where(
            repair,
            torch.lerp(small_score, high_target, blend),
            small_score,
        )

    small_delta = (small_score - original_small_score).clamp_min(0.0)
    delta = F.interpolate(
        small_delta[None, None], size=(height, width),
        mode='bilinear', align_corners=False,
    )[0, 0]
    confidence = F.interpolate(
        consensus_confidence[None, None], size=(height, width),
        mode='bilinear', align_corners=False,
    )[0, 0]
    corrected = (score + delta).clamp(0.0, 1.0)
    return corrected, delta, confidence


def normal_prior_supervision(
    camera,
    rendered_world_normal,
    foreground,
    surface_alpha,
    axis_sign=(-1.0, 1.0, 1.0),
    edge_suppression=2.0,
    min_alpha=0.05,
    pool_size=3,
):
    """Compare rendered normals with an aligned camera-space normal prior.

    RT-Splatting stores/render normals in world space, while the generated
    normal maps are camera-space. The same world-to-camera conversion used by
    mesh visualization is applied here. Truck's priors use the opposite image
    X convention, represented by the configurable default sign [-1, 1, 1].
    """
    prior_camera = camera.load_normal_prior(device=rendered_world_normal.device)
    if prior_camera is None:
        return None

    sign = rendered_world_normal.new_tensor([float(value) for value in axis_sign])
    if sign.numel() != 3:
        raise ValueError('normal_prior_axis_sign must contain exactly three values')
    sign = sign.reshape(3, 1, 1)
    aligned_prior = F.normalize(prior_camera * sign, dim=0, eps=1e-6)
    rendered_camera = -(
        rendered_world_normal.movedim(0, -1) @ camera.world_view_transform[:3, :3]
    ).movedim(-1, 0)
    rendered_camera = F.normalize(rendered_camera, dim=0, eps=1e-6)

    normal_edge = torch.stack([
        _sobel_edge(aligned_prior[channel]) for channel in range(3)
    ]).mean(dim=0, keepdim=True)
    edge_confidence = torch.exp(-float(edge_suppression) * normal_edge)
    prior_valid = (
        aligned_prior.square().sum(dim=0, keepdim=True) > 0.5
    ).to(aligned_prior.dtype)
    alpha_confidence = surface_alpha.detach().clamp(0.0, 1.0)
    alpha_confidence = alpha_confidence * (
        alpha_confidence >= float(min_alpha)
    ).to(alpha_confidence.dtype)
    pixel_confidence = (
        foreground.detach().clamp(0.0, 1.0) *
        alpha_confidence * prior_valid * edge_confidence
    )

    # Compare regional mean directions, not a blurred per-pixel loss. Merely
    # averaging the scalar loss would leave its global mean almost unchanged.
    # Pooling the normal vectors first allows local high-frequency detail while
    # constraining the overall orientation of each overlapping neighborhood.
    pool_size = max(1, int(pool_size))
    if pool_size % 2 == 0:
        pool_size += 1
    if pool_size > 1:
        padding = pool_size // 2
        pooled_confidence = F.avg_pool2d(
            pixel_confidence[None],
            kernel_size=pool_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )[0]
        pooled_prior_raw = F.avg_pool2d(
            (aligned_prior * pixel_confidence)[None],
            kernel_size=pool_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )[0] / pooled_confidence.clamp_min(1e-8)
        pooled_rendered_raw = F.avg_pool2d(
            (rendered_camera * pixel_confidence)[None],
            kernel_size=pool_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )[0] / pooled_confidence.clamp_min(1e-8)
        pooled_valid = (
            (pooled_prior_raw.norm(dim=0, keepdim=True) > 0.2) &
            (pooled_rendered_raw.norm(dim=0, keepdim=True) > 0.2) &
            (pooled_confidence > 1e-4)
        ).to(pixel_confidence.dtype)
        compared_prior = F.normalize(pooled_prior_raw, dim=0, eps=1e-6)
        compared_rendered = F.normalize(pooled_rendered_raw, dim=0, eps=1e-6)
        confidence = pooled_confidence * pooled_valid
    else:
        compared_prior = aligned_prior
        compared_rendered = rendered_camera
        confidence = pixel_confidence

    cosine = (
        compared_rendered * compared_prior
    ).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    error = 1.0 - cosine
    loss = (error * confidence).sum() / confidence.sum().clamp_min(1e-8)
    return {
        'loss': loss,
        'error': error,
        'confidence': confidence,
        'prior_camera': aligned_prior,
        'rendered_camera': rendered_camera,
        'compared_prior': compared_prior,
        'compared_rendered': compared_rendered,
    }


@torch.no_grad()
def build_prior_glossy_map(camera, opt, device='cuda', return_details=False):
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

    # ``render.py`` reconstructs its Namespace from ``cfg_args``, which only
    # contains ModelParams in existing checkpoints.  Keep every visualization
    # option backward-compatible so an already trained model can be rendered
    # without requiring its OptimizationParams to be present.
    roughness_threshold = min(max(float(getattr(
        opt, 'glossy_prior_roughness_threshold', 0.45
    )), 0.0), 1.0)
    roughness_gate = (roughness < roughness_threshold).to(roughness.dtype)
    roughness_score = (1.0 - roughness).pow(
        float(getattr(opt, 'glossy_roughness_power', 2.0))
    ) * roughness_gate
    near_levels = int(getattr(opt, 'glossy_wavelet_levels', 2))
    far_levels = int(getattr(opt, 'glossy_wavelet_far_levels', 4))
    wavelet_near = _haar_detail(gray, near_levels)
    wavelet_far = _haar_detail(gray, max(near_levels, far_levels))
    depth_scale_weight = _smoothstep(
        depth,
        getattr(opt, 'glossy_depth_scale_start', 0.35),
        getattr(opt, 'glossy_depth_scale_end', 0.85),
    )
    # Depth changes the image-space receptive field, never the material score
    # directly.  Far pixels use coarser Haar evidence because the same physical
    # feature covers fewer pixels there.
    wavelet_score = torch.lerp(wavelet_near, wavelet_far, depth_scale_weight).pow(
        float(getattr(opt, 'glossy_wavelet_power', 1.0))
    )
    highlight_score = _normalize01(gray).pow(
        float(getattr(opt, 'glossy_highlight_power', 1.0))
    )

    depth_edge = _sobel_edge(depth)
    normal_edge = torch.stack([_sobel_edge(normal[channel]) for channel in range(3)]).mean(dim=0)
    geometry_edge = (depth_edge + normal_edge).clamp(0.0, 1.0)
    geometry_confidence = torch.exp(
        -float(getattr(opt, 'glossy_geometry_suppression', 2.0)) * geometry_edge
    )

    # Low roughness remains the main gate. Haar detail and brightness strengthen
    # the evidence without suppressing broad, smooth reflections entirely.
    appearance_evidence = 0.35 + 0.35 * wavelet_score + 0.30 * highlight_score
    score_before_guided = roughness_score * appearance_evidence * geometry_confidence
    score_before_guided = torch.nan_to_num(
        score_before_guided, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    score_after_guided = _surface_guided_filter(
        score_before_guided,
        depth,
        normal,
        roughness,
        radius=getattr(opt, 'glossy_guided_filter_radius', 1),
        iterations=getattr(opt, 'glossy_guided_filter_iterations', 2),
        depth_sigma=getattr(opt, 'glossy_guided_depth_sigma', 0.03),
        normal_sigma=getattr(opt, 'glossy_guided_normal_sigma', 0.15),
        roughness_sigma=getattr(opt, 'glossy_guided_roughness_sigma', 0.08),
    )
    score_after_guided = torch.nan_to_num(
        score_after_guided, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0) * roughness_gate
    score, plane_consensus_delta, plane_consensus_confidence = _plane_majority_consensus(
        score_after_guided,
        depth,
        normal,
        threshold=getattr(opt, 'glossy_plane_consensus_threshold', 0.30),
        downsample=getattr(opt, 'glossy_plane_consensus_downsample', 8),
        radius=getattr(opt, 'glossy_plane_consensus_radius', 5),
        iterations=getattr(opt, 'glossy_plane_consensus_iterations', 1),
        majority=getattr(opt, 'glossy_plane_consensus_majority', 0.60),
        blend=getattr(opt, 'glossy_plane_consensus_blend', 0.75),
        normal_sigma=getattr(opt, 'glossy_plane_consensus_normal_sigma', 0.06),
        depth_sigma=getattr(opt, 'glossy_plane_consensus_depth_sigma', 0.05),
        depth_weight_floor=getattr(opt, 'glossy_plane_consensus_depth_floor', 0.50),
        min_support=getattr(opt, 'glossy_plane_consensus_min_support', 0.35),
    )
    # Plane consensus is allowed to repair reflected-texture holes only inside
    # the material gate. High-roughness pixels remain exactly zero and can
    # never be accumulated onto a Gaussian by the multi-view rasterizer.
    score = (
        torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        .clamp(0.0, 1.0) * roughness_gate
    )
    plane_consensus_delta = plane_consensus_delta * roughness_gate
    plane_consensus_confidence = plane_consensus_confidence * roughness_gate
    if not return_details:
        return score
    return {
        'score': score,
        'score_before_guided': score_before_guided,
        'score_after_guided': score_after_guided,
        'plane_consensus_delta': plane_consensus_delta,
        'plane_consensus_confidence': plane_consensus_confidence,
        'wavelet_near': wavelet_near,
        'wavelet_far': wavelet_far,
        'wavelet_adaptive': wavelet_score,
        'depth_scale_weight': depth_scale_weight,
        'geometry_confidence': geometry_confidence,
        'roughness_gate': roughness_gate,
    }


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
    view_direction_sum = torch.zeros(num_gaussians, 3, device=device)
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
        visible_directions = F.normalize(
            camera.camera_center[None] - gaussians.get_xyz[valid], dim=-1, eps=1e-6
        )
        view_direction_sum[valid] += visible_directions
        view_count[valid] += 1
        used_cameras += 1

    if used_cameras == 0:
        return None

    prior_score = (prior_numerator / total_weight.clamp_min(1e-8)).clamp(0.0, 1.0)
    mean_color = color_sum / view_count.clamp_min(1.0)[:, None]
    color_variance = color_sq_sum / view_count.clamp_min(1.0)[:, None] - mean_color.square()
    color_variance = color_variance.clamp_min(0.0).mean(dim=-1)
    mean_view_direction = view_direction_sum / view_count.clamp_min(1.0)[:, None]
    # For unit view directions, 1-|mean(d)|^2 is their directional variance.
    # Distant surfaces commonly have a smaller view-angle baseline, so their
    # raw RGB variance is corrected for observation geometry rather than being
    # boosted merely because their depth is large.
    view_angle_spread = (
        1.0 - mean_view_direction.square().sum(dim=-1)
    ).clamp(0.0, 1.0)
    reference_spread = max(float(getattr(opt, 'glossy_angle_reference_spread', 0.01)), 0.0)
    max_angle_compensation = max(
        float(getattr(opt, 'glossy_angle_max_compensation', 4.0)), 1.0
    )
    angle_compensation = torch.ones_like(view_angle_spread)
    enough_angles = view_count >= 2
    if reference_spread > 0.0 and enough_angles.any():
        minimum_spread = reference_spread / max_angle_compensation
        angle_compensation[enough_angles] = (
            reference_spread /
            view_angle_spread[enough_angles].clamp_min(minimum_spread)
        ).clamp(1.0, max_angle_compensation)
    corrected_color_variance = color_variance * angle_compensation
    color_score = 1.0 - torch.exp(
        -float(opt.glossy_color_var_scale) * corrected_color_variance
    )
    confidence = (view_count / max(1, int(opt.glossy_min_views))).clamp(0.0, 1.0)
    color_gate = float(opt.glossy_color_floor) + (1.0 - float(opt.glossy_color_floor)) * color_score
    fused_score = (prior_score * color_gate * confidence).clamp(0.0, 1.0)

    return {
        'fused': fused_score,
        'prior': prior_score,
        'color_variation': color_score,
        'color_variation_raw': color_variance,
        'view_angle_spread': view_angle_spread,
        'angle_compensation': angle_compensation,
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
            pkg = render(
                cam, gaussians, pipe, bg, enable_secondary_raytrace=False
            )
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
                          protection_map=protection_map,
                          enable_secondary_raytrace=False)
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
