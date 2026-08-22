import math

import numpy as np
import torch
import torch.nn.functional as F
from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.camera_utils import *
from utils.color_utils import *
from utils.general_utils import *
from utils.point_utils import *
from utils.sph_utils import *


@torch.no_grad()
def accumulate_gaussian_map(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, value_map: torch.Tensor):
    """Alpha-composite a scalar image map back onto the visible Gaussians.

    The CUDA rasterizer accumulates ``value_map[pixel] * alpha * transmittance``
    for every contributing Gaussian. Calling this once with an all-ones map
    provides the matching normalization weight. This lightweight path performs
    only the surface rasterization and skips RT-Splatting's volume/PBR passes.
    """
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
    if value_map.numel() != image_height * image_width:
        raise ValueError(
            f'value_map has {value_map.numel()} elements, expected '
            f'{image_height}x{image_width}'
        )

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device='cuda')
    _, _, radii, _, _, accumulated = rasterizer(
        means3D=pc.get_xyz,
        means2D=means2D,
        shs=pc.get_features,
        extras=None,
        opacities=pc.get_occupancy,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
        protection_map=value_map.reshape(-1).to(device='cuda', dtype=torch.float32).contiguous(),
    )
    return accumulated, radii > 0


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, metric_map=None, protection_map=None,
           enable_secondary_raytrace=True):  # [FASTGS]
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    occupancy = pc.get_occupancy
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # Forward pass

    extras = torch.cat([opacity], dim=-1)

    render_tran, volume_extras, _, volume_allmap, _, _ = rasterizer(  # [FASTGS] 末两位为 accum_metric_counts/accum_protection，volume pass 不使用
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        extras=extras,
        opacities=occupancy * opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    (volume_opacity,) = volume_extras.split([1], dim=0)

    volume_alpha = volume_allmap[1:2]

    volume_normal = volume_allmap[2:5]
    volume_normal = (volume_normal.movedim(0, -1) @ (viewpoint_camera.world_view_transform[:3, :3].T)).movedim(-1, 0)
    volume_normal = F.normalize(volume_normal, dim=0)

    volume_depth_median = volume_allmap[5:6]
    volume_depth_median = torch.nan_to_num(volume_depth_median, 0, 0)

    volume_depth_expected = volume_allmap[0:1]
    volume_depth_expected = volume_depth_expected / volume_alpha
    volume_depth_expected = torch.nan_to_num(volume_depth_expected, 0, 0)

    volume_depth = volume_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * volume_depth_median

    volume_depth_normal = depth_to_normal_sobel(viewpoint_camera, volume_depth.movedim(0, -1)).movedim(-1, 0)
    volume_depth_normal = volume_depth_normal * volume_alpha.detach()

    volume_dist = volume_allmap[6:7]

    # Deferred pass

    inside_mask = pc.get_inside_mask.float()
    extras = torch.cat([pc.get_roughness, pc.get_language_feature,
                        inside_mask, inside_mask * pc.get_reflectance,
                        opacity, inside_mask * pc.get_transmissivity,
                        pc.get_prior_glossy_score], dim=-1)

    # [FASTGS] surface pass：metric_map 用于高误差计数，protection_map 用于边缘保护分
    render_scat, surface_extras, radii, surface_allmap, accum_metric_counts, accum_protection = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        extras=extras,
        opacities=occupancy,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        metric_map=metric_map,
        protection_map=protection_map,
    )

    render_roughness, render_feature, foreground, render_reflectance, surface_opacity, render_transmissivity, render_glossy_score = surface_extras.split([1, 4, 1, 1, 1, 1, 1], dim=0)

    foreground = foreground.detach()

    surface_alpha = surface_allmap[1:2]

    surface_normal = surface_allmap[2:5]
    surface_normal = (surface_normal.movedim(0, -1) @ (viewpoint_camera.world_view_transform[:3, :3].T)).movedim(-1, 0)
    surface_normal = F.normalize(surface_normal, dim=0)

    surface_depth_median = surface_allmap[5:6]
    surface_depth_median = torch.nan_to_num(surface_depth_median, 0, 0)

    surface_depth_expected = surface_allmap[0:1]
    surface_depth_expected = surface_depth_expected / surface_alpha
    surface_depth_expected = torch.nan_to_num(surface_depth_expected, 0, 0)

    surface_depth = surface_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * surface_depth_median
    rays_o, viewdirs = camera_rays(viewpoint_camera)
    render_position = (
        surface_depth.movedim(0, -1) * viewdirs + rays_o
    ).movedim(-1, 0)

    surface_depth_normal = depth_to_normal_sobel(viewpoint_camera, surface_depth.movedim(0, -1)).movedim(-1, 0)
    surface_depth_normal = surface_depth_normal * surface_alpha.detach()

    surface_dist = surface_allmap[6:7]

    #####################################################################################################################

    with torch.no_grad():
        select_index = (foreground.flatten() > 0.05).nonzero(as_tuple=True)[0]

    render_spec = torch.zeros(3, image_height, image_width).cuda()
    render_attenuation = torch.zeros(1, image_height, image_width).cuda()
    secondary_radiance = torch.zeros(3, image_height, image_width).cuda()
    secondary_gate = torch.zeros(1, image_height, image_width).cuda()
    secondary_hit_opacity = torch.zeros(1, image_height, image_width).cuda()
    secondary_reliability = torch.zeros(1, image_height, image_width).cuda()

    viewdirs = F.normalize(viewdirs, dim=-1)
    normal_map = surface_normal.movedim(0, -1)
    wo = F.normalize(reflect(-viewdirs, normal_map), dim=-1)

    if len(select_index) > 0:
        wo = wo.reshape(-1, 3)[select_index]
        normal_map = normal_map.reshape(-1, 3)[select_index]
        roughness_map = render_roughness.movedim(0, -1).reshape(-1, 1)[select_index]

        feature_map = render_feature.movedim(0, -1).reshape(-1, pc.gsfeat_dim)[select_index]
        feature_map = F.normalize(feature_map, dim=-1)

        feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
        feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)

        """ Sph-Mip """
        wo_xy = (cart2sph(wo.reshape(-1, 3)[..., pc.XYZ])[..., 1:] / torch.Tensor([[np.pi, 2 * np.pi]]).cuda())[..., [1, 0]]
        wo_xyz = torch.stack(
            [wo_xy[:, None, :]],
            dim=0,
        )

        spec_level = roughness_map.reshape(-1, 1)

        spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
        spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
        spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)

        # Specular color
        wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim * pc.gsfeat_dim)
        input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
        mlp_output = pc.light_mlp(input_mlp)
        spec_light = torch.exp(mlp_output[..., :3] + np.log(0.5))
        spec_attenuation = torch.sigmoid(mlp_output[..., 3:4])

        training_iteration = getattr(pipe, 'training_iteration', None)
        raytrace_ready = (
            pc.secondary_raytrace_on and
            enable_secondary_raytrace and
            not getattr(pipe, 'disable_secondary_raytrace', False) and
            (
                training_iteration is None or
                training_iteration >= pc.secondary_raytrace_from_iter
            )
        )
        if raytrace_ready:
            glossy_score = render_glossy_score.reshape(-1, 1)[select_index].detach()
            gate_range = max(
                pc.secondary_raytrace_glossy_high -
                pc.secondary_raytrace_glossy_low,
                1e-6,
            )
            ray_gate = (
                (glossy_score - pc.secondary_raytrace_glossy_low) / gate_range
            ).clamp(0.0, 1.0)
            ray_gate = ray_gate.square() * (3.0 - 2.0 * ray_gate)
            ray_gate = ray_gate * (
                roughness_map.detach() <= pc.secondary_raytrace_roughness_max
            ).to(ray_gate.dtype)
            ray_local_index = (ray_gate[:, 0] > 0.0).nonzero(
                as_tuple=True
            )[0]
            if ray_local_index.numel() > 0:
                position_map = render_position.movedim(0, -1).reshape(
                    -1, 3
                )[select_index]
                trace_directions = wo[ray_local_index]
                trace_origins = (
                    position_map[ray_local_index] +
                    pc.secondary_raytrace_origin_epsilon * trace_directions
                )
                ray_radiance, ray_opacity, ray_reliability = (
                    pc.get_secondary_raytracer().trace(
                        pc,
                        trace_origins,
                        trace_directions,
                        return_reliability=getattr(
                            pipe,
                            'secondary_raytrace_reliability_debug',
                            False,
                        ),
                    )
                )
                material_weight = (
                    pc.secondary_raytrace_strength *
                    ray_gate[ray_local_index]
                ).clamp(0.0, 1.0)
                global_spec_light = spec_light[ray_local_index]
                # 3DGRT returns radiance composited over black. Preserve the
                # existing SphMip environment for the unoccluded transmittance
                # instead of multiplying the traced radiance by alpha twice.
                traced_spec_light = (
                    ray_radiance + global_spec_light * (1.0 - ray_opacity)
                )
                spec_light = spec_light.clone()
                spec_light[ray_local_index] = (
                    global_spec_light * (1.0 - material_weight) +
                    traced_spec_light * material_weight
                )
                pixel_index = select_index[ray_local_index]
                secondary_radiance.reshape(3, -1)[:, pixel_index] = (
                    ray_radiance.transpose(0, 1)
                )
                secondary_gate.reshape(1, -1)[:, pixel_index] = (
                    (material_weight * ray_opacity).transpose(0, 1)
                )
                secondary_hit_opacity.reshape(1, -1)[:, pixel_index] = (
                    ray_opacity.transpose(0, 1)
                )
                if ray_reliability is not None:
                    secondary_reliability.reshape(1, -1)[:, pixel_index] = (
                        ray_reliability.transpose(0, 1)
                    )

        render_spec.reshape(3, -1)[:, select_index] = spec_light.transpose(0, 1)
        render_attenuation.reshape(1, -1)[:, select_index] = spec_attenuation.transpose(0, 1)

    render_attenuation = 1 - (1 - render_attenuation) * foreground

    final_tran = render_tran * render_transmissivity
    final_scat = render_scat * (1 - render_transmissivity)
    # Preserve the 2026-08-11 SOTA glossy gain. With secondary tracing off,
    # render_spec is exactly the original global SphMip result.
    glossy_gain = 1.0 + pc.glossy_specular_boost * render_glossy_score.clamp(0.0, 1.0)
    final_spec = render_spec * render_reflectance * glossy_gain
    final_rendering = final_tran + final_scat

    if not pipe.init_stage:
        final_tran = final_tran * render_attenuation
        final_scat = final_scat * render_attenuation
        final_rendering = final_tran + final_scat + final_spec

    rets = {
        'final_rendering': final_rendering,
        'final_tran': final_tran,
        'final_scat': final_scat,
        'final_spec': final_spec,
        'render_spec': render_spec,
        'render_tran': render_tran,
        'render_scat': render_scat,
        'feature': render_feature,
        'roughness': render_roughness,
        'reflectance': render_reflectance,
        'glossy_score': render_glossy_score,
        'secondary_raytrace_radiance': secondary_radiance,
        'secondary_raytrace_gate': secondary_gate,
        'secondary_raytrace_hit_opacity': secondary_hit_opacity,
        'secondary_raytrace_reliability': secondary_reliability,
        'surface_position': render_position,
        'transmissivity': render_transmissivity,
        'attenuation': render_attenuation,
        'foreground': foreground,
        'surface_alpha': surface_alpha,
        'surface_depth': surface_depth,
        'surface_normal': surface_normal,
        'surface_depth_normal': surface_depth_normal,
        'surface_dist': surface_dist,
        'surface_opacity': surface_opacity,
        'volume_alpha': volume_alpha,
        'volume_depth': volume_depth,
        'volume_normal': volume_normal,
        'volume_depth_normal': volume_depth_normal,
        'volume_dist': volume_dist,
        'volume_opacity': volume_opacity,
        'viewspace_points': means2D,
        'visibility_filter': radii > 0,
        'radii': radii,
        'accum_metric_counts': accum_metric_counts,  # [FASTGS] per-Gaussian 高误差像素覆盖计数
        'accum_protection': accum_protection,         # [FASTGS] per-Gaussian alpha加权边缘保护分
    }

    return rets
