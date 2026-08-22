"""3DGRT adapter used only for glossy secondary reflection rays.

The CUDA/OptiX implementation lives in NVIDIA's official ``3dgrut`` project,
which is pinned as ``submodules/3dgrut``.  This module deliberately does not
replace RT-Splatting's primary rasterizer: it packs only selected reflection
rays and sends those rays through the 3DGRT BVH.
"""

import os
import platform
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_EXTENSION = None


def _official_root():
    override = os.environ.get('RTSPLAT_3DGRT_ROOT', '').strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / 'submodules' / '3dgrut'


def load_3dgrt_extension(verbose=True):
    """Build/load the pinned official 3DGRT CUDA extension lazily."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if CUDA_HOME is None:
        raise RuntimeError(
            '3DGRT requires a CUDA toolkit with nvcc. CUDA_HOME is not set.'
        )

    root = _official_root()
    tracer_root = root / 'threedgrt_tracer'
    optix_include = tracer_root / 'dependencies' / 'optix-dev' / 'include'
    required = [
        tracer_root / 'src' / 'optixTracer.cpp',
        tracer_root / 'src' / 'particlePrimitives.cu',
        tracer_root / 'bindings.cpp',
        optix_include / 'optix.h',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            'Official 3DGRT sources are incomplete. Run:\n'
            '  git submodule update --init submodules/3dgrut\n'
            '  git -C submodules/3dgrut submodule update --init '
            'threedgrt_tracer/dependencies/optix-dev\n'
            f'Missing: {missing}'
        )

    feature_defines = [
        '-DPARTICLE_FEATURE_DIM=48',
        '-DRAY_FEATURE_DIM=3',
        '-DFEATURE_TRANSFORM_TYPE=0',  # spherical harmonics
        '-DFEATURE_INTERPOLATION_TYPE=0',
        '-DFEATURE_INTERPOLATION_SUPPORT=0',
        '-DFEATURE_ACTIVATION_TYPE=0',
        '-DFEATURE_ACTIVATION_NUM_FREQUENCIES=1',
        '-DINTERP_POINT_FEATURE_DIM=3',
        '-DPARTICLE_FEATURE_HALF=0',
        '-DFEATURE_OUTPUT_HALF=0',
    ]
    kernel_defines = [
        *feature_defines,
        '-DPARTICLE_RADIANCE_NUM_COEFFS=16',
        '-DGAUSSIAN_PARTICLE_KERNEL_DEGREE=4',
        '-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY=0.0113',
        '-DGAUSSIAN_PARTICLE_MIN_ALPHA=0.00392156862745098',
        '-DGAUSSIAN_PARTICLE_MAX_ALPHA=0.99',
        '-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD=0.001',
    ]
    common_flags = ['-DNVDR_TORCH']
    cuda_flags = [
        '-DNVDR_TORCH',
        '-std=c++17',
        '--extended-lambda',
        '--expt-relaxed-constexpr',
        '-Xcompiler=-fno-strict-aliasing',
        '-diag-suppress=1444',
        '-diag-suppress=3287',
        *kernel_defines,
    ]
    if os.name == 'nt':
        common_flags.extend(['/DNOMINMAX', *feature_defines])
        link_flags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']
    else:
        common_flags.extend(feature_defines)
        cuda_arch = f'{platform.machine()}-linux'
        link_flags = [
            f'-L{Path(CUDA_HOME) / "lib" / "stubs"}',
            f'-L{Path(CUDA_HOME) / "targets" / cuda_arch / "lib"}',
            f'-L{Path(CUDA_HOME) / "targets" / cuda_arch / "lib" / "stubs"}',
            '-lcuda',
            '-lnvrtc',
        ]

    include_paths = [str(tracer_root / 'include'), str(optix_include)]
    targets = Path(CUDA_HOME) / 'targets'
    if targets.is_dir():
        include_paths.extend(
            str(path / 'include') for path in targets.iterdir()
            if (path / 'include').is_dir()
        )
    _EXTENSION = load(
        name='lib3dgrt_rtsplat_sh3',
        sources=[str(path) for path in required[:3]],
        extra_cflags=common_flags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=link_flags,
        extra_include_paths=include_paths,
        with_cuda=True,
        verbose=verbose,
    )
    return _EXTENSION


class _TraceRays(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, wrapper, frame_id, ray_to_world, ray_origins, ray_directions,
        positions, rotations, scales, densities, features, sh_degree,
        min_transmittance,
    ):
        particle_density = torch.cat(
            [positions, densities, rotations, scales, torch.zeros_like(densities)],
            dim=1,
        )
        outputs = wrapper.trace(
            frame_id,
            ray_to_world,
            ray_origins,
            ray_directions,
            particle_density,
            features,
            0,
            sh_degree,
            min_transmittance,
        )
        ray_features, ray_density, ray_hit, ray_normals, _, _ = outputs
        ctx.save_for_backward(
            ray_to_world,
            ray_origins,
            ray_directions,
            ray_features,
            ray_density,
            ray_hit,
            ray_normals,
            particle_density,
            features,
        )
        ctx.wrapper = wrapper
        ctx.frame_id = frame_id
        ctx.sh_degree = sh_degree
        ctx.min_transmittance = min_transmittance
        return ray_features.float(), ray_density

    @staticmethod
    def backward(ctx, feature_grad, density_grad):
        (
            ray_to_world, ray_origins, ray_directions, ray_features,
            ray_density, ray_hit, ray_normals, particle_density, features,
        ) = ctx.saved_tensors
        if feature_grad is None:
            feature_grad = torch.zeros_like(ray_features)
        if density_grad is None:
            density_grad = torch.zeros_like(ray_density)
        particle_grad, feature_parameter_grad = ctx.wrapper.trace_bwd(
            ctx.frame_id,
            ray_to_world,
            ray_origins,
            ray_directions,
            ray_features,
            ray_density,
            ray_hit,
            ray_normals,
            particle_density,
            features,
            feature_grad.contiguous(),
            density_grad.contiguous(),
            torch.zeros_like(ray_hit[..., :1]),
            torch.zeros_like(ray_normals),
            0,
            ctx.sh_degree,
            ctx.min_transmittance,
        )
        position_grad, density_parameter_grad, rotation_grad, scale_grad, _ = (
            torch.split(particle_grad, [3, 1, 4, 3, 1], dim=1)
        )
        return (
            None, None, None, None, None,
            position_grad, rotation_grad, scale_grad, density_parameter_grad,
            feature_parameter_grad, None, None,
        )


class ThreeDGRTSecondaryTracer:
    """Lazy OptiX BVH and arbitrary-ray tracing for one Gaussian model."""

    def __init__(self, thickness_ratio=0.10, rebuild_interval=1,
                 min_transmittance=0.03):
        extension = load_3dgrt_extension()
        root = _official_root() / 'threedgrt_tracer'
        self.wrapper = extension.OptixTracer(
            str(root),
            str(CUDA_HOME),
            'reference',
            'referenceBwd',
            'icosahedron',
            4.0,
            0.0113,
            0.99,
            True,
            3,
            False,
            True,
        )
        self.thickness_ratio = max(float(thickness_ratio), 1e-4)
        self.rebuild_interval = max(int(rebuild_interval), 1)
        self.min_transmittance = min(max(float(min_transmittance), 1e-5), 0.5)
        self._frame_id = 0
        self._build_count = 0
        self._last_gaussian_count = -1

    def _scales3d(self, gaussians):
        scales = gaussians.get_scaling
        if scales.shape[1] == 3:
            return scales
        if scales.shape[1] != 2:
            raise ValueError(f'3DGRT expects 2D/3D scales, got {tuple(scales.shape)}')
        thickness = scales.min(dim=1, keepdim=True).values * self.thickness_ratio
        return torch.cat([scales, thickness.clamp_min(1e-6)], dim=1)

    def _build_bvh(self, gaussians, scales):
        count = int(gaussians.get_xyz.shape[0])
        topology_changed = count != self._last_gaussian_count
        should_update = topology_changed or self._build_count % self.rebuild_interval == 0
        if should_update:
            # Density clamping changes each proxy's extent, so use the paper's
            # robust full rebuild path. ``rebuild_interval`` can deliberately
            # keep a BVH for several small geometry updates when requested.
            self.wrapper.build_bvh(
                gaussians.get_xyz.detach().contiguous(),
                gaussians.get_rotation.detach().contiguous(),
                scales.detach().contiguous(),
                gaussians.get_occupancy.detach().contiguous(),
                True,
                False,
            )
            self._last_gaussian_count = count
        self._build_count += 1

    def trace(
        self, gaussians, ray_origins, ray_directions,
        return_reliability=False,
    ):
        if ray_origins.ndim != 2 or ray_origins.shape[-1] != 3:
            raise ValueError('3DGRT ray origins must have shape [N, 3]')
        if ray_directions.shape != ray_origins.shape:
            raise ValueError('3DGRT ray directions must match ray origins')
        scales = self._scales3d(gaussians)
        self._build_bvh(gaussians, scales)

        features = gaussians.get_features.reshape(
            gaussians.get_features.shape[0], -1
        ).contiguous()
        if features.shape[1] != 48:
            raise ValueError(
                f'Pinned 3DGRT extension expects degree-3 SH (48 floats), '
                f'got {features.shape[1]}'
            )
        rays_o = ray_origins.reshape(1, 1, -1, 3).contiguous()
        rays_d = torch.nn.functional.normalize(
            ray_directions, dim=-1, eps=1e-6
        ).reshape(1, 1, -1, 3).contiguous()
        ray_to_world = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        radiance, opacity = _TraceRays.apply(
            self.wrapper,
            self._frame_id,
            ray_to_world,
            rays_o,
            rays_d,
            gaussians.get_xyz,
            gaussians.get_rotation,
            scales,
            gaussians.get_occupancy,
            features,
            min(int(gaussians.active_sh_degree), 3),
            self.min_transmittance,
        )
        reliability = None
        frame_advance = 1
        if return_reliability:
            # Encode scalar R_g as direction-independent RGB SH. The official
            # kernel evaluates C0 * coeff_dc + 0.5, so this produces exactly
            # R_g at every accepted Gaussian without changing CUDA code.
            sh_c0 = 0.28209479177387814
            reliability_features = torch.zeros_like(features)
            reliability_dc = (
                gaussians.get_raytrace_reliability.clamp(0.0, 1.0) - 0.5
            ) / sh_c0
            reliability_features[:, :3] = reliability_dc.expand(-1, 3)
            reliability_radiance, _ = _TraceRays.apply(
                self.wrapper,
                self._frame_id + 1,
                ray_to_world,
                rays_o,
                rays_d,
                gaussians.get_xyz,
                gaussians.get_rotation,
                scales,
                gaussians.get_occupancy,
                reliability_features,
                0,
                self.min_transmittance,
            )
            flat_opacity = opacity.reshape(-1, 1).clamp(0.0, 1.0)
            reliability_numerator = reliability_radiance.reshape(
                -1, 3
            ).mean(dim=1, keepdim=True)
            reliability = torch.where(
                flat_opacity > 1e-6,
                reliability_numerator / flat_opacity.clamp_min(1e-6),
                torch.zeros_like(flat_opacity),
            ).clamp(0.0, 1.0)
            frame_advance = 2
        self._frame_id += frame_advance
        return (
            radiance.reshape(-1, 3),
            opacity.reshape(-1, 1).clamp(0.0, 1.0),
            reliability,
        )
