"""
render_gaussian_heatmap.py  –  per-pixel Gaussian density heatmap

Projects every Gaussian centre onto the image plane for each camera and
accumulates a count into a [H, W] grid, saved as a false-colour PNG.

Usage
-----
    # Test cameras (default)
    python render_gaussian_heatmap.py -m ./output/truck

    # Overlay heatmap on rendered image + light blur
    python render_gaussian_heatmap.py -m ./output/truck --overlay --sigma 1.5

    # Training cameras, weighted by opacity × occupancy
    python render_gaussian_heatmap.py -m ./output/truck --cameras train --weight opacity

    # Pick three specific test cameras by 0-based index
    python render_gaussian_heatmap.py -m ./output/truck --cameras 0 4 9

    # Specific checkpoint iteration
    python render_gaussian_heatmap.py -m ./output/truck --iteration 30000 --overlay
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render as gs_render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap computation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_heatmap(gaussians, cam, weight_mode='count'):
    """
    Project Gaussian centres onto the image plane and bin into a [H, W] grid.

    weight_mode
        'count'   – each Gaussian contributes 1 per pixel it projects to
        'opacity' – each Gaussian contributes opacity × occupancy

    Returns a float32 numpy array with values normalised to [0, 1].
    """
    H = cam.image_height
    W = cam.image_width
    xyz = gaussians.get_xyz          # [N, 3]
    N   = xyz.shape[0]

    # Homogeneous coords → NDC (row-major: v @ M)
    ones  = torch.ones(N, 1, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)                  # [N, 4]
    ndc_h = xyz_h @ cam.full_proj_transform                # [N, 4]
    w_h   = ndc_h[:, 3:4].clamp(min=1e-6)
    ndc   = ndc_h[:, :3] / w_h                            # [N, 3]  (x,y,z in NDC)

    # Camera-space depth for frustum culling
    cam_z = (xyz_h @ cam.world_view_transform)[:, 2]      # [N]

    # NDC ∈ [-1, 1]  →  pixel coords (integer)
    px = ((ndc[:, 0] + 1.0) * 0.5 * W).long()
    py = ((ndc[:, 1] + 1.0) * 0.5 * H).long()

    valid = (cam_z > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    px_v = px[valid]
    py_v = py[valid]

    if weight_mode == 'opacity':
        weights = (gaussians.get_opacity[valid] *
                   gaussians.get_occupancy[valid]).squeeze().float()
    else:
        weights = torch.ones(valid.sum(), device=xyz.device, dtype=torch.float32)

    # Accumulate into flat pixel index, then reshape to [H, W]
    flat_idx = (py_v * W + px_v).clamp(0, H * W - 1)
    hmap = torch.zeros(H * W, device=xyz.device, dtype=torch.float32)
    hmap.scatter_add_(0, flat_idx, weights)
    hmap = hmap.view(H, W).cpu().numpy()

    vmax = hmap.max()
    return (hmap / vmax) if vmax > 0 else hmap


def blur_heatmap(hmap, sigma):
    """Optional Gaussian blur to smooth sparse heatmaps."""
    from scipy.ndimage import gaussian_filter
    out = gaussian_filter(hmap.astype(np.float32), sigma=sigma)
    vmax = out.max()
    return (out / vmax) if vmax > 0 else out


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_heatmap_image(hmap, out_path, rendered_np=None,
                       cmap='inferno', overlay_alpha=0.55,
                       title='', show=False):
    """
    Save (and optionally display) one heatmap.

    rendered_np  – if given, blend the heatmap over this [H,W,3] float image.
    """
    H, W = hmap.shape
    fig, axes = plt.subplots(
        1, 2 if rendered_np is not None else 1,
        figsize=((W * 2 / 100) if rendered_np is not None else (W / 100), H / 100),
        dpi=100,
    )

    if rendered_np is None:
        ax = axes
        ax.imshow(hmap, cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=7, pad=2)
    else:
        ax_render, ax_heat = axes
        # Left: overlay
        ax_render.imshow(rendered_np)
        ax_render.imshow(hmap, cmap=cmap, alpha=overlay_alpha, vmin=0, vmax=1)
        ax_render.set_axis_off()
        ax_render.set_title('overlay', fontsize=7, pad=2)
        # Right: heatmap only
        im = ax_heat.imshow(hmap, cmap=cmap, vmin=0, vmax=1)
        ax_heat.set_axis_off()
        ax_heat.set_title('heatmap', fontsize=7, pad=2)
        plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)

    if title and rendered_np is not None:
        fig.suptitle(title, fontsize=7, y=1.01)

    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = ArgumentParser(description='Render Gaussian density heatmap.')
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)

    parser.add_argument('--iteration', default=-1, type=int,
                        help='Checkpoint iteration to load (-1 = latest)')
    parser.add_argument('--cameras', nargs='+', default=['test'],
                        help='"train", "test", or integer indices into the test set')
    parser.add_argument('--weight', choices=['count', 'opacity'], default='count',
                        help='Per-Gaussian weight: "count" (uniform) or "opacity"')
    parser.add_argument('--overlay', action='store_true',
                        help='Render scene and blend heatmap over it')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Gaussian blur radius applied to heatmap (0 = none)')
    parser.add_argument('--cmap', default='inferno',
                        help='Matplotlib colormap name (default: inferno)')
    parser.add_argument('--output', default='',
                        help='Output directory (default: <model>/heatmap/<iter>)')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--show', action='store_true',
                        help='Display result interactively (blocks until closed)')

    args = get_combined_args(parser)

    if args.show:
        matplotlib.use('TkAgg')

    safe_state(args.quiet)

    dataset  = lp.extract(args)
    pipe     = pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene    = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    out_dir = args.output or os.path.join(
        dataset.model_path, 'heatmap', str(scene.loaded_iter)
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Select cameras ──────────────────────────────────────────────────────
    cam_spec = args.cameras
    if cam_spec == ['train']:
        cameras = scene.getTrainCameras()
        tag = 'train'
    elif cam_spec == ['test']:
        cameras = scene.getTestCameras()
        tag = 'test'
    else:
        all_test = scene.getTestCameras()
        try:
            idxs = [int(x) for x in cam_spec]
        except ValueError:
            print(f'ERROR: --cameras must be "train", "test", '
                  f'or integer indices; got {cam_spec}')
            sys.exit(1)
        cameras = [all_test[i] for i in idxs if i < len(all_test)]
        tag = 'select'

    n_gaussians = gaussians.get_xyz.shape[0]
    print(f'Model: {dataset.model_path}  iter={scene.loaded_iter}  '
          f'Gaussians={n_gaussians:,}')
    print(f'Cameras: {len(cameras)}  weight={args.weight}  '
          f'sigma={args.sigma}  output={out_dir}')

    # ── Process each camera ─────────────────────────────────────────────────
    for idx, cam in enumerate(cameras):
        name = getattr(cam, 'image_name', str(idx))
        if not args.quiet:
            print(f'  [{idx + 1}/{len(cameras)}] {name}  '
                  f'{cam.image_width}×{cam.image_height}')

        # Build heatmap
        hmap = compute_heatmap(gaussians, cam, weight_mode=args.weight)
        if args.sigma > 0:
            hmap = blur_heatmap(hmap, sigma=args.sigma)

        # Optionally render the scene for overlay
        rendered_np = None
        if args.overlay:
            with torch.no_grad():
                pkg = gs_render(cam, gaussians, pipe, background)
            rendered_np = (pkg['final_rendering']
                           .clamp(0, 1)
                           .permute(1, 2, 0)
                           .cpu()
                           .numpy())

        # Save
        out_path = os.path.join(out_dir, f'{tag}_{name}.png')
        title = (f'{name}  |  N={n_gaussians:,}  '
                 f'iter={scene.loaded_iter}  weight={args.weight}')
        save_heatmap_image(
            hmap, out_path,
            rendered_np=rendered_np,
            cmap=args.cmap,
            title=title,
            show=(args.show and idx == 0),
        )

        if not args.quiet:
            print(f'    → {out_path}')

    print(f'Done. {len(cameras)} heatmap(s) saved to {out_dir}')
