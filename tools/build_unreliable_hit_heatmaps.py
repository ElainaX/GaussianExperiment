"""Build unreliable-hit and routing maps from an existing render ``vis`` folder.

The score is ``hit_opacity * (1 - reliability)``. Therefore an untraced or
missed pixel remains zero instead of being mistaken for an unreliable hit.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


HIT_PREFIX = 'secondary_raytrace_hit_opacity_'
RELIABILITY_PREFIX = 'secondary_raytrace_reliability_'
RAW_PREFIX = 'secondary_raytrace_unreliable_hit_'
UNRELIABLE_HEATMAP_PREFIX = 'secondary_raytrace_unreliable_hit_heatmap_'
ROUTE_PREFIX = 'secondary_raytrace_route_score_'
ROUTE_HEATMAP_PREFIX = 'secondary_raytrace_route_heatmap_'


def compute_unreliable_hit(hit_opacity, reliability):
    """Return H * (1 - R), with both inputs interpreted in [0, 1]."""
    if hit_opacity.shape != reliability.shape:
        raise ValueError(
            f'Hit/reliability shapes differ: {hit_opacity.shape} vs '
            f'{reliability.shape}'
        )
    return np.clip(hit_opacity * (1.0 - reliability), 0.0, 1.0)


def compute_route_score(hit_opacity, reliability, low=0.35, high=0.65):
    """Return smooth RT routing confidence, with misses fixed to zero."""
    if hit_opacity.shape != reliability.shape:
        raise ValueError(
            f'Hit/reliability shapes differ: {hit_opacity.shape} vs '
            f'{reliability.shape}'
        )
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f'Routing thresholds must satisfy 0 <= low < high <= 1, '
            f'got low={low}, high={high}'
        )
    normalized = np.clip((reliability - low) / (high - low), 0.0, 1.0)
    route = normalized * normalized * (3.0 - 2.0 * normalized)
    return np.where(hit_opacity > 0.0, route, 0.0)


def load_gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def save_gray(score, path):
    Image.fromarray(np.rint(score * 255.0).astype(np.uint8), mode='L').save(path)


def save_heatmap(score, hit_opacity, path):
    heatmap = matplotlib.colormaps['turbo'](score)[..., :3]
    heatmap[hit_opacity <= 0.0] = 0.0
    Image.fromarray(np.rint(heatmap * 255.0).astype(np.uint8), mode='RGB').save(path)


def find_vis_dirs(root):
    if not root.exists():
        raise FileNotFoundError(f'Input path does not exist: {root}')
    if root.is_file():
        raise ValueError(f'Expected a directory, got file: {root}')
    if any(root.glob(f'{HIT_PREFIX}*.png')):
        return [root]
    return sorted({
        path.parent
        for path in root.rglob(f'{HIT_PREFIX}*.png')
        if not any(part.startswith('.') for part in path.relative_to(root).parts)
    })


def process_vis_dir(vis_dir, route_low=0.35, route_high=0.65, overwrite=False):
    written = 0
    missing = 0
    for hit_path in sorted(vis_dir.glob(f'{HIT_PREFIX}*.png')):
        suffix = hit_path.name[len(HIT_PREFIX):]
        reliability_path = vis_dir / f'{RELIABILITY_PREFIX}{suffix}'
        if not reliability_path.exists():
            print(f'[missing reliability] {reliability_path}')
            missing += 1
            continue

        raw_path = vis_dir / f'{RAW_PREFIX}{suffix}'
        heatmap_path = vis_dir / f'{UNRELIABLE_HEATMAP_PREFIX}{suffix}'
        route_path = vis_dir / f'{ROUTE_PREFIX}{suffix}'
        route_heatmap_path = vis_dir / f'{ROUTE_HEATMAP_PREFIX}{suffix}'
        output_paths = [raw_path, heatmap_path, route_path, route_heatmap_path]
        if not overwrite and all(path.exists() for path in output_paths):
            continue

        hit_opacity = load_gray(hit_path)
        reliability = load_gray(reliability_path)
        score = compute_unreliable_hit(hit_opacity, reliability)
        route_score = compute_route_score(
            hit_opacity,
            reliability,
            low=route_low,
            high=route_high,
        )
        save_gray(score, raw_path)
        save_heatmap(score, hit_opacity, heatmap_path)
        save_gray(route_score, route_path)
        save_heatmap(route_score, hit_opacity, route_heatmap_path)
        written += 1
    return written, missing


def main():
    parser = argparse.ArgumentParser(
        description='Build 3DGRT unreliable-hit and reliability-routing heatmaps.'
    )
    parser.add_argument(
        'root',
        type=Path,
        help='A vis folder or an experiment/model directory containing vis folders.',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--route-low', type=float, default=0.35)
    parser.add_argument('--route-high', type=float, default=0.65)
    args = parser.parse_args()

    vis_dirs = find_vis_dirs(args.root)
    if not vis_dirs:
        raise SystemExit(f'No {HIT_PREFIX}*.png files found below {args.root}')

    total_written = 0
    total_missing = 0
    for vis_dir in vis_dirs:
        written, missing = process_vis_dir(
            vis_dir,
            route_low=args.route_low,
            route_high=args.route_high,
            overwrite=args.overwrite,
        )
        total_written += written
        total_missing += missing
        print(f'{vis_dir}: wrote {written} pair(s), missing {missing}')
    print(f'Done: wrote {total_written} pair(s), missing {total_missing}')


if __name__ == '__main__':
    main()
