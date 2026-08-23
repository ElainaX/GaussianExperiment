"""Build unreliable-3DGRT-hit maps from an existing render ``vis`` folder.

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
HEATMAP_PREFIX = 'secondary_raytrace_unreliable_hit_heatmap_'


def compute_unreliable_hit(hit_opacity, reliability):
    """Return H * (1 - R), with both inputs interpreted in [0, 1]."""
    if hit_opacity.shape != reliability.shape:
        raise ValueError(
            f'Hit/reliability shapes differ: {hit_opacity.shape} vs '
            f'{reliability.shape}'
        )
    return np.clip(hit_opacity * (1.0 - reliability), 0.0, 1.0)


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
    return sorted({path.parent for path in root.rglob(f'{HIT_PREFIX}*.png')})


def process_vis_dir(vis_dir, overwrite=False):
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
        heatmap_path = vis_dir / f'{HEATMAP_PREFIX}{suffix}'
        if not overwrite and raw_path.exists() and heatmap_path.exists():
            continue

        hit_opacity = load_gray(hit_path)
        reliability = load_gray(reliability_path)
        score = compute_unreliable_hit(hit_opacity, reliability)
        save_gray(score, raw_path)
        save_heatmap(score, hit_opacity, heatmap_path)
        written += 1
    return written, missing


def main():
    parser = argparse.ArgumentParser(
        description='Combine 3DGRT hit opacity and reliability into an unreliable-hit heatmap.'
    )
    parser.add_argument(
        'root',
        type=Path,
        help='A vis folder or an experiment/model directory containing vis folders.',
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    vis_dirs = find_vis_dirs(args.root)
    if not vis_dirs:
        raise SystemExit(f'No {HIT_PREFIX}*.png files found below {args.root}')

    total_written = 0
    total_missing = 0
    for vis_dir in vis_dirs:
        written, missing = process_vis_dir(vis_dir, overwrite=args.overwrite)
        total_written += written
        total_missing += missing
        print(f'{vis_dir}: wrote {written} pair(s), missing {missing}')
    print(f'Done: wrote {total_written} pair(s), missing {total_missing}')


if __name__ == '__main__':
    main()
