"""
FastGS-style multi-view scoring for triangle densification and pruning.

Key idea (from FastGS): accumulate per-primitive "high-error pixel coverage"
across multiple training views. Triangles that consistently cover high-error
regions should be split (densification); triangles that are never seen or never
cover high-error regions are candidates for pruning.
"""

import torch
import torch.nn.functional as F


def compute_triangle_score(
    triangles,
    viewpoint_cameras,
    render_score_fn,
    pipe,
    bg_color: torch.Tensor,
    loss_thresh: float = 0.02,
    max_views: int = 16,
) -> torch.Tensor:
    """
    Accumulate per-triangle high-error pixel counts over a subset of views.

    Args:
        triangles: TriangleModel
        viewpoint_cameras: list of Camera objects (training cameras)
        render_score_fn: triangle_renderer.render_score_pass
        pipe: pipeline settings
        bg_color: background color tensor (GPU)
        loss_thresh: L1 per-pixel threshold that marks a pixel as "high error"
        max_views: cap how many cameras we score per call (avoid OOM)

    Returns:
        score [P] float32 — normalized to [0, 1], higher = more high-error views
    """
    P = triangles._triangle_indices.shape[0]
    accum = torch.zeros(P, dtype=torch.float32, device="cuda")
    n_views = min(max_views, len(viewpoint_cameras))

    # Randomly sample views so the selection is varied across training
    indices = torch.randperm(len(viewpoint_cameras))[:n_views].tolist()

    with torch.no_grad():
        for idx in indices:
            cam = viewpoint_cameras[idx]

            # Build metric map from last rendered image vs. ground truth.
            # We don't have the rendered image here, so we do a quick render.
            from triangle_renderer import render
            pkg = render(cam, triangles, pipe, bg_color)
            rendered = pkg["render"]  # [3, H, W]

            gt = cam.original_image.cuda()
            if gt.shape != rendered.shape:
                gt = F.interpolate(gt.unsqueeze(0), size=rendered.shape[1:], mode="bilinear", align_corners=False).squeeze(0)

            l1_map = (rendered - gt).abs().mean(dim=0)  # [H, W]
            metric_map = (l1_map > loss_thresh).float()  # binary, [H, W]

            counts = render_score_fn(cam, triangles, pipe, bg_color, metric_map)  # [P] int32
            accum += counts.float()

    # Normalize to [0, 1]
    max_val = accum.max().clamp(min=1.0)
    return accum / max_val


def score_weighted_probs(
    base_probs: torch.Tensor,
    score: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Blend existing importance-score probabilities with multi-view error scores.

    base_probs: existing probs [P] (e.g. importance_score from max_blending)
    score:      multi-view score [P] in [0, 1]
    alpha:      weight of multi-view score (0 = ignore score, 1 = use only score)
    """
    base = base_probs / (base_probs.sum().clamp(min=1e-8))
    scr = score / (score.sum().clamp(min=1e-8))
    blended = (1.0 - alpha) * base + alpha * scr
    return blended / blended.sum().clamp(min=1e-8)


def prune_low_score_triangles(triangles, score: torch.Tensor, keep_ratio: float = 0.9):
    """
    Prune triangles whose multi-view score falls below the keep_ratio quantile.

    Only prunes triangles that have *never* been high-error (score == 0) AND
    are also low-importance (max_blending small), to avoid pruning undertrained
    regions that simply haven't been visited yet.

    Returns number of triangles pruned.
    """
    with torch.no_grad():
        importance = triangles.importance_score
        if not isinstance(importance, torch.Tensor) or importance.numel() == 0:
            return 0

        # Only prune where BOTH score and importance are in the bottom fraction
        combined = score * 0.5 + importance / (importance.max().clamp(min=1e-8)) * 0.5
        threshold = torch.quantile(combined, 1.0 - keep_ratio)
        keep_mask = combined > threshold

        n_before = triangles._triangle_indices.shape[0]
        triangles.prune_triangles(keep_mask)
        return n_before - triangles._triangle_indices.shape[0]
