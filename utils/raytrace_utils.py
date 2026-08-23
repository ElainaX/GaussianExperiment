"""Small, backend-independent helpers for secondary-ray routing."""

import torch


def reliability_route_score(reliability, low=0.35, high=0.65):
    """Map hit reliability to a smooth environment-to-3DGRT route score."""
    low = float(low)
    high = float(high)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f'Routing thresholds must satisfy 0 <= low < high <= 1, '
            f'got low={low}, high={high}'
        )
    normalized = ((reliability - low) / (high - low)).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)
