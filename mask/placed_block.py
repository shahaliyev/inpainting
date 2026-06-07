"""
Deterministic square block masks at fixed spatial positions.

No random sampling — the mask is a pure function of (H, W, severity, placement).
This guarantees identical masks across models for the same image/domain/severity/placement.
"""

from typing import Tuple

import torch

PLACEMENTS = ("top_left", "top_right", "center", "bottom_left", "bottom_right")


def _block_side(H: int, W: int, severity: float) -> int:
    """Square side length that covers approximately `severity`% of the image area."""
    area = max(1.0, (severity / 100.0) * float(H * W))
    s = int(round(area ** 0.5))
    return max(1, min(s, H, W))


def _anchor(H: int, W: int, s: int, placement: str) -> Tuple[int, int]:
    """Return the top-left (y, x) corner for the block given a named placement."""
    if placement == "top_left":
        return 0, 0
    if placement == "top_right":
        return 0, W - s
    if placement == "center":
        return (H - s) // 2, (W - s) // 2
    if placement == "bottom_left":
        return H - s, 0
    if placement == "bottom_right":
        return H - s, W - s
    raise ValueError(
        f"Unknown placement: {placement!r}. "
        f"Choose from: {', '.join(PLACEMENTS)}"
    )


def make_placed_block_mask(
    H: int,
    W: int,
    severity: float,
    placement: str,
) -> torch.Tensor:
    """
    Build a deterministic (1, H, W) float32 mask.

    The block is 1.0 inside the masked region and 0.0 elsewhere.
    The result is a pure function of (H, W, severity, placement) — no randomness.

    Args:
        H:         Image height in pixels.
        W:         Image width in pixels.
        severity:  Target masked area as a percentage (e.g. 20 → ~20% masked).
        placement: One of PLACEMENTS.

    Returns:
        Tensor of shape (1, H, W), dtype=float32.
    """
    s = _block_side(H, W, severity)
    y, x = _anchor(H, W, s, placement)
    mask = torch.zeros((1, H, W), dtype=torch.float32)
    mask[:, y : y + s, x : x + s] = 1.0
    return mask
