"""
Evaluation metrics for inpainting: L1 (masked region), PSNR, SSIM, LPIPS (full image).
All expect normalized tensors (B, C, H, W); mean/std used to denormalize to [0, 1] for PSNR/SSIM.
LPIPS uses normalized [-1, 1] input by default.
"""

from typing import Optional

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize to [0, 1]. x, mean, std: (1, 3, 1, 1) or broadcastable."""
    return (x * std + mean).clamp(0.0, 1.0)


def l1_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss over the masked region only. Returns scalar (mean over batch)."""
    per_pix = torch.abs(pred - target)
    # mask: (B, 1, H, W), expand to channels
    denom = mask.sum() * target.shape[1] + 1e-8
    return (per_pix * mask).sum() / denom


def psnr_full(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """Full-image PSNR (dB). Recon = target outside mask, pred inside mask; compare to target."""
    recon = target * (1.0 - mask) + pred * mask
    recon_01 = _denorm(recon, mean, std)
    target_01 = _denorm(target, mean, std)
    mse = ((recon_01 - target_01) ** 2).view(recon.shape[0], -1).mean(dim=1)
    psnr_per = -10.0 * torch.log10(mse + 1e-10)
    return psnr_per.mean().item()


def ssim_full(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """Full-image SSIM (structural similarity). Per-image then mean."""
    recon = target * (1.0 - mask) + pred * mask
    recon_01 = _denorm(recon, mean, std)
    target_01 = _denorm(target, mean, std)
    scores = []
    for i in range(recon.shape[0]):
        r = recon_01[i].permute(1, 2, 0).cpu().numpy()
        t = target_01[i].permute(1, 2, 0).cpu().numpy()
        scores.append(ssim(r, t, data_range=1.0, channel_axis=2))
    return float(np.mean(scores))


def lpips_full(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lpips_net: torch.nn.Module,
) -> float:
    """Full-image LPIPS (perceptual). Recon vs target; expects normalized [-1, 1]."""
    recon = target * (1.0 - mask) + pred * mask
    with torch.no_grad():
        d = lpips_net(recon, target)
    return d.mean().item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lpips_net: Optional[torch.nn.Module] = None,
) -> dict[str, float]:
    """Compute L1 (mask), PSNR, SSIM, and optionally LPIPS (full image)."""
    out = {
        "l1_mask": l1_mask(pred, target, mask).item(),
        "psnr_full": psnr_full(pred, target, mask, mean, std),
        "ssim_full": ssim_full(pred, target, mask, mean, std),
    }
    if lpips_net is not None:
        out["lpips_full"] = lpips_full(pred, target, mask, lpips_net)
    return out
