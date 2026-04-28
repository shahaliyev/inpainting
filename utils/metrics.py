"""
Evaluation metrics for inpainting with configurable metric scope.
All expect normalized tensors (B, C, H, W). mean/std are used to denormalize
to [0, 1] for PSNR/SSIM. LPIPS uses normalized [-1, 1] input by default.
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


def l1_full(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 over the reconstructed full image."""
    recon = target * (1.0 - mask) + pred * mask
    return torch.abs(recon - target).mean()


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

def psnr_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """Masked PSNR (dB). Compare pred vs target only inside masked region."""
    pred_01 = _denorm(pred, mean, std)
    target_01 = _denorm(target, mean, std)

    se = (pred_01 - target_01).pow(2) * mask
    err_sum = se.sum(dim=[1, 2, 3])  # sum over C, H, W for each image
    denom = mask.sum(dim=[1, 2, 3]) * pred.shape[1] + 1e-8
    mse = err_sum / denom
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

def ssim_mask(
   pred: torch.Tensor,
   target: torch.Tensor,
   mask: torch.Tensor,
   mean: torch.Tensor,
   std: torch.Tensor,
) -> float:
   """Mask SSIM (structural similarity). Per-image then mean."""
   recon = target * (1.0 - mask) + pred * mask
   recon_01 = _denorm(recon, mean, std)
   target_01 = _denorm(target, mean, std)
   scores = []
   for i in range(recon.shape[0]):
       r = recon_01[i].permute(1, 2, 0).cpu().numpy()
       t = target_01[i].permute(1, 2, 0).cpu().numpy()
       m = mask[i, 0].detach().cpu().numpy().astype(np.float32)  # H, W

       _, ssim_map = ssim(r, t, data_range=1.0, channel_axis=2, full=True)
      
       # H,W,C -> H,W depending on skimage version and multichannel handling, full=True may 
       # return either a spatial map [H, W] or a per-channel-like map [H, W, C]
       if ssim_map.ndim == 3:  
           ssim_map = ssim_map.mean(axis=2)

       scores.append((ssim_map * m).sum() / (m.sum() + 1e-8))

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


def lpips_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lpips_net: torch.nn.Module,
) -> float:
    """Masked LPIPS by zeroing unmasked pixels in both images."""
    pred_masked = pred * mask
    target_masked = target * mask
    with torch.no_grad():
        d = lpips_net(pred_masked, target_masked)
    return d.mean().item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lpips_net: Optional[torch.nn.Module] = None,
    metric_scope: str = "mask",
    report_both: bool = True,
) -> dict[str, float]:
    """Compute scope-consistent primary metrics; optionally include both scopes."""
    scope = str(metric_scope).lower()
    if scope not in {"mask", "full"}:
        raise ValueError(f"Unsupported metric_scope: {metric_scope}. Use 'mask' or 'full'.")

    l1_m = l1_mask(pred, target, mask).item()
    l1_f = l1_full(pred, target, mask).item()
    psnr_m = psnr_mask(pred, target, mask, mean, std)
    psnr_f = psnr_full(pred, target, mask, mean, std)
    ssim_m = ssim_mask(pred, target, mask, mean, std)
    ssim_f = ssim_full(pred, target, mask, mean, std)

    out = {"metric_scope": scope}
    if scope == "mask":
        out["l1"] = l1_m
        out["psnr"] = psnr_m
        out["ssim"] = ssim_m
    else:
        out["l1"] = l1_f
        out["psnr"] = psnr_f
        out["ssim"] = ssim_f

    if lpips_net is not None:
        lpips_m = lpips_mask(pred, target, mask, lpips_net)
        lpips_f = lpips_full(pred, target, mask, lpips_net)
        out["lpips"] = lpips_m if scope == "mask" else lpips_f
        if report_both:
            out["lpips_mask"] = lpips_m
            out["lpips_full"] = lpips_f

    if report_both:
        out["l1_mask"] = l1_m
        out["l1_full"] = l1_f
        out["psnr_mask"] = psnr_m
        out["psnr_full"] = psnr_f
        out["ssim_mask"] = ssim_m
        out["ssim_full"] = ssim_f

    return out
