from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, l1_fn: nn.Module) -> torch.Tensor:
    per_pix = l1_fn(pred, target)
    denom = mask.sum() * target.shape[1] + 1e-8
    return (per_pix * mask).sum() / denom


def _to_01(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x * std + mean).clamp(0.0, 1.0)


def masked_lpips_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lpips_net: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    pred_01 = _to_01(pred, mean, std)
    target_01 = _to_01(target, mean, std)
    pred_masked = pred_01 * mask
    target_masked = target_01 * mask
    return lpips_net(pred_masked, target_masked, normalize=True).mean()


def tv_loss_on_hole(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Total variation on masked region only (encourages local smoothness in filled holes).
    dh = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]) * mask[:, :, 1:, :]
    dw = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]) * mask[:, :, :, 1:]
    denom = (mask[:, :, 1:, :].sum() + mask[:, :, :, 1:].sum()) * pred.shape[1] + 1e-8
    return (dh.sum() + dw.sum()) / denom


@dataclass
class LossBuilder:
    name: str
    w_l1: float
    w_perceptual: float
    w_tv: float
    lpips_net: nn.Module | None
    l1_fn: nn.Module

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        terms: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)

        if self.w_l1 > 0:
            l1 = masked_l1_loss(pred, target, mask, self.l1_fn)
            terms["loss_l1"] = l1
            total = total + self.w_l1 * l1

        if self.w_perceptual > 0:
            if self.lpips_net is None:
                raise ValueError("Perceptual loss requested but LPIPS network is not initialized.")
            p = masked_lpips_loss(pred, target, mask, self.lpips_net, mean, std)
            terms["loss_perceptual"] = p
            total = total + self.w_perceptual * p

        if self.w_tv > 0:
            tv = tv_loss_on_hole(pred, mask)
            terms["loss_tv"] = tv
            total = total + self.w_tv * tv

        if not terms:
            raise ValueError("No active loss terms. Set at least one positive loss weight.")

        terms["loss_total"] = total
        return total, terms


def _get_float(cfg: Any, key: str, default: float) -> float:
    raw = getattr(cfg, key, default)
    return float(raw)


def build_train_loss(train_cfg, device: torch.device):
    loss_cfg = getattr(train_cfg, "loss", None)
    if loss_cfg is None:
        # Backward-compatible default.
        loss_cfg = type("LossCfg", (), {})()
        loss_cfg.name = "l1"
        loss_cfg.weights = type("LossWeights", (), {"l1": 1.0, "perceptual": 0.0, "tv": 0.0})()

    name = str(getattr(loss_cfg, "name", "l1")).lower()
    weights = getattr(loss_cfg, "weights", None)
    if weights is None:
        raise ValueError("train.loss.weights is required.")

    w_l1 = _get_float(weights, "l1", 0.0)
    w_perceptual = _get_float(weights, "perceptual", 0.0)
    w_tv = _get_float(weights, "tv", 0.0)

    if min(w_l1, w_perceptual, w_tv) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if w_l1 == 0 and w_perceptual == 0 and w_tv == 0:
        raise ValueError("At least one loss weight must be > 0.")

    expected_by_name = {
        "l1": (w_l1 > 0 and w_perceptual == 0 and w_tv == 0),
        "l1_perceptual": (w_l1 > 0 and w_perceptual > 0 and w_tv == 0),
        "l1_perceptual_tv": (w_l1 > 0 and w_perceptual > 0 and w_tv > 0),
    }
    if name not in expected_by_name:
        raise ValueError(f"Unsupported train.loss.name: {name}")
    if not expected_by_name[name]:
        raise ValueError(
            f"train.loss.name='{name}' is inconsistent with weights "
            f"(l1={w_l1}, perceptual={w_perceptual}, tv={w_tv})."
        )

    lpips_net = None
    if w_perceptual > 0:
        try:
            import lpips  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "Perceptual loss requires lpips package. Install requirements or set perceptual weight to 0."
            ) from exc
        perc_cfg = getattr(loss_cfg, "perceptual", None)
        net = str(getattr(perc_cfg, "net", "vgg")) if perc_cfg is not None else "vgg"
        lpips_net = lpips.LPIPS(net=net).to(device).eval()
        for p in lpips_net.parameters():
            p.requires_grad = False

    l1_fn = nn.L1Loss(reduction="none")
    return LossBuilder(name=name, w_l1=w_l1, w_perceptual=w_perceptual, w_tv=w_tv, lpips_net=lpips_net, l1_fn=l1_fn)
