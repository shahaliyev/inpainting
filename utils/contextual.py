"""
Contextual Loss (CX) metric — Mechrez et al., ECCV 2018.

Implements L_CX = -log(CX) over VGG19 feature maps. Used as an optional
evaluation metric (lower is better), gated by eval CLI flags.

Reference formulas:
  relative distance, soft-max similarity, CX = mean_j max_i CX_ij
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# torchvision VGG19 features indices ending at each relu{k}_2
_VGG19_LAYER_SLICE = {
    "relu1_2": 4,
    "relu2_2": 9,
    "relu3_2": 16,
    "relu4_2": 23,
    "relu5_2": 30,
}

# ImageNet normalization expected by torchvision VGG weights
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def contextual_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    band_width: float = 0.5,
) -> torch.Tensor:
    """
    Contextual similarity CX(X, Y) per batch item.

    Args:
        x, y: feature maps (N, C, H, W) — same spatial size.
        band_width: h in the paper (default 0.5).

    Returns:
        (N,) tensor of CX values in (0, 1].
    """
    if x.shape != y.shape:
        raise ValueError(f"Feature shapes must match, got {tuple(x.shape)} vs {tuple(y.shape)}")

    dist = _cosine_distance(x, y)
    dist_tilde = _relative_distance(dist)
    cx = _feature_cx(dist_tilde, band_width)
    # Eq(1): for each target feature j, take max over generated features i
    return torch.mean(torch.max(cx, dim=1)[0], dim=1)


def contextual_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    band_width: float = 0.5,
) -> torch.Tensor:
    """Scalar L_CX = -log(CX) averaged over the batch."""
    cx = contextual_similarity(x, y, band_width=band_width)
    return torch.mean(-torch.log(cx.clamp(min=1e-5)))


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Channel-wise mean of y (paper footnote); average over batch + spatial.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_n = F.normalize(x - y_mu, p=2, dim=1)
    y_n = F.normalize(y - y_mu, p=2, dim=1)
    n, c, _, _ = x.shape
    x_flat = x_n.reshape(n, c, -1)
    y_flat = y_n.reshape(n, c, -1)
    # (N, HW, HW): d_ij between x_i and y_j
    return 1.0 - torch.bmm(x_flat.transpose(1, 2), y_flat)


def _relative_distance(dist: torch.Tensor) -> torch.Tensor:
    dist_min, _ = torch.min(dist, dim=2, keepdim=True)
    return dist / (dist_min + 1e-5)


def _feature_cx(dist_tilde: torch.Tensor, band_width: float) -> torch.Tensor:
    w = torch.exp((1.0 - dist_tilde) / band_width)
    return w / torch.sum(w, dim=2, keepdim=True)


class _VGG19Features(nn.Module):
    """Frozen VGG19 feature extractor for selected relu layers."""

    def __init__(self, layers: Sequence[str]):
        super().__init__()
        unknown = [l for l in layers if l not in _VGG19_LAYER_SLICE]
        if unknown:
            raise ValueError(
                f"Unknown CX layers {unknown}. Available: {sorted(_VGG19_LAYER_SLICE)}"
            )
        self.layers = list(layers)
        # End slice must cover the deepest requested layer
        end = max(_VGG19_LAYER_SLICE[l] for l in self.layers)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:end].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self._slice_ends = [_VGG19_LAYER_SLICE[l] for l in self.layers]
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x_01: torch.Tensor) -> list[torch.Tensor]:
        """x_01: RGB in [0, 1]. Returns one feature map per configured layer."""
        x = (x_01 - self.mean) / self.std
        collected: dict[int, torch.Tensor] = {}
        h = x
        target_ends = set(self._slice_ends)
        for i, layer in enumerate(self.vgg, start=1):
            h = layer(h)
            if i in target_ends:
                collected[i] = h
        return [collected[_VGG19_LAYER_SLICE[l]] for l in self.layers]


class ContextualMetric(nn.Module):
    """
    Evaluation wrapper: CX loss over VGG features of [0,1] RGB images.

    Optional binary mask (1 = hole) restricts comparison to masked feature
    locations after downsampling the mask to each feature map size.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("relu3_2", "relu4_2"),
        band_width: float = 0.5,
        max_samples: int = 65 * 65,
    ):
        super().__init__()
        self.band_width = float(band_width)
        self.max_samples = int(max_samples)
        self.features = _VGG19Features(layers)

    @torch.no_grad()
    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_01, target_01: (B, 3, H, W) in [0, 1]
            mask: optional (B, 1, H, W), 1 = region to score

        Returns:
            Scalar CX loss (lower is better).
        """
        pred_feats = self.features(pred_01.float())
        target_feats = self.features(target_01.float())
        layer_losses = []
        for pf, tf in zip(pred_feats, target_feats):
            if mask is None:
                pf_s, tf_s = _maybe_subsample_pair(pf, tf, self.max_samples)
                layer_losses.append(contextual_loss(pf_s, tf_s, band_width=self.band_width))
            else:
                # Per-image: masked feature counts can differ across the batch.
                per_img = []
                for i in range(pf.shape[0]):
                    pi, ti = _masked_feature_pair(
                        pf[i], tf[i], mask[i : i + 1], self.max_samples
                    )
                    per_img.append(contextual_loss(pi, ti, band_width=self.band_width))
                layer_losses.append(torch.stack(per_img).mean())
        return torch.stack(layer_losses).mean()


def build_contextual_metric(
    device: torch.device,
    layers: Iterable[str] = ("relu3_2", "relu4_2"),
    band_width: float = 0.5,
    max_samples: int = 65 * 65,
) -> ContextualMetric:
    net = ContextualMetric(
        layers=tuple(layers),
        band_width=band_width,
        max_samples=max_samples,
    )
    return net.to(device).eval()


def _masked_feature_pair(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    mask: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather masked feature vectors for one image into (1, C, 1, M) maps.

    pred_feat/target_feat: (C, Hf, Wf); mask: (1, 1, H, W) at image resolution.
    """
    c, hf, wf = pred_feat.shape
    mask_f = F.interpolate(mask.float(), size=(hf, wf), mode="nearest")[0, 0] > 0.5
    p = pred_feat.reshape(c, -1)[:, mask_f.reshape(-1)]
    t = target_feat.reshape(c, -1)[:, mask_f.reshape(-1)]
    m_count = p.shape[1]
    if m_count == 0:
        p = pred_feat.reshape(c, -1)
        t = target_feat.reshape(c, -1)
        m_count = p.shape[1]
    if m_count > max_samples:
        idx = torch.randperm(m_count, device=p.device)[:max_samples]
        p = p[:, idx]
        t = t[:, idx]
        m_count = max_samples
    return p.reshape(1, c, 1, m_count), t.reshape(1, c, 1, m_count)


def _maybe_subsample_pair(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, h, w = pred_feat.shape
    hw = h * w
    if hw <= max_samples:
        return pred_feat, target_feat
    # Shared random indices for pred/target so correspondence stays spatial
    idx = torch.randperm(hw, device=pred_feat.device)[:max_samples]
    p = pred_feat.reshape(n, c, hw)[:, :, idx].reshape(n, c, 1, max_samples)
    t = target_feat.reshape(n, c, hw)[:, :, idx].reshape(n, c, 1, max_samples)
    return p, t
