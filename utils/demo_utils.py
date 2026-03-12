"""Shared helpers for quick_test.ipynb: denorm, grid viz, dataset stats, model_info."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


def get_norm_from_cfg(cfg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) from cfg.norm."""
    norm = getattr(cfg, "norm", None)
    if norm is None:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    else:
        mean = np.array(getattr(norm, "mean", [0.5, 0.5, 0.5]))
        std = np.array(getattr(norm, "std", [0.5, 0.5, 0.5]))
    return mean, std


def denorm(
    tensor: Union[torch.Tensor, np.ndarray],
    mean: Union[np.ndarray, List[float]],
    std: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """Convert tensor to numpy and unnormalize (support 4D NCHW and 3D CHW)."""
    if torch.is_tensor(tensor):
        arr = tensor.cpu().numpy()
    else:
        arr = np.asarray(tensor)
    mean = np.asarray(mean, dtype=arr.dtype)
    std = np.asarray(std, dtype=arr.dtype)
    if arr.ndim == 4:
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
    elif arr.ndim == 3:
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    arr = np.clip(arr * std + mean, 0, 1).astype(np.float32)
    if arr.ndim == 4:
        arr = arr.transpose(0, 2, 3, 1)
    elif arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return arr


def show_grid(
    images: List[np.ndarray],
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
) -> None:
    """Plot images in a grid; no per-cell titles, axis off."""
    n = min(len(images), nrows * ncols)
    if figsize is None:
        figsize = (2 * ncols, 2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i < n:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=1)
            else:
                ax.imshow(img)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def dataset_stats(dataset_cfg: Any, split: str) -> dict:
    """Build dataset for split, return dict with len, sample_shape, ready."""
    from omegaconf import OmegaConf
    from data.build import build_base_dataset

    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)
    root = Path(dataset_cfg.root)
    ds = build_base_dataset(dataset_cfg, split)
    n = len(ds)
    if n == 0:
        return {"len": 0, "sample_shape": None, "ready": False}
    sample = ds[0]
    img = sample["image"]
    if hasattr(img, "shape"):
        shape = tuple(img.shape)
    else:
        shape = (getattr(img, "size", (0, 0))[1], getattr(img, "size", (0, 0))[0], 3)
    return {
        "len": n,
        "sample_shape": shape,
        "ready": n > 0 and root.exists(),
    }


def model_info(model: torch.nn.Module, x: torch.Tensor) -> dict:
    """Run model(x), return dict with output_shape and nparams."""
    with torch.no_grad():
        out = model(x)
    nparams = sum(p.numel() for p in model.parameters())
    return {
        "output_shape": tuple(out.shape),
        "nparams": nparams,
    }
