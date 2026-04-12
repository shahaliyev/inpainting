from typing import Any, Optional

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data.dataset import CarpetDataset, DTDDataset, ImageNetDataset
from data.inpainting import InpaintingDataset
from mask.build import build_mask_generator


def build_transform(spec: Optional[Any]):
    if spec is None:
        return None

    ops = []

    if getattr(spec, "resize", None) is not None:
        ops.append(T.Resize(spec.resize))

    if getattr(spec, "random_crop", None) is not None:
        ops.append(T.RandomCrop(spec.random_crop))

    if getattr(spec, "center_crop", None) is not None:
        ops.append(T.CenterCrop(spec.center_crop))

    if getattr(spec, "hflip", None) is not None:
        ops.append(T.RandomHorizontalFlip(p=float(spec.hflip)))

    if getattr(spec, "vflip", None) is not None:
        ops.append(T.RandomVerticalFlip(p=float(spec.vflip)))

    if getattr(spec, "to_tensor", False):
        ops.append(T.ToTensor())

    if getattr(spec, "normalize", None) is not None:
        n = spec.normalize
        ops.append(T.Normalize(mean=list(n.mean), std=list(n.std)))

    if not ops:
        return None

    return T.Compose(ops)


def _select_transform_cfg(dataset_cfg: Any, split: str):
    if split == "train":
        return getattr(dataset_cfg.transform, "train", None)
    return (
        getattr(dataset_cfg.transform, "eval", None)
        or getattr(dataset_cfg.transform, "val", None)
        or getattr(dataset_cfg.transform, "test", None)
    )


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader_kwargs(loader_cfg: Any, split: str):
    is_train = split == "train"

    batch_size = int(getattr(loader_cfg, "batch_size", 1))

    split_cfg = getattr(loader_cfg, "train" if is_train else "eval", None)
    dev_cfg = getattr(loader_cfg, "cuda" if torch.cuda.is_available() else "cpu", None)

    def pick(key: str, default):
        if split_cfg is not None and getattr(split_cfg, key, None) is not None:
            return getattr(split_cfg, key)
        if dev_cfg is not None and getattr(dev_cfg, key, None) is not None:
            return getattr(dev_cfg, key)
        v = getattr(loader_cfg, key, None)
        return default if v is None else v

    num_workers = int(pick("num_workers", 0))
    pin_memory = bool(pick("pin_memory", False))
    persistent_workers = bool(pick("persistent_workers", False)) and num_workers > 0
    prefetch_factor = pick("prefetch_factor", None)
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)

    shuffle = bool(pick("shuffle", is_train)) if is_train else False
    drop_last = bool(pick("drop_last", is_train)) if is_train else False

    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0 and prefetch_factor is not None:
        kw["prefetch_factor"] = prefetch_factor
    return kw


def build_base_dataset(dataset_cfg: Any, split: str):
    tcfg = _select_transform_cfg(dataset_cfg, split)
    transform = build_transform(tcfg)
    dataset_type = getattr(dataset_cfg, "type", getattr(dataset_cfg, "name", "dtd"))
    if dataset_type == "carpet":
        return CarpetDataset(dataset_cfg, split=split, transform=transform)
    if dataset_type == "imagenet":
        return ImageNetDataset(dataset_cfg, split=split, transform=transform)
    return DTDDataset(dataset_cfg, split=split, transform=transform)


def build_dataloader(dataset_cfg: Any, loader_cfg: Any, split: str, mask_cfg: Optional[Any] = None, global_seed: int = 42, eval_seed: int = 777):
    base_ds = build_base_dataset(dataset_cfg, split)
    ds = base_ds

    if mask_cfg is not None:
        mask_gen = build_mask_generator(mask_cfg, split=split, train_seed=global_seed, eval_seed=eval_seed)
        ds = InpaintingDataset(dataset=base_ds, mask_gen=mask_gen)

    kw = _loader_kwargs(loader_cfg, split)

    if split == "train":
        g = torch.Generator()
        g.manual_seed(int(global_seed) & 0x7FFFFFFF)
        kw["generator"] = g
        if kw["num_workers"] > 0:
            kw["worker_init_fn"] = _seed_worker

    return DataLoader(ds, **kw)