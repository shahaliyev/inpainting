import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class CarpetDataset(Dataset):
    """Dataset for folder-based layout: root/images/{split}/{camera}/...images
    Example: DATA_PATH/carpet/images/train/Cam_L, images/train/Cam_R, images/val/Cam_L, images/val/Cam_R.
    Returns same interface as DTDDataset: {"image": img, "path": str(path)}.
    """

    def __init__(self, cfg, split="train", transform=None):
        self.root = Path(cfg.root)
        self.images_dir = self.root / cfg.images_dir 
        self.split = split
        camera_folders = list(getattr(cfg, "camera_folders"))
        exts = tuple(getattr(cfg, "image_exts"))

        samples = []
        for cam in camera_folders:
            folder = self.images_dir / split / cam
            if folder.exists():
                for ext in exts:
                    samples.extend(folder.glob(f"*.{ext}"))
                    # Match uppercase too (e.g. .JPG on case-sensitive filesystems)
                    if ext.lower() == ext:
                        samples.extend(folder.glob(f"*.{ext.upper()}"))

        samples = sorted(set(samples))

        limit = getattr(cfg, "limit", None)
        if limit is not None:
            limit = int(limit)
            if limit < len(samples):
                shuffle = bool(getattr(cfg, "limit_shuffle", True))
                seed = int(getattr(cfg, "limit_seed", 123))
                if shuffle:
                    rng = random.Random(seed)
                    rng.shuffle(samples)
                samples = samples[:limit]

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "path": str(path)}


class DTDDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.root = Path(cfg.root)
        self.images_dir = self.root / cfg.images_dir
        self.labels_dir = self.root / cfg.labels_dir

        split_file = self.labels_dir / cfg.split_files[split]

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        samples = [self.images_dir / line for line in lines]

        limit = getattr(cfg, "limit", None)
        if limit is not None:
            limit = int(limit)
            if limit < len(samples):
                shuffle = bool(getattr(cfg, "limit_shuffle", True))
                seed = int(getattr(cfg, "limit_seed", 123))
                if shuffle:
                    rng = random.Random(seed)
                    rng.shuffle(samples)
                samples = samples[:limit]

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "path": str(path)}


class ImageNetDataset(Dataset):
    """
    ImageNet subset with layout:
      train: root/train.X{1,2,3,4}/<class_id>/*.JPEG
      val:   root/val.X/<class_id>/*.JPEG

    Classification labels are not used (inpainting only).
    Returns {"image": tensor, "path": str}.
    """

    def __init__(self, cfg, split="train", transform=None):
        self.root = Path(cfg.root)
        exts = list(getattr(cfg, "image_exts", ["JPEG", "jpeg", "jpg"]))

        if split == "train":
            search_dirs = [self.root / d for d in getattr(cfg, "train_dirs", ["train.X1", "train.X2", "train.X3", "train.X4"])]
        else:
            search_dirs = [self.root / str(getattr(cfg, "val_dir", "val.X"))]

        samples = []
        for top_dir in search_dirs:
            if not top_dir.exists():
                continue
            for class_dir in sorted(top_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for ext in exts:
                    samples.extend(class_dir.glob(f"*.{ext}"))

        samples = sorted(set(samples))

        limit = getattr(cfg, "limit", None)
        if limit is not None:
            limit = int(limit)
            if limit < len(samples):
                shuffle = bool(getattr(cfg, "limit_shuffle", True))
                seed = int(getattr(cfg, "limit_seed", 123))
                if shuffle:
                    rng = random.Random(seed)
                    rng.shuffle(samples)
                samples = samples[:limit]

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "path": str(path)}