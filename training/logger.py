import csv
from pathlib import Path

import torch

from utils.vis import save_triplet


class MetricsLogger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.images_dir = self.run_dir / "images"

        self.images_dir.mkdir(parents=True, exist_ok=True)

        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "split", "loss", "lr"])

    def log(self, epoch, step, split, loss, lr):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, split, loss, lr])

    def save_train_triplet(
        self,
        step,
        img,
        mask,
        recon,
        mean,
        std,
    ):
        path = self.images_dir / f"train_step_{step:06d}.png"

        save_triplet(
            path,
            img[:1],
            mask[:1],
            recon[:1],
            mean,
            std,
        )

    def save_val_triplet(
        self,
        epoch,
        img,
        mask,
        recon,
        mean,
        std,
    ):
        path = self.images_dir / f"val_epoch_{epoch:03d}.png"

        save_triplet(
            path,
            img[:1],
            mask[:1],
            recon[:1],
            mean,
            std,
        )