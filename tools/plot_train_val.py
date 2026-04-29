import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot train/val loss curves from metrics.csv")
    ap.add_argument("--metrics", required=True, help="Path to runs/<run>/metrics.csv")
    ap.add_argument("--out", default=None, help="Output image path (default: sibling loss_curves.png)")
    ap.add_argument("--x", choices=["step", "epoch"], default="step", help="X-axis for the curves")
    ap.add_argument("--show_train_running", action="store_true", help="Also plot running train split")
    return ap.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    out_path = Path(args.out) if args.out else (metrics_path.parent / "loss_curves.png")

    train_epoch_x, train_epoch_y = [], []
    val_x, val_y = [], []
    train_running_x, train_running_y = [], []

    with open(metrics_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", ""))
            x_raw = row.get(args.x, "")
            y_raw = row.get("loss", "")
            if x_raw in {"", None} or y_raw in {"", None}:
                continue
            x_val = float(x_raw)
            y_val = float(y_raw)
            if split == "train_epoch":
                train_epoch_x.append(x_val)
                train_epoch_y.append(y_val)
            elif split == "val":
                val_x.append(x_val)
                val_y.append(y_val)
            elif split == "train":
                train_running_x.append(x_val)
                train_running_y.append(y_val)

    if not train_epoch_x and not val_x:
        raise ValueError("metrics.csv has no train_epoch/val rows to plot.")

    plt.figure(figsize=(8, 4.5))
    if train_epoch_x:
        plt.plot(train_epoch_x, train_epoch_y, label="train_epoch", linewidth=2.0)
    if val_x:
        plt.plot(val_x, val_y, label="val", linewidth=2.0)
    if args.show_train_running and train_running_x:
        plt.plot(train_running_x, train_running_y, label="train_running", alpha=0.45, linewidth=1.0)

    plt.xlabel(args.x)
    plt.ylabel("loss")
    plt.title("Training / Validation Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
