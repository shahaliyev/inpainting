"""
Build ImageNet-Simple and/or ImageNet-Complex from a raw ImageNet subset.

Usage:
  # Build both subsets for all folders (default)
  python tools/imagenet.py

  # Build only one subset
  python tools/imagenet.py --subset simple
  python tools/imagenet.py --subset complex

  # Process a single folder (e.g. for a quick test)
  python tools/imagenet.py --train_dirs train.X1

  # Combine: one subset, specific folders
  python tools/imagenet.py --subset complex --train_dirs train.X2 train.X4

Idempotency:
  - Raw scores are cached in DATA_PATH/imagenet/.scores_cache.npz so images
    are never re-scored across runs.
  - A class output folder that already has >= k files is silently skipped,
    so runs on different folders accumulate safely.

Expected input  (DATA_PATH/imagenet/):
  train.X1/<class_id>/*.JPEG  ...  train.X4/<class_id>/*.JPEG
  val.X/<class_id>/*.JPEG

Output  (DATA_PATH/):
  imagenet-simple/train/<class_id>/...
  imagenet-simple/val/<class_id>/...
  imagenet-complex/train/<class_id>/...
  imagenet-complex/val/<class_id>/...
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import sobel as skimage_sobel


VAL_DIR = "val.X"
ALL_TRAIN_DIRS = ["train.X1", "train.X2", "train.X3", "train.X4"]
IMAGE_EXTS = {".jpeg", ".jpg"}
CACHE_FILE = ".scores_cache.npz"


def parse_args():
    data_path = os.environ.get("DATA_PATH", "")
    ap = argparse.ArgumentParser(
        description="Build ImageNet-Simple / ImageNet-Complex complexity-ranked subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default=os.path.join(data_path, "imagenet"),
                    help="Path to raw ImageNet folder (default: $DATA_PATH/imagenet)")
    ap.add_argument("--out_dir", default=data_path,
                    help="Parent output dir (default: $DATA_PATH)")
    ap.add_argument("--subset", choices=["simple", "complex"], default=None,
                    help="Which subset to build. Omit to build both.")
    ap.add_argument("--train_dirs", nargs="+", default=ALL_TRAIN_DIRS,
                    help="Train subdirectory names to include")
    ap.add_argument("--no_val", action="store_true",
                    help="Skip the val split")
    ap.add_argument("--train_k", type=int, default=500,
                    help="Images per class per subset for train (gap: middle 300 discarded)")
    ap.add_argument("--val_k", type=int, default=25,
                    help="Images per class per subset for val (no gap)")
    ap.add_argument("--min_size", type=int, default=100,
                    help="Skip images where min(W,H) < min_size")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Gradient threshold for edge-density metric e(I)")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Parallel worker processes for scoring")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict:
    """Return {path_str: (v, g, e)}."""
    if not cache_path.exists():
        return {}
    d = np.load(cache_path)
    return {str(p): (float(v), float(g), float(e))
            for p, v, g, e in zip(d["paths"], d["v"], d["g"], d["e"])}


def save_cache(cache_path: Path, cache: dict) -> None:
    paths = list(cache.keys())
    np.savez(
        cache_path,
        paths=np.array(paths),
        v=np.array([cache[p][0] for p in paths], dtype=np.float32),
        g=np.array([cache[p][1] for p in paths], dtype=np.float32),
        e=np.array([cache[p][2] for p in paths], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_paths(data_dir: Path, train_dirs: list, include_val: bool):
    records = []
    for d in train_dirs:
        top = data_dir / d
        if not top.exists():
            print(f"  WARNING: not found: {top}", file=sys.stderr)
            continue
        for cls in sorted(top.iterdir()):
            if cls.is_dir():
                for p in cls.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        records.append(("train", cls.name, p))
    if include_val:
        val_top = data_dir / VAL_DIR
        if not val_top.exists():
            print(f"  WARNING: not found: {val_top}", file=sys.stderr)
        else:
            for cls in sorted(val_top.iterdir()):
                if cls.is_dir():
                    for p in cls.iterdir():
                        if p.suffix.lower() in IMAGE_EXTS:
                            records.append(("val", cls.name, p))
    return records


# ---------------------------------------------------------------------------
# Scoring (worker)
# ---------------------------------------------------------------------------

def _score_one(args):
    path_str, min_size, tau = args
    try:
        img = Image.open(path_str)
        W, H = img.size
        if min(W, H) < min_size:
            return None
        scale = 256.0 / min(W, H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        img = img.crop(((nw - 224) // 2, (nh - 224) // 2,
                        (nw - 224) // 2 + 224, (nh - 224) // 2 + 224))
        Y = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        G = skimage_sobel(Y)
        return (path_str, float(np.var(Y)), float(G.mean()), float((G > tau).mean()))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _zscore(arr):
    a = np.asarray(arr, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)


def _already_done(folder: Path, k: int) -> bool:
    return folder.exists() and sum(1 for _ in folder.iterdir()) >= k


def _copy(paths, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, dst / Path(p).name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    cache_path = data_dir / CACHE_FILE
    subsets = ["simple", "complex"] if args.subset is None else [args.subset]

    if not data_dir.exists():
        sys.exit(f"ERROR: data_dir does not exist: {data_dir}")
    if not out_dir.exists():
        sys.exit(f"ERROR: out_dir does not exist: {out_dir}")

    # ── collect ────────────────────────────────────────────────────────────
    print(f"Subsets     : {subsets}")
    print(f"Train dirs  : {args.train_dirs}")
    records = collect_paths(data_dir, args.train_dirs, include_val=not args.no_val)
    if not records:
        sys.exit("ERROR: no images found.")
    print(f"Images found: {len(records):,}")

    # ── cache ──────────────────────────────────────────────────────────────
    cache = load_cache(cache_path)
    print(f"Cache       : {len(cache):,} images already scored")

    uncached = [(split, cls, p) for split, cls, p in records if str(p) not in cache]
    print(f"To score    : {len(uncached):,} new images")

    n_skipped = 0
    if uncached:
        print(f"Scoring with {args.num_workers} worker(s) ...")
        with Pool(args.num_workers) as pool:
            todo = [(str(p), args.min_size, args.tau) for _, _, p in uncached]
            for i, r in enumerate(pool.imap_unordered(_score_one, todo, chunksize=128), 1):
                if i % 10_000 == 0 or i == len(todo):
                    print(f"  {i:,}/{len(todo):,}", flush=True)
                if r is not None:
                    cache[r[0]] = (r[1], r[2], r[3])
                else:
                    n_skipped += 1
        print(f"Scored {len(cache):,} total  |  skipped {n_skipped} (too small/unreadable)")
        save_cache(cache_path, cache)
        print(f"Cache saved → {cache_path}")
    else:
        print("All images already cached.")

    # ── global z-score ─────────────────────────────────────────────────────
    all_paths = list(cache.keys())
    c_map = {p: float(c) for p, c in zip(
        all_paths,
        (_zscore([cache[p][0] for p in all_paths]) +
         _zscore([cache[p][1] for p in all_paths]) +
         _zscore([cache[p][2] for p in all_paths])) / 3.0
    )}

    # ── group by (split, class_id) ─────────────────────────────────────────
    groups = defaultdict(list)
    for split, class_id, p in records:
        if str(p) in cache:
            groups[(split, class_id)].append(str(p))

    # ── per-subset copy ────────────────────────────────────────────────────
    for subset in subsets:
        subset_dir = out_dir / f"imagenet-{subset}"
        print(f"\nBuilding imagenet-{subset} → {subset_dir}")
        done_count = copied_count = small_count = 0

        for (split, class_id), paths in sorted(groups.items()):
            k = args.train_k if split == "train" else args.val_k
            class_out = subset_dir / split / class_id

            if _already_done(class_out, k):
                done_count += 1
                continue

            if len(paths) < 2 * k:
                small_count += 1
                continue

            paths.sort(key=lambda p: c_map[p])
            chosen = paths[:k] if subset == "simple" else paths[-k:]
            _copy(chosen, class_out)
            copied_count += k

        print(f"  Copied : {copied_count:,}  |  already done: {done_count}  |  too small: {small_count}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
