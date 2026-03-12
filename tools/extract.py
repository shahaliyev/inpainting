"""
Dataset Extraction Utility

Purpose
-------
Extract compressed dataset archives into a standardized location defined by
the DATA_PATH environment variable. This keeps datasets organized and avoids
hardcoding paths inside training code.

Destination layout
------------------
DATA_PATH/
    dataset_name/
        files...

The dataset folder name can be provided explicitly or inferred from the
archive filename.

Supported archive formats
-------------------------
.zip
.tar
.tar.gz
.tgz
.tar.bz2
.tbz2

Usage
-----
1. Set the DATA_PATH environment variable.

Linux / macOS
export DATA_PATH=/data

Windows (PowerShell)
$env:DATA_PATH="D:\\datasets"

Windows (cmd)
set DATA_PATH=D:\datasets

2. Run extraction.

python extract.py --archive /path/to/archive.tar.gz
python extract.py --archive /path/to/archive.tar.gz --name dataset_name

Examples
--------
python extract.py --archive downloads/dtd-r1.0.1.tar.gz
python extract.py --archive downloads/carpet.zip --name carpet

Result
------
DATA_PATH/
    dtd/
    carpet/

Behavior
--------
• Creates DATA_PATH/dataset_name if it does not exist
• Prevents overwriting existing datasets unless --force is used
• Adds a ".extracted" marker file after successful extraction
• Protects against unsafe archive paths

Options
-------
--archive PATH
    Path to archive file.

--name NAME
    Optional dataset folder name.
    If omitted, inferred from archive filename.

--force
    Overwrite existing extracted dataset.

--no-marker
    Do not create the ".extracted" marker file.

Typical workflow
----------------
Download archive → run this script → dataset appears inside DATA_PATH →
training code reads datasets from that location.
"""

import argparse
import os
import tarfile
import zipfile
from pathlib import Path


def _env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} environment variable is not set")
    p = Path(v)
    if not p.exists():
        raise RuntimeError(f"{name} points to a non-existent path: {p}")
    if not p.is_dir():
        raise RuntimeError(f"{name} must point to a directory: {p}")
    return p


def _infer_name(archive: Path) -> str:
    n = archive.name
    for ext in (".tar.gz", ".tar.bz2", ".tgz", ".tbz2", ".tar", ".zip", ".gz", ".bz2"):
        if n.lower().endswith(ext):
            n = n[: -len(ext)]
            break
    return n


def _safe_within_base(base: Path, target: Path) -> bool:
    base = base.resolve()
    target = target.resolve()
    return str(target).startswith(str(base))


def _safe_extract_tar(archive_path: Path, out_dir: Path):
    with tarfile.open(archive_path, "r:*") as tar:
        for m in tar.getmembers():
            t = out_dir / m.name
            if not _safe_within_base(out_dir, t):
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tar.extractall(out_dir)


def _safe_extract_zip(archive_path: Path, out_dir: Path):
    with zipfile.ZipFile(archive_path, "r") as z:
        for name in z.namelist():
            t = out_dir / name
            if not _safe_within_base(out_dir, t):
                raise RuntimeError(f"Unsafe path in zip: {name}")
        z.extractall(out_dir)


def extract_archive(archive_path: Path, out_dir: Path):
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        _safe_extract_zip(archive_path, out_dir)
        return
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        _safe_extract_tar(archive_path, out_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset archives into $DATA_PATH/<dataset_name>")
    parser.add_argument("--archive", type=Path, required=True, help="Path to archive (.zip, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2)")
    parser.add_argument("--name", type=str, default=None, help="Dataset folder name (default: inferred from archive filename)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset folder contents")
    parser.add_argument("--no-marker", action="store_true", help="Do not write .extracted marker file")

    args = parser.parse_args()

    if not args.archive.exists():
        raise FileNotFoundError(f"Archive not found: {args.archive}")

    data_path = _env_path("DATA_PATH")
    ds_name = args.name or _infer_name(args.archive)
    out_dir = data_path / ds_name
    marker = out_dir / ".extracted"

    if marker.exists() and not args.force:
        print(f"Already extracted (marker present): {out_dir}")
        return

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"Output not empty: {out_dir} (use --force to overwrite)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force and out_dir.exists():
        for p in out_dir.iterdir():
            if p.name == ".extracted":
                continue
            if p.is_dir():
                for sub in sorted(p.rglob("*"), reverse=True):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                    elif sub.is_dir():
                        try:
                            sub.rmdir()
                        except OSError:
                            pass
                try:
                    p.rmdir()
                except OSError:
                    pass
            else:
                p.unlink()

    print(f"DATA_PATH: {data_path}")
    print(f"Extracting: {args.archive}")
    print(f"Destination: {out_dir}")

    extract_archive(args.archive, out_dir)

    if not args.no_marker:
        marker.write_text("ok\n", encoding="utf-8")

    print("Extraction complete")


if __name__ == "__main__":
    main()