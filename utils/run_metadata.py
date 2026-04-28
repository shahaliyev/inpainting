import json
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _sanitize_token(value: str) -> str:
    clean = []
    for ch in value.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            clean.append(ch)
        else:
            clean.append("-")
    token = "".join(clean).strip("-_")
    return token or "na"


def _stem(path_like: str) -> str:
    return _sanitize_token(Path(path_like).stem)


def _collect_ratio_tokens(mask_cfg: Any) -> list[str]:
    name = str(getattr(mask_cfg, "name", "")).lower()
    tokens = []
    ratios = list(getattr(mask_cfg, "ratios", []) or [])
    if ratios:
        ratios_sorted = "-".join(str(int(r)) for r in sorted(int(r) for r in ratios))
        tokens.append(f"r{ratios_sorted}")
    if name == "mixed":
        for gen in list(getattr(mask_cfg, "generators", []) or []):
            gen_ratios = list(getattr(gen, "ratios", []) or [])
            if gen_ratios:
                gen_name = str(getattr(gen, "name", "g")).lower()
                gen_ratio_str = "-".join(str(int(r)) for r in sorted(int(r) for r in gen_ratios))
                tokens.append(f"{_sanitize_token(gen_name)}r{gen_ratio_str}")
    return sorted(set(tokens))


def build_train_run_name(model_yaml: str, dataset_yaml: str, mask_cfg: Any, seed: int) -> str:
    model = _stem(model_yaml)
    dataset = _stem(dataset_yaml)
    mask_name = _sanitize_token(str(getattr(mask_cfg, "name", "mask")))
    ratio_tokens = _collect_ratio_tokens(mask_cfg)
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    parts = [model, dataset, mask_name]
    parts.extend(ratio_tokens)
    parts.append(f"s{int(seed)}")
    parts.append(timestamp)
    return "__".join(parts)


def save_run_metadata(
    run_dir: Path,
    *,
    run_name: str,
    seed: int,
    args_dict: dict[str, Any],
    config_paths: dict[str, str],
    resolved_cfg: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "config_paths": config_paths,
        "args": args_dict,
        "resolved": resolved_cfg,
    }
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_resolved_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(resolved_path))
