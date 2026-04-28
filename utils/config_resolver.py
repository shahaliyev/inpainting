from pathlib import Path


CONFIGS_DIR = Path("configs")


def resolve_config_path(group: str, name: str) -> str:
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown {group} config '{name}': {path}")
    return str(path)


def require_cfg_fields(cfg, field_paths, context: str):
    for field_path in field_paths:
        node = cfg
        for part in field_path.split("."):
            if not hasattr(node, part):
                raise ValueError(f"{context} is missing required field '{field_path}'")
            node = getattr(node, part)
