from pathlib import Path


def cfg_name(path_like: str | None) -> str:
    if not path_like:
        return "unknown"
    return Path(str(path_like)).stem


def startup_summary_line(*, dataset_yaml: str | None, mask_yaml: str | None, model_yaml: str | None, train_yaml: str | None = None, metric_scope: str | None = None) -> str:
    parts = [
        f"dataset={cfg_name(dataset_yaml)}",
        f"mask={cfg_name(mask_yaml)}",
        f"model={cfg_name(model_yaml)}",
    ]
    if train_yaml is not None:
        parts.append(f"train={cfg_name(train_yaml)}")
    if metric_scope is not None:
        parts.append(f"metric_scope={metric_scope}")
    return "startup: " + " ".join(parts)
