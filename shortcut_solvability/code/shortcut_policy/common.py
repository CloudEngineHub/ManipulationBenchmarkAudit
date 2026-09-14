from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


STEP_RE = re.compile(r"^step(\d+)\.pt$")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"expected YAML mapping in {path}")
    return cfg


def cfg_get(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def expand_placeholders(
    cfg: dict[str, Any],
    *,
    data_root: str | None = None,
    checkpoint_root: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    replacements = {
        "<DATA_ROOT>": data_root,
        "<CHECKPOINT_ROOT>": checkpoint_root,
        "<OUTPUT_ROOT>": output_root,
    }
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, str):
            text = value
            for token, replacement in replacements.items():
                if token in text:
                    if replacement is None:
                        raise ValueError(f"{key} contains {token}; pass the matching CLI root")
                    text = text.replace(token, replacement)
            out[key] = text
        else:
            out[key] = value
    return out


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("PyTorch is required for training/evaluation commands") from exc
    return torch


def require_torchvision() -> Any:
    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("torchvision is required for ResNet/tiny visual backbones") from exc
    return torchvision


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def device_from_arg(device: str | None) -> Any:
    torch = require_torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_step(path: Path) -> int | None:
    m = STEP_RE.match(path.name)
    return int(m.group(1)) if m else None


def find_checkpoint(checkpoint_dir: str | Path, step: int | None = None) -> Path:
    root = Path(checkpoint_dir)
    if step is not None:
        path = root / f"step{int(step)}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates: list[tuple[int, Path]] = []
    for p in root.glob("step*.pt"):
        s = checkpoint_step(p)
        if s is not None:
            candidates.append((s, p))
    if not candidates:
        raise FileNotFoundError(f"no step*.pt checkpoints under {root}")
    return sorted(candidates)[-1][1]


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    torch = require_torch()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


def write_csv(path: str | Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_continuous_actions(actions: Any, mean: Any, std: Any, dims: int = 6) -> Any:
    out = actions.clone()
    out[..., :dims] = (out[..., :dims] - mean[:dims]) / std[:dims]
    return out


def unnormalize_continuous_actions(actions: Any, mean: Any, std: Any, dims: int = 6) -> Any:
    out = actions.clone()
    out[..., :dims] = out[..., :dims] * std[:dims] + mean[:dims]
    return out
