from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_source_dependency.widowx.tasks import TASK_ORDER, parse_tasks


@dataclass(frozen=True)
class WidowXSample:
    path: Path
    step: int
    task_id: int
    episode_idx: int | None = None


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    value = data[key]
    return value.item() if value.shape == () else value


def _npz_array_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(member, "r") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _fortran_order, dtype = np.lib.format._read_array_header(f, version)
    return tuple(int(x) for x in shape), np.dtype(dtype)


def find_episode_paths(
    dataset_root: str | Path,
    tasks: list[str] | tuple[str, ...] | str | None = None,
    limit_episodes_per_task: int | None = None,
) -> dict[str, list[Path]]:
    root = Path(dataset_root)
    task_keys = parse_tasks(tasks)
    out: dict[str, list[Path]] = {}
    for task_key in task_keys:
        paths = sorted((root / task_key / "episodes").glob("*.npz"))
        if limit_episodes_per_task is not None:
            paths = paths[: int(limit_episodes_per_task)]
        if not paths:
            raise FileNotFoundError(f"No .npz episodes found under {root / task_key / 'episodes'}")
        out[task_key] = paths
    return out


def validate_episode(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    image_shape, image_dtype = _npz_array_header(path, "images")
    action_shape, action_dtype = _npz_array_header(path, "actions_env")
    proprio_shape, proprio_dtype = _npz_array_header(path, "proprios_20d")
    with np.load(path, allow_pickle=True) as data:
        required = {
            "images",
            "actions_env",
            "proprios_20d",
            "task_key",
            "grid_episode_id",
            "official_horizon",
            "success",
            "steps",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")

        task_key = str(_scalar(data, "task_key"))
        steps = int(_scalar(data, "steps"))
        official_horizon = int(_scalar(data, "official_horizon"))
        success = bool(_scalar(data, "success"))
        grid_episode_id = int(_scalar(data, "grid_episode_id"))

        if len(image_shape) != 4 or image_shape[-1] != 3 or image_dtype != np.uint8:
            raise ValueError(f"{path}: expected images uint8 (T,H,W,3), got {image_shape} {image_dtype}")
        if len(action_shape) != 2 or action_shape[-1] != 7 or action_dtype != np.float32:
            raise ValueError(
                f"{path}: expected actions_env float32 (T,7), got {action_shape} {action_dtype}"
            )
        if len(proprio_shape) != 2 or proprio_shape[-1] != 20 or proprio_dtype != np.float32:
            raise ValueError(
                f"{path}: expected proprios_20d float32 (T,20), got {proprio_shape} {proprio_dtype}"
            )
        if not (image_shape[0] == action_shape[0] == proprio_shape[0] == steps):
            raise ValueError(
                f"{path}: length mismatch images={image_shape[0]} actions={action_shape[0]} "
                f"proprios={proprio_shape[0]} steps={steps}"
            )
        if not success:
            raise ValueError(f"{path}: expected success=True")
        if steps > official_horizon:
            raise ValueError(f"{path}: steps={steps} exceeds official_horizon={official_horizon}")

        actions = data["actions_env"]
        return {
            "path": str(path),
            "task_key": task_key,
            "steps": steps,
            "official_horizon": official_horizon,
            "grid_episode_id": grid_episode_id,
            "image_shape": list(image_shape[1:]),
            "actions_min": actions.min(axis=0).astype(float).tolist(),
            "actions_max": actions.max(axis=0).astype(float).tolist(),
        }


def summarize_dataset(
    dataset_root: str | Path,
    tasks: list[str] | tuple[str, ...] | str | None = None,
    chunk_size: int = 10,
    limit_episodes_per_task: int | None = None,
    action_key: str = "actions_env",
    pad_terminal_actions: bool = False,
) -> dict[str, Any]:
    paths_by_task = find_episode_paths(dataset_root, tasks, limit_episodes_per_task)
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "chunk_size": int(chunk_size),
        "action_key": action_key,
        "pad_terminal_actions": bool(pad_terminal_actions),
        "tasks": {},
        "total_episodes": 0,
        "total_steps": 0,
        "total_train_samples": 0,
    }
    all_actions: list[np.ndarray] = []
    all_proprios: list[np.ndarray] = []
    for task_key, paths in paths_by_task.items():
        steps: list[int] = []
        grid_ids: set[int] = set()
        image_shapes: set[tuple[int, ...]] = set()
        for path in paths:
            info = validate_episode(path)
            steps.append(int(info["steps"]))
            grid_ids.add(int(info["grid_episode_id"]))
            image_shapes.add(tuple(info["image_shape"]))
            with np.load(path, allow_pickle=True) as data:
                if action_key not in data.files:
                    raise ValueError(f"{path}: missing action_key={action_key}")
                actions = np.asarray(data[action_key], dtype=np.float32)
                if actions.shape[0] != int(info["steps"]):
                    raise ValueError(
                        f"{path}: action_key={action_key} length {actions.shape[0]} "
                        f"does not match steps={info['steps']}"
                    )
                all_actions.append(actions)
                all_proprios.append(np.asarray(data["proprios_20d"], dtype=np.float32))
        if pad_terminal_actions:
            train_samples = sum(steps)
        else:
            train_samples = sum(max(0, n - int(chunk_size) + 1) for n in steps)
        task_summary = {
            "episodes": len(paths),
            "steps_min": min(steps),
            "steps_max": max(steps),
            "steps_mean": float(np.mean(steps)),
            "unique_grid_ids": len(grid_ids),
            "image_shapes": [list(x) for x in sorted(image_shapes)],
            "train_samples": train_samples,
        }
        summary["tasks"][task_key] = task_summary
        summary["total_episodes"] += len(paths)
        summary["total_steps"] += int(sum(steps))
        summary["total_train_samples"] += int(train_samples)

    actions = np.concatenate(all_actions, axis=0)
    summary["action_mean"] = actions.mean(axis=0).astype(float).tolist()
    summary["action_std"] = actions.std(axis=0).astype(float).tolist()
    summary["action_min"] = actions.min(axis=0).astype(float).tolist()
    summary["action_max"] = actions.max(axis=0).astype(float).tolist()
    proprios = np.concatenate(all_proprios, axis=0)
    summary["proprio_mean"] = proprios.mean(axis=0).astype(float).tolist()
    summary["proprio_std"] = proprios.std(axis=0).astype(float).tolist()
    summary["proprio_min"] = proprios.min(axis=0).astype(float).tolist()
    summary["proprio_max"] = proprios.max(axis=0).astype(float).tolist()
    return summary


class WidowXGridDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        tasks: list[str] | tuple[str, ...] | str | None,
        chunk_size: int,
        limit_episodes_per_task: int | None = None,
        use_proprio: bool = False,
        preload_to_memory: bool = False,
        action_key: str = "actions_env",
        pad_terminal_actions: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_keys = parse_tasks(tasks)
        self.chunk_size = int(chunk_size)
        self.use_proprio = bool(use_proprio)
        self.preload_to_memory = bool(preload_to_memory)
        self.action_key = str(action_key)
        self.pad_terminal_actions = bool(pad_terminal_actions)
        self.task_to_id = {task: i for i, task in enumerate(self.task_keys)}
        self.paths_by_task = find_episode_paths(
            self.dataset_root,
            self.task_keys,
            limit_episodes_per_task=limit_episodes_per_task,
        )

        samples: list[WidowXSample] = []
        self._episodes: list[dict[str, np.ndarray]] = []
        for task_key in self.task_keys:
            for path in self.paths_by_task[task_key]:
                info = validate_episode(path)
                episode_idx = None
                if self.preload_to_memory:
                    with np.load(path, allow_pickle=True) as data:
                        episode_idx = len(self._episodes)
                        self._episodes.append(
                            {
                                "images": np.asarray(data["images"], dtype=np.uint8),
                                "actions": np.asarray(data[self.action_key], dtype=np.float32),
                                "proprios_20d": np.asarray(data["proprios_20d"], dtype=np.float32),
                            }
                        )
                if self.pad_terminal_actions:
                    max_start = int(info["steps"]) - 1
                else:
                    max_start = int(info["steps"]) - self.chunk_size
                    if max_start < 0:
                        continue
                for step in range(max_start + 1):
                    samples.append(
                        WidowXSample(
                            path=path,
                            step=step,
                            task_id=self.task_to_id[task_key],
                            episode_idx=episode_idx,
                        )
                    )
        if not samples:
            raise ValueError(f"No train samples found in {self.dataset_root} with chunk_size={chunk_size}")
        self.samples = samples

    def _action_chunk(self, actions: np.ndarray, t: int) -> np.ndarray:
        chunk = np.asarray(actions[t : t + self.chunk_size], dtype=np.float32)
        if chunk.shape[0] == self.chunk_size:
            return chunk
        if not self.pad_terminal_actions:
            raise ValueError(
                f"Short action chunk with pad_terminal_actions=False: "
                f"start={t} len={chunk.shape[0]} chunk_size={self.chunk_size}"
            )
        if chunk.shape[0] == 0:
            raise ValueError(f"Empty action chunk at start={t}")
        pad = np.repeat(chunk[-1:][...], self.chunk_size - chunk.shape[0], axis=0)
        return np.concatenate([chunk, pad], axis=0).astype(np.float32, copy=False)

    @property
    def task_vocab(self) -> list[str]:
        return list(self.task_keys)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        sample = self.samples[idx]
        if sample.episode_idx is not None:
            episode = self._episodes[sample.episode_idx]
            t = sample.step
            image = np.ascontiguousarray(episode["images"][t])
            actions = self._action_chunk(episode["actions"], t)
            out: dict[str, np.ndarray] = {
                "image": image,
                "actions": actions,
                "task_id": np.int64(sample.task_id),
            }
            if self.use_proprio:
                out["proprio"] = np.asarray(episode["proprios_20d"][t], dtype=np.float32)
            return out

        with np.load(sample.path, allow_pickle=True) as data:
            t = sample.step
            image = np.ascontiguousarray(data["images"][t])
            actions = self._action_chunk(data[self.action_key], t)
            out: dict[str, np.ndarray] = {
                "image": image,
                "actions": actions,
                "task_id": np.int64(sample.task_id),
            }
            if self.use_proprio:
                out["proprio"] = np.asarray(data["proprios_20d"][t], dtype=np.float32)
            return out


def widowx_collate(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    images = torch.from_numpy(np.stack([b["image"] for b in batch], axis=0)).permute(0, 3, 1, 2)
    actions = torch.from_numpy(np.stack([b["actions"] for b in batch], axis=0)).to(dtype=torch.float32)
    task_id = torch.from_numpy(np.stack([b["task_id"] for b in batch], axis=0)).to(dtype=torch.long)
    out = {"image": images.contiguous(), "actions": actions, "task_id": task_id}
    if "proprio" in batch[0]:
        out["proprio"] = torch.from_numpy(np.stack([b["proprio"] for b in batch], axis=0)).to(
            dtype=torch.float32
        )
    return out


def build_widowx_loader(cfg: dict[str, Any]) -> tuple[DataLoader, WidowXGridDataset]:
    dataset = WidowXGridDataset(
        dataset_root=cfg["dataset_root"],
        tasks=cfg.get("tasks", TASK_ORDER),
        chunk_size=int(cfg["chunk_size"]),
        limit_episodes_per_task=cfg.get("limit_episodes_per_task"),
        use_proprio=bool(cfg.get("use_proprio", False)),
        preload_to_memory=bool(cfg.get("preload_to_memory", False)),
        action_key=str(cfg.get("action_key", "actions_env")),
        pad_terminal_actions=bool(cfg.get("pad_terminal_actions", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(cfg.get("num_workers", 4)) > 0,
        collate_fn=widowx_collate,
    )
    return loader, dataset


def stats_from_summary(summary: dict[str, Any], std_floor: float = 1e-6) -> dict[str, np.ndarray]:
    action_mean = np.asarray(summary["action_mean"], dtype=np.float32)
    action_std = np.asarray(summary["action_std"], dtype=np.float32)
    action_std = np.maximum(action_std, float(std_floor)).astype(np.float32)
    action_min = np.asarray(summary["action_min"], dtype=np.float32)
    action_max = np.asarray(summary["action_max"], dtype=np.float32)
    out = {
        "action_mean": action_mean,
        "action_std": action_std,
        "action_min": action_min,
        "action_max": action_max,
    }
    if "proprio_mean" in summary:
        proprio_mean = np.asarray(summary["proprio_mean"], dtype=np.float32)
        proprio_std = np.asarray(summary["proprio_std"], dtype=np.float32)
        out.update(
            {
                "proprio_mean": proprio_mean,
                "proprio_std": np.maximum(proprio_std, float(std_floor)).astype(np.float32),
                "proprio_min": np.asarray(summary["proprio_min"], dtype=np.float32),
                "proprio_max": np.asarray(summary["proprio_max"], dtype=np.float32),
            }
        )
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
