from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

EXPECTED_PROTOCOL = "calvin_official_d_table_resets_v1"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


class CalvinResetBank:
    def __init__(self, path: str | os.PathLike[str], eval_start: int = 0, expected_protocol: str = EXPECTED_PROTOCOL):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"CALVIN_RESET_BANK does not exist: {self.path}")
        data = np.load(self.path, allow_pickle=False)
        try:
            self.metadata = json.loads(str(data["metadata_json"].item()))
        except Exception as exc:
            raise RuntimeError(f"invalid reset-bank metadata in {self.path}: {exc}") from exc
        protocol = self.metadata.get("protocol")
        if protocol != expected_protocol:
            raise RuntimeError(f"reset-bank protocol mismatch: got {protocol!r}, expected {expected_protocol!r}")
        n = int(self.metadata.get("num_sequences", 0))
        self.robot_obs = np.asarray(data["robot_obs"], dtype=np.float64)
        self.scene_obs = np.asarray(data["scene_obs"], dtype=np.float64)
        self.official_robot_obs = np.asarray(data["official_robot_obs"], dtype=np.float64)
        self.official_scene_obs = np.asarray(data["official_scene_obs"], dtype=np.float64)
        self.initial_state_json = np.asarray(data["initial_state_json"], dtype=str)
        self.table_signature = np.asarray(data["table_signature"], dtype=str)
        if n <= 0:
            n = len(self.initial_state_json)
        for name, shape in {
            "robot_obs": (n, 15),
            "scene_obs": (n, 24),
            "official_robot_obs": (n, 15),
            "official_scene_obs": (n, 24),
        }.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise RuntimeError(f"{self.path} has {name} shape {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"{self.path} has non-finite values in {name}")
        if self.initial_state_json.shape != (n,):
            raise RuntimeError(f"{self.path} has initial_state_json shape {self.initial_state_json.shape}, expected {(n,)}")
        if eval_start < 0 or eval_start > n:
            raise ValueError(f"eval_start={eval_start} outside reset bank length {n}")
        self.next_idx = int(eval_start)

    def next_reset(self, initial_condition: Any, original_fn: Callable[[Any], tuple[np.ndarray, np.ndarray]]):
        idx = self.next_idx
        if idx >= len(self.initial_state_json):
            raise RuntimeError(f"reset bank exhausted at index {idx}; bank={self.path}")
        actual = stable_json(initial_condition)
        expected = str(self.initial_state_json[idx])
        if actual != expected:
            raise RuntimeError(f"CALVIN reset-bank sequence mismatch at index {idx}: actual={actual}, expected={expected}, bank={self.path}")
        official_robot, official_scene = original_fn(initial_condition)
        official_robot = np.asarray(official_robot, dtype=np.float64).reshape(-1)
        official_scene = np.asarray(official_scene, dtype=np.float64).reshape(-1)
        if not np.allclose(official_robot, self.official_robot_obs[idx], atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"official robot reset drift at index {idx}; bank={self.path}")
        if not np.allclose(official_scene, self.official_scene_obs[idx], atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"official scene reset drift at index {idx}; signature={self.table_signature[idx]}; bank={self.path}")
        self.next_idx += 1
        return self.robot_obs[idx].copy(), self.scene_obs[idx].copy()


def patch_module(module: Any, reset_bank_path: str | os.PathLike[str], eval_start: int = 0, attr_name: str = "get_env_state_for_initial_condition", expected_protocol: str = EXPECTED_PROTOCOL) -> CalvinResetBank:
    if not hasattr(module, attr_name):
        raise AttributeError(f"module {module!r} has no {attr_name}")
    original_fn = getattr(module, attr_name)
    bank = CalvinResetBank(reset_bank_path, eval_start=eval_start, expected_protocol=expected_protocol)

    def patched(initial_condition):
        return bank.next_reset(initial_condition, original_fn)

    setattr(module, attr_name, patched)
    print(f"CALVIN reset override active: bank={bank.path} eval_start={eval_start} protocol={expected_protocol}", flush=True)
    return bank


def patch_module_from_env(module: Any, attr_name: str = "get_env_state_for_initial_condition", eval_start: int | None = None):
    reset_bank_path = os.environ.get("CALVIN_RESET_BANK")
    if not reset_bank_path:
        return None
    protocol = os.environ.get("CALVIN_RESET_PROTOCOL", EXPECTED_PROTOCOL)
    if eval_start is None:
        eval_start = int(os.environ.get("CALVIN_RESET_EVAL_START", "0"))
    return patch_module(module, reset_bank_path, eval_start=eval_start, attr_name=attr_name, expected_protocol=protocol)
