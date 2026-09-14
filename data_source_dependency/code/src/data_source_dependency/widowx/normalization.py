from __future__ import annotations

from typing import Any

import numpy as np
import torch


BRIDGE_ACTION_Q01 = np.array(
    [
        -0.02872725307941437,
        -0.04170349963009357,
        -0.026093858778476715,
        -0.08092105075716972,
        -0.09288699507713317,
        -0.20718276381492615,
        0.0,
    ],
    dtype=np.float32,
)
BRIDGE_ACTION_Q99 = np.array(
    [
        0.028309678435325586,
        0.040855254605412394,
        0.040161586627364146,
        0.08192047759890528,
        0.07792850524187081,
        0.20382574498653397,
        1.0,
    ],
    dtype=np.float32,
)
BRIDGE_ACTION_MASK = np.array([True, True, True, True, True, True, False])
XVLA_GRIPPER_CHANNELS = (9, 19)


def _cfg(payload_or_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(payload_or_cfg.get("cfg", payload_or_cfg))


def _torch_const(array: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(array, dtype=ref.dtype, device=ref.device)


def normalize_action_tensor(
    actions: torch.Tensor,
    cfg: dict[str, Any],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    mode = str(cfg.get("action_norm", "dataset_standard"))
    action_key = str(cfg.get("action_key", "actions_env"))
    out = actions.clone()
    if mode == "dataset_standard":
        dims = int(cfg.get("action_normalize_dims", 6))
        if dims > 0:
            out[..., :dims] = (actions[..., :dims] - action_mean[:dims]) / action_std[:dims]
        return out
    if mode == "bridge_q01_q99":
        if action_key != "actions_env" or out.shape[-1] != 7:
            raise ValueError("bridge_q01_q99 action_norm requires action_key=actions_env and action_dim=7")
        low = _torch_const(BRIDGE_ACTION_Q01, actions)
        high = _torch_const(BRIDGE_ACTION_Q99, actions)
        mask = _torch_const(BRIDGE_ACTION_MASK, actions).bool()
        out[..., mask] = torch.clamp(
            2.0 * (actions[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0,
            min=-1.0,
            max=1.0,
        )
        out[..., 6] = (actions[..., 6] > 0.0).to(dtype=actions.dtype)
        return out
    if mode == "none":
        return out
    raise ValueError(f"Unsupported action_norm={mode}")


def denormalize_action_array(
    chunk: np.ndarray,
    payload: dict[str, Any],
    threshold_gripper: bool,
    clip: bool,
) -> np.ndarray:
    cfg = _cfg(payload)
    mode = str(cfg.get("action_norm", "dataset_standard"))
    out = np.asarray(chunk, dtype=np.float32).copy()
    if mode == "dataset_standard":
        dims = int(cfg.get("action_normalize_dims", 6))
        if dims > 0:
            out[..., :dims] = out[..., :dims] * payload["action_std"][:dims] + payload["action_mean"][:dims]
        if str(cfg.get("action_key", "actions_env")) == "actions_env" and threshold_gripper:
            out[..., 6] = np.where(out[..., 6] >= 0.0, 1.0, -1.0)
        if clip:
            out = np.clip(out, payload["action_min"], payload["action_max"])
        return out
    if mode == "bridge_q01_q99":
        if str(cfg.get("action_key", "actions_env")) != "actions_env" or out.shape[-1] != 7:
            raise ValueError("bridge_q01_q99 action_norm requires action_key=actions_env and action_dim=7")
        mask = BRIDGE_ACTION_MASK
        if clip:
            out[..., mask] = np.clip(out[..., mask], -1.0, 1.0)
            out[..., 6] = np.clip(out[..., 6], 0.0, 1.0)
        out[..., mask] = 0.5 * (out[..., mask] + 1.0) * (
            BRIDGE_ACTION_Q99[mask] - BRIDGE_ACTION_Q01[mask]
        ) + BRIDGE_ACTION_Q01[mask]
        if threshold_gripper:
            out[..., 6] = np.where(out[..., 6] >= 0.5, 1.0, -1.0)
        else:
            out[..., 6] = 2.0 * out[..., 6] - 1.0
        return out
    if mode == "none":
        return out
    raise ValueError(f"Unsupported action_norm={mode}")


def preprocess_proprio_tensor(
    proprio: torch.Tensor | None,
    cfg: dict[str, Any],
    proprio_mean: torch.Tensor | None = None,
    proprio_std: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if proprio is None:
        return None
    out = proprio.clone()
    norm = str(cfg.get("proprio_norm", "none"))
    if norm == "dataset_standard":
        if proprio_mean is None or proprio_std is None:
            raise ValueError("proprio_norm=dataset_standard requires proprio_mean/proprio_std")
        out = (out - proprio_mean) / proprio_std
    elif norm != "none":
        raise ValueError(f"Unsupported proprio_norm={norm}")

    preprocess = str(cfg.get("proprio_preprocess", "none"))
    if preprocess == "xvla_zero_gripper":
        out[..., list(XVLA_GRIPPER_CHANNELS)] = 0.0
    elif preprocess != "none":
        raise ValueError(f"Unsupported proprio_preprocess={preprocess}")
    return out


def preprocess_proprio_array(proprio: np.ndarray | None, payload: dict[str, Any]) -> np.ndarray | None:
    if proprio is None:
        return None
    cfg = _cfg(payload)
    out = np.asarray(proprio, dtype=np.float32).copy()
    norm = str(cfg.get("proprio_norm", "none"))
    if norm == "dataset_standard":
        if "proprio_mean" not in payload or "proprio_std" not in payload:
            raise ValueError("proprio_norm=dataset_standard requires proprio_mean/proprio_std in checkpoint")
        out = (out - payload["proprio_mean"]) / payload["proprio_std"]
    elif norm != "none":
        raise ValueError(f"Unsupported proprio_norm={norm}")

    preprocess = str(cfg.get("proprio_preprocess", "none"))
    if preprocess == "xvla_zero_gripper":
        out[..., list(XVLA_GRIPPER_CHANNELS)] = 0.0
    elif preprocess != "none":
        raise ValueError(f"Unsupported proprio_preprocess={preprocess}")
    return out
