from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotate6d_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")


def euler_xyz_to_rotate6d(euler_xyz: np.ndarray) -> np.ndarray:
    matrix = R.from_euler("xyz", np.asarray(euler_xyz)).as_matrix()
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def xvla_first10_to_env_action(action_pred: np.ndarray) -> np.ndarray:
    action_pred = np.asarray(action_pred, dtype=np.float32)
    if action_pred.shape[-1] < 10:
        raise ValueError(f"Expected action_pred last dim >= 10, got {action_pred.shape}")
    return np.concatenate(
        [
            action_pred[:3],
            rotate6d_to_euler_xyz(action_pred[3:9]) + np.array([0.0, math.pi / 2.0, 0.0]),
            np.array([1.0 if action_pred[9] < 0.7 else -1.0]),
        ]
    ).astype(np.float32)


def env_action_to_xvla_first10(action_env: np.ndarray) -> np.ndarray:
    action_env = np.asarray(action_env, dtype=np.float32)
    if action_env.shape[-1] != 7:
        raise ValueError(f"Expected action_env last dim 7, got {action_env.shape}")
    raw_rot_euler = action_env[3:6] - np.array([0.0, math.pi / 2.0, 0.0], dtype=np.float32)
    gripper = np.array([0.0 if action_env[6] > 0.0 else 1.0], dtype=np.float32)
    return np.concatenate(
        [action_env[:3], euler_xyz_to_rotate6d(raw_rot_euler).astype(np.float32), gripper]
    ).astype(np.float32)


def initial_xvla_proprio(obs: dict[str, Any]) -> np.ndarray:
    from sapien.core import Pose

    ee_pose_wrt_base = Pose(
        p=obs["agent"]["base_pose"][:3],
        q=obs["agent"]["base_pose"][3:],
    ).inv() * Pose(
        p=obs["extra"]["tcp_pose"][:3],
        q=obs["extra"]["tcp_pose"][3:],
    )
    proprio = torch.from_numpy(
        np.concatenate([ee_pose_wrt_base.p, np.array([1, 0, 0, 1, 0, 0, 0])])
    ).to(dtype=torch.float32)
    return torch.cat([proprio, torch.zeros_like(proprio)], dim=-1).numpy()


def update_xvla_proprio_from_env_action(proprio: np.ndarray, action_env: np.ndarray) -> np.ndarray:
    updated = np.asarray(proprio, dtype=np.float32).copy()
    updated[:10] = env_action_to_xvla_first10(action_env)
    return updated


def update_xvla_proprio_from_raw_action(proprio: np.ndarray, action_pred: np.ndarray) -> np.ndarray:
    action_pred = np.asarray(action_pred, dtype=np.float32)
    if action_pred.shape[-1] < 10:
        raise ValueError(f"Expected raw X-VLA action dim >= 10, got {action_pred.shape}")
    updated = np.asarray(proprio, dtype=np.float32).copy()
    updated[:10] = action_pred[:10]
    return updated
