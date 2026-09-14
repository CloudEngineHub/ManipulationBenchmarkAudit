from __future__ import annotations

import argparse
import json
import math
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_source_dependency.widowx.proprio import (
    env_action_to_xvla_first10,
    initial_xvla_proprio,
    update_xvla_proprio_from_env_action,
)
from data_source_dependency.widowx.tasks import TASKS


DEFAULT_TASK_KEY = "stack"
DEFAULT_SOURCE_POLICY = "scripted_widowx_stack_v0"
DEFAULT_GRASP_Z_OFFSET_M = 0.012
TASK_GRASP_Z_OFFSET_M = {"spoon": 0.006}
OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
DEFAULT_ROT_EULER = np.array([0.0, math.pi / 2.0, 0.0], dtype=np.float32)


@dataclass
class Rollout:
    success: bool
    steps: int
    reward: float
    instruction: str
    images: np.ndarray
    actions_env: np.ndarray
    actions_xvla_first10: np.ndarray
    proprios_20d: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    elapsed_sec: float
    reset_metadata: dict[str, Any]
    phase_trace: list[str]


def _parse_episode_ids(args: argparse.Namespace) -> list[int]:
    if args.episode_ids:
        raw_ids = str(args.episode_ids).replace(";", ",")
        grid_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        if not grid_ids:
            raise ValueError("--episode-ids was provided but no ids were parsed")
        duplicates = sorted({x for x in grid_ids if grid_ids.count(x) > 1})
        if duplicates:
            raise ValueError(f"--episode-ids contains duplicate ids: {duplicates}")
        if any(x < 0 for x in grid_ids):
            raise ValueError(f"--episode-ids must be nonnegative, got {grid_ids}")
        return grid_ids
    return list(range(args.episode_start, args.episode_end))


def _unwrap_task_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    for _ in range(30):
        if id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "source_obj_pose") and hasattr(current, "target_obj_pose"):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Could not find unwrapped stack env with source_obj_pose/target_obj_pose")


def _object_config_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    for _ in range(20):
        if id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "_xy_configs") and hasattr(current, "_quat_configs"):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Object perturbation requires an env with _xy_configs and _quat_configs")


def _episode_xy_quat(env: Any, episode_id: int) -> tuple[Any, np.ndarray, np.ndarray]:
    env = _object_config_env(env)
    if not hasattr(env, "_dsd_base_xy_configs") or not hasattr(env, "_dsd_base_quat_configs"):
        xy_configs = getattr(env, "_xy_configs", None)
        quat_configs = getattr(env, "_quat_configs", None)
        if xy_configs is None or quat_configs is None:
            raise RuntimeError("Object perturbation requires an env with _xy_configs and _quat_configs")
        env._dsd_base_xy_configs = [np.asarray(xy, dtype=np.float32).copy() for xy in xy_configs]  # noqa: SLF001
        env._dsd_base_quat_configs = [  # noqa: SLF001
            np.asarray(quat, dtype=np.float32).copy() for quat in quat_configs
        ]
    xy_configs = env._dsd_base_xy_configs  # noqa: SLF001
    quat_configs = env._dsd_base_quat_configs  # noqa: SLF001
    num_quats = len(quat_configs)
    total = len(xy_configs) * num_quats
    if total == 0:
        raise RuntimeError("Object perturbation requires nonempty object configs")
    if episode_id < 0 or episode_id >= total:
        raise ValueError(
            f"Grid episode id {episode_id} is outside available object configs 0..{total - 1}"
        )
    xy = np.asarray(xy_configs[(episode_id % total) // num_quats], dtype=np.float32)
    quat = np.asarray(quat_configs[episode_id % num_quats], dtype=np.float32)
    return env, xy.copy(), quat.copy()


def _yaw_quats(yaws_rad: np.ndarray) -> np.ndarray:
    half_yaws = yaws_rad / 2.0
    zeros = np.zeros_like(half_yaws)
    return np.stack([np.cos(half_yaws), zeros, zeros, np.sin(half_yaws)], axis=-1).astype(np.float32)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot normalize zero-norm quaternion")
    return (quats / norms).astype(np.float32)


def _quat_yaw_wxyz(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"Expected one wxyz quaternion, got shape {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Cannot read yaw from zero-norm quaternion")
    w, x, y, z = (quat / norm).astype(float)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _canonical_parallel_jaw_yaw(yaw_rad: float) -> float:
    return float((float(yaw_rad) + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _install_robot_xy_reset(env: Any, robot_xy: np.ndarray) -> None:
    def _additional_prepackaged_config_reset(self: Any, options: dict[str, Any]) -> bool:
        options["robot_init_options"] = {
            "init_xy": np.asarray(robot_xy, dtype=float).tolist(),
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    env._additional_prepackaged_config_reset = types.MethodType(  # noqa: SLF001
        _additional_prepackaged_config_reset,
        env,
    )


def _prepare_reset(
    env: Any,
    grid_episode_id: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_options: dict[str, Any] = {"obj_init_options": {"episode_id": int(grid_episode_id)}}
    metadata: dict[str, Any] = {
        "base_grid_episode_id": int(grid_episode_id),
        "object_xy_jitter_m": float(args.object_xy_jitter_m),
        "object_yaw_jitter_deg": float(args.object_yaw_jitter_deg),
        "robot_xy_jitter_m": float(args.robot_xy_jitter_m),
    }

    if args.object_xy_jitter_m > 0.0 or args.object_yaw_jitter_deg > 0.0:
        config_env, base_xy, base_quat = _episode_xy_quat(env, grid_episode_id)
        delta_xy = np.zeros_like(base_xy, dtype=np.float32)
        delta_yaws_rad = np.zeros(base_quat.shape[:-1], dtype=np.float32)
        reset_xy = base_xy.copy()
        reset_quat = base_quat.copy()
        if args.object_xy_jitter_m > 0.0:
            delta_xy = rng.uniform(
                -float(args.object_xy_jitter_m),
                float(args.object_xy_jitter_m),
                size=base_xy.shape,
            ).astype(np.float32)
            reset_xy = base_xy + delta_xy
        if args.object_yaw_jitter_deg > 0.0:
            max_yaw_rad = math.radians(float(args.object_yaw_jitter_deg))
            delta_yaws_rad = rng.uniform(0.0, max_yaw_rad, size=base_quat.shape[:-1]).astype(np.float32)
            reset_quat = _normalize_quats(
                _quat_multiply_wxyz(_yaw_quats(delta_yaws_rad), base_quat)
            )
        metadata.update(
            {
                "object_base_xys": base_xy.astype(float).tolist(),
                "object_delta_xys": delta_xy.astype(float).tolist(),
                "object_reset_xys": reset_xy.astype(float).tolist(),
                "object_base_quats": base_quat.astype(float).tolist(),
                "object_delta_yaws_deg": np.degrees(delta_yaws_rad).astype(float).tolist(),
                "object_reset_quats": reset_quat.astype(float).tolist(),
            }
        )
        config_env._xy_configs = [reset_xy]  # noqa: SLF001
        config_env._quat_configs = [reset_quat]  # noqa: SLF001
        reset_options["obj_init_options"]["episode_id"] = 0

    if args.robot_xy_jitter_m > 0.0:
        robot_base_xy = np.array([args.robot_base_x, args.robot_base_y], dtype=np.float32)
        robot_delta_xy = rng.uniform(
            -float(args.robot_xy_jitter_m),
            float(args.robot_xy_jitter_m),
            size=(2,),
        ).astype(np.float32)
        robot_reset_xy = robot_base_xy + robot_delta_xy
        _install_robot_xy_reset(env, robot_reset_xy)
        metadata.update(
            {
                "robot_base_xy": robot_base_xy.astype(float).tolist(),
                "robot_delta_xy": robot_delta_xy.astype(float).tolist(),
                "robot_reset_xy": robot_reset_xy.astype(float).tolist(),
            }
        )

    return reset_options, metadata


def _base_pose(obs: dict[str, Any]) -> Any:
    from sapien.core import Pose

    return Pose(p=obs["agent"]["base_pose"][:3], q=obs["agent"]["base_pose"][3:])


def _base_target_from_world(obs: dict[str, Any], world_pos: np.ndarray) -> np.ndarray:
    pose = _base_pose(obs).inv() * Pose(p=np.asarray(world_pos, dtype=np.float32))
    return pose.p.astype(np.float32)


def _limited_next_pos(current: np.ndarray, goal: np.ndarray, max_step_m: float) -> np.ndarray:
    delta = np.asarray(goal, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    dist = float(np.linalg.norm(delta))
    if dist > max_step_m and dist > 1e-8:
        delta = delta / dist * float(max_step_m)
    return (np.asarray(current, dtype=np.float32) + delta).astype(np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return value.astype(float).tolist()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class StackScript:
    def __init__(self, args: argparse.Namespace, task_key: str = DEFAULT_TASK_KEY) -> None:
        self.args = args
        self.task_key = task_key
        self.phase = "move_above_source"
        self.phase_step = 0
        self.done = False
        self.grasp_tcp_minus_source: np.ndarray | None = None

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_step = 0

    def _source_target(self, core_env: Any) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(core_env.source_obj_pose.p, dtype=np.float32)
        target = np.asarray(core_env.target_obj_pose.p, dtype=np.float32)
        return source, target

    def _rot_euler(self, core_env: Any) -> np.ndarray:
        if self.task_key == "stack":
            return DEFAULT_ROT_EULER.copy()
        source_yaw = _quat_yaw_wxyz(np.asarray(core_env.source_obj_pose.q, dtype=np.float32))
        source_yaw = _canonical_parallel_jaw_yaw(source_yaw)
        return np.array([0.0, math.pi / 2.0, source_yaw], dtype=np.float32)

    def _goal_world(
        self,
        tcp: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        cube_size = float(self.args.cube_size_m)
        if self.phase == "move_above_source":
            goal = source + np.array([0.0, 0.0, self.args.approach_height_m], dtype=np.float32)
            gripper = OPEN_GRIPPER
            if np.linalg.norm(tcp - goal) < self.args.phase_tolerance_m:
                self._advance("descend_to_grasp")
        elif self.phase == "descend_to_grasp":
            goal = source + np.array([0.0, 0.0, self.args.grasp_z_offset_m], dtype=np.float32)
            gripper = OPEN_GRIPPER
            if np.linalg.norm(tcp - goal) < self.args.phase_tolerance_m:
                self._advance("close_gripper")
        elif self.phase == "close_gripper":
            goal = source + np.array([0.0, 0.0, self.args.grasp_z_offset_m], dtype=np.float32)
            gripper = CLOSE_GRIPPER
            if self.phase_step >= self.args.close_steps:
                self.grasp_tcp_minus_source = tcp - source
                self._advance("lift")
        elif self.phase == "lift":
            goal = source + np.array([0.0, 0.0, self.args.lift_height_m], dtype=np.float32)
            gripper = CLOSE_GRIPPER
            if source[2] > target[2] + cube_size * 1.6 or self.phase_step >= self.args.lift_steps:
                self._advance("move_above_target")
        elif self.phase == "move_above_target":
            offset = self.grasp_tcp_minus_source
            if offset is None:
                offset = np.zeros(3, dtype=np.float32)
            desired_source = target + np.array(
                [0.0, 0.0, cube_size + self.args.transport_height_m], dtype=np.float32
            )
            goal = desired_source + offset
            gripper = CLOSE_GRIPPER
            if np.linalg.norm(tcp - goal) < self.args.phase_tolerance_m:
                self._advance("descend_to_place")
        elif self.phase == "descend_to_place":
            offset = self.grasp_tcp_minus_source
            if offset is None:
                offset = np.zeros(3, dtype=np.float32)
            desired_source = target + np.array(
                [0.0, 0.0, cube_size + self.args.place_clearance_m], dtype=np.float32
            )
            goal = desired_source + offset
            gripper = CLOSE_GRIPPER
            if np.linalg.norm(tcp - goal) < self.args.phase_tolerance_m:
                self._advance("open_gripper")
        elif self.phase == "open_gripper":
            offset = self.grasp_tcp_minus_source
            if offset is None:
                offset = np.zeros(3, dtype=np.float32)
            desired_source = target + np.array(
                [0.0, 0.0, cube_size + self.args.place_clearance_m], dtype=np.float32
            )
            goal = desired_source + offset
            gripper = OPEN_GRIPPER
            if self.phase_step >= self.args.open_steps:
                if self.args.finish_after_open:
                    self.done = True
                else:
                    self._advance("retreat")
        elif self.phase == "retreat":
            goal = target + np.array([0.0, 0.0, self.args.retreat_height_m], dtype=np.float32)
            gripper = OPEN_GRIPPER
            if self.phase_step >= self.args.retreat_steps:
                self.done = True
        else:
            raise RuntimeError(f"Unknown phase {self.phase}")
        self.phase_step += 1
        return goal.astype(np.float32), float(gripper)

    def action(self, obs: dict[str, Any], core_env: Any) -> np.ndarray:
        tcp = np.asarray(obs["extra"]["tcp_pose"][:3], dtype=np.float32)
        source, target = self._source_target(core_env)
        goal_world, gripper = self._goal_world(tcp, source, target)
        next_world = _limited_next_pos(tcp, goal_world, float(self.args.max_pos_step_m))
        target_base = _base_target_from_world(obs, next_world)
        return np.concatenate(
            [target_base, self._rot_euler(core_env), np.array([gripper], dtype=np.float32)]
        ).astype(np.float32)


def rollout_once(
    env: Any,
    task_key: str,
    grid_episode_id: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Rollout:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    start_time = time.time()
    reset_options, perturb_metadata = _prepare_reset(env, grid_episode_id, args, rng)
    obs, reset_info = env.reset(options=reset_options)
    core_env = _unwrap_task_env(env)
    initial_source, initial_target = StackScript(args, task_key)._source_target(core_env)
    instruction = env.get_language_instruction()
    script = StackScript(args, task_key)
    proprio = initial_xvla_proprio(obs)

    images: list[np.ndarray] = []
    actions_env: list[np.ndarray] = []
    actions_xvla_first10: list[np.ndarray] = []
    proprios_20d: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    phase_trace: list[str] = []
    success_seen = False
    script_completed = False
    terminal_pad_steps_added = 0
    reward = 0.0

    def step_env(action_env: np.ndarray, phase_label: str) -> tuple[bool, bool]:
        nonlocal obs, proprio, reward, success_seen
        image = get_image_from_maniskill2_obs_dict(env, obs)
        action_xvla = env_action_to_xvla_first10(action_env)

        proprios_20d.append(np.asarray(proprio, dtype=np.float32).copy())
        phase_trace.append(phase_label)
        obs, reward, done, truncated, _info = env.step(action_env)

        images.append(np.asarray(image, dtype=np.uint8).copy())
        actions_env.append(action_env.astype(np.float32))
        actions_xvla_first10.append(action_xvla.astype(np.float32))
        rewards.append(float(reward))
        dones.append(bool(done))
        proprio = update_xvla_proprio_from_env_action(proprio, action_env)
        if done:
            success_seen = True
        return bool(done), bool(truncated)

    for _step_idx in range(int(args.max_steps)):
        phase_before = script.phase
        action_env = script.action(obs, core_env)
        done, truncated = step_env(action_env, phase_before)

        if done:
            if not args.ignore_env_success_until_script_done:
                break
        if script.done:
            script_completed = True
            break
        if truncated:
            break

    if not actions_env:
        raise RuntimeError(f"Script produced zero actions for grid {grid_episode_id}")

    if script_completed and args.terminal_pad_steps > 0:
        last_action_env = actions_env[-1].copy()
        for _pad_idx in range(int(args.terminal_pad_steps)):
            if len(actions_env) >= int(args.max_steps):
                break
            _done, truncated = step_env(last_action_env, "terminal_pad")
            terminal_pad_steps_added += 1
            if truncated:
                break

    success = success_seen
    if args.require_script_done_for_success:
        success = success and script_completed
    if args.terminal_pad_steps > 0:
        success = success and terminal_pad_steps_added == int(args.terminal_pad_steps)

    source, target = StackScript(args, task_key)._source_target(core_env)
    reset_metadata = {
        "base_grid_episode_id": int(grid_episode_id),
        "reset_options": reset_options,
        "reset_info": _json_safe(reset_info),
        "source_policy": str(args.source_policy),
        "perturb_metadata": _json_safe(perturb_metadata),
        "source_obj_initial_pose": initial_source.astype(float).tolist(),
        "target_obj_initial_pose": initial_target.astype(float).tolist(),
        "source_obj_final_pose": source.astype(float).tolist(),
        "target_obj_final_pose": target.astype(float).tolist(),
        "success_seen": bool(success_seen),
        "script_completed": bool(script_completed),
        "terminal_pad_steps_requested": int(args.terminal_pad_steps),
        "terminal_pad_steps_added": int(terminal_pad_steps_added),
        "script_args": {
            "max_pos_step_m": float(args.max_pos_step_m),
            "cube_size_m": float(args.cube_size_m),
            "approach_height_m": float(args.approach_height_m),
            "grasp_z_offset_m": float(args.grasp_z_offset_m),
            "lift_height_m": float(args.lift_height_m),
            "transport_height_m": float(args.transport_height_m),
            "place_clearance_m": float(args.place_clearance_m),
            "retreat_height_m": float(args.retreat_height_m),
            "close_steps": int(args.close_steps),
            "lift_steps": int(args.lift_steps),
            "open_steps": int(args.open_steps),
            "retreat_steps": int(args.retreat_steps),
            "ignore_env_success_until_script_done": bool(
                args.ignore_env_success_until_script_done
            ),
            "require_script_done_for_success": bool(args.require_script_done_for_success),
            "terminal_pad_steps": int(args.terminal_pad_steps),
            "finish_after_open": bool(args.finish_after_open),
            "align_gripper_yaw_to_source": bool(task_key != "stack"),
        },
    }
    return Rollout(
        success=success,
        steps=len(actions_env),
        reward=float(reward),
        instruction=instruction,
        images=np.stack(images).astype(np.uint8),
        actions_env=np.stack(actions_env).astype(np.float32),
        actions_xvla_first10=np.stack(actions_xvla_first10).astype(np.float32),
        proprios_20d=np.stack(proprios_20d).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        elapsed_sec=time.time() - start_time,
        reset_metadata=reset_metadata,
        phase_trace=phase_trace,
    )


def resize_images_256(images: np.ndarray) -> np.ndarray:
    resized = [
        np.asarray(Image.fromarray(img).resize((256, 256), Image.Resampling.BILINEAR))
        for img in images
    ]
    return np.stack(resized).astype(np.uint8)


def save_episode(
    task_dir: Path,
    task_key: str,
    grid_episode_id: int,
    attempt_index: int,
    success_index: int,
    rollout: Rollout,
    args: argparse.Namespace,
) -> Path:
    episode_dir = task_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / (
        f"success_{success_index:05d}_grid{grid_episode_id:02d}_attempt{attempt_index:06d}.npz"
    )
    task = TASKS[task_key]
    payload: dict[str, Any] = {
        "images": rollout.images,
        "actions_env": rollout.actions_env,
        "actions_xvla_first10": rollout.actions_xvla_first10,
        "actions_xvla_raw": rollout.actions_xvla_first10,
        "proprios_20d": rollout.proprios_20d,
        "rewards": rollout.rewards,
        "dones": rollout.dones,
        "task_key": np.array(task_key),
        "task_name": np.array(task.env_task),
        "instruction": np.array(rollout.instruction),
        "grid_episode_id": np.array(grid_episode_id, dtype=np.int32),
        "attempt_index": np.array(attempt_index, dtype=np.int32),
        "success_index": np.array(success_index, dtype=np.int32),
        "official_horizon": np.array(task.official_horizon, dtype=np.int32),
        "source_policy": np.array(str(args.source_policy)),
        "image_camera": np.array("3rd_view_camera"),
        "success": np.array(True),
        "steps": np.array(rollout.steps, dtype=np.int32),
        "reset_metadata_json": np.array(json.dumps(rollout.reset_metadata, sort_keys=True)),
        "phase_trace": np.asarray(rollout.phase_trace),
    }
    if args.save_resized_256:
        payload["images_256"] = resize_images_256(rollout.images)
    if args.compression == "compressed":
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def collect(args: argparse.Namespace) -> None:
    import simpler_env

    task_key = str(args.task_key)
    task = TASKS[task_key]
    grid_ids = _parse_episode_ids(args)
    target_per_grid = int(args.target_successes_per_grid_id)
    run_root = args.output_root / args.run_tag
    task_dir = run_root / task_key
    task_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_tag": args.run_tag,
        "source_policy": str(args.source_policy),
        "task_key": task_key,
        "task_name": task.env_task,
        "episode_ids": grid_ids,
        "target_successes_per_grid_id": target_per_grid,
        "max_attempts_per_grid_id": int(args.max_attempts_per_grid_id),
        "output_root": str(args.output_root),
        "run_root": str(run_root),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "storage": args.compression,
        "positive_demo_rule": (
            "success seen, optional script completion and terminal padding satisfied, "
            "and steps <= official_horizon"
        ),
        "object_xy_jitter_m": float(args.object_xy_jitter_m),
        "object_yaw_jitter_deg": float(args.object_yaw_jitter_deg),
        "robot_xy_jitter_m": float(args.robot_xy_jitter_m),
        "perturb_seed": int(args.perturb_seed),
        "ignore_env_success_until_script_done": bool(
            args.ignore_env_success_until_script_done
        ),
        "require_script_done_for_success": bool(args.require_script_done_for_success),
        "terminal_pad_steps": int(args.terminal_pad_steps),
        "finish_after_open": bool(args.finish_after_open),
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    successes_by_grid_id = {grid_id: 0 for grid_id in grid_ids}
    attempts_by_grid_id = {grid_id: 0 for grid_id in grid_ids}
    success_count = 0
    attempt_index = 0
    start_time = time.time()
    rng = np.random.default_rng(int(args.perturb_seed))
    env = simpler_env.make(task.env_task)
    try:
        for grid_id in grid_ids:
            while successes_by_grid_id[grid_id] < target_per_grid:
                if attempts_by_grid_id[grid_id] >= int(args.max_attempts_per_grid_id):
                    raise RuntimeError(
                        f"Grid {grid_id} ended before target: "
                        f"successes={successes_by_grid_id[grid_id]} "
                        f"attempts={attempts_by_grid_id[grid_id]}"
                    )
                attempt_index += 1
                attempts_by_grid_id[grid_id] += 1
                record: dict[str, Any] = {
                    "task_key": task_key,
                    "task_name": task.env_task,
                    "grid_episode_id": int(grid_id),
                    "attempt_index": int(attempt_index),
                    "source_policy": str(args.source_policy),
                }
                rollout = rollout_once(env, task_key, grid_id, args, rng)
                record.update(
                    {
                        "success": rollout.success,
                        "steps": rollout.steps,
                        "reward": rollout.reward,
                        "elapsed_sec": rollout.elapsed_sec,
                        "phase_trace": rollout.phase_trace,
                    }
                )
                if rollout.success:
                    episode_path = save_episode(
                        task_dir=task_dir,
                        task_key=task_key,
                        grid_episode_id=grid_id,
                        attempt_index=attempt_index,
                        success_index=success_count,
                        rollout=rollout,
                        args=args,
                    )
                    record["episode_path"] = str(episode_path)
                    record["success_index"] = success_count
                    success_count += 1
                    successes_by_grid_id[grid_id] += 1
                append_jsonl(task_dir / "manifest.jsonl", record)
                append_jsonl(run_root / "manifest.jsonl", record)
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        env.close()

    summary = {
        **metadata,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "successes": success_count,
        "attempts": attempt_index,
        "successes_by_grid_id": successes_by_grid_id,
        "attempts_by_grid_id": attempts_by_grid_id,
        "official_horizon": task.official_horizon,
        "elapsed_sec": time.time() - start_time,
        "summaries": [
            {
                "task_key": task_key,
                "task_name": task.env_task,
                "successes": success_count,
                "attempts": attempt_index,
                "successes_by_grid_id": successes_by_grid_id,
                "attempts_by_grid_id": attempts_by_grid_id,
                "official_horizon": task.official_horizon,
                "task_dir": str(task_dir),
            }
        ],
    }
    (task_dir / "summary.json").write_text(
        json.dumps(summary["summaries"][0], indent=2, sort_keys=True) + "\n"
    )
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted WidowX put-on demos.")
    parser.add_argument("--task-key", choices=sorted(TASKS), default=DEFAULT_TASK_KEY)
    parser.add_argument("--source-policy", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--target-successes-per-grid-id", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=24)
    parser.add_argument("--episode-ids", type=str)
    parser.add_argument("--max-attempts-per-grid-id", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--compression", choices=["stored", "compressed"], default="stored")
    parser.add_argument("--save-resized-256", action="store_true")
    parser.add_argument("--object-xy-jitter-m", type=float, default=0.0)
    parser.add_argument("--object-yaw-jitter-deg", type=float, default=0.0)
    parser.add_argument("--robot-xy-jitter-m", type=float, default=0.0)
    parser.add_argument("--robot-base-x", type=float, default=0.147)
    parser.add_argument("--robot-base-y", type=float, default=0.028)
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--max-pos-step-m", type=float, default=0.035)
    parser.add_argument("--phase-tolerance-m", type=float, default=0.018)
    parser.add_argument("--cube-size-m", type=float, default=0.03)
    parser.add_argument("--approach-height-m", type=float, default=0.12)
    parser.add_argument("--grasp-z-offset-m", type=float)
    parser.add_argument("--lift-height-m", type=float, default=0.13)
    parser.add_argument("--transport-height-m", type=float, default=0.09)
    parser.add_argument("--place-clearance-m", type=float, default=0.006)
    parser.add_argument("--retreat-height-m", type=float, default=0.13)
    parser.add_argument("--close-steps", type=int, default=8)
    parser.add_argument("--lift-steps", type=int, default=10)
    parser.add_argument("--open-steps", type=int, default=6)
    parser.add_argument("--retreat-steps", type=int, default=6)
    parser.add_argument("--ignore-env-success-until-script-done", action="store_true")
    parser.add_argument("--require-script-done-for-success", action="store_true")
    parser.add_argument("--terminal-pad-steps", type=int, default=0)
    parser.add_argument("--finish-after-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_policy is None:
        args.source_policy = (
            DEFAULT_SOURCE_POLICY
            if args.task_key == DEFAULT_TASK_KEY
            else f"scripted_widowx_{args.task_key}_v0"
        )
    if args.max_steps is None:
        args.max_steps = TASKS[args.task_key].official_horizon
    if args.grasp_z_offset_m is None:
        args.grasp_z_offset_m = TASK_GRASP_Z_OFFSET_M.get(
            args.task_key,
            DEFAULT_GRASP_Z_OFFSET_M,
        )
    if args.target_successes_per_grid_id <= 0:
        raise ValueError("--target-successes-per-grid-id must be positive")
    if args.max_attempts_per_grid_id <= 0:
        raise ValueError("--max-attempts-per-grid-id must be positive")
    if args.max_steps <= 0 or args.max_steps > TASKS[args.task_key].official_horizon:
        raise ValueError("--max-steps must be in 1..official_horizon")
    if args.terminal_pad_steps < 0:
        raise ValueError("--terminal-pad-steps must be nonnegative")
    for name in ("close_steps", "lift_steps", "open_steps", "retreat_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.object_xy_jitter_m < 0.0:
        raise ValueError("--object-xy-jitter-m must be nonnegative")
    if args.object_yaw_jitter_deg < 0.0:
        raise ValueError("--object-yaw-jitter-deg must be nonnegative")
    if args.robot_xy_jitter_m < 0.0:
        raise ValueError("--robot-xy-jitter-m must be nonnegative")
    collect(args)


if __name__ == "__main__":
    main()
