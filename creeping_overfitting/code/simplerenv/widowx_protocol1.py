"""Runtime patch for SimplerEnv WidowX randomized-position protocols."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


_PROTOCOL3_FLAG = "SIMPLERENV_PROTOCOL3_STACK_YELLOW_ON_GREEN"
_PROTOCOL3_NAME = "widowx_protocol3_stack_yellow_on_green_protocol1_positions"
_PROTOCOL3_INSTRUCTION = "stack the yellow block on the green block"
_PROTOCOL3_STACK_PAIR = ("baked_green_cube_3cm", "baked_yellow_cube_3cm")

_ABCDE_NAME = "simplerenv_protocol_abcde_stack_v1"
_ABCDE_CONDITION_ENV = "SIMPLERENV_PROTOCOL_CONDITION"
_ABCDE_NUM_EPISODES_PER_CONDITION = 288
_ABCDE_CONDITIONS = (
    "protocol_A",
    "protocol_B",
    "protocol_C1_yellow_on_green",
    "protocol_C2_blue_on_red",
    "protocol_C3_red_on_blue",
    "protocol_D",
    "protocol_E",
)
_ABCDE_EPISODE_ATTR = "_simplerenv_protocol_abcde_episode"
_ABCDE_CONDITION_ATTR = "_simplerenv_protocol_abcde_condition"
_ABCDE_INSTRUCTION_ATTR = "_simplerenv_protocol_abcde_instruction"

_CONFIG: dict[str, Any] | None = None
_INSTALLED = False
_PROTOCOL3_INSTALLED = False
_ABCDE_STACK_PATCH_INSTALLED = False
_ORIGINAL_RESET = None
_ABCDE_ORIGINAL_STACK_EVALUATE = None

_TASK_BY_MODEL_PAIR = {
    ("baked_green_cube_3cm", "baked_yellow_cube_3cm"): "stack",
    ("bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"): "carrot",
    ("bridge_spoon_generated_modified", "table_cloth_generated_shorter"): "spoon",
}

_SUPPORTED_PROTOCOLS = {
    "widowx_protocol1_random_positions",
    "widowx_protocol2_random_positions_robot_initial_states",
}

_MAT_TRANSFORM = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_finite_array(value: Any, shape: tuple[int, ...], context: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric") from exc
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{context} must have finite shape {shape}, got {arr.shape}: {value!r}")
    return arr


def _require_obj_id(value: Any, n_objects: int, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer, got {value!r}")
    if value < 0 or value >= n_objects:
        raise RuntimeError(f"{context}={value} outside object range [0, {n_objects})")
    return value


def _require_nonnegative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{context} must be a nonnegative integer, got {value!r}")
    return value


def _validate_legacy_config(config: dict[str, Any]) -> None:
    protocol_name = config.get("name")
    tasks = config.get("tasks")
    if set(tasks or {}) != {"stack", "carrot", "spoon"}:
        raise RuntimeError(f"Protocol must contain stack/carrot/spoon tasks, got {sorted((tasks or {}).keys())}")

    for task_name, task in tasks.items():
        episodes = task.get("episodes", [])
        if len(episodes) != int(task.get("num_episodes", -1)):
            raise RuntimeError(f"{task_name} episode count mismatch in protocol config")
        if len(episodes) != int(config.get("num_episodes_per_task", -1)):
            raise RuntimeError(f"{task_name} does not match num_episodes_per_task")
        for idx, episode in enumerate(episodes):
            if int(episode.get("episode_id", -1)) != idx:
                raise RuntimeError(f"{task_name} episode id mismatch at index {idx}")
            if protocol_name == "widowx_protocol2_random_positions_robot_initial_states":
                state = episode.get("robot_initial_state")
                if not isinstance(state, dict):
                    raise RuntimeError(f"{task_name} episode {idx} is missing robot_initial_state")
                vec = state.get("state_xyz_rpy_pad_gripper")
                if len(vec or []) != 8:
                    raise RuntimeError(f"{task_name} episode {idx} has invalid robot_initial_state vector")


def _validate_abcde_episode(condition: str, idx: int, episode: Any) -> None:
    episode = _require_mapping(episode, f"{condition} episode {idx}")

    model_ids = episode.get("model_ids")
    if (
        not isinstance(model_ids, list)
        or not model_ids
        or not all(isinstance(model_id, str) and model_id for model_id in model_ids)
    ):
        raise RuntimeError(f"{condition} episode {idx} must contain nonempty string model_ids")
    n_objects = len(model_ids)

    source_obj_id = _require_obj_id(episode.get("source_obj_id"), n_objects, f"{condition} episode {idx} source_obj_id")
    target_obj_id = _require_obj_id(episode.get("target_obj_id"), n_objects, f"{condition} episode {idx} target_obj_id")
    if source_obj_id == target_obj_id:
        raise RuntimeError(f"{condition} episode {idx} source_obj_id and target_obj_id must differ")

    if condition == "protocol_D":
        if source_obj_id != 0 or target_obj_id != 1:
            raise RuntimeError(
                f"{condition} episode {idx} must use source_obj_id=0 and target_obj_id=1, "
                f"got {source_obj_id} and {target_obj_id}"
            )
        expected_source = {"color": "green", "model_id": "baked_green_cube_3cm", "obj_id": 0}
        expected_target = {"color": "yellow", "model_id": "baked_yellow_cube_3cm", "obj_id": 1}
        if model_ids[0] != expected_source["model_id"] or episode.get("source") != expected_source:
            raise RuntimeError(f"{condition} episode {idx} source must be green object 0")
        if model_ids[1] != expected_target["model_id"] or episode.get("target") != expected_target:
            raise RuntimeError(f"{condition} episode {idx} target must be yellow object 1")

        source_support_blocks = _require_nonnegative_int(
            episode.get("source_support_blocks"),
            f"{condition} episode {idx} source_support_blocks",
        )
        target_support_blocks = _require_nonnegative_int(
            episode.get("target_support_blocks"),
            f"{condition} episode {idx} target_support_blocks",
        )
        source_tower_height = _require_nonnegative_int(
            episode.get("source_tower_height"),
            f"{condition} episode {idx} source_tower_height",
        )
        target_tower_height = _require_nonnegative_int(
            episode.get("target_tower_height"),
            f"{condition} episode {idx} target_tower_height",
        )
        if source_support_blocks + target_support_blocks < 1:
            raise RuntimeError(f"{condition} episode {idx} must have at least one support block")
        if target_support_blocks > 2:
            raise RuntimeError(f"{condition} episode {idx} target_support_blocks must be <= 2")
        if target_tower_height > 3:
            raise RuntimeError(f"{condition} episode {idx} target_tower_height must be <= 3")
        if source_tower_height != source_support_blocks + 1:
            raise RuntimeError(f"{condition} episode {idx} source tower/support count mismatch")
        if target_tower_height != target_support_blocks + 1:
            raise RuntimeError(f"{condition} episode {idx} target tower/support count mismatch")

        source_support_colors = episode.get("source_support_colors")
        target_support_colors = episode.get("target_support_colors")
        if not isinstance(source_support_colors, list) or not isinstance(target_support_colors, list):
            raise RuntimeError(f"{condition} episode {idx} support color fields must be lists")
        if len(source_support_colors) != source_support_blocks:
            raise RuntimeError(f"{condition} episode {idx} source support color count mismatch")
        if len(target_support_colors) != target_support_blocks:
            raise RuntimeError(f"{condition} episode {idx} target support color count mismatch")
        allowed_support_colors = {"blue", "red", "white"}
        for support_field, support_colors in (
            ("source_support_colors", source_support_colors),
            ("target_support_colors", target_support_colors),
        ):
            unexpected_support_colors = sorted(set(support_colors) - allowed_support_colors)
            if unexpected_support_colors:
                raise RuntimeError(
                    f"{condition} episode {idx} has unexpected {support_field}: "
                    f"{unexpected_support_colors}"
                )
            if len(support_colors) != len(set(support_colors)):
                raise RuntimeError(f"{condition} episode {idx} repeats a {support_field} color")
        if "post_set_poses" not in episode:
            raise RuntimeError(f"{condition} episode {idx} must contain post_set_poses")

    instruction = episode.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError(f"{condition} episode {idx} must contain a nonempty instruction")

    _require_finite_array(episode.get("init_xys"), (n_objects, 2), f"{condition} episode {idx} init_xys")
    _require_finite_array(
        episode.get("init_rot_quats"),
        (n_objects, 4),
        f"{condition} episode {idx} init_rot_quats",
    )

    if "episode_id" in episode and episode["episode_id"] != idx:
        raise RuntimeError(f"{condition} episode index {idx} has episode_id={episode['episode_id']!r}")

    model_scales = episode.get("model_scales")
    if model_scales is not None:
        _require_finite_array(model_scales, (n_objects,), f"{condition} episode {idx} model_scales")

    robot_qpos = episode.get("robot_qpos")
    if robot_qpos is not None:
        try:
            qpos = np.asarray(robot_qpos, dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{condition} episode {idx} robot_qpos must be numeric") from exc
        if qpos.ndim != 1 or not np.isfinite(qpos).all():
            raise RuntimeError(f"{condition} episode {idx} robot_qpos must be a finite vector")
        if qpos.size == 0:
            raise RuntimeError(f"{condition} episode {idx} robot_qpos must be nonempty")

    robot_initial_state = episode.get("robot_initial_state")
    if robot_initial_state is not None:
        state = _require_mapping(robot_initial_state, f"{condition} episode {idx} robot_initial_state")
        has_base_init = "init_xy" in state or "init_rot_quat" in state
        if has_base_init and ("init_xy" not in state or "init_rot_quat" not in state):
            raise RuntimeError(
                f"{condition} episode {idx} robot_initial_state base init requires both "
                "init_xy and init_rot_quat"
            )
        if "init_xy" in state:
            _require_finite_array(state["init_xy"], (2,), f"{condition} episode {idx} robot_initial_state.init_xy")
        if "init_rot_quat" in state:
            _require_finite_array(
                state["init_rot_quat"],
                (4,),
                f"{condition} episode {idx} robot_initial_state.init_rot_quat",
            )
        if "state_xyz_rpy_pad_gripper" in state:
            _require_finite_array(
                state["state_xyz_rpy_pad_gripper"],
                (8,),
                f"{condition} episode {idx} robot_initial_state.state_xyz_rpy_pad_gripper",
            )
        if (
            not has_base_init
            and "state_xyz_rpy_pad_gripper" not in state
        ):
            raise RuntimeError(
                f"{condition} episode {idx} robot_initial_state must contain init_xy/init_rot_quat "
                "or state_xyz_rpy_pad_gripper"
            )

    robot_ik_state = episode.get("robot_ik_initial_state")
    if robot_ik_state is not None:
        state = _require_mapping(robot_ik_state, f"{condition} episode {idx} robot_ik_initial_state")
        _require_finite_array(
            state.get("state_xyz_rpy_pad_gripper"),
            (8,),
            f"{condition} episode {idx} robot_ik_initial_state.state_xyz_rpy_pad_gripper",
        )

    robot_init_xy = episode.get("robot_init_xy")
    if robot_init_xy is not None:
        _require_finite_array(robot_init_xy, (2,), f"{condition} episode {idx} robot_init_xy")

    post_set_poses = episode.get("post_set_poses")
    if post_set_poses is not None:
        if not isinstance(post_set_poses, list) or len(post_set_poses) != n_objects:
            raise RuntimeError(f"{condition} episode {idx} post_set_poses must have {n_objects} entries")
        for pose_idx, pose in enumerate(post_set_poses):
            pose = _require_mapping(pose, f"{condition} episode {idx} post_set_poses[{pose_idx}]")
            _require_finite_array(pose.get("p"), (3,), f"{condition} episode {idx} post_set_poses[{pose_idx}].p")
            _require_finite_array(pose.get("q"), (4,), f"{condition} episode {idx} post_set_poses[{pose_idx}].q")

    metadata = episode.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise RuntimeError(f"{condition} episode {idx} metadata must be an object when present")


def _validate_abcde_config(config: dict[str, Any]) -> None:
    conditions = _require_mapping(config.get("conditions"), f"{_ABCDE_NAME}.conditions")
    if set(conditions) != set(_ABCDE_CONDITIONS):
        raise RuntimeError(
            f"{_ABCDE_NAME} conditions must be exactly {list(_ABCDE_CONDITIONS)}, "
            f"got {sorted(conditions)}"
        )

    for condition in _ABCDE_CONDITIONS:
        condition_config = _require_mapping(conditions[condition], f"{_ABCDE_NAME}.{condition}")
        episodes = condition_config.get("episodes")
        if not isinstance(episodes, list):
            raise RuntimeError(f"{condition}.episodes must be a list")
        if len(episodes) != _ABCDE_NUM_EPISODES_PER_CONDITION:
            raise RuntimeError(
                f"{condition}.episodes must contain {_ABCDE_NUM_EPISODES_PER_CONDITION} episodes, "
                f"got {len(episodes)}"
            )
        if "num_episodes" in condition_config and int(condition_config["num_episodes"]) != len(episodes):
            raise RuntimeError(f"{condition}.num_episodes does not match episodes length")
        for idx, episode in enumerate(episodes):
            _validate_abcde_episode(condition, idx, episode)


def _load_config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path_value = os.environ.get("SIMPLERENV_PROTOCOL_CONFIG", "").strip()
    if not path_value:
        raise RuntimeError("SIMPLERENV_PROTOCOL_CONFIG is not set")

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Protocol config does not exist: {path}")

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = os.environ.get("SIMPLERENV_PROTOCOL_SHA256", "").strip()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Protocol config SHA mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    config = json.loads(raw)
    config = _require_mapping(config, f"Protocol config {path}")
    protocol_name = config.get("name")
    if protocol_name in _SUPPORTED_PROTOCOLS:
        _validate_legacy_config(config)
    elif protocol_name == _ABCDE_NAME:
        _validate_abcde_config(config)
    else:
        raise RuntimeError(f"Unexpected protocol name: {config.get('name')!r}")

    _CONFIG = config
    print(f"[widowx_protocol] loaded {path} sha256={actual_sha256}", flush=True)
    return config


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_abcde_config(config: dict[str, Any]) -> bool:
    return config.get("name") == _ABCDE_NAME


def _selected_abcde_condition(config: dict[str, Any]) -> str:
    condition = os.environ.get(_ABCDE_CONDITION_ENV, "").strip()
    if not condition:
        raise RuntimeError(f"{_ABCDE_NAME} requires {_ABCDE_CONDITION_ENV}")
    conditions = config["conditions"]
    if condition not in conditions:
        raise RuntimeError(
            f"{_ABCDE_CONDITION_ENV}={condition!r} is not in protocol config; "
            f"expected one of {list(_ABCDE_CONDITIONS)}"
        )
    return condition


def _clear_abcde_episode_state(env: Any) -> None:
    setattr(env, _ABCDE_EPISODE_ATTR, None)
    setattr(env, _ABCDE_CONDITION_ATTR, None)
    setattr(env, _ABCDE_INSTRUCTION_ATTR, None)


def _require_abcde_episode_id(condition: str, obj_init_options: dict[str, Any]) -> int:
    if "episode_id" not in obj_init_options:
        raise RuntimeError(
            f"{_ABCDE_NAME}/{condition} requires obj_init_options.episode_id"
        )
    try:
        episode_id = int(obj_init_options["episode_id"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{_ABCDE_NAME}/{condition} episode_id must be an integer, "
            f"got {obj_init_options['episode_id']!r}"
        ) from exc
    return episode_id


def _get_abcde_episode(
    config: dict[str, Any], obj_init_options: dict[str, Any]
) -> tuple[str, int, dict[str, Any]]:
    condition = _selected_abcde_condition(config)
    episode_id = _require_abcde_episode_id(condition, obj_init_options)
    episodes = config["conditions"][condition]["episodes"]
    if episode_id < 0 or episode_id >= len(episodes):
        raise RuntimeError(
            f"{_ABCDE_NAME}/{condition} episode_id {episode_id} out of range [0, {len(episodes)})"
        )
    return condition, episode_id, episodes[episode_id]


def _resolve_task_key(env: Any) -> str:
    pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    task_key = _TASK_BY_MODEL_PAIR.get(pair)
    if task_key is None:
        raise RuntimeError(
            "Protocol 1 only applies to stack/carrot/spoon. "
            f"Got source/target pair {pair!r}"
        )
    return task_key


def _require_protocol3_config(config: dict[str, Any]) -> None:
    if config.get("name") != "widowx_protocol1_random_positions":
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} must use the Protocol 1 randomized-position config; "
            f"got {config.get('name')!r}"
        )


def _require_protocol3_stack_pair(env: Any) -> None:
    pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    if pair != _PROTOCOL3_STACK_PAIR:
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} only supports the baked stack-cube env with "
            f"source/target pair {_PROTOCOL3_STACK_PAIR!r}; got {pair!r}"
        )


def _protocol3_actor(env: Any, actor_name: str) -> tuple[int, Any]:
    objs = getattr(env, "episode_objs", None)
    if objs is None:
        raise RuntimeError(f"{_PROTOCOL3_NAME} could not read env.episode_objs")
    matches = [(idx, obj) for idx, obj in enumerate(objs) if getattr(obj, "name", None) == actor_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} expected exactly one actor named {actor_name!r}; "
            f"found {len(matches)} among {[getattr(obj, 'name', None) for obj in objs]!r}"
        )
    return matches[0]


def _protocol3_actor_bbox(env: Any, actor_name: str) -> np.ndarray:
    if actor_name == getattr(getattr(env, "episode_source_obj", None), "name", None):
        bbox = getattr(env, "episode_source_obj_bbox_world", None)
    elif actor_name == getattr(getattr(env, "episode_target_obj", None), "name", None):
        bbox = getattr(env, "episode_target_obj_bbox_world", None)
    else:
        raise RuntimeError(f"{_PROTOCOL3_NAME} cannot resolve bbox for actor {actor_name!r}")

    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (3,) or not np.isfinite(bbox).all():
        raise RuntimeError(f"{_PROTOCOL3_NAME} invalid bbox for {actor_name!r}: {bbox}")
    return bbox


def _protocol3_actor_pose(actor: Any) -> sapien.Pose:
    return actor.pose.transform(actor.cmass_local_pose)


def _protocol3_src_on_target(
    env: Any,
    source_name: str,
    target_name: str,
    *,
    success_require_src_completely_on_target: bool,
    z_flag_required_offset: float,
) -> bool:
    _, source_obj = _protocol3_actor(env, source_name)
    _, target_obj = _protocol3_actor(env, target_name)
    source_pose = _protocol3_actor_pose(source_obj)
    target_pose = _protocol3_actor_pose(target_obj)

    tgt_obj_half_length_bbox = _protocol3_actor_bbox(env, target_name) / 2
    src_obj_half_length_bbox = _protocol3_actor_bbox(env, source_name) / 2

    offset = source_pose.p - target_pose.p
    xy_flag = (
        np.linalg.norm(offset[:2])
        <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
    )
    z_flag = (offset[2] > 0) and (
        offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
        <= z_flag_required_offset
    )
    src_on_target = bool(xy_flag and z_flag)

    if success_require_src_completely_on_target:
        contacts = env._scene.get_contacts()
        flag = True
        robot_link_names = [x.name for x in env.agent.robot.get_links()]
        ignore_actor_names = [target_obj.name] + robot_link_names
        for contact in contacts:
            actor_0, actor_1 = contact.actor0, contact.actor1
            other_obj_contact_actor_name = None
            if actor_0.name == source_obj.name:
                other_obj_contact_actor_name = actor_1.name
            elif actor_1.name == source_obj.name:
                other_obj_contact_actor_name = actor_0.name
            if other_obj_contact_actor_name is not None:
                contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
                if (other_obj_contact_actor_name not in ignore_actor_names) and (
                    np.linalg.norm(contact_impulse) > 1e-6
                ):
                    flag = False
                    break
        src_on_target = bool(src_on_target and flag)

    return src_on_target


def _protocol3_moved_flags(env: Any, source_name: str) -> tuple[bool, bool]:
    source_idx, source_obj = _protocol3_actor(env, source_name)
    settled_xyzs = getattr(env, "episode_obj_xyzs_after_settle", None)
    objs = getattr(env, "episode_objs", None)
    if settled_xyzs is None or objs is None or len(settled_xyzs) != len(objs):
        raise RuntimeError(f"{_PROTOCOL3_NAME} invalid settled object state")

    source_obj_xy_move_dist = np.linalg.norm(
        np.asarray(settled_xyzs[source_idx], dtype=float)[:2] - source_obj.pose.p[:2]
    )
    other_obj_xy_move_dist = []
    for idx, obj in enumerate(objs):
        if idx == source_idx:
            continue
        other_obj_xy_move_dist.append(
            np.linalg.norm(np.asarray(settled_xyzs[idx], dtype=float)[:2] - obj.pose.p[:2])
        )
    moved_correct_obj = (source_obj_xy_move_dist > 0.03) and all(
        x < source_obj_xy_move_dist for x in other_obj_xy_move_dist
    )
    moved_wrong_obj = any(x > 0.03 for x in other_obj_xy_move_dist) and any(
        x > source_obj_xy_move_dist for x in other_obj_xy_move_dist
    )
    return bool(moved_correct_obj), bool(moved_wrong_obj)


def _as_xy(episode: dict[str, Any]) -> np.ndarray:
    return np.array([episode["source_xy"], episode["target_xy"]], dtype=float)


def _as_quat(episode: dict[str, Any]) -> np.ndarray:
    return np.array([episode["source_quat_wxyz"], episode["target_quat_wxyz"]], dtype=float)


def _validate_xy(task_key: str, task: dict[str, Any], xys: np.ndarray, *, settled: bool) -> None:
    center = np.array(task["square_center"], dtype=float)
    half_edge = float(task["half_edge_m"])
    min_separation = float(task["min_separation_m"])
    tol = 0.01 if settled else 1e-9

    if xys.shape != (2, 2):
        raise RuntimeError(f"{task_key} protocol xy has invalid shape: {xys.shape}")
    if np.any(xys < center - half_edge - tol) or np.any(xys > center + half_edge + tol):
        phase = "settled" if settled else "initial"
        raise RuntimeError(f"{task_key} {phase} xy outside Protocol 1 square: {xys.tolist()}")

    distance = float(np.linalg.norm(xys[0] - xys[1]))
    min_allowed = min_separation - (0.02 if settled else 1e-9)
    if distance < min_allowed:
        phase = "settled" if settled else "initial"
        raise RuntimeError(
            f"{task_key} {phase} source-target distance {distance:.4f} "
            f"is below allowed minimum {min_allowed:.4f}"
        )


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)))


def _robot_pose_from_state(robot_state: dict[str, Any]) -> sapien.Pose:
    state = np.array(robot_state["state_xyz_rpy_pad_gripper"], dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid protocol robot_initial_state: {state}")
    quat_wxyz = mat2quat(euler2mat(*state[3:6]) @ _MAT_TRANSFORM)
    return sapien.Pose(p=state[:3], q=quat_wxyz)


def _apply_robot_initial_state(env: Any, robot_state: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
    controller = env.agent.controller.controllers["arm"]
    target_pose = _robot_pose_from_state(robot_state)
    init_arm_qpos = controller.compute_ik(target_pose)
    if init_arm_qpos is None:
        raise RuntimeError(
            "Protocol IK failed for robot_initial_state "
            f"candidate_index={robot_state.get('candidate_index')} "
            f"episode_index={robot_state.get('episode_index')}"
        )

    cur_qpos = env.agent.robot.get_qpos()
    cur_qpos[controller.joint_indices] = init_arm_qpos
    env.agent.reset(cur_qpos)

    actual_pose = controller.ee_pose_at_base
    pos_error = float(np.linalg.norm(actual_pose.p - target_pose.p))
    rot_error = _quat_angle(np.asarray(actual_pose.q, dtype=float), np.asarray(target_pose.q, dtype=float))
    if pos_error > 0.035 or rot_error > 0.60:
        raise RuntimeError(
            "Protocol IK error too large for robot_initial_state "
            f"candidate_index={robot_state.get('candidate_index')} "
            f"episode_index={robot_state.get('episode_index')}: "
            f"pos_error={pos_error:.6f}, rot_error={rot_error:.6f}"
        )

    return env.get_obs(), pos_error, rot_error


def _abcde_robot_ik_state(episode: dict[str, Any]) -> dict[str, Any] | None:
    robot_ik_state = episode.get("robot_ik_initial_state")
    if robot_ik_state is not None:
        return robot_ik_state
    robot_initial_state = episode.get("robot_initial_state")
    if isinstance(robot_initial_state, dict) and "state_xyz_rpy_pad_gripper" in robot_initial_state:
        return robot_initial_state
    return None


def _apply_robot_qpos(env: Any, episode: dict[str, Any]) -> dict[str, Any]:
    qpos = np.asarray(episode["robot_qpos"], dtype=float)
    current_qpos = np.asarray(env.agent.robot.get_qpos(), dtype=float)
    if qpos.shape != current_qpos.shape:
        raise RuntimeError(
            f"{_ABCDE_NAME} robot_qpos shape {qpos.shape} does not match env qpos shape {current_qpos.shape}"
        )
    env.agent.reset(qpos)
    return env.get_obs()


def _apply_post_set_poses(
    env: Any,
    episode: dict[str, Any],
    *,
    source_obj_id: int,
    target_obj_id: int,
) -> dict[str, Any]:
    post_set_poses = episode.get("post_set_poses")
    if post_set_poses is None:
        return env.get_obs()
    if len(post_set_poses) != len(env.episode_objs):
        raise RuntimeError(
            f"{_ABCDE_NAME} post_set_poses count {len(post_set_poses)} "
            f"does not match actor count {len(env.episode_objs)}"
        )

    for actor, pose in zip(env.episode_objs, post_set_poses):
        actor.set_pose(
            sapien.Pose(
                np.asarray(pose["p"], dtype=float),
                np.asarray(pose["q"], dtype=float),
            )
        )
        actor.set_velocity(np.zeros(3))
        actor.set_angular_velocity(np.zeros(3))

    env.episode_obj_xyzs_after_settle = [
        np.asarray(obj.pose.p, dtype=float).copy() for obj in env.episode_objs
    ]
    env.episode_source_obj_xyz_after_settle = env.episode_obj_xyzs_after_settle[source_obj_id]
    env.episode_target_obj_xyz_after_settle = env.episode_obj_xyzs_after_settle[target_obj_id]
    env.episode_source_obj_bbox_world = (
        quat2mat(np.asarray(env.episode_source_obj.pose.q, dtype=float))
        @ np.asarray(env.episode_model_bbox_sizes[source_obj_id], dtype=float)
    )
    env.episode_target_obj_bbox_world = (
        quat2mat(np.asarray(env.episode_target_obj.pose.q, dtype=float))
        @ np.asarray(env.episode_model_bbox_sizes[target_obj_id], dtype=float)
    )
    return env.get_obs()


def _reset_abcde(
    env: Any,
    config: dict[str, Any],
    parent_reset: Any,
    *,
    seed: Any,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _clear_abcde_episode_state(env)
    if options is None:
        options = {}
    else:
        options = options.copy()
    obj_init_options = dict(options.get("obj_init_options") or {})

    if "episode_id" not in obj_init_options and options.get("reconfigure") is True:
        return _ORIGINAL_RESET(env, seed=seed, options=options)

    env_pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    if _TASK_BY_MODEL_PAIR.get(env_pair) != "stack":
        raise RuntimeError(f"{_ABCDE_NAME} only supports the stack env, got source/target pair {env_pair!r}")

    condition, episode_id, episode = _get_abcde_episode(config, obj_init_options)
    model_ids = list(episode["model_ids"])
    missing_model_ids = [model_id for model_id in model_ids if model_id not in env.model_db]
    if missing_model_ids:
        raise RuntimeError(
            f"{_ABCDE_NAME}/{condition} episode {episode_id} model_ids missing from env.model_db: "
            f"{missing_model_ids}"
        )

    n_objects = len(model_ids)
    source_obj_id = int(episode["source_obj_id"])
    target_obj_id = int(episode["target_obj_id"])

    options["model_ids"] = model_ids
    options["model_scales"] = list(episode.get("model_scales") or [1.0] * n_objects)

    obj_init_options["source_obj_id"] = source_obj_id
    obj_init_options["target_obj_id"] = target_obj_id
    obj_init_options["init_xys"] = np.asarray(episode["init_xys"], dtype=float)
    obj_init_options["init_rot_quats"] = np.asarray(episode["init_rot_quats"], dtype=float)
    options["obj_init_options"] = obj_init_options

    robot_init_options = dict(options.get("robot_init_options") or {})
    robot_initial_state = episode.get("robot_initial_state")
    if isinstance(robot_initial_state, dict):
        if robot_initial_state.get("init_xy") is not None:
            robot_init_options["init_xy"] = np.asarray(robot_initial_state["init_xy"], dtype=float)
        if robot_initial_state.get("init_rot_quat") is not None:
            robot_init_options["init_rot_quat"] = np.asarray(
                robot_initial_state["init_rot_quat"], dtype=float
            )
    if episode.get("robot_init_xy") is not None:
        robot_init_options["init_xy"] = np.asarray(episode["robot_init_xy"], dtype=float)
    if robot_init_options:
        options["robot_init_options"] = robot_init_options

    obs, info = parent_reset(env, seed=seed, options=options)

    if episode.get("robot_qpos") is not None:
        obs = _apply_robot_qpos(env, episode)
        info["protocol_robot_qpos_applied"] = True
    elif (robot_ik_state := _abcde_robot_ik_state(episode)) is not None:
        obs, ik_pos_error, ik_rot_error = _apply_robot_initial_state(env, robot_ik_state)
        info.update(
            {
                "protocol_robot_initial_state_applied": True,
                "protocol_robot_ik_pos_error_m": ik_pos_error,
                "protocol_robot_ik_rot_error_rad": ik_rot_error,
            }
        )

    if episode.get("post_set_poses") is not None:
        obs = _apply_post_set_poses(
            env,
            episode,
            source_obj_id=source_obj_id,
            target_obj_id=target_obj_id,
        )
        info["protocol_post_set_poses_applied"] = True

    setattr(env, _ABCDE_EPISODE_ATTR, episode)
    setattr(env, _ABCDE_CONDITION_ATTR, condition)
    setattr(env, _ABCDE_INSTRUCTION_ATTR, episode["instruction"])

    info.update(
        {
            "protocol": config["name"],
            "protocol_version": config.get("version"),
            "protocol_condition": condition,
            "protocol_episode_id": episode_id,
            "protocol_instruction": episode["instruction"],
            "protocol_model_ids": model_ids,
            "protocol_source_obj_id": source_obj_id,
            "protocol_target_obj_id": target_obj_id,
            "protocol_source_obj_name": model_ids[source_obj_id],
            "protocol_target_obj_name": model_ids[target_obj_id],
            "protocol_metadata": episode.get("metadata", {}),
        }
    )
    if "case_id" in episode:
        info["protocol_case_id"] = episode["case_id"]
    return obs, info


def _install_abcde_stack_patch(put_on_in_scene: Any) -> None:
    global _ABCDE_STACK_PATCH_INSTALLED
    global _ABCDE_ORIGINAL_STACK_EVALUATE
    if _ABCDE_STACK_PATCH_INSTALLED:
        return

    stack_cls = put_on_in_scene.StackGreenCubeOnYellowCubeInScene
    _ABCDE_ORIGINAL_STACK_EVALUATE = stack_cls.evaluate

    def abcde_language(self, **kwargs):
        instruction = getattr(self, _ABCDE_INSTRUCTION_ATTR, None)
        if not isinstance(instruction, str) or not instruction:
            raise RuntimeError(
                f"{_ABCDE_NAME} language requested before a configured episode reset; "
                "reset with obj_init_options.episode_id first"
            )
        return instruction

    def abcde_evaluate(self, *args, **kwargs):
        episode = getattr(self, _ABCDE_EPISODE_ATTR, None)
        condition = getattr(self, _ABCDE_CONDITION_ATTR, None)
        if episode is None or condition is None:
            raise RuntimeError(
                f"{_ABCDE_NAME} success requested before a configured episode reset; "
                "reset with obj_init_options.episode_id first"
            )

        result = _ABCDE_ORIGINAL_STACK_EVALUATE(self, *args, **kwargs)
        if condition == "protocol_C1_yellow_on_green":
            success_require_src_completely_on_target = (
                args[0]
                if len(args) >= 1
                else kwargs.get("success_require_src_completely_on_target", True)
            )
            z_flag_required_offset = (
                args[1] if len(args) >= 2 else kwargs.get("z_flag_required_offset", 0.02)
            )
            green_on_yellow = _protocol3_src_on_target(
                self,
                "baked_green_cube_3cm",
                "baked_yellow_cube_3cm",
                success_require_src_completely_on_target=success_require_src_completely_on_target,
                z_flag_required_offset=z_flag_required_offset,
            )
            self.episode_stats["env_success_green_on_yellow"] = green_on_yellow
            result["env_success_green_on_yellow"] = green_on_yellow
            result["episode_stats"] = self.episode_stats
        return result

    stack_cls.get_language_instruction = abcde_language
    stack_cls.evaluate = abcde_evaluate
    _ABCDE_STACK_PATCH_INSTALLED = True
    print(f"[widowx_protocol] {_ABCDE_NAME} stack language/success patch installed", flush=True)


def _install_protocol3_stack_patch(put_on_in_scene: Any, config: dict[str, Any]) -> None:
    global _PROTOCOL3_INSTALLED
    if _PROTOCOL3_INSTALLED:
        return

    _require_protocol3_config(config)
    stack_cls = put_on_in_scene.StackGreenCubeOnYellowCubeInScene

    def protocol3_language(self, **kwargs):
        _require_protocol3_stack_pair(self)
        return _PROTOCOL3_INSTRUCTION

    def protocol3_evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        _require_protocol3_stack_pair(self)
        moved_correct_obj, moved_wrong_obj = _protocol3_moved_flags(self, "baked_yellow_cube_3cm")
        _, yellow_obj = _protocol3_actor(self, "baked_yellow_cube_3cm")

        is_src_obj_grasped = self.agent.check_grasp(yellow_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        yellow_on_green = _protocol3_src_on_target(
            self,
            "baked_yellow_cube_3cm",
            "baked_green_cube_3cm",
            success_require_src_completely_on_target=success_require_src_completely_on_target,
            z_flag_required_offset=z_flag_required_offset,
        )
        green_on_yellow = _protocol3_src_on_target(
            self,
            "baked_green_cube_3cm",
            "baked_yellow_cube_3cm",
            success_require_src_completely_on_target=success_require_src_completely_on_target,
            z_flag_required_offset=z_flag_required_offset,
        )
        success = yellow_on_green

        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = yellow_on_green
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )
        self.episode_stats["protocol3_success_yellow_on_green"] = yellow_on_green
        self.episode_stats["env_success_green_on_yellow"] = green_on_yellow

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=yellow_on_green,
            protocol3_success_yellow_on_green=yellow_on_green,
            env_success_green_on_yellow=green_on_yellow,
            episode_stats=self.episode_stats,
            success=success,
        )

    stack_cls.get_language_instruction = protocol3_language
    stack_cls.evaluate = protocol3_evaluate
    _PROTOCOL3_INSTALLED = True
    print(f"[widowx_protocol] {_PROTOCOL3_NAME} patch installed", flush=True)


def install_from_env(*, required: bool = False) -> dict[str, Any] | None:
    """Install the WidowX protocol reset patch when SIMPLERENV_PROTOCOL_CONFIG is set."""

    global _INSTALLED, _ORIGINAL_RESET
    if not os.environ.get("SIMPLERENV_PROTOCOL_CONFIG"):
        if required:
            raise RuntimeError("Protocol injection requested but SIMPLERENV_PROTOCOL_CONFIG is not set")
        return None

    config = _load_config()
    if _is_abcde_config(config):
        if _env_flag_enabled(_PROTOCOL3_FLAG):
            raise RuntimeError(f"{_PROTOCOL3_FLAG} cannot be combined with {_ABCDE_NAME}")
        _selected_abcde_condition(config)

    if _INSTALLED:
        if _is_abcde_config(config):
            from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

            _install_abcde_stack_patch(put_on_in_scene)
        elif _env_flag_enabled(_PROTOCOL3_FLAG):
            from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

            _install_protocol3_stack_patch(put_on_in_scene, config)
        return config

    from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

    cls = put_on_in_scene.PutOnBridgeInSceneEnv
    _ORIGINAL_RESET = cls.reset
    abcde_parent_reset = put_on_in_scene.PutOnInSceneEnv.reset

    def protocol_reset(self, seed=None, options=None):
        if _is_abcde_config(config):
            return _reset_abcde(
                self,
                config,
                abcde_parent_reset,
                seed=seed,
                options=options,
            )

        task_key = _resolve_task_key(self)
        if _env_flag_enabled(_PROTOCOL3_FLAG) and task_key != "stack":
            raise RuntimeError(f"{_PROTOCOL3_NAME} only supports stack, got task {task_key!r}")
        task = config["tasks"][task_key]

        if options is None:
            options = {}
        else:
            options = options.copy()
        obj_init_options = dict(options.get("obj_init_options") or {})

        if "episode_id" not in obj_init_options:
            # ManiSkill performs a constructor-time reconfigure reset before evaluators pick an episode.
            if options.get("reconfigure") is True:
                return _ORIGINAL_RESET(self, seed=seed, options=options)
            raise RuntimeError(f"{config['name']} requires obj_init_options.episode_id for {task_key}")
        episode_id = int(obj_init_options["episode_id"])
        episodes = task["episodes"]
        if episode_id < 0 or episode_id >= len(episodes):
            raise RuntimeError(
                f"{task_key} {config['name']} episode_id {episode_id} out of range [0, {len(episodes)})"
            )

        episode = episodes[episode_id]
        xys = _as_xy(episode)
        quats = _as_quat(episode)
        _validate_xy(task_key, task, xys, settled=False)

        self._xy_configs = [xys]
        self._quat_configs = [quats]
        options["obj_init_options"] = obj_init_options

        obs, info = _ORIGINAL_RESET(self, seed=seed, options=options)

        settled_xys = np.array(
            [
                self.episode_source_obj.pose.p[:2],
                self.episode_target_obj.pose.p[:2],
            ],
            dtype=float,
        )
        _validate_xy(task_key, task, settled_xys, settled=True)

        if config["name"] == "widowx_protocol2_random_positions_robot_initial_states":
            robot_state = episode["robot_initial_state"]
            obs, ik_pos_error, ik_rot_error = _apply_robot_initial_state(self, robot_state)
            info.update(
                {
                    "protocol_robot_state_candidate_index": int(robot_state["candidate_index"]),
                    "protocol_robot_state_episode_index": int(robot_state["episode_index"]),
                    "protocol_robot_state_task": robot_state["task"],
                    "protocol_robot_ik_pos_error_m": ik_pos_error,
                    "protocol_robot_ik_rot_error_rad": ik_rot_error,
                }
            )

        info.update(
            {
                "protocol": config["name"],
                "protocol_version": config["version"],
                "protocol_task": task_key,
                "protocol_episode_id": episode_id,
            }
        )
        return obs, info

    cls.reset = protocol_reset
    _INSTALLED = True
    if _is_abcde_config(config):
        _install_abcde_stack_patch(put_on_in_scene)
    elif _env_flag_enabled(_PROTOCOL3_FLAG):
        _install_protocol3_stack_patch(put_on_in_scene, config)

    print("[widowx_protocol] reset patch installed", flush=True)
    return config
