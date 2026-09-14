#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calvin_sequence_manifest import load_manifest, validate_bank_metadata_against_manifest

PROTOCOL = "calvin_official_d_table_resets_v1"
BLOCK_SLICES = {"red_block": (6, 12), "blue_block": (12, 18), "pink_block": (18, 24)}


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


def hash_rows(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_calvin(calvin_root: Path):
    for rel in ("calvin_models", "calvin_env"):
        path = calvin_root / rel
        if not path.exists():
            raise FileNotFoundError(f"missing CALVIN path: {path}")
        sys.path.insert(0, str(path))
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    return get_sequences, get_env_state_for_initial_condition


def load_validation_env(calvin_root: Path, dataset_dir: Path):
    for rel in ("calvin_models", "calvin_env"):
        sys.path.insert(0, str(calvin_root / rel))
    import hydra
    from omegaconf import OmegaConf

    conf_path = dataset_dir / "validation" / ".hydra" / "merged_config.yaml"
    if not conf_path.exists():
        raise FileNotFoundError(f"missing CALVIN validation config: {conf_path}")
    render_conf = OmegaConf.load(conf_path)
    render_conf.env.use_egl = False
    render_conf.cameras = {}
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=True)


def load_scene_d_table_surface(calvin_root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    from omegaconf import OmegaConf

    scene_path = calvin_root / "calvin_env" / "conf" / "scene" / "calvin_scene_D.yaml"
    if not scene_path.exists():
        raise FileNotFoundError(f"missing scene-D config: {scene_path}")
    cfg = OmegaConf.load(scene_path)
    surface = np.asarray(cfg.surfaces.table, dtype=np.float64)
    if surface.shape != (2, 3):
        raise RuntimeError(f"unexpected scene-D table surface shape: {surface.shape} in {scene_path}")
    if not np.allclose(surface[:, 2], surface[0, 2]):
        raise RuntimeError(f"scene-D table z surface is not flat: {surface.tolist()}")
    return surface[0], surface[1], scene_path


def table_blocks(initial_state: dict[str, Any]) -> tuple[str, ...]:
    blocks = tuple(block for block in BLOCK_SLICES if initial_state.get(block) == "table")
    if len(blocks) not in (1, 2):
        raise ValueError(f"expected one or two table blocks, got {blocks}: {initial_state}")
    return blocks


def in_bounds(pos: np.ndarray, low: np.ndarray, high: np.ndarray, tol: float) -> bool:
    return bool(np.all(pos >= low - tol) and np.all(pos <= high + tol))


def movable_pair_contacts(env) -> int:
    count = 0
    for obj_a, obj_b in itertools.combinations(env.scene.movable_objects, 2):
        contacts = env.p.getContactPoints(bodyA=obj_a.uid, bodyB=obj_b.uid, physicsClientId=env.cid)
        count += len(contacts)
    return count


def validate_scene(env, intended_scene: np.ndarray, readback_scene: np.ndarray, blocks: tuple[str, ...], table_low: np.ndarray, table_high: np.ndarray, args) -> str | None:
    if readback_scene.shape != (24,):
        return f"readback scene_obs shape {readback_scene.shape}, expected (24,)"
    if not np.isfinite(readback_scene).all():
        return "readback scene_obs has non-finite values"
    pair_contacts = movable_pair_contacts(env)
    if pair_contacts:
        return f"pairwise movable-object contacts={pair_contacts}"
    for block in blocks:
        start, stop = BLOCK_SLICES[block]
        pos = readback_scene[start : start + 3]
        if not in_bounds(pos, table_low, table_high, float(args.bounds_tolerance)):
            return f"{block} readback table position out of bounds: {pos.tolist()}"
        if not np.allclose(pos, intended_scene[start : start + 3], atol=float(args.position_tolerance), rtol=0.0):
            return f"{block} readback position drift"
        if not np.isfinite(readback_scene[stop - 1]):
            return f"{block} readback yaw is non-finite"
    for block in set(BLOCK_SLICES) - set(blocks):
        start, stop = BLOCK_SLICES[block]
        if not np.allclose(readback_scene[start:stop], intended_scene[start:stop], atol=float(args.non_table_tolerance), rtol=0.0):
            return f"{block} non-table state drifted"
    return None


def sample_table_scene(rng: np.random.Generator, official_scene: np.ndarray, blocks: tuple[str, ...], table_low: np.ndarray, table_high: np.ndarray) -> np.ndarray:
    scene = official_scene.copy()
    for block in blocks:
        start, stop = BLOCK_SLICES[block]
        scene[start : start + 3] = rng.uniform(table_low, table_high)
        scene[start + 3 : stop] = 0.0
        scene[stop - 1] = float(rng.uniform(-np.pi, np.pi))
    return scene


def build_bank(args):
    dataset_dir = Path(args.dataset_dir).resolve()
    calvin_root = Path(args.calvin_root).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir does not exist: {dataset_dir}")
    get_sequences, official_reset = load_calvin(calvin_root)
    table_low, table_high, scene_cfg = load_scene_d_table_surface(calvin_root)
    env = load_validation_env(calvin_root, dataset_dir)

    eval_sequences = list(get_sequences(int(args.num_sequences), num_workers=int(args.sequence_workers)))
    if len(eval_sequences) != int(args.num_sequences):
        raise ValueError(f"get_sequences returned {len(eval_sequences)}, expected {args.num_sequences}")

    rng = np.random.default_rng(int(args.seed))
    validation_fail_counts: dict[str, int] = {}
    rows = {k: [] for k in [
        "robot_obs", "scene_obs", "official_robot_obs", "official_scene_obs", "table_signature",
        "initial_state_json", "eval_sequence_json", "sample_attempts"
    ]}

    try:
        for eval_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
            blocks = tuple(sorted(table_blocks(initial_state)))
            official_robot, official_scene = official_reset(initial_state)
            official_robot = np.asarray(official_robot, dtype=np.float64).reshape(-1)
            official_scene = np.asarray(official_scene, dtype=np.float64).reshape(-1)
            if official_robot.shape != (15,) or official_scene.shape != (24,):
                raise ValueError(f"bad official reset shapes at {eval_idx}: robot={official_robot.shape} scene={official_scene.shape}")
            if not np.isfinite(official_robot).all() or not np.isfinite(official_scene).all():
                raise ValueError(f"non-finite official reset at index {eval_idx}")

            accepted_scene = None
            last_reason = "not attempted"
            for attempt in range(1, int(args.max_attempts) + 1):
                candidate_scene = sample_table_scene(rng, official_scene, blocks, table_low, table_high)
                obs = env.reset(robot_obs=official_robot, scene_obs=candidate_scene)
                readback_scene = np.asarray(obs["scene_obs"], dtype=np.float64).reshape(-1)
                reason = validate_scene(env, candidate_scene, readback_scene, blocks, table_low, table_high, args)
                if reason is None:
                    accepted_scene = readback_scene.copy()
                    rows["sample_attempts"].append(attempt)
                    break
                last_reason = reason
                key = reason.split(":")[0]
                validation_fail_counts[key] = validation_fail_counts.get(key, 0) + 1
            if accepted_scene is None:
                raise RuntimeError(f"failed to sample valid official-D table reset at eval index {eval_idx}, blocks={blocks}, last_reason={last_reason}")

            rows["robot_obs"].append(official_robot)
            rows["scene_obs"].append(accepted_scene)
            rows["official_robot_obs"].append(official_robot)
            rows["official_scene_obs"].append(official_scene)
            rows["table_signature"].append(",".join(blocks))
            rows["initial_state_json"].append(stable_json(initial_state))
            rows["eval_sequence_json"].append(stable_json(eval_sequence))
            if args.progress_every and (eval_idx + 1) % args.progress_every == 0:
                print(f"built {eval_idx + 1}/{args.num_sequences} official-D table resets", flush=True)
    finally:
        env.close()

    metadata = {
        "protocol": PROTOCOL,
        "dataset_dir": str(dataset_dir),
        "calvin_root": str(calvin_root),
        "scene_config": str(scene_cfg),
        "num_sequences": int(args.num_sequences),
        "seed": int(args.seed),
        "sequence_workers": int(args.sequence_workers),
        "table_surface": {"low": table_low.tolist(), "high": table_high.tolist()},
        "orientation_sampler": "uniform_yaw_-pi_pi",
        "max_attempts": int(args.max_attempts),
        "position_tolerance": float(args.position_tolerance),
        "non_table_tolerance": float(args.non_table_tolerance),
        "bounds_tolerance": float(args.bounds_tolerance),
        "validation_fail_counts": validation_fail_counts,
        "sample_attempts": {
            "max": int(max(rows["sample_attempts"])) if rows["sample_attempts"] else 0,
            "mean": float(np.mean(rows["sample_attempts"])) if rows["sample_attempts"] else 0.0,
        },
        "initial_state_sha256": hash_rows(rows["initial_state_json"]),
        "eval_sequence_sha256": hash_rows(rows["eval_sequence_json"]),
    }
    if args.sequence_manifest:
        manifest = load_manifest(args.sequence_manifest)
        validate_bank_metadata_against_manifest(metadata, manifest, "calvin_reset_bank")
        metadata["sequence_manifest"] = str(Path(args.sequence_manifest).resolve())
    return {
        "robot_obs": np.asarray(rows["robot_obs"], dtype=np.float64),
        "scene_obs": np.asarray(rows["scene_obs"], dtype=np.float64),
        "official_robot_obs": np.asarray(rows["official_robot_obs"], dtype=np.float64),
        "official_scene_obs": np.asarray(rows["official_scene_obs"], dtype=np.float64),
        "table_signature": np.asarray(rows["table_signature"], dtype=str),
        "sample_attempts": np.asarray(rows["sample_attempts"], dtype=np.int64),
        "initial_state_json": np.asarray(rows["initial_state_json"], dtype=str),
        "eval_sequence_json": np.asarray(rows["eval_sequence_json"], dtype=str),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True), dtype=str),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a CALVIN CALVIN resampled-pose reset bank using official scene-D table randomization.")
    parser.add_argument("--dataset-dir", required=True, help="CALVIN task_ABC_D dataset dir containing validation/.hydra/merged_config.yaml")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument("--sequence-manifest", default=None)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--position-tolerance", type=float, default=5e-3)
    parser.add_argument("--non-table-tolerance", type=float, default=5e-3)
    parser.add_argument("--bounds-tolerance", type=float, default=5e-3)
    args = parser.parse_args()
    if args.num_sequences <= 0:
        raise ValueError(f"--num-sequences must be positive, got {args.num_sequences}")
    if args.sequence_workers <= 0:
        raise ValueError(f"--sequence-workers must be positive, got {args.sequence_workers}")
    if args.max_attempts <= 0:
        raise ValueError(f"--max-attempts must be positive, got {args.max_attempts}")
    return args


def main():
    args = parse_args()
    bank = build_bank(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **bank)
    tmp.replace(output)
    print(str(bank["metadata_json"].item()))
    print(f"wrote reset bank: {output}")


if __name__ == "__main__":
    main()
