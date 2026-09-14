from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_source_dependency.widowx.model import WidowXViTSMLPPolicy
from data_source_dependency.widowx.normalization import denormalize_action_array, preprocess_proprio_array
from data_source_dependency.widowx.proprio import (
    initial_xvla_proprio,
    update_xvla_proprio_from_raw_action,
    update_xvla_proprio_from_env_action,
    xvla_first10_to_env_action,
)
from data_source_dependency.widowx.tasks import TASKS, parse_tasks


def _step_from_name(path: Path) -> int:
    name = path.stem
    if not name.startswith("step") or not name[4:].isdigit():
        return -1
    return int(name[4:])


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    paths = sorted(Path(checkpoint_dir).glob("step*.pt"), key=_step_from_name)
    paths = [p for p in paths if _step_from_name(p) >= 0]
    if not paths:
        raise FileNotFoundError(f"No step*.pt checkpoints under {checkpoint_dir}")
    return paths[-1]


def load_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[WidowXViTSMLPPolicy, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"])
    cfg["pretrained_visual"] = False
    model = WidowXViTSMLPPolicy(cfg=cfg, num_tasks=len(ckpt["task_vocab"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    payload = {
        "cfg": cfg,
        "task_vocab": list(ckpt["task_vocab"]),
        "action_mean": np.asarray(ckpt["action_mean"], dtype=np.float32),
        "action_std": np.asarray(ckpt["action_std"], dtype=np.float32),
        "action_min": np.asarray(ckpt["action_min"], dtype=np.float32),
        "action_max": np.asarray(ckpt["action_max"], dtype=np.float32),
        "step": int(ckpt["step"]),
    }
    for key in ("proprio_mean", "proprio_std", "proprio_min", "proprio_max"):
        value = ckpt.get(key)
        if value is not None:
            payload[key] = np.asarray(value, dtype=np.float32)
    return model, payload


def unnormalize_action_chunk(chunk: np.ndarray, payload: dict[str, Any], threshold_gripper: bool, clip: bool) -> np.ndarray:
    return denormalize_action_array(chunk, payload, threshold_gripper=threshold_gripper, clip=clip)


def decode_policy_chunk(
    chunk: np.ndarray,
    payload: dict[str, Any],
    threshold_gripper: bool,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    action_chunk = unnormalize_action_chunk(chunk, payload, threshold_gripper, clip)
    action_key = str(payload["cfg"].get("action_key", "actions_env"))
    if action_key == "actions_env":
        return action_chunk.astype(np.float32), None
    if action_key == "actions_xvla_first10":
        env_chunk = np.stack([xvla_first10_to_env_action(x) for x in action_chunk], axis=0)
        return env_chunk.astype(np.float32), action_chunk.astype(np.float32)
    raise ValueError(f"Unsupported action_key={action_key}")


def rollout_one(
    model: WidowXViTSMLPPolicy,
    payload: dict[str, Any],
    task_key: str,
    grid_episode_id: int,
    replan_steps: int,
    threshold_gripper: bool,
    clip_action: bool,
) -> dict[str, Any]:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    task = TASKS[task_key]
    env = None
    try:
        env = simpler_env.make(task.env_task)
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": int(grid_episode_id)}})
        task_id = payload["task_vocab"].index(task_key)
        raw_proprio = initial_xvla_proprio(obs) if model.use_proprio else None
        proprio = preprocess_proprio_array(raw_proprio, payload)
        pending: list[tuple[np.ndarray, np.ndarray | None]] = []
        reward = 0.0
        success = False
        for step_idx in range(task.official_horizon):
            if not pending:
                image = get_image_from_maniskill2_obs_dict(env, obs)
                pred = model.infer_chunk(image, task_id=task_id, proprio_np=proprio)
                env_chunk, raw_chunk = decode_policy_chunk(pred, payload, threshold_gripper, clip_action)
                if raw_chunk is None:
                    pending = [(x.astype(np.float32), None) for x in env_chunk[:replan_steps]]
                else:
                    pending = [
                        (env_x.astype(np.float32), raw_x.astype(np.float32))
                        for env_x, raw_x in zip(env_chunk[:replan_steps], raw_chunk[:replan_steps], strict=True)
                    ]
            action, raw_action = pending.pop(0)
            obs, reward, done, truncated, _info = env.step(action)
            if raw_proprio is not None:
                if raw_action is not None:
                    raw_proprio = update_xvla_proprio_from_raw_action(raw_proprio, raw_action)
                else:
                    raw_proprio = update_xvla_proprio_from_env_action(raw_proprio, action)
                proprio = preprocess_proprio_array(raw_proprio, payload)
            if done:
                success = True
                return {
                    "task_key": task_key,
                    "grid_episode_id": int(grid_episode_id),
                    "success": True,
                    "steps": step_idx + 1,
                    "reward": float(reward),
                }
            if truncated:
                break
        return {
            "task_key": task_key,
            "grid_episode_id": int(grid_episode_id),
            "success": success,
            "steps": task.official_horizon,
            "reward": float(reward),
        }
    finally:
        if env is not None:
            env.close()


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Evaluation requires CUDA for SimplerEnv rendering/model inference.")
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(args.checkpoint_dir)
    model, payload = load_policy(checkpoint, device)
    cfg = payload["cfg"]
    tasks = parse_tasks(args.tasks if args.tasks else cfg.get("tasks"))
    replan_steps = int(args.replan_steps if args.replan_steps is not None else cfg.get("replan_steps", cfg["chunk_size"]))
    episode_ids = list(range(int(args.episode_start), int(args.episode_end)))

    if args.output_dir:
        eval_dir = Path(args.output_dir)
    else:
        base_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
        eval_dir = base_dir / "eval" / f"step{payload['step']}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_key in tasks:
        for grid_episode_id in episode_ids:
            row = rollout_one(
                model=model,
                payload=payload,
                task_key=task_key,
                grid_episode_id=grid_episode_id,
                replan_steps=replan_steps,
                threshold_gripper=bool(args.threshold_gripper),
                clip_action=bool(args.clip_action),
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = eval_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_key", "grid_episode_id", "success", "steps", "reward"])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": payload["step"],
        "tasks": tasks,
        "episode_ids": episode_ids,
        "replan_steps": replan_steps,
        "threshold_gripper": bool(args.threshold_gripper),
        "clip_action": bool(args.clip_action),
        "overall_success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "num_trials": len(rows),
        "by_task": {},
    }
    for task_key in tasks:
        task_rows = [r for r in rows if r["task_key"] == task_key]
        summary["by_task"][task_key] = {
            "successes": int(sum(bool(r["success"]) for r in task_rows)),
            "trials": len(task_rows),
            "success_rate": float(np.mean([r["success"] for r in task_rows])) if task_rows else 0.0,
        }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WidowX ViT-S + MLP policy on official grid.")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=24)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--threshold-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint and not args.checkpoint_dir:
        raise ValueError("Provide --checkpoint or --checkpoint-dir")
    evaluate(args)


if __name__ == "__main__":
    main()
