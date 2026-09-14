from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_source_dependency.widowx.tasks import TASKS, parse_tasks


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    value = data[key]
    return value.item() if value.shape == () else value


def build_replay_index(dataset_root: str | Path, tasks: list[str]) -> dict[tuple[str, int], Path]:
    root = Path(dataset_root)
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for task_key in tasks:
        for path in sorted((root / task_key / "episodes").glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                if not bool(_scalar(data, "success")):
                    continue
                key = (str(_scalar(data, "task_key")), int(_scalar(data, "grid_episode_id")))
                grouped[key].append(path)
    return {key: paths[0] for key, paths in grouped.items()}


@dataclass(frozen=True)
class ReplayCase:
    task_key: str
    grid_episode_id: int
    episode_path: Path | None
    case_index: int


def _episode_cases(dataset_root: str | Path, tasks: list[str]) -> list[ReplayCase]:
    root = Path(dataset_root)
    cases: list[ReplayCase] = []
    for task_key in tasks:
        for path in sorted((root / task_key / "episodes").glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                if not bool(_scalar(data, "success")):
                    continue
                cases.append(
                    ReplayCase(
                        task_key=str(_scalar(data, "task_key")),
                        grid_episode_id=int(_scalar(data, "grid_episode_id")),
                        episode_path=path,
                        case_index=len(cases),
                    )
                )
    if not cases:
        raise FileNotFoundError(f"No successful .npz demos found under {root}")
    return cases


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
    raise RuntimeError("Saved perturbed replay requires an env with _xy_configs and _quat_configs")


def _reset_options_from_saved_metadata(env: Any, data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "reset_metadata_json" not in data.files:
        raise ValueError("Episode is missing reset_metadata_json; cannot use saved reset metadata")
    metadata = json.loads(str(_scalar(data, "reset_metadata_json")))
    reset_options = metadata.get("reset_options")
    if not isinstance(reset_options, dict):
        raise ValueError("reset_metadata_json is missing dict reset_options")
    perturb = metadata.get("perturb_metadata", metadata)
    if isinstance(perturb, dict) and "object_reset_xys" in perturb:
        if "object_reset_quats" not in perturb:
            raise ValueError("Saved object_reset_xys exists without object_reset_quats")
        config_env = _object_config_env(env)
        config_env._xy_configs = [np.asarray(perturb["object_reset_xys"], dtype=np.float32)]  # noqa: SLF001
        config_env._quat_configs = [np.asarray(perturb["object_reset_quats"], dtype=np.float32)]  # noqa: SLF001
    return reset_options


def rollout_replay(
    task_key: str,
    grid_episode_id: int,
    episode_path: Path | None,
    case_index: int,
    use_saved_reset_metadata: bool,
) -> dict[str, Any]:
    import simpler_env

    task = TASKS[task_key]
    if episode_path is None:
        return {
            "case_index": int(case_index),
            "task_key": task_key,
            "grid_episode_id": int(grid_episode_id),
            "has_demo": False,
            "success": False,
            "steps": 0,
            "reward": 0.0,
            "episode_path": "",
        }

    with np.load(episode_path, allow_pickle=True) as data:
        actions = np.asarray(data["actions_env"], dtype=np.float32)

    env = None
    try:
        env = simpler_env.make(task.env_task)
        with np.load(episode_path, allow_pickle=True) as data:
            if use_saved_reset_metadata:
                reset_options = _reset_options_from_saved_metadata(env, data)
            else:
                reset_options = {"obj_init_options": {"episode_id": int(grid_episode_id)}}
        obs, _ = env.reset(options=reset_options)
        del obs
        reward = 0.0
        for step_idx, action in enumerate(actions[: task.official_horizon]):
            _obs, reward, done, truncated, _info = env.step(action)
            if done:
                return {
                    "case_index": int(case_index),
                    "task_key": task_key,
                    "grid_episode_id": int(grid_episode_id),
                    "has_demo": True,
                    "success": True,
                    "steps": step_idx + 1,
                    "reward": float(reward),
                    "episode_path": str(episode_path),
                }
            if truncated:
                break
        return {
            "case_index": int(case_index),
            "task_key": task_key,
            "grid_episode_id": int(grid_episode_id),
            "has_demo": True,
            "success": False,
            "steps": min(len(actions), task.official_horizon),
            "reward": float(reward),
            "episode_path": str(episode_path),
        }
    finally:
        if env is not None:
            env.close()


def evaluate(args: argparse.Namespace) -> None:
    tasks = parse_tasks(args.tasks)
    episode_ids = list(range(int(args.episode_start), int(args.episode_end)))
    if args.all_demos:
        cases = _episode_cases(args.dataset_root, tasks)
    else:
        replay_index = build_replay_index(args.dataset_root, tasks)
        cases = []
        for task_key in tasks:
            for grid_episode_id in episode_ids:
                cases.append(
                    ReplayCase(
                        task_key=task_key,
                        grid_episode_id=grid_episode_id,
                        episode_path=replay_index.get((task_key, grid_episode_id)),
                        case_index=len(cases),
                    )
                )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in cases:
        row = rollout_replay(
            task_key=case.task_key,
            grid_episode_id=case.grid_episode_id,
            episode_path=case.episode_path,
            case_index=case.case_index,
            use_saved_reset_metadata=bool(args.use_saved_reset_metadata),
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_index",
                "task_key",
                "grid_episode_id",
                "has_demo",
                "success",
                "steps",
                "reward",
                "episode_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "episode_ids": episode_ids,
        "tasks": tasks,
        "all_demos": bool(args.all_demos),
        "use_saved_reset_metadata": bool(args.use_saved_reset_metadata),
        "num_trials": len(rows),
        "demo_coverage": int(sum(bool(r["has_demo"]) for r in rows)),
        "successes": int(sum(bool(r["success"]) for r in rows)),
        "overall_success_rate": float(np.mean([bool(r["success"]) for r in rows])) if rows else 0.0,
        "by_task": {},
    }
    for task_key in tasks:
        task_rows = [r for r in rows if r["task_key"] == task_key]
        summary["by_task"][task_key] = {
            "demo_coverage": int(sum(bool(r["has_demo"]) for r in task_rows)),
            "successes": int(sum(bool(r["success"]) for r in task_rows)),
            "trials": len(task_rows),
            "success_rate": float(np.mean([bool(r["success"]) for r in task_rows])) if task_rows else 0.0,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved successful WidowX demo actions on the official grid.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
        help="Root containing <task>/episodes/*.npz successful demos.",
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--tasks", default="stack,carrot,spoon,eggplant")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=24)
    parser.add_argument("--all-demos", action="store_true")
    parser.add_argument("--use-saved-reset-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
