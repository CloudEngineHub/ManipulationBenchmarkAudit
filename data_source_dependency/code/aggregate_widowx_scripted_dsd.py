#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ("stack", "carrot", "spoon", "eggplant")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        fail(f"could not read {path.name}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.name}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path.name} must contain a JSON object")
    return payload


def require_int(row: dict[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} has non-integer {key}: {value!r}")
    return value


def require_rate(row: dict[str, Any], key: str, label: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} has non-numeric {key}: {value!r}")
    return float(value)


def check_rate(label: str, successes: int, trials: int, rate: float) -> None:
    if trials <= 0:
        fail(f"{label} has non-positive trials: {trials}")
    expected = successes / trials
    if not math.isclose(rate, expected, rel_tol=0.0, abs_tol=1e-9):
        fail(f"{label} success_rate={rate!r} does not match {successes}/{trials}={expected!r}")


def parse_expected_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(task.strip() for task in raw.split(",") if task.strip())
    if not tasks:
        fail("--expected-tasks parsed to an empty task list")
    if len(set(tasks)) != len(tasks):
        fail(f"--expected-tasks contains duplicates: {tasks}")
    return tasks


def read_task_row(path: Path, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    by_task = payload.get("by_task")
    if by_task is not None:
        if not isinstance(by_task, dict):
            fail(f"{path.name} field by_task must be an object")
        if len(by_task) != 1:
            fail(f"{path.name} must contain exactly one by_task entry, found {len(by_task)}")
        task, row = next(iter(by_task.items()))
        if not isinstance(row, dict):
            fail(f"{path.name}:{task} by_task entry must be an object")
        return task, row

    task = payload.get("task")
    if task is None:
        fail(f"{path.name} must contain either by_task or top-level task metrics")
    return task, payload


def validate_grid_lists(path: Path, task: str, payload: dict[str, Any], successes: int, trials: int) -> None:
    success_grid_ids = payload.get("success_grid_ids")
    failure_grid_ids = payload.get("failure_grid_ids")
    if success_grid_ids is None and failure_grid_ids is None:
        return
    if not isinstance(success_grid_ids, list) or not isinstance(failure_grid_ids, list):
        fail(f"{path.name}:{task} success_grid_ids and failure_grid_ids must both be lists")
    if len(success_grid_ids) != successes:
        fail(f"{path.name}:{task} success_grid_ids length does not match successes={successes}")
    if len(failure_grid_ids) != trials - successes:
        fail(f"{path.name}:{task} failure_grid_ids length does not match failures={trials - successes}")
    combined = success_grid_ids + failure_grid_ids
    if any(isinstance(x, bool) or not isinstance(x, int) for x in combined):
        fail(f"{path.name}:{task} grid id lists must contain only integers")
    if len(set(combined)) != len(combined):
        fail(f"{path.name}:{task} grid id lists contain duplicates")
    episode_ids = payload.get("episode_ids")
    if isinstance(episode_ids, list) and sorted(combined) != sorted(episode_ids):
        fail(f"{path.name}:{task} success/failure grid ids do not cover episode_ids")


def validate_one_file(path: Path, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    payload = load_json(path)
    task, row = read_task_row(path, payload)
    if not isinstance(task, str) or not task:
        fail(f"{path.name} has invalid task key {task!r}")

    tasks_field = payload.get("tasks")
    if tasks_field is not None and tasks_field != [task]:
        fail(f"{path.name} tasks field {tasks_field!r} does not match by_task key {task!r}")

    successes = require_int(row, "successes", f"{path.name}:{task}")
    trials = require_int(row, "trials", f"{path.name}:{task}")
    success_rate = require_rate(row, "success_rate", f"{path.name}:{task}")
    if successes < 0 or successes > trials:
        fail(f"{path.name}:{task} has invalid successes/trials: {successes}/{trials}")
    check_rate(f"{path.name}:{task}", successes, trials, success_rate)

    num_trials = payload.get("num_trials")
    if num_trials is not None and num_trials != trials:
        fail(f"{path.name} num_trials={num_trials!r} does not match by_task trials={trials}")
    overall_rate = payload.get("overall_success_rate")
    if overall_rate is not None:
        check_rate(f"{path.name}:overall", successes, trials, require_rate(payload, "overall_success_rate", path.name))
    validate_grid_lists(path, task, payload, successes, trials)

    if args.trials_per_task is not None and trials != args.trials_per_task:
        fail(f"{path.name}:{task} trials={trials}, expected {args.trials_per_task}")
    if args.expected_replan_steps is not None:
        replan_steps = payload.get("replan_steps")
        if replan_steps != args.expected_replan_steps:
            fail(f"{path.name} replan_steps={replan_steps!r}, expected {args.expected_replan_steps}")
    if args.expected_grid_count is not None:
        episode_ids = payload.get("episode_ids")
        expected_ids = list(range(args.expected_grid_count))
        if episode_ids != expected_ids:
            fail(f"{path.name} episode_ids do not match official grid ids {expected_ids}")

    return task, {
        "source_file": path.name,
        "successes": successes,
        "trials": trials,
        "success_rate": success_rate,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.summary_dir.is_dir():
        fail(f"not a directory: {args.summary_dir}")

    paths = sorted(args.summary_dir.glob("*.json"))
    expected_tasks = parse_expected_tasks(args.expected_tasks)
    if len(paths) != len(expected_tasks):
        fail(f"expected {len(expected_tasks)} JSON summaries, found {len(paths)}")

    per_task: dict[str, dict[str, Any]] = {}
    for path in paths:
        task, row = validate_one_file(path, args)
        if task in per_task:
            fail(f"duplicate task across summaries: {task}")
        per_task[task] = row

    missing = [task for task in expected_tasks if task not in per_task]
    unexpected = sorted(set(per_task) - set(expected_tasks))
    if missing or unexpected:
        fail(f"task mismatch: missing={missing}, unexpected={unexpected}")

    total_successes = sum(row["successes"] for row in per_task.values())
    total_trials = sum(row["trials"] for row in per_task.values())
    if args.expected_successes is not None and total_successes != args.expected_successes:
        fail(f"total_successes={total_successes}, expected {args.expected_successes}")
    if args.expected_trials is not None and total_trials != args.expected_trials:
        fail(f"total_trials={total_trials}, expected {args.expected_trials}")

    return {
        "overall_success_rate": total_successes / total_trials,
        "per_task": {task: per_task[task] for task in expected_tasks},
        "total_successes": total_successes,
        "total_trials": total_trials,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate four WidowX scripted-demo eval summary JSON files."
    )
    parser.add_argument("summary_dir", type=Path)
    parser.add_argument("--expected-tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--trials-per-task", type=int, default=24)
    parser.add_argument("--expected-grid-count", type=int, default=24)
    parser.add_argument("--expected-replan-steps", type=int, default=5)
    parser.add_argument("--expected-successes", type=int)
    parser.add_argument("--expected-trials", type=int)
    return parser.parse_args()


def main() -> None:
    result = aggregate(parse_args())
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
