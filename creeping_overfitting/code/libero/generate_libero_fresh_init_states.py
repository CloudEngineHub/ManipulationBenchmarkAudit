#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--num-states-per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--allow-empty-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_env(task, env_img_res: int) -> OffScreenRenderEnv:
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=env_img_res,
        camera_widths=env_img_res,
    )


def assert_clean_out_root(out_root: Path, allow_empty_existing: bool) -> None:
    if not out_root.exists():
        out_root.mkdir(parents=True)
        return
    children = list(out_root.iterdir())
    if children:
        raise RuntimeError(f"Refusing to write into non-empty output root: {out_root}")
    if not allow_empty_existing:
        raise RuntimeError(
            f"Output root already exists, even though it is empty: {out_root}. "
            "Pass --allow-empty-existing to acknowledge this."
        )


def generate_suite(suite_name: str, suite_seed: int, out_root: Path, num_states: int, env_img_res: int) -> dict:
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise KeyError(f"Unknown LIBERO suite {suite_name}; available keys include {sorted(benchmark_dict)[:20]}")

    set_seed(suite_seed)
    task_suite = benchmark_dict[suite_name]()
    suite_dir = out_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=False)

    task_records = []
    progress = tqdm(total=task_suite.n_tasks * num_states, desc=suite_name, unit="state")
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        out_file = suite_dir / f"{task.name}.pruned_init"
        if out_file.exists():
            raise RuntimeError(f"Refusing to overwrite existing task file: {out_file}")

        states = []
        env = make_env(task, env_img_res)
        try:
            for _ in range(num_states):
                env.reset()
                reset_attempts = 1
                while env.check_success():
                    if reset_attempts >= 100:
                        raise RuntimeError(f"Task {task.name} stayed successful after 100 reset attempts")
                    env.reset()
                    reset_attempts += 1
                states.append(env.get_sim_state())
                progress.update(1)
        finally:
            env.close()

        states_np = np.stack(states, axis=0).astype(np.float64)
        if states_np.ndim != 2 or states_np.shape[0] != num_states:
            raise RuntimeError(f"Unexpected state shape for {task.name}: {states_np.shape}")
        if states_np.shape[1] <= 0:
            raise RuntimeError(f"Unexpected empty state width for {task.name}: {states_np.shape}")
        if states_np.dtype != np.float64:
            raise RuntimeError(f"Unexpected state dtype for {task.name}: {states_np.dtype}")
        torch.save(states_np, out_file)
        task_records.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "file": str(out_file.relative_to(out_root)),
                "shape": list(states_np.shape),
                "dtype": str(states_np.dtype),
                "sha256": sha256_file(out_file),
            }
        )
    progress.close()

    if len(task_records) != 10:
        raise RuntimeError(f"Expected 10 tasks for {suite_name}, got {len(task_records)}")

    return {
        "suite": suite_name,
        "suite_seed": suite_seed,
        "num_tasks": len(task_records),
        "states_per_task": num_states,
        "total_states": len(task_records) * num_states,
        "tasks": task_records,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    assert_clean_out_root(out_root, args.allow_empty_existing)

    start_time = time.time()
    manifest = {
        "artifact_type": "libero_iid_fresh_init_states",
        "out_root": str(out_root),
        "suites": args.suites,
        "num_states_per_task": args.num_states_per_task,
        "base_seed": args.seed,
        "env_img_res": args.env_img_res,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_context": "sanitized anonymous-review reference implementation",
        "suite_records": [],
    }

    for suite_idx, suite_name in enumerate(args.suites):
        suite_seed = args.seed + suite_idx * 100000
        manifest["suite_records"].append(
            generate_suite(
                suite_name=suite_name,
                suite_seed=suite_seed,
                out_root=out_root,
                num_states=args.num_states_per_task,
                env_img_res=args.env_img_res,
            )
        )

    total_files = sum(record["num_tasks"] for record in manifest["suite_records"])
    total_states = sum(record["total_states"] for record in manifest["suite_records"])
    expected_files = len(args.suites) * 10
    expected_states = expected_files * args.num_states_per_task
    if total_files != expected_files:
        raise RuntimeError(f"Expected {expected_files} task files, got {total_files}")
    if total_states != expected_states:
        raise RuntimeError(f"Expected {expected_states} states, got {total_states}")

    manifest["total_task_files"] = total_files
    manifest["total_states"] = total_states
    manifest["elapsed_s"] = round(time.time() - start_time, 3)
    manifest["validation_status"] = "passed"
    manifest_path = out_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "total_states": total_states}, indent=2))


if __name__ == "__main__":
    main()
