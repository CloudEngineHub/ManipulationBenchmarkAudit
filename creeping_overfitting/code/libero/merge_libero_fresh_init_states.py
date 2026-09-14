#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from libero.libero import benchmark


SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--part-root", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--base-config-dir", required=True)
    parser.add_argument("--num-states-per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_names_for_suite(suite_name: str) -> list[str]:
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    return [task_suite.get_task(task_id).name for task_id in range(task_suite.n_tasks)]


def copy_suite_from_part(final_root: Path, part_root: Path, suite_name: str) -> None:
    final_suite = final_root / suite_name
    if final_suite.exists():
        return
    part_suite = part_root / suite_name / suite_name
    if not part_suite.is_dir():
        raise RuntimeError(f"Missing part suite directory: {part_suite}")
    shutil.copytree(part_suite, final_suite)


def validate_suite(final_root: Path, suite_name: str, num_states: int) -> dict:
    suite_dir = final_root / suite_name
    if not suite_dir.is_dir():
        raise RuntimeError(f"Missing suite directory: {suite_dir}")
    expected_names = task_names_for_suite(suite_name)
    files = sorted(suite_dir.glob("*.pruned_init"))
    if [path.stem for path in files] != sorted(expected_names):
        raise RuntimeError(
            f"Task files for {suite_name} do not match benchmark task names: "
            f"files={[path.stem for path in files]}, expected={sorted(expected_names)}"
        )

    records = []
    for task_id, task_name in enumerate(expected_names):
        path = suite_dir / f"{task_name}.pruned_init"
        states = torch.load(path, map_location="cpu")
        shape = list(states.shape)
        dtype = str(states.dtype)
        if len(shape) != 2 or shape[0] != num_states or shape[1] <= 0:
            raise RuntimeError(f"Unexpected state shape for {path}: {shape}")
        if dtype != "float64":
            raise RuntimeError(f"Unexpected dtype for {path}: {dtype}")
        records.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "file": str(path.relative_to(final_root)),
                "shape": shape,
                "dtype": dtype,
                "sha256": sha256_file(path),
            }
        )
    return {
        "suite": suite_name,
        "num_tasks": len(records),
        "states_per_task": num_states,
        "total_states": len(records) * num_states,
        "tasks": records,
    }


def install_rollout_copy(final_root: Path, rollout_root: Path) -> None:
    if rollout_root.exists():
        children = list(rollout_root.iterdir())
        if children:
            raise RuntimeError(f"Refusing to overwrite non-empty rollout root: {rollout_root}")
    rollout_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(final_root, rollout_root, dirs_exist_ok=True)


def write_config(base_config_dir: Path, config_dir: Path, rollout_root: Path) -> None:
    base = base_config_dir / "config.yaml"
    dst = config_dir / "config.yaml"
    if not base.exists():
        raise RuntimeError(f"Missing base config: {base}")
    config_dir.mkdir(parents=True, exist_ok=True)
    out = []
    replaced = False
    for line in base.read_text().splitlines():
        if line.startswith("init_states:"):
            out.append(f"init_states: {rollout_root}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError(f"Base config had no init_states line: {base}")
    dst.write_text("\n".join(out) + "\n")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    final_root = Path(args.final_root).expanduser().resolve()
    part_root = Path(args.part_root).expanduser().resolve()
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    base_config_dir = Path(args.base_config_dir).expanduser().resolve()

    final_root.mkdir(parents=True, exist_ok=True)
    for suite_name in SUITE_ORDER:
        copy_suite_from_part(final_root, part_root, suite_name)

    suite_records = [validate_suite(final_root, suite, args.num_states_per_task) for suite in SUITE_ORDER]
    total_files = sum(record["num_tasks"] for record in suite_records)
    total_states = sum(record["total_states"] for record in suite_records)
    if total_files != 40:
        raise RuntimeError(f"Expected 40 task files, got {total_files}")
    if total_states != 10000:
        raise RuntimeError(f"Expected 10000 states, got {total_states}")

    manifest = {
        "artifact_type": "libero_iid_fresh_init_states",
        "out_root": str(final_root),
        "suites": list(SUITE_ORDER),
        "num_states_per_task": args.num_states_per_task,
        "base_seed": args.seed,
        "suite_seed_rule": "suite_seed = base_seed + suite_index * 100000",
        "generation_layout": "suite-parallel generation runs; serial generation within each suite",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_context": "sanitized anonymous-review reference implementation",
        "suite_records": suite_records,
        "total_task_files": total_files,
        "total_states": total_states,
        "elapsed_s": round(time.time() - start_time, 3),
        "validation_status": "passed",
    }
    (final_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    install_rollout_copy(final_root, rollout_root)
    write_config(base_config_dir, config_dir, rollout_root)
    rollout_manifest = rollout_root / "MANIFEST.json"
    if json.loads(rollout_manifest.read_text())["total_states"] != 10000:
        raise RuntimeError(f"Rollout manifest failed total_states check: {rollout_manifest}")
    print(json.dumps({"manifest": str(final_root / "MANIFEST.json"), "rollout_root": str(rollout_root)}, indent=2))


if __name__ == "__main__":
    main()
