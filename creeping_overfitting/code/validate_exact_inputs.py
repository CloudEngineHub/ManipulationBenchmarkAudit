#!/usr/bin/env python3
"""Validate the released exact-input artifacts for creeping-overfitting checks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CREEPING = REPO / "creeping_overfitting"
STAT_SHARED = REPO / "statistical_significance" / "libero_goal_5x5k" / "shared"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_calvin_manifests() -> None:
    manifest_dir = CREEPING / "results/calvin/manifests"
    official = load_json(manifest_dir / "abc_d_official_1000seq_workers4.json")
    official_rows = official["sequences"]
    require(official["num_sequences"] == 1000, "CALVIN official manifest num_sequences mismatch")
    require(len(official_rows) == 1000, "CALVIN official manifest row count mismatch")
    require([r["index"] for r in official_rows] == list(range(1000)), "CALVIN official indices are not 0..999")
    require(all(len(r["eval_sequence"]) == 5 for r in official_rows), "CALVIN official chain length mismatch")

    expected_order = ["led", "lightbulb", "slider", "drawer", "red_block", "blue_block", "pink_block", "grasped"]
    signatures = []
    for name in [
        "abc_d_fresh_sequence_1000seq_seed2026052601.json",
        "abc_d_fresh_sequence_1000seq_seed2026052602.json",
    ]:
        data = load_json(manifest_dir / name)
        rows = data["sequences"]
        require(data["num_sequences"] == 1000, f"{name} num_sequences mismatch")
        require(data.get("calvin_reset_bank_used") is False, f"{name} unexpectedly records reset-bank use")
        require(len(rows) == 1000, f"{name} row count mismatch")
        require([r["global_index"] for r in rows] == list(range(1000)), f"{name} global indices are not 0..999")
        require(all(len(r["eval_sequence"]) == 5 for r in rows), f"{name} chain length mismatch")
        require(all([k for k, _ in r["initial_state_items"]] == expected_order for r in rows), f"{name} initial-state key order changed")

        by_state: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_state[int(row["source_state_index"])].append(row)
        require(len(by_state) == 192, f"{name} does not cover 192 symbolic states")
        quota_counter = Counter()
        for state_rows in by_state.values():
            quotas = {int(r["state_quota"]) for r in state_rows}
            require(len(quotas) == 1, f"{name} has inconsistent quota for a symbolic state")
            quota = quotas.pop()
            require(len(state_rows) == quota, f"{name} state row count does not match state_quota")
            quota_counter[quota] += 1
        require(dict(quota_counter) == {5: 152, 6: 40}, f"{name} quota structure changed: {dict(quota_counter)}")
        signatures.append({(r["initial_state_ordered_json"], r["eval_sequence_json"]) for r in rows})
    official_signatures = {(json.dumps(list(r["initial_state"].items()), separators=(",", ":")), r["eval_sequence_json"]) for r in official_rows}
    require(signatures[0] != signatures[1], "fresh CALVIN manifests have identical content")
    require(signatures[0] != official_signatures and signatures[1] != official_signatures, "fresh CALVIN manifest matches official content")


def check_calvin_reset_bank() -> None:
    import numpy as np

    path = CREEPING / "results/calvin/reset_banks/abc_d_official_d_table_resets_1000seq_seed0.npz"
    bank = np.load(path, allow_pickle=False)
    required = {
        "robot_obs": (1000, 15),
        "scene_obs": (1000, 24),
        "official_robot_obs": (1000, 15),
        "official_scene_obs": (1000, 24),
        "table_signature": (1000,),
        "sample_attempts": (1000,),
        "initial_state_json": (1000,),
        "eval_sequence_json": (1000,),
    }
    for key, shape in required.items():
        require(key in bank.files, f"CALVIN reset bank missing {key}")
        require(tuple(bank[key].shape) == shape, f"CALVIN reset bank {key} shape mismatch: {bank[key].shape}")
    metadata = json.loads(str(bank["metadata_json"].item()))
    require(metadata["protocol"] == "calvin_official_d_table_resets_v1", "CALVIN reset bank protocol mismatch")
    require(metadata["num_sequences"] == 1000 and metadata["seed"] == 0, "CALVIN reset bank metadata mismatch")

    official = load_json(CREEPING / "results/calvin/manifests/abc_d_official_1000seq_workers4.json")
    rows = official["sequences"]
    for index, row in enumerate(rows):
        require(row["index"] == index, f"CALVIN official manifest index mismatch at row {index}")
        require(row["initial_state_json"] == str(bank["initial_state_json"][index]), f"CALVIN reset-bank initial_state mismatch at row {index}")
        require(row["eval_sequence_json"] == str(bank["eval_sequence_json"][index]), f"CALVIN reset-bank eval_sequence mismatch at row {index}")


def load_torch_shape(path: Path) -> tuple[tuple[int, ...], str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to validate LIBERO .pruned_init payload contents; "
            "use a CPU torch environment or set PYTHONPATH to the task scratch deps"
        ) from exc

    # These release artifacts are trusted local inputs copied from the project scratch.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    require(shape is not None, f"{path} did not load as an ndarray-like init-state payload")
    return tuple(int(v) for v in shape), str(dtype)


def check_libero_init_tree(root: Path, expected_suites: dict[str, int], expected_states_per_task: int) -> None:
    require((root / "MANIFEST.json").is_file(), f"{root} missing MANIFEST.json")
    manifest = load_json(root / "MANIFEST.json")
    files = list(root.glob("*/*.pruned_init"))
    require(len(files) == sum(expected_suites.values()), f"{root} .pruned_init file count mismatch")
    for suite, count in expected_suites.items():
        suite_files = list((root / suite).glob("*.pruned_init"))
        require(len(suite_files) == count, f"{root}/{suite} file count mismatch")

    if "tasks" in manifest:
        require(manifest["total_episodes"] == expected_states_per_task * sum(expected_suites.values()), f"{root} manifest total mismatch")
        require(all(task["states"] == expected_states_per_task for task in manifest["tasks"].values()), f"{root} manifest per-task state count mismatch")
        for task_name, record in manifest["tasks"].items():
            path = root / manifest["task_directory"] / f"{task_name}.pruned_init"
            shape, _dtype = load_torch_shape(path)
            require(shape[0] == record["states"], f"{path} loaded state count mismatch: {shape}")
    else:
        require(manifest["total_task_files"] == sum(expected_suites.values()), f"{root} manifest task-file count mismatch")
        require(manifest["total_states"] == expected_states_per_task * sum(expected_suites.values()), f"{root} manifest total state count mismatch")
        require(manifest["validation_status"] == "passed", f"{root} manifest validation did not pass")
        for suite_record in manifest["suite_records"]:
            require(suite_record["states_per_task"] == expected_states_per_task, f"{root} manifest suite state count mismatch")
            for task in suite_record["tasks"]:
                path = root / task["file"]
                shape, dtype = load_torch_shape(path)
                require(shape == tuple(task["shape"]), f"{path} loaded shape mismatch: {shape} vs {task['shape']}")
                require(dtype == task["dtype"], f"{path} loaded dtype mismatch: {dtype} vs {task['dtype']}")


def check_libero_outcomes() -> None:
    expected = {
        "fresh_init_state/selected_episode_outcomes.csv": {
            "counts": {"spatial_forcing": 10000, "simvla": 10000, "pi05_lerobot": 10000},
            "successes": {"spatial_forcing": 9718, "simvla": 9760, "pi05_lerobot": 9741},
        },
        "official_calibration/selected_episode_outcomes.csv": {
            "counts": {"spatial_forcing": 2000, "simvla": 2000, "pi05_lerobot": 2000},
            "successes": {"spatial_forcing": 1956, "simvla": 1946, "pi05_lerobot": 1951},
        },
    }
    for rel, exp in expected.items():
        path = CREEPING / "results/libero" / rel
        counts: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        per_task_counts: Counter[tuple[str, str, int]] = Counter()
        protocol_keys: Counter[tuple[str, str, str, int, int]] = Counter()
        row_kind = rel.split("/", 1)[0]
        if row_kind == "fresh_init_state":
            expected_per_task = 250
        else:
            expected_per_task = 50
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                task_id = int(row["task_id"])
                episode_idx = int(row["episode_idx"])
                expected_instance_id = f"{row['suite']}/{row['task_name']}/{episode_idx:04d}"
                require(row["row_kind"] == row_kind, f"{path} row_kind mismatch")
                require(row["instance_id"] == expected_instance_id, f"{path} instance_id mismatch")
                counts[row["policy"]] += 1
                successes[row["policy"]] += int(row["success"])
                per_task_counts[(row["policy"], row["suite"], task_id)] += 1
                protocol_keys[(row["row_kind"], row["policy"], row["suite"], task_id, episode_idx)] += 1
                require(row["source_run_group"], f"{path} row missing source_run_group")
        require(dict(counts) == exp["counts"], f"{path} policy row counts mismatch: {dict(counts)}")
        require(dict(successes) == exp["successes"], f"{path} policy successes mismatch: {dict(successes)}")
        require(all(v == expected_per_task for v in per_task_counts.values()), f"{path} per-task row count mismatch")
        duplicate_keys = [key for key, value in protocol_keys.items() if value != 1]
        require(not duplicate_keys, f"{path} duplicate protocol instance keys: {duplicate_keys[:3]}")
    manifest = load_json(CREEPING / "results/libero/selected_episode_outcomes_manifest.json")
    replacements = manifest["fresh"]["retry_replacements"]
    require(len(replacements) == 2, "LIBERO fresh retry replacement record count mismatch")
    require({(r["suite"], r["task_id"]) for r in replacements} == {("libero_spatial", 1), ("libero_goal", 5)}, "LIBERO retry replacement identities changed")
    replacement_keys = {(r["policy"], r["suite"], int(r["task_id"]), r["replacement_run_group"]) for r in replacements}
    replacement_counts: Counter[tuple[str, str, int, str]] = Counter()
    with (CREEPING / "results/libero/fresh_init_state/selected_episode_outcomes.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["selection_note"] == "retry_replacement":
                replacement_counts[(row["policy"], row["suite"], int(row["task_id"]), row["source_run_group"])] += 1
    require(set(replacement_counts) == replacement_keys, "LIBERO retry replacement source keys changed")
    require(all(v == 250 for v in replacement_counts.values()), "LIBERO retry replacement row counts changed")


def check_simplerenv_inputs() -> None:
    config = CREEPING / "configs/simplerenv_protocol_abcde_stack_v1.json"
    expected_hash = (CREEPING / "configs/simplerenv_protocol_abcde_stack_v1.json.sha256").read_text().split()[0]
    require(sha256(config) == expected_hash, "SimplerEnv Protocol A-E config SHA256 mismatch")
    data = load_json(config)
    require(data["episodes_per_condition"] == 288, "SimplerEnv episodes_per_condition mismatch")
    require(set(data["conditions"]) == {
        "protocol_A",
        "protocol_B",
        "protocol_C1_yellow_on_green",
        "protocol_C2_blue_on_red",
        "protocol_C3_red_on_blue",
        "protocol_D",
        "protocol_E",
    }, "SimplerEnv condition set mismatch")
    for name, condition in data["conditions"].items():
        require(len(condition["episodes"]) == 288, f"{name} episode count mismatch")
    d_episodes = data["conditions"]["protocol_D"]["episodes"]
    require(all(len(set(ep["source_support_colors"])) == len(ep["source_support_colors"]) for ep in d_episodes), "Protocol D repeats a source tower support color")
    require(all(len(set(ep["target_support_colors"])) == len(ep["target_support_colors"]) for ep in d_episodes), "Protocol D repeats a target tower support color")
    require(not any("gray" in ep["source_support_colors"] + ep["target_support_colors"] for ep in d_episodes), "Protocol D includes gray support blocks")
    e_pairs = {(ep["source"]["color"], ep["target"]["color"]) for ep in data["conditions"]["protocol_E"]["episodes"]}
    require(len(e_pairs) == 20, "Protocol E does not cover all 20 ordered color pairs")
    require(all(src != dst for src, dst in e_pairs), "Protocol E includes a same-color pair")

    assets = CREEPING / "artifacts/simplerenv/protocol_abcde/assets/custom/models"
    for model in [
        "render_candidate_blue_hybrid_v4",
        "render_candidate_red_corrected_v6e",
        "render_candidate_white_offwhite_hybrid_v4",
    ]:
        require((assets / model / "collision.obj").is_file(), f"missing collision.obj for {model}")
        require((assets / model / "textured.dae").is_file(), f"missing textured.dae for {model}")
    for rel in [
        "code/simplerenv/widowx_protocol1.py",
        "code/simplerenv/protocol_abcde_common.py",
        "code/simplerenv/stage_protocol_assets.py",
    ]:
        require((CREEPING / rel).is_file(), f"missing SimplerEnv support code {rel}")


def main() -> None:
    check_calvin_manifests()
    check_calvin_reset_bank()
    check_libero_init_tree(STAT_SHARED / "init_state_goal_5000", {"libero_goal": 10}, 500)
    check_libero_init_tree(
        CREEPING / "artifacts/libero/init_state_layer2_4suite_2500_seed20260519",
        {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10},
        250,
    )
    check_libero_outcomes()
    check_simplerenv_inputs()
    print("exact input validation passed")


if __name__ == "__main__":
    main()
