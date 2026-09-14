#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/share/data/ripl/tianchong/projects/Eval_Policies_CoRL"))
PROTOCOL_NAME = "simplerenv_protocol_abcde_stack_v1"
EXPECTED_PROTOCOL_SHA256 = "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
CONDITIONS = (
    "protocol_A",
    "protocol_B",
    "protocol_C1_yellow_on_green",
    "protocol_C2_blue_on_red",
    "protocol_C3_red_on_blue",
    "protocol_D",
    "protocol_E",
)
POLICIES = ("cogact", "spatialvla", "internvla_m1", "xvla", "dexbotic")
EPISODE_IDS = tuple(range(288))
HORIZON = 60
BASE_SEED = 202605250

CSV_COLUMNS = [
    "policy",
    "condition",
    "episode_id",
    "seed",
    "success",
    "steps",
    "error",
    "timeout",
    "horizon",
    "job_id",
    "video_path",
    "protocol_name",
    "protocol_sha256",
    "instruction",
    "source_model_id",
    "target_model_id",
    "source_color",
    "target_color",
    "run_id",
]


def protocol_config_path() -> Path:
    return Path(
        os.environ.get(
            "SIMPLERENV_PROTOCOL_CONFIG",
            PROJECT_ROOT / "configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json",
        )
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_protocol_config() -> dict[str, Any]:
    path = protocol_config_path()
    if not path.is_file():
        raise RuntimeError(f"missing Protocol A-E config: {path}")
    actual = sha256_file(path)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"Protocol A-E config SHA mismatch: {actual} != {EXPECTED_PROTOCOL_SHA256}")
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing Protocol A-E SHA sidecar: {sidecar}")
    declared = sidecar.read_text().split()[0]
    if declared != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"Protocol A-E sidecar SHA mismatch: {declared} != {EXPECTED_PROTOCOL_SHA256}")
    config = json.loads(path.read_text())
    if config.get("name") != PROTOCOL_NAME:
        raise RuntimeError(f"unexpected protocol name: {config.get('name')!r}")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise RuntimeError(f"unexpected Protocol A-E conditions: {sorted((conditions or {}).keys())}")
    for condition in CONDITIONS:
        episodes = conditions[condition].get("episodes")
        if not isinstance(episodes, list) or len(episodes) != len(EPISODE_IDS):
            raise RuntimeError(f"{condition} must have exactly 288 episodes")
        for idx, episode in enumerate(episodes):
            if episode.get("episode_id", idx) != idx:
                raise RuntimeError(f"{condition} episode index {idx} records episode_id={episode.get('episode_id')!r}")
    return config


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy {policy!r}; expected {list(POLICIES)}")
    return policy


def validate_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition {condition!r}; expected {list(CONDITIONS)}")
    return condition


def seed_for(policy: str, condition: str, episode_id: int) -> int:
    validate_policy(policy)
    validate_condition(condition)
    if episode_id not in EPISODE_IDS:
        raise ValueError(f"episode_id must be 0..287, got {episode_id}")
    return BASE_SEED + POLICIES.index(policy) * 1_000_000 + CONDITIONS.index(condition) * 10_000 + episode_id


def server_seed_for(policy: str, condition: str) -> int:
    validate_policy(policy)
    validate_condition(condition)
    return BASE_SEED + POLICIES.index(policy) * 1_000_000 + CONDITIONS.index(condition) * 10_000 + 9_000


def episode_metadata(config: dict[str, Any], condition: str, episode_id: int) -> dict[str, str]:
    episode = config["conditions"][condition]["episodes"][episode_id]
    model_ids = episode.get("model_ids") or []
    source = episode.get("source") or {}
    target = episode.get("target") or {}
    source_id = int(episode.get("source_obj_id"))
    target_id = int(episode.get("target_obj_id"))
    return {
        "instruction": str(episode.get("instruction", "")),
        "source_model_id": str(source.get("model_id") or model_ids[source_id]),
        "target_model_id": str(target.get("model_id") or model_ids[target_id]),
        "source_color": str(source.get("color", "")),
        "target_color": str(target.get("color", "")),
    }


def manifest_rows(policy: str, condition: str, run_id: str) -> list[dict[str, Any]]:
    config = load_protocol_config()
    rows = []
    for episode_id in EPISODE_IDS:
        row = {
            "policy": policy,
            "condition": condition,
            "episode_id": episode_id,
            "seed": seed_for(policy, condition, episode_id),
            "horizon": HORIZON,
            "protocol_name": PROTOCOL_NAME,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "run_id": run_id,
        }
        row.update(episode_metadata(config, condition, episode_id))
        rows.append(row)
    return rows


def write_manifest(path: Path, policy: str, condition: str, run_id: str) -> None:
    rows = manifest_rows(policy, condition, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def init_results_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != CSV_COLUMNS:
            raise RuntimeError(f"results CSV header mismatch: {header}")
        return
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)
        f.flush()
        os.fsync(f.fileno())


def append_result(path: Path, row: dict[str, Any]) -> None:
    init_results_csv(path)
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow({key: row.get(key, "") for key in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def count_video_steps(video_path: Path, *, subtract_initial_frame: bool) -> int | None:
    if not video_path.is_file():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        frames = int(proc.stdout.strip().splitlines()[0])
    except Exception:
        return None
    steps = frames - 1 if subtract_initial_frame else frames
    return max(steps, 0)


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except Exception:
        return ""


def stat_digest(path: Path) -> str:
    try:
        st = path.stat()
    except Exception:
        return ""
    return sha256_text(f"{path.resolve()}\t{st.st_size}\t{st.st_mtime_ns}")


def runtime_asset_evidence(policy: str) -> dict[str, Any]:
    manifest_text = os.environ.get("ASSET_MANIFEST", "")
    evidence: dict[str, Any] = {"manifest_path": manifest_text}
    if not manifest_text:
        return evidence
    manifest = Path(manifest_text)
    if manifest.is_file():
        evidence["manifest_sha256"] = sha256_file(manifest)
        try:
            payload = json.loads(manifest.read_text())
        except Exception as exc:
            evidence["manifest_parse_error"] = str(exc)
            return evidence
        evidence["accepted_asset_ids"] = payload.get("accepted_asset_ids") or payload.get("asset_ids") or []
        evidence["runtime_paths"] = [
            entry for entry in payload.get("runtime_paths", []) if entry.get("policy") == policy
        ]
    return evidence


def write_runtime_metadata(
    path: Path,
    *,
    policy: str,
    condition: str,
    run_id: str,
    checkpoint_identity: str,
    checkpoint_path: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    config_path = protocol_config_path()
    support_root_text = os.environ.get("SUPPORT_ROOT", "")
    run_root_text = os.environ.get("RUN_ROOT", "")
    support_hash_file = Path(run_root_text) / "bundle/submission_support_files.sha256" if run_root_text else Path("")
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    metadata: dict[str, Any] = {
        "policy": policy,
        "condition": condition,
        "run_id": run_id,
        "horizon": HORIZON,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "simplerenv_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_config_path": str(config_path),
        "config_snapshot_sha256": sha256_file(config_path) if config_path.is_file() else "",
        "project_repo_commit": git_head(PROJECT_ROOT),
        "support_root": support_root_text,
        "launch_snapshot_path": support_root_text,
        "code_snapshot_sha256": sha256_file(support_hash_file) if support_hash_file.is_file() else "",
        "checkpoint_model_id": checkpoint_identity,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else "",
        "checkpoint_identity_sha256": sha256_text(checkpoint_identity) if checkpoint_identity else "",
        "checkpoint_stat_sha256": stat_digest(checkpoint) if checkpoint is not None and checkpoint.exists() else "",
        "checkpoint_hash_caveat": (
            "checkpoint_identity_sha256 is a digest of the model/checkpoint identity string; "
            "checkpoint_stat_sha256 is present only for local checkpoint files."
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "runtime_env": {
            "node": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
            "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "runtime_assets": runtime_asset_evidence(policy),
    }
    if extra:
        metadata.update(extra)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def base_row(policy: str, condition: str, episode_id: int, run_id: str) -> dict[str, Any]:
    metadata = episode_metadata(load_protocol_config(), condition, episode_id)
    row: dict[str, Any] = {
        "policy": policy,
        "condition": condition,
        "episode_id": episode_id,
        "seed": seed_for(policy, condition, episode_id),
        "success": "",
        "steps": "",
        "error": "",
        "timeout": "0",
        "horizon": HORIZON,
        "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
        "video_path": "",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "run_id": run_id,
    }
    row.update(metadata)
    return row


def validate_results_csv(path: Path, policy: str, condition: str, report_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    if not path.is_file():
        errors.append(f"missing results CSV: {path}")
    else:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_COLUMNS:
                errors.append(f"header mismatch: {reader.fieldnames}")
            else:
                rows = list(reader)
    expected_ids = set(EPISODE_IDS)
    ids: list[int] = []
    successes = 0
    error_rows = 0
    timeout_rows = 0
    integer_step_rows = 0
    non_integer_step_rows = 0
    for idx, row in enumerate(rows, start=2):
        try:
            episode_id = int(row["episode_id"])
            ids.append(episode_id)
            expected_seed = seed_for(policy, condition, episode_id)
            if int(row["seed"]) != expected_seed:
                errors.append(f"line {idx}: seed mismatch {row['seed']} != {expected_seed}")
            if int(row["horizon"]) != HORIZON:
                errors.append(f"line {idx}: horizon mismatch {row['horizon']} != {HORIZON}")
        except Exception as exc:
            errors.append(f"line {idx}: parse error {exc}")
            continue
        if row["policy"] != policy or row["condition"] != condition:
            errors.append(f"line {idx}: wrong policy/condition {row['policy']}/{row['condition']}")
        if row["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
            errors.append(f"line {idx}: protocol SHA mismatch {row['protocol_sha256']}")
        if row["success"] not in {"0", "1"}:
            errors.append(f"line {idx}: success must be 0/1, got {row['success']!r}")
        else:
            successes += int(row["success"])
        try:
            steps = int(str(row["steps"]).strip())
            integer_step_rows += 1
            if steps < 0 or steps > HORIZON:
                errors.append(f"line {idx}: steps {steps} outside 0..{HORIZON}")
        except Exception:
            non_integer_step_rows += 1
            errors.append(f"line {idx}: steps must be an integer, got {row['steps']!r}")
        if row["error"]:
            error_rows += 1
        if row["timeout"] not in {"0", ""}:
            timeout_rows += 1
        if not row["video_path"]:
            errors.append(f"line {idx}: missing video_path")
        elif not Path(row["video_path"]).is_file():
            errors.append(f"line {idx}: video_path missing on disk: {row['video_path']}")
    actual_ids = set(ids)
    if len(rows) != len(EPISODE_IDS):
        errors.append(f"expected 288 rows, found {len(rows)}")
    if len(ids) != len(actual_ids):
        errors.append("duplicate episode_id rows detected")
    if actual_ids != expected_ids:
        errors.append(
            f"episode IDs mismatch missing={sorted(expected_ids - actual_ids)[:20]} "
            f"extra={sorted(actual_ids - expected_ids)[:20]}"
        )
    if error_rows or timeout_rows:
        errors.append(f"error/timeout rows present: error_rows={error_rows} timeout_rows={timeout_rows}")
    report = {
        "policy": policy,
        "condition": condition,
        "result_path": str(path),
        "expected_rows": len(EPISODE_IDS),
        "actual_rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "error_rows": error_rows,
        "timeout_rows": timeout_rows,
        "integer_step_rows": integer_step_rows,
        "non_integer_step_rows": non_integer_step_rows,
        "validation_status": "passed" if not errors else "failed",
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        raise RuntimeError(f"validation failed for {policy}/{condition}: {errors[:5]}")
    return report


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["EVAL_ROLLOUT_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
