#!/usr/bin/env python3
"""Regenerate selected paper-style analysis outputs from release files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist, stdev, variance


Z_975 = NormalDist().inv_cdf(0.975)

PAPER_BUCKETS = [
    ("negative", "No improvement"),
    ("not_significant", "Provably not significant"),
    ("possibly_significant", "Inconclusive"),
    ("certainly_significant", "Provably significant"),
]
BUCKET_KEYS = [key for key, _ in PAPER_BUCKETS]
RELEASE_TO_PAPER_BUCKET = {
    "no_improvement": "negative",
    "provably_not_significant": "not_significant",
    "indeterminate": "possibly_significant",
    "provably_significant": "certainly_significant",
}
PIE_SPECS = [
    {
        "benchmark": "LIBERO",
        "protocol": "Spatial + Object + Goal",
        "release_tracks": [("LIBERO", "spatial"), ("LIBERO", "object"), ("LIBERO", "goal")],
        "paper_files": [
            "stat_sig_libero_spatial_cutoff_data.csv",
            "stat_sig_libero_object_cutoff_data.csv",
            "stat_sig_libero_goal_cutoff_data.csv",
        ],
    },
    {
        "benchmark": "CALVIN",
        "protocol": "ABC->D",
        "release_tracks": [("CALVIN", "abc_d_atc")],
        "paper_files": ["stat_sig_calvin_abc_d_cutoff_data.csv"],
    },
    {
        "benchmark": "SimplerEnv",
        "protocol": "WidowX-Bridge",
        "release_tracks": [("SimplerEnv", "widowx_bridge")],
        "paper_files": ["stat_sig_simplerenv_widowx_bridge_cutoff_data.csv"],
    },
    {
        "benchmark": "RoboCasa",
        "protocol": "RSS24 protocol",
        "release_tracks": [("RoboCasa", "rss24")],
        "paper_files": ["stat_sig_robocasa_rss24_cutoff_data.csv"],
    },
    {
        "benchmark": "RoboTwin 2.0",
        "protocol": "50 clean + 500 randomized demos",
        "release_tracks": [("RoboTwin2", "hard_randomized")],
        "paper_files": ["stat_sig_robotwin2_hard_randomized_cutoff_data.csv"],
    },
]

SIMPLERENV_POLICIES = [
    ("cogact", "CogACT-Base", "CogACT"),
    ("internvla_m1", "InternVLA-M1", "InternVLA"),
    ("xvla", "X-VLA-WidowX", "X-VLA"),
    ("dexbotic", "Dexbotic / DB-MemVLA", "DB-MemVLA"),
]
SIMPLERENV_CONDITIONS = [
    ("original_stack", "Original stack", "fixed_grid_calibration/per_task_summary.csv", "stack"),
    ("language_swap", "Reverse language", "distribution_overfitting/per_policy_condition_summary.csv", "reverse_language"),
    ("stacked_support", "Stacked support blocks", "distribution_overfitting/per_policy_condition_summary.csv", "stacked_support"),
    ("pose_arm_randomized", "Random pose + arm", "distribution_overfitting/per_policy_condition_summary.csv", "random_pose_arm"),
]
SIMPLERENV_CI_CONDITIONS = [
    ("reverse_language", "Reverse language"),
    ("stacked_support", "Stacked support"),
    ("random_pose_arm", "Random pose + arm"),
]
SIMPLERENV_CI_POLICY_LABELS = {
    "cogact": "CogACT-Base",
    "internvla_m1": "InternVLA-M1",
    "xvla": "X-VLA-WidowX",
    "dexbotic": "Dexbotic (DB-MemVLA)",
}
LIBERO_CI_POLICY_LABELS = {
    "spatial_forcing": "Spatial Forcing",
    "simvla": "SimVLA",
    "pi05_lerobot": r"$\pi_{0.5}$ / LeRobot",
}


class AnalysisError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AnalysisError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"{path} has no header")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_keyed(path: Path, key_fields: tuple[str, ...], context: str) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in read_csv(path):
        key = tuple(row[field] for field in key_fields)
        if key in out:
            fail(f"{context}: duplicate row for key {key}")
        out[key] = row
    return out


def int_field(row: dict[str, str], field: str, context: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid integer field {field!r}") from exc


def float_or_none(value: str) -> float | None:
    value = value.strip()
    if not value or value == "unknown":
        return None
    return float(value)


def check_count(successes: int, total: int, context: str) -> None:
    if total <= 0:
        fail(f"{context}: total must be positive")
    if not 0 <= successes <= total:
        fail(f"{context}: successes {successes} outside [0, {total}]")


def format_signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def format_plain(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def wilson_interval(successes: int, total: int, z: float = Z_975) -> tuple[float, float]:
    check_count(successes, total, "Wilson interval")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - half, center + half


def newcombe_wilson_diff(x_c: int, n_c: int, x_a: int, n_a: int) -> tuple[float, float, float]:
    p_c = x_c / n_c
    p_a = x_a / n_a
    l_c, u_c = wilson_interval(x_c, n_c)
    l_a, u_a = wilson_interval(x_a, n_a)
    drop = p_c - p_a
    lower = drop - math.sqrt((p_c - l_c) ** 2 + (u_a - p_a) ** 2)
    upper = drop + math.sqrt((u_c - p_c) ** 2 + (p_a - l_a) ** 2)
    return drop, lower, upper


def release_significance_rows(root: Path) -> list[dict[str, str]]:
    return read_csv(root / "statistical_significance" / "significance_categories" / "all_comparisons.csv")


def release_pie_counts(root: Path) -> list[dict[str, str]]:
    by_track: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in release_significance_rows(root):
        by_track.setdefault((row["benchmark"], row["track"]), []).append(row)

    rows: list[dict[str, str]] = []
    for spec in PIE_SPECS:
        buckets: Counter[str] = Counter()
        release_categories: Counter[str] = Counter()
        for track in spec["release_tracks"]:
            track_rows = by_track.get(track)
            if track_rows is None:
                fail(f"missing release significance track {track}")
            for row in track_rows:
                category = row["category"]
                if category not in RELEASE_TO_PAPER_BUCKET:
                    fail(f"unexpected release category {category!r}")
                release_categories[category] += 1
                buckets[RELEASE_TO_PAPER_BUCKET[category]] += 1
        rows.append(pie_count_row(spec, buckets, release_categories))
    return rows


def paper_pie_counts(paper_data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in PIE_SPECS:
        buckets: Counter[str] = Counter()
        for filename in spec["paper_files"]:
            for row in read_csv(paper_data_root / filename):
                bucket = row.get("delta_bin", "")
                if bucket not in BUCKET_KEYS:
                    fail(f"{filename}: unexpected delta_bin {bucket!r}")
                buckets[bucket] += 1
        rows.append(pie_count_row(spec, buckets, Counter()))
    return rows


def pie_count_row(spec: dict[str, object], buckets: Counter[str], release_categories: Counter[str]) -> dict[str, str]:
    total = sum(buckets.values())
    if total <= 0:
        fail(f"{spec['benchmark']} pie has no rows")
    release_values = {
        "release_no_improvement": "",
        "release_provably_not_significant": "",
        "release_indeterminate": "",
        "release_provably_significant": "",
    }
    if release_categories:
        release_values = {
            "release_no_improvement": str(release_categories["no_improvement"]),
            "release_provably_not_significant": str(release_categories["provably_not_significant"]),
            "release_indeterminate": str(release_categories["indeterminate"]),
            "release_provably_significant": str(release_categories["provably_significant"]),
        }
    return {
        "benchmark": str(spec["benchmark"]),
        "protocol": str(spec["protocol"]),
        "negative": str(buckets["negative"]),
        "not_significant": str(buckets["not_significant"]),
        "possibly_significant": str(buckets["possibly_significant"]),
        "certainly_significant": str(buckets["certainly_significant"]),
        "total": str(total),
        **release_values,
    }


def release_simplerenv_plot_rows(root: Path) -> list[dict[str, str]]:
    base = root / "creeping_overfitting" / "results" / "simplerenv"
    fixed_rel = "fixed_grid_calibration/per_task_summary.csv"
    dist_rel = "distribution_overfitting/per_policy_condition_summary.csv"
    fixed = read_keyed(base / fixed_rel, ("policy", "task"), "SimplerEnv fixed-grid per-task summary")
    dist = read_keyed(base / dist_rel, ("policy", "condition"), "SimplerEnv distribution per-policy-condition summary")

    rows: list[dict[str, str]] = []
    for policy, policy_label, plot_label in SIMPLERENV_POLICIES:
        for condition, condition_label, source_rel, release_key in SIMPLERENV_CONDITIONS:
            row = fixed.get((policy, release_key)) if source_rel == fixed_rel else dist.get((policy, release_key))
            if row is None:
                fail(f"missing SimplerEnv {policy}/{release_key}")
            successes = int_field(row, "successes", f"SimplerEnv {policy}/{release_key}")
            total = int_field(row, "total", f"SimplerEnv {policy}/{release_key}")
            check_count(successes, total, f"SimplerEnv {policy}/{release_key}")
            rows.append(
                {
                    "policy": policy,
                    "policy_label": policy_label,
                    "plot_label": plot_label,
                    "condition": condition,
                    "condition_label": condition_label,
                    "successes": str(successes),
                    "total": str(total),
                    "release_source_csv": f"creeping_overfitting/results/simplerenv/{source_rel}",
                    "release_source_key": release_key,
                }
            )
    return rows


def paper_simplerenv_plot_rows(paper_data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(paper_data_root / "creeping_overfitting_simplerenv.csv"):
        successes = int_field(row, "successes", f"paper SimplerEnv {row.get('policy')}/{row.get('condition')}")
        total = int_field(row, "total", f"paper SimplerEnv {row.get('policy')}/{row.get('condition')}")
        check_count(successes, total, f"paper SimplerEnv {row.get('policy')}/{row.get('condition')}")
        rows.append(
            {
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "plot_label": row["plot_label"],
                "condition": row["condition"],
                "condition_label": row["condition_label"],
                "successes": str(successes),
                "total": str(total),
                "release_source_csv": "",
                "release_source_key": "",
            }
        )
    return rows


def simplerenv_ci_rows(root: Path) -> list[dict[str, str]]:
    base = root / "creeping_overfitting" / "results" / "simplerenv"
    fixed = read_keyed(base / "fixed_grid_calibration" / "per_task_summary.csv", ("policy", "task"), "SimplerEnv fixed-grid per-task summary")
    dist = read_keyed(base / "distribution_overfitting" / "per_policy_condition_summary.csv", ("policy", "condition"), "SimplerEnv distribution per-policy-condition summary")

    rows: list[dict[str, str]] = []
    for policy, _, _ in SIMPLERENV_POLICIES:
        cal = fixed.get((policy, "stack"))
        if cal is None:
            fail(f"missing SimplerEnv stack calibration for {policy}")
        cal_successes = int_field(cal, "successes", f"SimplerEnv {policy} stack")
        cal_total = int_field(cal, "total", f"SimplerEnv {policy} stack")
        for condition, label in SIMPLERENV_CI_CONDITIONS:
            altered = dist.get((policy, condition))
            if altered is None:
                fail(f"missing SimplerEnv altered condition {policy}/{condition}")
            altered_successes = int_field(altered, "successes", f"SimplerEnv {policy}/{condition}")
            altered_total = int_field(altered, "total", f"SimplerEnv {policy}/{condition}")
            drop, lower, upper = newcombe_wilson_diff(cal_successes, cal_total, altered_successes, altered_total)
            rows.append(ci_row("distribution overfitting", "SimplerEnv", label, SIMPLERENV_CI_POLICY_LABELS[policy], "success rate", f"{cal_successes}/{cal_total}", f"{altered_successes}/{altered_total}", drop * 100.0, lower * 100.0, upper * 100.0, "percentage points", "Newcombe-Wilson difference of proportions", 2))
    return rows


def calvin_ci_rows(root: Path) -> list[dict[str, str]]:
    base = root / "creeping_overfitting" / "results" / "calvin"
    resampled_rows = read_csv(base / "resampled_pose_per_sequence.csv")
    fresh_rows = read_csv(base / "fresh_sequence_per_sequence.csv")
    policies = ["X-VLA", "GR-1", "RoboFlamingo"]
    rows: list[dict[str, str]] = []

    for policy in policies:
        policy_resampled = [row for row in resampled_rows if row["policy"] == policy]
        conditions = sorted({row["condition"] for row in policy_resampled})
        if conditions != ["calibration", "resampled_pose"]:
            fail(f"unexpected CALVIN conditions for {policy}: {conditions}")
        cal = {int_field(row, "global_index", f"CALVIN {policy} calibration"): row for row in policy_resampled if row["condition"] == "calibration"}
        altered = {int_field(row, "global_index", f"CALVIN {policy} resampled_pose"): row for row in policy_resampled if row["condition"] == "resampled_pose"}
        if len(cal) != 1000 or set(cal) != set(altered):
            fail(f"CALVIN {policy}: expected paired 1000-row calibration/resampled global_index sets")

        indices = sorted(cal)
        cal_tasks = [int_field(cal[idx], "tasks_completed", f"CALVIN {policy} calibration {idx}") for idx in indices]
        alt_tasks = [int_field(altered[idx], "tasks_completed", f"CALVIN {policy} resampled {idx}") for idx in indices]
        task_diffs = [a - b for a, b in zip(cal_tasks, alt_tasks)]
        drop = sum(task_diffs) / len(task_diffs)
        half = Z_975 * stdev(task_diffs) / math.sqrt(len(task_diffs))
        rows.append(ci_row("distribution overfitting", "CALVIN", "resampled-pose", policy, "ATC", format_plain(sum(cal_tasks) / len(cal_tasks), 3), format_plain(sum(alt_tasks) / len(alt_tasks), 3), drop, drop - half, drop + half, "tasks completed out of 5", "paired row-level Wald", 3))

        cal_chain = [int_field(cal[idx], "chain_success", f"CALVIN {policy} calibration chain {idx}") for idx in indices]
        alt_chain = [int_field(altered[idx], "chain_success", f"CALVIN {policy} resampled chain {idx}") for idx in indices]
        chain_diffs = [100.0 * (a - b) for a, b in zip(cal_chain, alt_chain)]
        chain_drop = sum(chain_diffs) / len(chain_diffs)
        chain_half = Z_975 * stdev(chain_diffs) / math.sqrt(len(chain_diffs))
        rows.append(ci_row("distribution overfitting", "CALVIN", "resampled-pose", policy, "5/5 chain success", f"{sum(cal_chain)}/{len(cal_chain)}", f"{sum(alt_chain)}/{len(alt_chain)}", chain_drop, chain_drop - chain_half, chain_drop + chain_half, "percentage points", "paired row-level Wald", 2))

    for policy in policies:
        cal_values = [int_field(row, "tasks_completed", f"CALVIN {policy} calibration") for row in resampled_rows if row["policy"] == policy and row["condition"] == "calibration"]
        fresh_values = [int_field(row, "tasks_completed", f"CALVIN {policy} fresh") for row in fresh_rows if row["policy"] == policy]
        if len(cal_values) != 1000 or len(fresh_values) != 2000:
            fail(f"CALVIN {policy}: expected 1000 calibration and 2000 pooled fresh rows")
        cal_mean = sum(cal_values) / len(cal_values)
        fresh_mean = sum(fresh_values) / len(fresh_values)
        drop = cal_mean - fresh_mean
        half = Z_975 * math.sqrt(variance(cal_values) / len(cal_values) + variance(fresh_values) / len(fresh_values))
        rows.append(ci_row("sample overfitting", "CALVIN", "fresh-sequence", policy, "ATC", format_plain(cal_mean, 3), format_plain(fresh_mean, 4), drop, drop - half, drop + half, "tasks completed out of 5", "independent two-sample Wald", 4))
    return rows


def libero_ci_rows(root: Path) -> list[dict[str, str]]:
    rows_by_policy = read_keyed(root / "creeping_overfitting" / "results" / "libero" / "sample_overfitting_summary.csv", ("policy",), "LIBERO sample-overfitting summary")
    rows: list[dict[str, str]] = []
    for policy in ["spatial_forcing", "simvla", "pi05_lerobot"]:
        row = rows_by_policy.get((policy,))
        if row is None:
            fail(f"missing LIBERO sample-overfitting policy {policy}")
        official_successes = int_field(row, "official_successes", f"LIBERO {policy}")
        official_rows = int_field(row, "official_rows", f"LIBERO {policy}")
        fresh_successes = int_field(row, "fresh_successes", f"LIBERO {policy}")
        fresh_rows = int_field(row, "fresh_rows", f"LIBERO {policy}")
        drop, lower, upper = newcombe_wilson_diff(official_successes, official_rows, fresh_successes, fresh_rows)
        rows.append(ci_row("sample overfitting", "LIBERO", "fresh-init-state", LIBERO_CI_POLICY_LABELS[policy], "success rate", f"{official_successes}/{official_rows}", f"{fresh_successes}/{fresh_rows}", drop * 100.0, lower * 100.0, upper * 100.0, "percentage points", "Newcombe-Wilson difference of proportions", 2))
    return rows


def ci_row(overfitting_type: str, benchmark: str, test_set: str, policy: str, metric: str, calibration: str, comparison: str, drop: float, lower: float, upper: float, unit: str, method: str, digits: int) -> dict[str, str]:
    return {
        "overfitting_type": overfitting_type,
        "benchmark": benchmark,
        "test_set": test_set,
        "policy": policy,
        "metric": metric,
        "calibration": calibration,
        "comparison_result": comparison,
        "drop": format_signed(drop, digits),
        "ci_lower": format_signed(lower, digits),
        "ci_upper": format_signed(upper, digits),
        "unit": unit,
        "ci_method": method,
    }


def confidence_interval_rows(root: Path) -> list[dict[str, str]]:
    calvin_rows = calvin_ci_rows(root)
    return [*simplerenv_ci_rows(root), *calvin_rows[:6], *libero_ci_rows(root), *calvin_rows[6:]]


def compare_release_ci(root: Path, generated: list[dict[str, str]]) -> None:
    release_rows = read_csv(root / "creeping_overfitting" / "results" / "confidence_intervals.csv")
    if release_rows != generated:
        fail("generated confidence intervals differ from creeping_overfitting/results/confidence_intervals.csv")


def write_current_values(root: Path, output_dir: Path, paper_data_root: Path) -> list[str]:
    pie_rows = release_pie_counts(root)
    simplerenv_rows = release_simplerenv_plot_rows(root)
    ci_rows = confidence_interval_rows(root)
    compare_release_ci(root, ci_rows)
    files = write_value_files(output_dir, pie_rows, simplerenv_rows, ci_rows)
    write_bucket_comparison(output_dir, paper_pie_counts(paper_data_root), pie_rows)
    files.append("statistical_significance_bucket_comparison.csv")
    return files


def write_paper_input_values(root: Path, paper_data_root: Path, output_dir: Path) -> list[str]:
    pie_rows = paper_pie_counts(paper_data_root)
    simplerenv_rows = paper_simplerenv_plot_rows(paper_data_root)
    ci_rows = confidence_interval_rows(root)
    compare_release_ci(root, ci_rows)
    return write_value_files(output_dir, pie_rows, simplerenv_rows, ci_rows)


def write_value_files(output_dir: Path, pie_rows: list[dict[str, str]], simplerenv_rows: list[dict[str, str]], ci_rows: list[dict[str, str]]) -> list[str]:
    files = []
    write_csv(output_dir / "statistical_significance_pie_counts.csv", pie_rows, ["benchmark", "protocol", *BUCKET_KEYS, "total", "release_no_improvement", "release_provably_not_significant", "release_indeterminate", "release_provably_significant"])
    files.append("statistical_significance_pie_counts.csv")
    write_csv(output_dir / "creeping_overfitting_simplerenv_plot_data.csv", simplerenv_rows, ["policy", "policy_label", "plot_label", "condition", "condition_label", "successes", "total", "release_source_csv", "release_source_key"])
    files.append("creeping_overfitting_simplerenv_plot_data.csv")
    if ci_rows:
        write_csv(output_dir / "creeping_overfitting_confidence_intervals.csv", ci_rows, ["overfitting_type", "benchmark", "test_set", "policy", "metric", "calibration", "comparison_result", "drop", "ci_lower", "ci_upper", "unit", "ci_method"])
        files.append("creeping_overfitting_confidence_intervals.csv")
    return files


def write_bucket_comparison(output_dir: Path, paper_rows: list[dict[str, str]], current_rows: list[dict[str, str]]) -> None:
    paper_by_key = {(row["benchmark"], row["protocol"]): row for row in paper_rows}
    current_by_key = {(row["benchmark"], row["protocol"]): row for row in current_rows}
    if set(paper_by_key) != set(current_by_key):
        fail("paper and release-current pie count keys differ")

    rows = []
    for benchmark, protocol in [(str(spec["benchmark"]), str(spec["protocol"])) for spec in PIE_SPECS]:
        key = (benchmark, protocol)
        paper = paper_by_key[key]
        current = current_by_key[key]
        if paper["total"] != current["total"]:
            fail(f"{benchmark}: paper total {paper['total']} differs from release-current total {current['total']}")
        out = {"benchmark": benchmark, "protocol": protocol, "bucket_order": ";".join(BUCKET_KEYS)}
        for bucket in BUCKET_KEYS:
            out[f"paper_{bucket}"] = paper[bucket]
        for bucket in BUCKET_KEYS:
            out[f"current_{bucket}"] = current[bucket]
        out["total"] = paper["total"]
        rows.append(out)

    write_csv(
        output_dir / "statistical_significance_bucket_comparison.csv",
        rows,
        [
            "benchmark",
            "protocol",
            "bucket_order",
            "paper_negative",
            "paper_not_significant",
            "paper_possibly_significant",
            "paper_certainly_significant",
            "current_negative",
            "current_not_significant",
            "current_possibly_significant",
            "current_certainly_significant",
            "total",
        ],
    )


def compare_expected_dir(expected_dir: Path, generated_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        expected_path = expected_dir / filename
        generated_path = generated_dir / filename
        if not expected_path.exists():
            fail(f"expected output missing: {expected_path}")
        if expected_path.read_bytes() != generated_path.read_bytes():
            fail(f"generated {generated_path} differs from expected {expected_path}")


def compare_paper_to_release(root: Path, paper_data_root: Path, output_dir: Path) -> tuple[int, list[str]]:
    release_rows = release_significance_rows(root)
    release_by_track: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in release_rows:
        release_by_track.setdefault((row["benchmark"], row["track"]), []).append(row)

    differences: list[dict[str, str]] = []
    notes: list[str] = []
    for spec in PIE_SPECS:
        for release_track, paper_file in zip(spec["release_tracks"], spec["paper_files"]):
            release_track_rows = release_by_track.get(release_track)
            if release_track_rows is None:
                fail(f"missing release track {release_track}")
            paper_rows = read_csv(paper_data_root / paper_file)
            if len(release_track_rows) != len(paper_rows):
                fail(f"{release_track}: release has {len(release_track_rows)} rows, paper input has {len(paper_rows)} rows")
            for line_number, (release_row, paper_row) in enumerate(zip(release_track_rows, paper_rows), start=2):
                if release_row["paper_name"] != paper_row["paper_name"]:
                    fail(f"{release_track}: paper name mismatch at line {line_number}")
                release_bucket = RELEASE_TO_PAPER_BUCKET[release_row["category"]]
                paper_bucket = paper_row["delta_bin"]
                if release_bucket == paper_bucket:
                    continue
                differences.append(
                    {
                        "benchmark": str(spec["benchmark"]),
                        "track": release_track[1],
                        "source_csv": paper_file,
                        "source_line": str(line_number),
                        "paper_name": release_row["paper_name"],
                        "current_score": release_row["current_score"],
                        "previous_sota_score": release_row["previous_sota_score"],
                        "reported_delta": release_row["reported_delta"],
                        "release_category": release_row["category"],
                        "release_paper_bucket": release_bucket,
                        "paper_delta_bin": paper_bucket,
                        "previous_sota_scaled_count": release_row["previous_sota_scaled_count"],
                        "current_scaled_count": release_row["current_scaled_count"],
                        "previous_sota_count": release_row["previous_sota_count"],
                        "current_count": release_row["current_count"],
                        "previous_sota_rounding_residual": release_row["previous_sota_rounding_residual"],
                        "current_rounding_residual": release_row["current_rounding_residual"],
                        "necessary_score_delta": release_row["necessary_score_delta"],
                        "sufficient_score_delta": release_row["sufficient_score_delta"],
                        "paper_delta_necessary": paper_row.get("Delta_necessary", ""),
                        "paper_delta_sufficient": paper_row.get("Delta_sufficient", ""),
                        "inferred_cause": infer_significance_difference_cause(release_row),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paper_vs_release_significance_differences.csv", differences, ["benchmark", "track", "source_csv", "source_line", "paper_name", "current_score", "previous_sota_score", "reported_delta", "release_category", "release_paper_bucket", "paper_delta_bin", "previous_sota_scaled_count", "current_scaled_count", "previous_sota_count", "current_count", "previous_sota_rounding_residual", "current_rounding_residual", "necessary_score_delta", "sufficient_score_delta", "paper_delta_necessary", "paper_delta_sufficient", "inferred_cause"])
    summary_rows = []
    for key, count in sorted(Counter(row["inferred_cause"] for row in differences).items()):
        summary_rows.append({"inferred_cause": key, "rows": str(count)})
    write_csv(output_dir / "paper_vs_release_significance_summary.csv", summary_rows, ["inferred_cause", "rows"])
    if differences:
        notes.append(f"{len(differences)} significance classification rows differ between release-current generation and paper input CSVs.")
    return len(differences), notes


def infer_significance_difference_cause(row: dict[str, str]) -> str:
    delta = float(row["reported_delta"])
    if delta <= 0:
        return "zero_or_negative_gain_definition"
    if row["necessary_score_delta"] == "unknown" or row["sufficient_score_delta"] == "unknown":
        return "saturated_or_undefined_cutoff"
    rounding_residual = abs(float(row["previous_sota_rounding_residual"])) + abs(float(row["current_rounding_residual"]))
    if rounding_residual > 1e-12:
        return "integer_count_rounding"
    necessary = float_or_none(row["necessary_score_delta"])
    sufficient = float_or_none(row["sufficient_score_delta"])
    if (necessary is not None and math.isclose(delta, necessary, rel_tol=0.0, abs_tol=1e-12)) or (sufficient is not None and math.isclose(delta, sufficient, rel_tol=0.0, abs_tol=1e-12)):
        return "threshold_equality"
    return "classification_logic_difference"


def render_figures(values_dir: Path, figures_dir: Path, prefix: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    pie_rows = read_csv(values_dir / "statistical_significance_pie_counts.csv")
    colors = {
        "negative": "#9CA3AF",
        "not_significant": "#F4A261",
        "possibly_significant": "#E9C46A",
        "certainly_significant": "#2A9D8F",
    }
    fig, axes = plt.subplots(1, len(pie_rows), figsize=(5.85, 2.25))
    for ax, row in zip(axes, pie_rows):
        values = [int(row[key]) for key in BUCKET_KEYS]
        protocol = row["protocol"].replace("50 clean + 500 randomized demos", "50 clean + 500\nrandomized demos")
        ax.pie(values, colors=[colors[key] for key in BUCKET_KEYS], startangle=90, counterclock=False, autopct=lambda pct: f"{pct:.1f}%" if pct >= 7.0 else "", pctdistance=0.66, radius=1.02, wedgeprops={"edgecolor": "white", "linewidth": 0.45}, textprops={"fontsize": 6.0, "fontweight": "bold"})
        ax.text(0.5, 1.18, row["benchmark"], ha="center", va="bottom", fontsize=7.0, transform=ax.transAxes, clip_on=False)
        ax.text(0.5, 1.02, protocol, ha="center", va="bottom", fontsize=6.2, color="#374151", linespacing=0.92, multialignment="center", transform=ax.transAxes, clip_on=False)
        ax.text(0, -1.22, f"n={row['total']}", ha="center", va="top", fontsize=6.5, color="#374151", transform=ax.transData)
        ax.axis("equal")
    fig.legend(handles=[Patch(facecolor=colors[key], edgecolor="none", label=label) for key, label in PAPER_BUCKETS], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=0.9, columnspacing=0.85, fontsize=6.5)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.72, bottom=0.19, wspace=0.18)
    pie_path = figures_dir / f"statistical_significance_pies_{prefix}.svg"
    fig.savefig(pie_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    outputs.append(str(pie_path))

    simplerenv_rows = read_csv(values_dir / "creeping_overfitting_simplerenv_plot_data.csv")
    render_simplerenv_dumbbell(plt, simplerenv_rows, figures_dir / f"creeping_overfitting_simplerenv_{prefix}.svg")
    outputs.append(str(figures_dir / f"creeping_overfitting_simplerenv_{prefix}.svg"))
    return outputs


def render_simplerenv_dumbbell(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    from matplotlib.lines import Line2D

    by_key = {(row["policy"], row["condition"]): row for row in rows}
    policy_order = ["cogact", "internvla_m1", "dexbotic", "xvla"]
    condition_order = ["language_swap", "stacked_support", "pose_arm_randomized"]
    condition_labels = {
        "language_swap": "Reverse\nlanguage",
        "stacked_support": "Stacked\nsupport blocks",
        "pose_arm_randomized": "Random pose\n+ arm",
    }
    colors = {"cogact": "#264653", "internvla_m1": "#E9C46A", "xvla": "#E76F51", "dexbotic": "#2A9D8F"}
    fig, axes = plt.subplots(1, len(condition_order), figsize=(6.2, 2.45), sharey=True)
    for idx, condition in enumerate(condition_order):
        ax = axes[idx]
        for x, policy in enumerate(policy_order):
            baseline = by_key[(policy, "original_stack")]
            changed = by_key[(policy, condition)]
            baseline_pct = 100.0 * int(baseline["successes"]) / int(baseline["total"])
            changed_pct = 100.0 * int(changed["successes"]) / int(changed["total"])
            ax.vlines(x, min(baseline_pct, changed_pct), max(baseline_pct, changed_pct), colors=colors[policy], linewidth=1.7, alpha=0.7)
            ax.scatter([x], [baseline_pct], s=18, facecolors="white", edgecolors=colors[policy], linewidths=1.1, zorder=3)
            ax.scatter([x], [changed_pct], s=20, color=colors[policy], edgecolors="none", zorder=4)
            ax.text(x, max(baseline_pct, changed_pct) + 1.2, f"{max(baseline_pct, changed_pct):.1f}", ha="center", va="bottom", fontsize=6, color=colors[policy])
            ax.text(x, min(baseline_pct, changed_pct) - 1.2, f"{min(baseline_pct, changed_pct):.1f}", ha="center", va="top", fontsize=6, color=colors[policy])
        ax.set_title(condition_labels[condition], fontsize=7)
        ax.set_xticks(range(len(policy_order)))
        ax.set_xticklabels([by_key[(policy, "original_stack")]["plot_label"] for policy in policy_order], rotation=30, ha="right", fontsize=6)
        ax.set_ylim(0, 70)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Stack success (%)", fontsize=7)
    legend_handles = [
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", markeredgecolor="#374151", linestyle="None", markersize=4.5, label="Open = official baseline"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#374151", markeredgecolor="#374151", linestyle="None", markersize=4.5, label="Filled = perturbed"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, handletextpad=0.45, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_tables(values_dir: Path, tables_dir: Path, prefix: str) -> list[str]:
    ci_path = values_dir / "creeping_overfitting_confidence_intervals.csv"
    if not ci_path.exists():
        return []
    rows = read_csv(ci_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output = tables_dir / f"creeping_overfitting_confidence_intervals_{prefix}.tex"
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Benchmark & Test set & Policy & Metric & Drop & 95\% CI \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                tex_escape(value)
                for value in [
                    row["benchmark"],
                    row["test_set"],
                    row["policy"],
                    row["metric"],
                    row["drop"],
                    f"[{row['ci_lower']}, {row['ci_upper']}]",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines))
    return [str(output)]


def tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%")


def default_output_dir(script_dir: Path, mode: str) -> Path:
    if mode == "current":
        return script_dir / "release_current_values"
    if mode == "paper-inputs":
        return script_dir / "paper_input_values"
    return script_dir / "paper_comparison"


def default_paper_data_root(script_dir: Path) -> Path:
    return script_dir / "paper_inputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["current", "paper-inputs", "compare-paper"], default="current")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--paper-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--tables-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    script_dir = Path(__file__).resolve().parent
    paper_data_root = (args.paper_data_root or default_paper_data_root(script_dir)).resolve()
    output_dir = (args.output_dir or default_output_dir(script_dir, args.mode)).resolve()
    try:
        if args.mode == "current":
            files = write_current_values(root, output_dir, paper_data_root)
            if args.expected_dir is not None:
                compare_expected_dir(args.expected_dir.resolve(), output_dir, files)
            figures = render_figures(output_dir, args.figures_dir.resolve(), "release_current") if args.figures_dir else []
            tables = render_tables(output_dir, args.tables_dir.resolve(), "release_current") if args.tables_dir else []
            payload = {"status": "passed", "mode": args.mode, "output_dir": str(output_dir), "paper_data_root": str(paper_data_root), "files": files, "figures": figures, "tables": tables}
            code = 0
        elif args.mode == "paper-inputs":
            files = write_paper_input_values(root, paper_data_root, output_dir)
            figures = render_figures(output_dir, args.figures_dir.resolve(), "paper_inputs") if args.figures_dir else []
            tables = render_tables(output_dir, args.tables_dir.resolve(), "paper_inputs") if args.tables_dir else []
            payload = {"status": "passed", "mode": args.mode, "output_dir": str(output_dir), "paper_data_root": str(paper_data_root), "files": files, "figures": figures, "tables": tables}
            code = 0
        else:
            mismatch_count, notes = compare_paper_to_release(root, paper_data_root, output_dir)
            status = "failed" if mismatch_count else "passed"
            payload = {"status": status, "mode": args.mode, "output_dir": str(output_dir), "paper_data_root": str(paper_data_root), "mismatch_count": mismatch_count, "notes": notes}
            code = 1 if mismatch_count else 0
    except AnalysisError as exc:
        print(json.dumps({"status": "failed", "mode": args.mode, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
