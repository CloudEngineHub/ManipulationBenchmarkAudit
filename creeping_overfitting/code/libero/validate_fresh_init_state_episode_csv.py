#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--require-no-errors", action="store_true")
    parser.add_argument("--require-no-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.episode_csv)
    if not csv_path.exists():
        raise SystemExit(f"missing episode CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, got {len(rows)} in {csv_path}")

    successes = 0
    errors = []
    for idx, row in enumerate(rows):
        if row.get("suite") != args.suite:
            raise SystemExit(f"row {idx} has suite {row.get('suite')}, expected {args.suite}")
        if row.get("policy_name") != args.policy_name:
            raise SystemExit(f"row {idx} has policy {row.get('policy_name')}, expected {args.policy_name}")
        if not row.get("instance_id", "").startswith(args.suite + "/"):
            raise SystemExit(f"row {idx} has unstable instance_id: {row.get('instance_id')}")
        if row.get("success") not in {"0", "1", 0, 1}:
            raise SystemExit(f"row {idx} has invalid success value: {row.get('success')}")
        successes += int(row["success"])
        if row.get("error_type", ""):
            errors.append({"row": idx, "error_type": row.get("error_type")})

    if args.require_no_errors and errors:
        raise SystemExit(f"expected no episode errors, found {len(errors)}; first={errors[0]}")

    videos = list(csv_path.parent.rglob("*.mp4"))
    if args.require_no_videos and videos:
        raise SystemExit(f"expected no videos, found {len(videos)} under {csv_path.parent}")

    summary = {
        "policy_name": args.policy_name,
        "suite": args.suite,
        "episode_csv": str(csv_path),
        "rows": len(rows),
        "successes": successes,
        "errors": len(errors),
        "videos": len(videos),
        "validation_status": "passed",
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
