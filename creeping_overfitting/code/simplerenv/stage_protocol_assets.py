#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from protocol_abcde_common import POLICIES, PROJECT_ROOT, load_protocol_config, sha256_file

ASSET_IDS = (
    "render_candidate_blue_hybrid_v4",
    "render_candidate_red_corrected_v6e",
    "render_candidate_white_offwhite_hybrid_v4",
)
SOURCE_CUSTOM = PROJECT_ROOT / "third_party/xvla_simpler_env/ManiSkill2_real2sim/data/custom"
TARGET_CUSTOM = {
    "cogact": PROJECT_ROOT / "third_party/simpler_env/ManiSkill2_real2sim/data/custom",
    "spatialvla": PROJECT_ROOT / "scratch/repeated_fixed_grid_calibration_288_20260523_v2/sources/simplerenv_openvla_ccfe380/ManiSkill2_real2sim/data/custom",
    "internvla_m1": Path("/share/data/ripl/tianchong/projects/InternVLA-M1/cache/Projects/SimplerEnv/ManiSkill2_real2sim/data/custom"),
    "xvla": SOURCE_CUSTOM,
    "dexbotic": PROJECT_ROOT / "scratch/strict_calibrations_20260523/sources/dexbotic-benchmark_54a6c749_simpler4ab717_clean_20260523T0152Z/simpler/ManiSkill2_real2sim/data/custom",
}
INFO_NAME = "info_bridge_custom_baked_tex_v0.json"


def file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"missing asset directory: {root}")
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def target_report(policy: str) -> dict[str, Any]:
    target = TARGET_CUSTOM[policy]
    report: dict[str, Any] = {
        "policy": policy,
        "target_custom_dir": str(target),
        "target_info": str(target / INFO_NAME),
        "assets": {},
        "status": "unknown",
    }
    source_info = SOURCE_CUSTOM / INFO_NAME
    source = json.loads(source_info.read_text()) if source_info.is_file() else {}
    target_info = target / INFO_NAME
    target_data = json.loads(target_info.read_text()) if target_info.is_file() else {}
    errors: list[str] = []
    for asset_id in ASSET_IDS:
        source_dir = SOURCE_CUSTOM / "models" / asset_id
        target_dir = target / "models" / asset_id
        asset_record: dict[str, Any] = {
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "source_file_sha256": file_hashes(source_dir) if source_dir.is_dir() else {},
            "source_info_entry": source.get(asset_id),
            "target_exists": target_dir.is_dir(),
            "target_info_entry": target_data.get(asset_id),
        }
        if target_dir.is_dir():
            asset_record["target_file_sha256"] = file_hashes(target_dir)
            if asset_record["target_file_sha256"] != asset_record["source_file_sha256"]:
                errors.append(f"{policy}:{asset_id}: target file hashes do not match source")
        else:
            errors.append(f"{policy}:{asset_id}: missing target asset dir")
        if target_data.get(asset_id) != source.get(asset_id):
            errors.append(f"{policy}:{asset_id}: target info entry missing or mismatched")
        report["assets"][asset_id] = asset_record
    report["errors"] = errors
    report["status"] = "passed" if not errors else "missing_or_mismatch"
    return report


def write_manifest(path: Path, *, policies: list[str], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_info = SOURCE_CUSTOM / INFO_NAME
    runtime_paths: list[dict[str, Any]] = []
    for policy in policies:
        target = TARGET_CUSTOM[policy]
        maniskill_root = target.parent.parent
        asset_files: list[dict[str, str]] = []
        for asset_id in ASSET_IDS:
            target_dir = target / "models" / asset_id
            if target_dir.is_dir():
                for rel, digest in file_hashes(target_dir).items():
                    asset_files.append(
                        {
                            "asset_id": asset_id,
                            "relative_path": str(Path("data/custom/models") / asset_id / rel),
                            "sha256": digest,
                        }
                    )
        runtime_paths.append(
            {
                "policy": policy,
                "maniskill2_real2sim": str(maniskill_root),
                "info_json": str(target / INFO_NAME),
                "info_json_sha256": sha256_file(target / INFO_NAME) if (target / INFO_NAME).is_file() else "",
                "asset_files": asset_files,
            }
        )
    payload: dict[str, Any] = {
        "mode": mode,
        "source_custom_dir": str(SOURCE_CUSTOM),
        "source_info": str(source_info),
        "asset_ids": list(ASSET_IDS),
        "accepted_asset_ids": list(ASSET_IDS),
        "runtime_paths": runtime_paths,
        "source_info_sha256": sha256_file(source_info) if source_info.is_file() else None,
        "source_asset_file_sha256": {
            asset_id: file_hashes(SOURCE_CUSTOM / "models" / asset_id) for asset_id in ASSET_IDS
        },
        "policy_targets": [target_report(policy) for policy in policies],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_asset_dir(source: Path, target: Path, *, write: bool) -> None:
    source_hashes = file_hashes(source)
    if target.exists():
        target_hashes = file_hashes(target)
        if target_hashes != source_hashes:
            raise RuntimeError(f"asset directory hash mismatch: {target}")
        return
    if not write:
        raise RuntimeError(f"missing target asset directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    if file_hashes(target) != source_hashes:
        raise RuntimeError(f"copied asset directory failed hash check: {target}")


def update_info(source_info: Path, target_info: Path, *, write: bool) -> None:
    if not source_info.is_file():
        raise RuntimeError(f"missing source asset info: {source_info}")
    source = json.loads(source_info.read_text())
    for asset_id in ASSET_IDS:
        if asset_id not in source:
            raise RuntimeError(f"source info missing {asset_id}")
    if not target_info.is_file():
        raise RuntimeError(f"missing target asset info: {target_info}")
    target = json.loads(target_info.read_text())
    changed = False
    for asset_id in ASSET_IDS:
        if asset_id in target:
            if target[asset_id] != source[asset_id]:
                raise RuntimeError(f"target info entry mismatch for {asset_id}: {target_info}")
        else:
            if not write:
                raise RuntimeError(f"target info missing {asset_id}: {target_info}")
            target[asset_id] = source[asset_id]
            changed = True
    if changed:
        tmp = target_info.with_suffix(target_info.suffix + ".tmp")
        tmp.write_text(json.dumps(target, indent=2, sort_keys=True) + "\n")
        tmp.replace(target_info)


def check_source() -> None:
    load_protocol_config()
    source_info = SOURCE_CUSTOM / INFO_NAME
    if not source_info.is_file():
        raise RuntimeError(f"missing source info JSON: {source_info}")
    for asset_id in ASSET_IDS:
        copy_asset_dir(SOURCE_CUSTOM / "models" / asset_id, SOURCE_CUSTOM / "models" / asset_id, write=False)


def stage_policy(policy: str, *, write: bool, allow_missing_targets: bool = False) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy {policy}")
    target = TARGET_CUSTOM[policy]
    source_info = SOURCE_CUSTOM / INFO_NAME
    target_info = target / INFO_NAME
    try:
        for asset_id in ASSET_IDS:
            copy_asset_dir(SOURCE_CUSTOM / "models" / asset_id, target / "models" / asset_id, write=write)
        update_info(source_info, target_info, write=write)
    except RuntimeError:
        if not allow_missing_targets:
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="stage missing assets; default is verify-only")
    parser.add_argument("--source-only", action="store_true", help="only verify the source asset bundle and frozen config")
    parser.add_argument("--allow-missing-targets", action="store_true", help="record target status without failing on missing assets")
    parser.add_argument("--manifest", type=Path, default=None, help="write runtime asset manifest JSON")
    parser.add_argument("--policies", nargs="*", default=list(POLICIES))
    args = parser.parse_args()
    check_source()
    if args.source_only:
        if args.manifest is not None:
            write_manifest(args.manifest, policies=list(args.policies), mode="source-only")
        print("source_asset_status=passed")
        return
    for policy in args.policies:
        stage_policy(policy, write=args.write, allow_missing_targets=args.allow_missing_targets)
    if args.manifest is not None:
        mode = "write" if args.write else "verify"
        if args.allow_missing_targets:
            mode += "_allow_missing_targets"
        write_manifest(args.manifest, policies=list(args.policies), mode=mode)
    print(("asset_stage_status=written" if args.write else "asset_stage_status=verified"))


if __name__ == "__main__":
    main()
