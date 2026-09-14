#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from itertools import product
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROTOCOL = "calvin_official_sequence_manifest_v1"
FRESH_SAMPLE_KIND = "fresh_sequence_stratified"
OFFICIAL_SAMPLE_KIND = "official_seed0"
OFFICIAL_SEQUENCE_SOURCE = "official_generated"
MANIFEST_SEQUENCE_SOURCE = "manifest_direct"
RESET_PROTOCOL_ENV_VARS = ("CALVIN_RESET_BANK", "RESET_BANK", "CALVIN_RESET_PROTOCOL")
INITIAL_STATE_KEYS = [
    "led",
    "lightbulb",
    "slider",
    "drawer",
    "red_block",
    "blue_block",
    "pink_block",
    "grasped",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def ordered_state_items(state: dict[str, Any]) -> list[list[Any]]:
    return [[str(k), jsonable(v)] for k, v in state.items()]


def ordered_state_json(state: dict[str, Any]) -> str:
    return json.dumps(ordered_state_items(state), separators=(",", ":"))


def hash_rows(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sample_kind(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return OFFICIAL_SAMPLE_KIND
    return str(manifest.get("sample_kind") or OFFICIAL_SAMPLE_KIND)


def is_fresh_manifest(manifest: dict[str, Any] | None) -> bool:
    return manifest_sample_kind(manifest) == FRESH_SAMPLE_KIND


def sequence_source_for_manifest(manifest: dict[str, Any] | None) -> str:
    return MANIFEST_SEQUENCE_SOURCE if is_fresh_manifest(manifest) else OFFICIAL_SEQUENCE_SOURCE


def _row_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"{context}: expected integer, got {value!r}") from exc


def _normalize_manifest_row(row: dict[str, Any], position: int, manifest_path: Path, require_ordered: bool) -> None:
    context = f"{manifest_path} row {position}"
    row_index = row.get("index", position)
    if row_index is None:
        if require_ordered:
            raise RuntimeError(f"{context}: missing index")
        row_index = position
    row_index = _row_int(row_index, f"{context} index")
    if row_index != position:
        raise RuntimeError(f"{context}: index mismatch: got {row_index}, expected {position}")
    row["index"] = row_index

    global_index = row.get("global_index", row_index)
    if global_index is None:
        if require_ordered:
            raise RuntimeError(f"{context}: missing global_index")
        global_index = row_index
    global_index = _row_int(global_index, f"{context} global_index")
    if global_index != position:
        raise RuntimeError(f"{context}: global_index mismatch: got {global_index}, expected {position}")
    row["global_index"] = global_index

    items = row.get("initial_state_items")
    if items is None:
        if require_ordered:
            raise RuntimeError(f"{context}: fresh manifest row missing initial_state_items")
        state = row.get("initial_state")
        if not isinstance(state, dict):
            raise RuntimeError(f"{context}: missing initial_state")
    else:
        if not isinstance(items, list) or not all(isinstance(item, list) and len(item) == 2 for item in items):
            raise RuntimeError(f"{context}: invalid ordered initial_state_items")
        keys = [str(key) for key, _value in items]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"{context}: duplicate keys in initial_state_items")
        state = {str(key): value for key, value in items}
        if row.get("initial_state") is not None and stable_json(row["initial_state"]) != stable_json(state):
            raise RuntimeError(f"{context}: initial_state does not match ordered initial_state_items")
        row["initial_state"] = state
        expected_ordered_json = row.get("initial_state_ordered_json")
        actual_ordered_json = ordered_state_json(state)
        if expected_ordered_json is not None and expected_ordered_json != actual_ordered_json:
            raise RuntimeError(f"{context}: initial_state_ordered_json mismatch")
        expected_values_repr = row.get("initial_state_values_repr")
        actual_values_repr = str(state.values())
        if expected_values_repr is not None and expected_values_repr != actual_values_repr:
            raise RuntimeError(f"{context}: initial_state_values_repr mismatch")

    expected_initial_json = row.get("initial_state_json")
    actual_initial_json = stable_json(row["initial_state"])
    if expected_initial_json is not None and expected_initial_json != actual_initial_json:
        raise RuntimeError(f"{context}: initial_state_json mismatch")

    eval_sequence = row.get("eval_sequence")
    if not isinstance(eval_sequence, list):
        raise RuntimeError(f"{context}: missing eval_sequence list")
    expected_eval_json = row.get("eval_sequence_json")
    actual_eval_json = stable_json(eval_sequence)
    if expected_eval_json is not None and expected_eval_json != actual_eval_json:
        raise RuntimeError(f"{context}: eval_sequence_json mismatch")


def load_manifest_from_env(context: str) -> dict[str, Any] | None:
    path = os.environ.get("CALVIN_SEQUENCE_MANIFEST")
    expected_sha256 = os.environ.get("CALVIN_SEQUENCE_MANIFEST_SHA256")
    if not path:
        if expected_sha256:
            raise RuntimeError(f"{context}: CALVIN_SEQUENCE_MANIFEST_SHA256 is set but CALVIN_SEQUENCE_MANIFEST is empty")
        return None
    manifest = load_manifest(path, expected_sha256=expected_sha256, context=context)
    if is_fresh_manifest(manifest):
        errors = []
        if not expected_sha256:
            errors.append(f"CALVIN_SEQUENCE_MANIFEST_SHA256 is required for fresh-sequence manifest {path}")
        offenders = [f"{name}={os.environ[name]!r}" for name in RESET_PROTOCOL_ENV_VARS if name in os.environ]
        if offenders:
            errors.append("fresh-sequence manifest mode forbids reset-bank/protocol env vars: " + ", ".join(offenders))
        if errors:
            raise RuntimeError(f"{context}: " + "; ".join(errors))
    return manifest


def manifest_sequences(
    manifest: dict[str, Any],
    eval_start: int = 0,
    eval_end: int | None = None,
    context: str = "manifest",
) -> list[tuple[dict[str, Any], list[Any]]]:
    rows = manifest["sequences"]
    if eval_end is None:
        eval_end = len(rows)
    if eval_start < 0 or eval_start > eval_end or eval_end > len(rows):
        raise RuntimeError(f"{context}: invalid manifest slice {eval_start}:{eval_end} for {len(rows)} rows")
    selected = []
    seen: set[int] = set()
    for idx in range(int(eval_start), int(eval_end)):
        row = rows[idx]
        global_index = int(row["global_index"])
        if global_index != idx:
            raise RuntimeError(f"{context}: row/global_index mismatch at {idx}: got {global_index}")
        if global_index in seen:
            raise RuntimeError(f"{context}: duplicate global_index={global_index} in selected manifest slice")
        seen.add(global_index)
        selected.append((row["initial_state"], list(row["eval_sequence"])))
    return selected


def runtime_metadata(manifest: dict[str, Any] | None, sequence_source: str) -> dict[str, Any]:
    generation = manifest.get("generation", {}) if isinstance(manifest, dict) else {}
    if not isinstance(generation, dict):
        generation = {}
    return {
        "calvin_sequence_manifest": manifest.get("_manifest_path", "") if manifest else os.environ.get("CALVIN_SEQUENCE_MANIFEST", ""),
        "calvin_sequence_manifest_sha256": manifest.get("_manifest_sha256", "") if manifest else "",
        "calvin_sequence_manifest_sha256_expected": manifest.get("_manifest_sha256_expected", "") if manifest else os.environ.get("CALVIN_SEQUENCE_MANIFEST_SHA256", ""),
        "calvin_sequence_sample_kind": manifest_sample_kind(manifest) if manifest else OFFICIAL_SAMPLE_KIND,
        "calvin_sequence_sample_seed": generation.get("sample_seed", ""),
        "calvin_sequence_sample_condition": generation.get("sample_condition", generation.get("condition", manifest.get("sample_condition", "") if manifest else "")),
        "calvin_sequence_source": sequence_source,
        "calvin_reset_bank": os.environ.get("CALVIN_RESET_BANK", ""),
        "reset_bank": os.environ.get("RESET_BANK", ""),
        "calvin_reset_protocol": os.environ.get("CALVIN_RESET_PROTOCOL", ""),
    }


def result_row(
    global_index: int,
    success: int,
    initial_state: dict[str, Any],
    eval_sequence: Any,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "global_index": int(global_index),
        "success": int(success),
        "initial_state_json": stable_json(initial_state),
        "eval_sequence": jsonable(eval_sequence),
        "eval_sequence_json": stable_json(eval_sequence),
    }
    if manifest is not None:
        expected = manifest["sequences"][int(global_index)]
        if expected.get("initial_state_ordered_json") is not None:
            row["initial_state_ordered_json"] = ordered_state_json(initial_state)
        if expected.get("initial_state_values_repr") is not None:
            row["initial_state_values_repr"] = str(initial_state.values())
    return row


def load_calvin_multistep_sequences(calvin_root: Path):
    for rel in ("calvin_models", "calvin_env"):
        path = calvin_root / rel
        if not path.exists():
            raise FileNotFoundError(f"missing CALVIN path: {path}")
        sys.path.insert(0, str(path))
    return importlib.import_module("calvin_agent.evaluation.multistep_sequences")


def load_calvin_get_sequences(calvin_root: Path):
    module = load_calvin_multistep_sequences(calvin_root)
    return module.get_sequences


def official_initial_states() -> list[dict[str, Any]]:
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    if list(possible_conditions) != INITIAL_STATE_KEYS:
        raise RuntimeError("local CALVIN initial-state key order drifted")

    def valid_state(values: tuple[Any, ...]) -> bool:
        return (
            values.count("table") in [1, 2]
            and values.count("slider_right") < 2
            and values.count("slider_left") < 2
        )

    return [
        dict(zip(possible_conditions.keys(), values))
        for values in filter(valid_state, product(*possible_conditions.values()))
    ]


def quota_by_state(num_sequences: int, num_states: int) -> list[int]:
    return list(map(len, np.array_split(range(int(num_sequences)), int(num_states))))


def quota_metadata(quotas: list[int]) -> dict[str, Any]:
    counts = {str(k): int(v) for k, v in sorted(Counter(quotas).items())}
    metadata: dict[str, Any] = {
        "quota_counts": counts,
        "quota_by_state_sha256": hash_rows([str(quota) for quota in quotas]),
    }
    if 0 in quotas:
        metadata["quota_caveat"] = (
            f"official np.array_split quota is degenerate for this sample size: "
            f"{counts.get('1', 0)} symbolic states get one chain and {counts.get('0', 0)} get zero"
        )
    return metadata


def derived_seed(sample_seed: int, label: str, index: int | None = None) -> int:
    material = f"calvin-fresh-sequence-v1:{int(sample_seed)}:{label}:{index if index is not None else ''}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "little")


def shuffle_rows(rows: list[Any], seed: int) -> list[Any]:
    shuffled = list(rows)
    np_state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        np.random.shuffle(shuffled)
    finally:
        np.random.set_state(np_state)
    return shuffled


def normalize_sequence(seq: Any) -> tuple[str, ...]:
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    return tuple(str(task) for task in seq)


def build_fresh_sequences(
    module: Any,
    num_sequences: int,
    sample_seed: int,
    sequence_workers: int,
) -> tuple[list[Any], dict[str, Any]]:
    initial_states = official_initial_states()
    if len(initial_states) != 192:
        raise RuntimeError(f"expected 192 CALVIN symbolic initial states, got {len(initial_states)}")
    quotas = quota_by_state(num_sequences, len(initial_states))
    unshuffled = []
    chain_seeds = []
    worker_args = []
    for state_index, (state, quota) in enumerate(zip(initial_states, quotas)):
        chain_seed = derived_seed(sample_seed, "state-chain", state_index)
        chain_seeds.append(chain_seed)
        worker_args.append((state, quota, chain_seed))
    with ProcessPoolExecutor(max_workers=int(sequence_workers)) as executor:
        per_state_sequences = list(executor.map(module.get_sequences_for_state2, worker_args))
    for state_index, (state, quota, chain_seed, seqs) in enumerate(
        zip(initial_states, quotas, chain_seeds, per_state_sequences)
    ):
        if len(seqs) != int(quota):
            raise RuntimeError(f"state {state_index} returned {len(seqs)} chains, expected {quota}")
        for state_chain_index, seq in enumerate(seqs):
            unshuffled.append((state, normalize_sequence(seq), state_index, state_chain_index, quota, chain_seed))
    if len(unshuffled) != int(num_sequences):
        raise RuntimeError(f"fresh generator returned {len(unshuffled)}, expected {num_sequences}")
    shuffle_seed = derived_seed(sample_seed, "shuffle")
    metadata = {
        "sample_seed": int(sample_seed),
        "chain_seed_strategy": "sha256('calvin-fresh-sequence-v1:{sample_seed}:state-chain:{state_index}')[:32bits]",
        "shuffle_seed_strategy": "sha256('calvin-fresh-sequence-v1:{sample_seed}:shuffle:')[:32bits]",
        "shuffle_seed": int(shuffle_seed),
        "initial_state_count": len(initial_states),
        "initial_state_key_order": INITIAL_STATE_KEYS,
        "sequence_workers": int(sequence_workers),
        "state_chain_seed_sha256": hash_rows([str(seed) for seed in chain_seeds]),
    }
    metadata.update(quota_metadata(quotas))
    return shuffle_rows(unshuffled, shuffle_seed), metadata


def sequence_rows(eval_sequences: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(eval_sequences):
        initial_state, eval_sequence = item[:2]
        row = {
            "index": index,
            "global_index": index,
            "initial_state": jsonable(initial_state),
            "initial_state_items": ordered_state_items(initial_state),
            "initial_state_values_repr": str(initial_state.values()),
            "eval_sequence": jsonable(eval_sequence),
            "initial_state_json": stable_json(initial_state),
            "initial_state_ordered_json": ordered_state_json(initial_state),
            "eval_sequence_json": stable_json(eval_sequence),
        }
        if len(item) == 6:
            row.update({
                "source_state_index": int(item[2]),
                "state_chain_index": int(item[3]),
                "state_quota": int(item[4]),
                "state_chain_seed": int(item[5]),
            })
        rows.append(row)
    return rows


def build_manifest(
    calvin_root: Path,
    num_sequences: int,
    sequence_workers: int,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    module = load_calvin_multistep_sequences(calvin_root)
    if sample_seed is None:
        eval_sequences = list(module.get_sequences(int(num_sequences), num_workers=int(sequence_workers)))
        sample_kind = OFFICIAL_SAMPLE_KIND
        generation = {
            "sample_seed": 0,
            "initial_state_count": 192,
            "initial_state_key_order": INITIAL_STATE_KEYS,
        }
        generation.update(quota_metadata(quota_by_state(num_sequences, 192)))
    else:
        eval_sequences, generation = build_fresh_sequences(
            module,
            int(num_sequences),
            int(sample_seed),
            int(sequence_workers),
        )
        sample_kind = FRESH_SAMPLE_KIND
    if len(eval_sequences) != int(num_sequences):
        raise RuntimeError(f"sequence generator returned {len(eval_sequences)}, expected {num_sequences}")
    rows = sequence_rows(eval_sequences)
    initial_rows = [row["initial_state_json"] for row in rows]
    ordered_initial_rows = [row["initial_state_ordered_json"] for row in rows]
    eval_rows = [row["eval_sequence_json"] for row in rows]
    module_path = Path(module.__file__).resolve()
    utils_module = importlib.import_module("calvin_agent.evaluation.utils")
    utils_path = Path(utils_module.__file__).resolve()
    return {
        "protocol": PROTOCOL,
        "sample_kind": sample_kind,
        "calvin_root": str(calvin_root.resolve()),
        "official_generator": {
            "path": str(module_path),
            "sha256": file_sha256(module_path),
        },
        "official_sources": {
            "multistep_sequences.py": {
                "path": str(module_path),
                "sha256": file_sha256(module_path),
            },
            "utils.py": {
                "path": str(utils_path),
                "sha256": file_sha256(utils_path),
            },
        },
        "calvin_reset_bank_used": False,
        "generation": generation,
        "num_sequences": int(num_sequences),
        "sequence_workers": int(sequence_workers),
        "initial_state_sha256": hash_rows(initial_rows),
        "initial_state_ordered_sha256": hash_rows(ordered_initial_rows),
        "eval_sequence_sha256": hash_rows(eval_rows),
        "sequences": rows,
    }


def load_manifest(
    path: str | Path,
    expected_sha256: str | None = None,
    context: str = "CALVIN sequence manifest",
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing CALVIN sequence manifest: {manifest_path}")
    actual_sha256 = file_sha256(manifest_path)
    if expected_sha256:
        expected = expected_sha256.strip().lower()
        if actual_sha256.lower() != expected:
            raise RuntimeError(
                f"{context}: CALVIN_SEQUENCE_MANIFEST_SHA256 mismatch for {manifest_path}: "
                f"got {actual_sha256}, expected {expected}"
            )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"manifest protocol mismatch in {manifest_path}: {manifest.get('protocol')!r}")
    num_sequences = int(manifest.get("num_sequences", -1))
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or len(rows) != num_sequences:
        raise RuntimeError(f"manifest sequence count mismatch in {manifest_path}: rows={len(rows) if isinstance(rows, list) else type(rows)}, num_sequences={num_sequences}")
    require_ordered = is_fresh_manifest(manifest)
    seen: dict[int, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{manifest_path} row {position}: expected object, got {type(row)}")
        _normalize_manifest_row(row, position, manifest_path, require_ordered=require_ordered)
        global_index = int(row["global_index"])
        if global_index in seen:
            raise RuntimeError(f"{manifest_path}: duplicate global_index={global_index} at rows {seen[global_index]} and {position}")
        seen[global_index] = position
    missing = [idx for idx in range(num_sequences) if idx not in seen]
    if missing:
        raise RuntimeError(f"{manifest_path}: missing global_index rows, first missing={missing[:10]}")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = actual_sha256
    manifest["_manifest_sha256_expected"] = expected_sha256 or ""
    return manifest


def validate_sequences_against_manifest(eval_sequences: list[Any], manifest: dict[str, Any], context: str) -> None:
    rows = sequence_rows(eval_sequences)
    expected_n = int(manifest["num_sequences"])
    if len(rows) != expected_n:
        raise RuntimeError(f"{context}: sequence count mismatch: got {len(rows)}, expected {expected_n}")
    seen: set[int] = set()
    for idx, row in enumerate(rows):
        expected = manifest["sequences"][idx]
        global_index = int(expected["global_index"])
        if global_index != idx:
            raise RuntimeError(f"{context}: manifest global_index mismatch at row {idx}: got {global_index}")
        if global_index in seen:
            raise RuntimeError(f"{context}: duplicate manifest global_index={global_index}")
        seen.add(global_index)
        if row["initial_state_json"] != expected["initial_state_json"]:
            raise RuntimeError(f"{context}: initial-state mismatch at global_index={global_index}")
        if expected.get("initial_state_ordered_json") is not None and row["initial_state_ordered_json"] != expected["initial_state_ordered_json"]:
            raise RuntimeError(f"{context}: ordered initial-state mismatch at global_index={global_index}")
        if expected.get("initial_state_values_repr") is not None and row["initial_state_values_repr"] != expected["initial_state_values_repr"]:
            raise RuntimeError(f"{context}: initial-state value-order mismatch at global_index={global_index}")
        if row["eval_sequence_json"] != expected["eval_sequence_json"]:
            raise RuntimeError(f"{context}: eval-sequence mismatch at global_index={global_index}")
    initial_hash = hash_rows([row["initial_state_json"] for row in rows])
    eval_hash = hash_rows([row["eval_sequence_json"] for row in rows])
    if initial_hash != manifest["initial_state_sha256"]:
        raise RuntimeError(f"{context}: initial-state sequence hash mismatch: got {initial_hash}, expected {manifest['initial_state_sha256']}")
    if eval_hash != manifest["eval_sequence_sha256"]:
        raise RuntimeError(f"{context}: eval-sequence hash mismatch: got {eval_hash}, expected {manifest['eval_sequence_sha256']}")


def validate_result_rows_against_manifest(rows: list[dict[str, Any]], manifest: dict[str, Any] | None, context: str) -> None:
    if manifest is None:
        return
    seen: dict[int, int] = {}
    for row_num, row in enumerate(rows):
        if "global_index" not in row:
            raise RuntimeError(f"{context}: result row {row_num} missing global_index")
        idx = _row_int(row["global_index"], f"{context} result row {row_num} global_index")
        if idx in seen:
            raise RuntimeError(f"{context}: duplicate result global_index={idx} at rows {seen[idx]} and {row_num}")
        seen[idx] = row_num
        if idx < 0 or idx >= int(manifest["num_sequences"]):
            raise RuntimeError(f"{context}: result global_index={idx} outside manifest range")
        expected = manifest["sequences"][idx]
        if row.get("initial_state_json") != expected.get("initial_state_json"):
            raise RuntimeError(f"{context}: result initial_state_json mismatch at global_index={idx}")
        expected_ordered = expected.get("initial_state_ordered_json")
        if expected_ordered is not None and row.get("initial_state_ordered_json") != expected_ordered:
            raise RuntimeError(f"{context}: result ordered initial-state mismatch at global_index={idx}")
        expected_values = expected.get("initial_state_values_repr")
        if expected_values is not None and row.get("initial_state_values_repr") != expected_values:
            raise RuntimeError(f"{context}: result initial-state values mismatch at global_index={idx}")
        expected_eval_json = expected.get("eval_sequence_json")
        actual_eval_json = row.get("eval_sequence_json", stable_json(row.get("eval_sequence")))
        if actual_eval_json != expected_eval_json:
            raise RuntimeError(f"{context}: result eval_sequence mismatch at global_index={idx}")


def validate_bank_metadata_against_manifest(metadata: dict[str, Any], manifest: dict[str, Any], context: str) -> None:
    for key in ("num_sequences", "sequence_workers", "initial_state_sha256", "eval_sequence_sha256"):
        actual = metadata.get(key)
        expected = manifest.get(key)
        if str(actual) != str(expected):
            raise RuntimeError(f"{context}: reset-bank/manifest mismatch for {key}: got {actual!r}, expected {expected!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a fail-loud CALVIN sequence manifest.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Generate a fresh-sequence sample with this seed. Omit to preserve official seed-0 get_sequences().",
    )
    args = parser.parse_args()
    if args.num_sequences <= 0:
        raise ValueError(f"--num-sequences must be positive, got {args.num_sequences}")
    if args.sequence_workers <= 0:
        raise ValueError(f"--sequence-workers must be positive, got {args.sequence_workers}")
    if args.sample_seed is not None and args.sample_seed < 0:
        raise ValueError(f"--sample-seed must be non-negative, got {args.sample_seed}")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        Path(args.calvin_root).resolve(),
        args.num_sequences,
        args.sequence_workers,
        args.sample_seed,
    )
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(output)
    print(json.dumps({
        "manifest": str(output),
        "sample_kind": manifest["sample_kind"],
        "sample_seed": manifest["generation"]["sample_seed"],
        "num_sequences": manifest["num_sequences"],
        "sequence_workers": manifest["sequence_workers"],
        "initial_state_sha256": manifest["initial_state_sha256"],
        "initial_state_ordered_sha256": manifest["initial_state_ordered_sha256"],
        "eval_sequence_sha256": manifest["eval_sequence_sha256"],
        "quota_counts": manifest["generation"]["quota_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
