from __future__ import annotations

import argparse
import json

from data_source_dependency.widowx.data import summarize_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an X-VLA WidowX grid demo dataset.")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--tasks", type=str, default="stack,carrot,spoon,eggplant")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--limit-episodes-per-task", type=int)
    parser.add_argument("--pad-terminal-actions", action="store_true")
    parser.add_argument("--write-json", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(
        args.dataset_root,
        tasks=args.tasks,
        chunk_size=args.chunk_size,
        limit_episodes_per_task=args.limit_episodes_per_task,
        pad_terminal_actions=args.pad_terminal_actions,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.write_json:
        write_json(args.write_json, summary)


if __name__ == "__main__":
    main()
