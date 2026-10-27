#!/usr/bin/env python3
"""CLI: filter CIT jsonl to usable rows (and future balancing rules).

Examples::

  python -m qwen_finetune.filter_dataset \\
    --dataset $CIT_DATA_ROOT/procthor_v9/natural \\
    --out /tmp/procthor_v7_usable.jsonl

  python -m qwen_finetune.filter_dataset --dataset ... --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_CIT = _ROOT.parent
for p in (_CIT, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_config import (  # noqa: E402
    DatasetFilterConfig,
    summarize_filter,
    write_filtered_jsonl,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root (contains dataset.jsonl + images/)",
    )
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="Write filtered jsonl here")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--categories",
        default=None,
        help="Comma-separated allow-list (optional)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.dataset)
    src = Path(args.jsonl) if args.jsonl else root / "dataset.jsonl"
    cats = None
    if args.categories:
        cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    cfg = DatasetFilterConfig(categories=cats, seed=int(args.seed))

    stats = summarize_filter(src, dataset_root=root, cfg=cfg)
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_only:
        return
    if args.out is None:
        raise SystemExit("Pass --out PATH or --stats-only")
    counts = write_filtered_jsonl(
        src, Path(args.out), dataset_root=root, cfg=cfg, limit=int(args.limit or 0)
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
