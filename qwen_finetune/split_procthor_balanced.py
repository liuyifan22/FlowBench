#!/usr/bin/env python3
"""Stratified train/val/test split for a balanced procthor SoM jsonl.

Mirrors the v9 split_meta recipe:
  - per category: hold out ``per_cat_hold`` for test and the same for val
    (capped by available rows); remainder → train
  - eval.jsonl is a copy of test.jsonl
  - optional train shards train_00.jsonl …
"""

from __future__ import annotations

import os
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="Balanced dataset.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cat-hold", type=int, default=625, help="Per-cat test and val size")
    ap.add_argument("--n-train-splits", type=int, default=10)
    args = ap.parse_args()

    rows = _load(args.src)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r.get("category") or "unknown")].append(r)

    rng = random.Random(args.seed)
    test: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    per_cat: dict[str, Any] = {}

    for cat in sorted(by):
        pool = list(by[cat])
        rng.shuffle(pool)
        hold = min(args.per_cat_hold, max(0, len(pool) // 5))  # never take >20% each
        hold = min(hold, args.per_cat_hold, len(pool) // 3 if len(pool) >= 3 else 0)
        # Prefer requested hold when enough data (v9 used 625 with large pools)
        if len(pool) >= 2 * args.per_cat_hold + 10:
            hold = args.per_cat_hold
        te = pool[:hold]
        va = pool[hold : 2 * hold]
        tr = pool[2 * hold :]
        test.extend(te)
        val.extend(va)
        train.extend(tr)
        per_cat[cat] = {
            "n_total": len(pool),
            "n_test": len(te),
            "n_val": len(va),
            "n_train_pool": len(tr),
        }

    rng.shuffle(train)
    rng.shuffle(test)
    rng.shuffle(val)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "train.jsonl", train)
    _write(out / "val.jsonl", val)
    _write(out / "test.jsonl", test)
    _write(out / "eval.jsonl", test)  # held-out alias, same as v9

    split_sizes: list[int] = []
    if args.n_train_splits > 0 and train:
        n = args.n_train_splits
        for i in range(n):
            shard = train[i::n]
            _write(out / f"train_{i:02d}.jsonl", shard)
            split_sizes.append(len(shard))

    # Symlink images dir next to splits if source has a sibling images/
    src_images = args.src.parent / "images"
    if not src_images.is_dir():
        # balanced out may live under qwen_training/.../dataset.jsonl with images/
        src_images = args.src.parent / "images"
    # Also try dataset root images via first row path
    dst_images = out / "images"
    if not dst_images.exists():
        # Prefer balanced pack images; else original procthor images
        candidates = [
            args.src.parent / "images",
            Path(
                os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v10/som/images"
            ),
        ]
        for c in candidates:
            if c.is_dir():
                dst_images.symlink_to(c.resolve())
                break

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_balanced": str(args.src),
        "out_dir": str(out),
        "seed": args.seed,
        "per_cat_hold": args.per_cat_hold,
        "n_train_splits": args.n_train_splits,
        "n_test": len(test),
        "n_val": len(val),
        "n_train": len(train),
        "train_split_sizes": split_sizes,
        "per_category": per_cat,
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
