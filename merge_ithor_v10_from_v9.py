#!/usr/bin/env python3
"""Build ithor_v10 benchmark jsonl: keep v10 contact_wall+feasibility, copy other cats from v9.

Also copies referenced ``images/`` files from v9 into v10 so relative paths resolve.

Usage:
  python merge_ithor_v10_from_v9.py
  python merge_ithor_v10_from_v9.py --dry-run
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

V9 = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v9")
V10 = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v10")
REPLACE = {"contact_wall", "feasibility"}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _copy_images(rows: list[dict], src_root: Path, dst_root: Path, dry: bool) -> int:
    n = 0
    for r in rows:
        for key in ("image_before", "image_after"):
            rel = r.get(key)
            if not rel:
                continue
            src = src_root / rel
            dst = dst_root / rel
            if not src.is_file():
                continue
            if dst.is_file():
                continue
            if dry:
                n += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n


def _merge_split(split: str, dry: bool) -> dict:
    v9_jsonl = V9 / split / "dataset.jsonl"
    v10_jsonl = V10 / split / "dataset.jsonl"
    v9_rows = _load(v9_jsonl)
    v10_rows = _load(v10_jsonl)

    kept_v10 = [r for r in v10_rows if r.get("category") in REPLACE]
    from_v9 = [r for r in v9_rows if r.get("category") not in REPLACE]
    # Drop any stale non-REPLACE rows already in v10 (shouldn't exist for fresh gen).
    out_rows = kept_v10 + from_v9

    img_copied = _copy_images(from_v9, V9 / split, V10 / split, dry=dry)

    stats = {
        "split": split,
        "v10_replace_rows": len(kept_v10),
        "v9_other_rows": len(from_v9),
        "out_rows": len(out_rows),
        "images_copied": img_copied,
        "by_cat": dict(Counter(r.get("category") for r in out_rows)),
        "v10_replace_by_cat": dict(Counter(r.get("category") for r in kept_v10)),
    }

    if not dry:
        bak = v10_jsonl.with_name(v10_jsonl.name + ".bak_pre_v9_merge")
        if v10_jsonl.exists():
            shutil.copy2(v10_jsonl, bak)
            stats["backup"] = str(bak)
        _dump(v10_jsonl, out_rows)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = {}
    for split in ("natural", "som"):
        results[split] = _merge_split(split, dry=args.dry_run)
        print(f"=== {split} {'(dry-run)' if args.dry_run else ''} ===")
        for k, v in results[split].items():
            print(f"  {k}: {v}")

    out = Path("outputs/ithor_v10_merge_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
