#!/usr/bin/env python3
"""Build ProcTHOR v11 colour-SoM MCQ set by merging v9 + v10 (no AI2-THOR / VLM).

v11 = v9 {closer_to, closest_three, count_near, left_right_landmark,
near_object, occlusion} + v10 {feasibility, contact_wall} (v10 supersedes
v9 for those two). Rows come from each source's ``som/dataset_static_som.jsonl``
and are filtered to ``gt_complete`` + letter answer, as for the v9 splits.
Images are re-drawn with colour SoM marks (``cit_pipeline.color_som``).

Layout::

  <out>/som/images/*_rgb{0,1}_som.png
  <out>/som/dataset.jsonl          # all usable rows
  <out>/som/dataset_unresolved.jsonl  # rows whose text references an undrawn mark
  <out>/som/dataset_removed_structural.jsonl  # window/door/room/cabinet/drawer/sink/faucet movers
  <out>/som/build_meta.json
  <out>/{train0,train1,train2,val,test}.jsonl + cit_psi2_split_meta.json

Resumable: per-shard outputs in ``som/_shards``; finished shards are skipped.

  python build_v11_colorsom_procthor.py --workers 48
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import random
import re
import time
import traceback
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from cit_pipeline.color_som import colorize_row
from cit_pipeline.som import SOM_COLOR_PALETTE, SOM_COLOR_RADIUS, SOM_COLOR_RIM

SRC_ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
OUT_ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
SOURCES: dict[str, tuple[str, ...]] = {
    "procthor_v9": (
        "closer_to",
        "closest_three",
        "count_near",
        "left_right_landmark",
        "near_object",
        "occlusion",
    ),
    "procthor_v10": ("feasibility", "contact_wall"),
}
SRC_JSONL = "som/dataset_static_som.jsonl"
SPLIT_SIZES = {"test": 5000, "val": 2000}
N_TRAIN_SPLITS = 3

# Filled in the parent before forking; workers read it copy-on-write.
_SHARDS: list[list[tuple[str, dict[str, Any]]]] = []
_OUT: Path = OUT_ROOT


# Movers that are part of the room structure or of fixed furniture are dropped:
# windows, doors (doorway / doorframe), the room shell ("the floor"), cabinet /
# drawer parts, sinks, faucets and countertops.
STRUCTURAL_MOVER_TYPES = {"window", "door", "room", "wall", "sink", "sinkbasin", "faucet", "countertop"}
STRUCTURAL_MOVER_PHRASES = {
    "the window", "the doorway", "the doorframe", "the door", "the cabinet", "the cupboard",
    "the drawer", "the wall", "the sink", "the faucet", "the countertop", "the counter top",
}
_MOVER_PHRASE_RE = re.compile(r"move (.+?) \(marked \w+\)")


def structural_mover(r: dict[str, Any]) -> bool:
    if str(r.get("mover_object_id") or "").split("|")[0].lower() in STRUCTURAL_MOVER_TYPES:
        return True
    m = _MOVER_PHRASE_RE.match(str(r.get("motion_command") or ""))
    return bool(m) and m[1] in STRUCTURAL_MOVER_PHRASES


def usable(r: dict[str, Any]) -> bool:
    return bool(r.get("gt_complete")) and bool(re.fullmatch(r"[A-F]+", str(r.get("answer") or "")))


def load_rows(n_shards: int) -> tuple[list[list], dict[str, Any]]:
    shards: list[list] = [[] for _ in range(n_shards)]
    stats: dict[str, Any] = collections.defaultdict(collections.Counter)
    for src, cats in SOURCES.items():
        with open(SRC_ROOT / src / SRC_JSONL) as f:
            for line in f:
                r = json.loads(line)
                cat = r["category"]
                if cat not in cats:
                    stats["dropped_category"][f"{src}/{cat}"] += 1
                    continue
                if not usable(r):
                    stats["dropped_unusable"][f"{src}/{cat}"] += 1
                    continue
                stats["kept"][f"{src}/{cat}"] += 1
                shards[zlib.crc32(r["sample_id"].encode()) % n_shards].append((src, r))
    return shards, {k: dict(v) for k, v in stats.items()}


def run_shard(idx: int) -> tuple[int, int, int, float]:
    done = _OUT / "som" / "_shards" / f"shard_{idx:04d}.jsonl"
    if done.exists():
        return idx, -1, 0, 0.0
    t0 = time.time()
    tmp = done.with_suffix(".jsonl.tmp")
    n_err = 0
    rows = sorted(_SHARDS[idx], key=lambda sr: sr[1]["sample_id"])
    cache: tuple[str, dict] | None = None
    with open(tmp, "w") as f:
        for src, r in rows:
            try:
                if cache is None or cache[0] != r["sample_id"]:
                    path = SRC_ROOT / src / "natural" / r["sample_id"] / "sample.json"
                    cache = (r["sample_id"], json.loads(path.read_text()))
                new = colorize_row(r, cache[1], _OUT / "som")
                new["source_version"] = src
                f.write(json.dumps(new) + "\n")
            except Exception:
                n_err += 1
                with open(_OUT / "som" / "_logs" / f"errors_{idx:04d}.log", "a") as ef:
                    ef.write(f"{r['id']}\n{traceback.format_exc()}\n")
    os.replace(tmp, done)
    return idx, len(rows), n_err, time.time() - t0


def merge_and_split(n_shards: int, seed: int, load_stats: dict[str, Any]) -> None:
    som = _OUT / "som"
    rows, unresolved, structural = [], [], []
    for i in range(n_shards):
        with open(som / "_shards" / f"shard_{i:04d}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                drop = r.pop("_drop", None)
                if r.pop("_leftover"):
                    unresolved.append(r)
                elif drop:
                    structural.append({**r, "removed_reason": drop})
                elif structural_mover(r):
                    structural.append(r)
                else:
                    rows.append(r)
    order = {src: i for i, src in enumerate(SOURCES)}
    rows.sort(key=lambda r: (order[r["source_version"]], r["id"]))

    def dump(path: Path, rs: list[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    dump(som / "dataset.jsonl", rows)
    dump(som / "dataset_unresolved.jsonl", unresolved)
    dump(som / "dataset_removed_structural.jsonl", structural)

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    splits: dict[str, list[int]] = {}
    pos = 0
    for name, n in SPLIT_SIZES.items():
        splits[name] = idx[pos : pos + n]
        pos += n
    rest = idx[pos:]
    for k in range(N_TRAIN_SPLITS):
        splits[f"train{k}"] = rest[k::N_TRAIN_SPLITS]
    paths = {}
    for name, ids in splits.items():
        paths[name] = str(_OUT / f"{name}.jsonl")
        dump(_OUT / f"{name}.jsonl", [rows[i] for i in sorted(ids)])

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (_OUT / "cit_psi2_split_meta.json").write_text(json.dumps({
        "created_utc": now,
        "source_jsonl": str(som / "dataset.jsonl"),
        "som_root": str(som),
        "split_dir": str(_OUT),
        "seed": seed,
        "n_usable": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "paths": paths,
        "filter": "gt_complete + letter answer + all mark refs drawn + no structural mover",
    }, indent=2))
    per_cat = collections.Counter(f"{r['source_version']}/{r['category']}" for r in rows)
    (som / "build_meta.json").write_text(json.dumps({
        "created_utc": now,
        "sources": {src: {"jsonl": str(SRC_ROOT / src / SRC_JSONL), "categories": list(c)}
                    for src, c in SOURCES.items()},
        "som_style": "color",
        "palette": SOM_COLOR_PALETTE,
        "mark_radius": SOM_COLOR_RADIUS,
        "mark_rim": SOM_COLOR_RIM,
        "text_ref_format": "the <object> (marked <colour>)",
        "load_stats": load_stats,
        "n_rows": len(rows),
        "n_unresolved": len(unresolved),
        "n_removed_structural": len(structural),
        "structural_mover_rule": {"types": sorted(STRUCTURAL_MOVER_TYPES),
                                  "phrases": sorted(STRUCTURAL_MOVER_PHRASES)},
        "rows_per_category": dict(sorted(per_cat.items())),
    }, indent=2))
    print(f"merged {len(rows)} rows ({len(unresolved)} unresolved, "
          f"{len(structural)} structural movers removed) ->", som / "dataset.jsonl")
    print("splits:", {k: len(v) for k, v in splits.items()})


def main() -> None:
    global _SHARDS, _OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--n-shards", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=0, help="only run the first K shards (testing)")
    args = ap.parse_args()

    _OUT = args.out
    for d in ("som/images", "som/_shards", "som/_logs"):
        (_OUT / d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _SHARDS, load_stats = load_rows(args.n_shards)
    print(f"loaded {sum(map(len, _SHARDS))} rows in {time.time() - t0:.0f}s:", json.dumps(load_stats), flush=True)

    todo = list(range(args.limit_shards or args.n_shards))
    n_rows = n_err = n_done = 0
    with Pool(args.workers) as pool:
        for idx, n, e, dt in pool.imap_unordered(run_shard, todo):
            n_done += 1
            if n >= 0:
                n_rows += n
                n_err += e
            print(f"[{n_done}/{len(todo)}] shard {idx} rows={n} err={e} {dt:.0f}s "
                  f"| total rows={n_rows} err={n_err} elapsed={time.time() - t0:.0f}s", flush=True)

    if args.limit_shards:
        print("limit-shards set; skipping merge")
        return
    merge_and_split(args.n_shards, args.seed, load_stats)


if __name__ == "__main__":
    main()
