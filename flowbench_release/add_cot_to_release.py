#!/usr/bin/env python3
"""Add the three reasoning annotations to the FlowBench sim release (after integrate_c0_v11.py).

New row fields, in dataset.jsonl of every style and in the parquet shards:
  cot_gt   c_gt  rule-based reasoning from simulator metadata
  cot_1    c_1   VLM reasoning written from the before AND after images (+ c_gt)
  cot_0    c_0   VLM reasoning written from the before image and the motion only (c_1 as a private hint)
They are set when the id has an accepted c_0 (procthor_v11/filters/c0_ids.txt). The CoTs were written for
the natural question, so colour_som / red_som rows get them only if the SoM question names the same objects
(filters/som_cot_compatible_ids.txt); otherwise null. Test (iTHOR) rows get null fields.

Parquet shards are rewritten with the three columns appended (image bytes copied as they are, no re-encoding)
and the HF feature schema updated. Old jsonl files are kept in sim/<split>/_build/pre_cot/.

    python flowbench_release/add_cot_to_release.py --workers 24
"""
from __future__ import annotations

import argparse, json, os, shutil, time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SIM = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim")
V11 = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
STYLES = ("natural", "colour_som", "red_som")
FIELDS = ("cot_gt", "cot_1", "cot_0")
FILES = {"cot_gt": "cold", "cot_1": "warm", "cot_0": "pretended"}


def read_cot(rid: str, kind: str) -> str:
    return (V11 / "cot" / f"{rid}_{kind}.txt").read_text().replace("<|endoftext|>", "").strip()


def cot_table() -> dict:
    ok = set(open(V11 / "filters/c0_ids.txt").read().split())
    compat = set(open(V11 / "filters/som_cot_compatible_ids.txt").read().split())
    cots = {rid: {f: read_cot(rid, FILES[f]) for f in FIELDS} for rid in ok}
    return {"natural": cots, "som": {rid: v for rid, v in cots.items() if rid in compat}}


def patch_jsonl(split: str, style: str, table: dict) -> int:
    p = SIM / split / style / "dataset.jsonl"
    bak = SIM / split / "_build" / "pre_cot" / style / "dataset.jsonl"
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copyfile(p, bak)
    src = table["natural"] if style == "natural" else table["som"]
    n = 0
    tmp = p.with_suffix(".jsonl.tmp")
    with open(bak) as fi, open(tmp, "w") as fo:
        for line in fi:
            r = json.loads(line)
            c = src.get(r["id"]) if split == "train" else None
            for f in FIELDS:
                r[f] = c[f] if c else None
            n += c is not None
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return n


def patch_shard(job) -> str:
    path, style, split = job
    t = pq.read_table(path)
    if "cot_0" in t.column_names:
        return f"skip {path.name}"
    src = _TABLE["natural"] if style == "natural" else _TABLE["som"]
    ids = t.column("id").to_pylist()
    for f in FIELDS:
        vals = [(src.get(i) or {}).get(f) if split == "train" else None for i in ids]
        t = t.append_column(pa.field(f, pa.string()), pa.array(vals, pa.string()))
    md = dict(t.schema.metadata or {})
    hf = json.loads(md[b"huggingface"])
    for f in FIELDS:
        hf["info"]["features"][f] = {"dtype": "string", "_type": "Value"}
    md[b"huggingface"] = json.dumps(hf).encode()
    t = t.replace_schema_metadata(md)
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, row_group_size=250, compression="zstd")
    os.replace(tmp, path)
    return f"patched {style}/{path.name}"


_TABLE: dict = {}


def init(table):
    global _TABLE
    _TABLE = table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-parquet", action="store_true")
    a = ap.parse_args()
    table = cot_table()
    print(f"CoTs: natural {len(table['natural'])}, SoM {len(table['som'])}", flush=True)
    for split in ("train", "test"):
        for style in STYLES:
            print(f"jsonl {split}/{style}: {patch_jsonl(split, style, table)} rows with CoT", flush=True)
    if a.skip_parquet:
        return
    jobs = [(p, style, p.name.split("-")[0]) for style in STYLES for p in sorted((SIM / "parquet" / style).glob("*.parquet"))]
    t0 = time.time()
    init(table)                      # fork start: workers share the table copy-on-write
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(patch_shard, jobs)):
            if i % 20 == 0 or i == len(jobs) - 1:
                print(f"{time.strftime('%H:%M:%S')} [{i + 1}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
