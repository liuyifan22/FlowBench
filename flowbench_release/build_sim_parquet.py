#!/usr/bin/env python3
"""Hugging Face parquet shards for the FlowBench sim release (run after build_sim_release.py).

  sim/parquet/<style>/<split>-NNNNN-of-MMMMM.parquet      style in natural / colour_som / red_som

One row per question, both images embedded as HF ``Image`` features (struct of bytes + path), with the
``datasets`` feature schema stored in the parquet metadata, so ``load_dataset(..., name="<style>")``
decodes them. Test images are the original PNG bytes; train images are re-encoded as JPEG q95 to keep
the train parquet at ~1/4 of the PNG size (the PNGs stay in the folder release).

  python flowbench_release/build_sim_parquet.py --split test
  python flowbench_release/build_sim_parquet.py --split train --workers 48
"""

from __future__ import annotations

import os
import argparse
import io
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SIM = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim")
STYLES = ("natural", "colour_som", "red_som")
ROWS_PER_SHARD = {"train": 2500, "test": 1000}
ROW_GROUP = 250
JPEG_QUALITY = 95

STR = ("id", "sample_id", "scene", "view_id", "question_id", "category", "mcq_type", "motion_command",
       "question", "static_question", *(f"choice_{c}" for c in "ABCDEF"), "answer", "source_version",
       "mover_object_id", "internal_split")
EXTRA = {  # style -> (column, arrow type, HF dtype); dict-valued fields are stored as JSON strings
    "natural": [("choices_match_som", pa.bool_(), "bool")],
    "colour_som": [("som_colors", pa.string(), "string")],
    "red_som": [("som_labels", pa.string(), "string"), ("som_relabelled", pa.bool_(), "bool")],
}
IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def schema(style: str) -> pa.Schema:
    fields = [pa.field(k, pa.string()) for k in STR[:2]]
    fields += [pa.field("image_before", IMAGE), pa.field("image_after", IMAGE)]
    fields += [pa.field(k, pa.string()) for k in STR[2:]]
    fields += [pa.field("n_choices", pa.int32()), pa.field("occludee_px", pa.int32()),
               pa.field("occlusion_cover", pa.float32()), pa.field("occlusion_relabelled", pa.bool_())]
    fields += [pa.field(k, t) for k, t, _ in EXTRA[style]]
    # reasoning annotations (add_cot_to_release.py): c_gt, c_1, c_0; null where not provided
    fields += [pa.field(k, pa.string()) for k in ("cot_gt", "cot_1", "cot_0")]
    hf = {}
    for f in fields:
        if f.name.startswith("image_"):
            hf[f.name] = {"_type": "Image"}
        else:
            dt = {pa.string(): "string", pa.int32(): "int32", pa.float32(): "float32", pa.bool_(): "bool"}[f.type]
            hf[f.name] = {"dtype": dt, "_type": "Value"}
    return pa.schema(fields, metadata={"huggingface": json.dumps({"info": {"features": hf}})})


def image_cell(path: Path, jpeg: bool) -> dict:
    if not jpeg:
        return {"bytes": path.read_bytes(), "path": path.name}
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return {"bytes": buf.getvalue(), "path": path.with_suffix(".jpg").name}


def write_shard(job: tuple) -> str:
    split, style, k, n, rows = job
    out = SIM / "parquet" / style / f"{split}-{k:05d}-of-{n:05d}.parquet"
    if out.exists():
        return f"exists {out.name}"
    base = SIM / split / style
    sch = schema(style)
    cols: dict = {f.name: [] for f in sch}
    for r in rows:
        for key in ("image_before", "image_after"):
            cols[key].append(image_cell(base / r[key], jpeg=(split == "train")))
        for f in sch:
            if f.name in ("image_before", "image_after"):
                continue
            v = r.get(f.name)
            cols[f.name].append(json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
    table = pa.table(cols, schema=sch)
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, row_group_size=ROW_GROUP, compression="zstd")
    tmp.rename(out)
    return f"wrote {out.name} ({out.stat().st_size / 1e6:.0f} MB)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    jobs = []
    for style in STYLES:
        (SIM / "parquet" / style).mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in open(SIM / args.split / style / "dataset.jsonl")]
        step = ROWS_PER_SHARD[args.split]
        n = (len(rows) + step - 1) // step
        jobs += [(args.split, style, k, n, rows[k * step:(k + 1) * step]) for k in range(n)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(write_shard, jobs)):
            print(f"{time.strftime('%H:%M:%S')} [{i + 1}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
