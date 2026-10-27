#!/usr/bin/env python3
"""Write the recovered c_gt (recover_cgt.py) and the fixed CoT availability into procthor_v11 and the release.

procthor_v11 (FlowBench-Sim-Train as used for training):
  cot/<id>_cold.txt          <- recovered c_gt in natural wording (old files moved to cot/_orig_cold/)
  som/dataset.jsonl, train0/1/2/val/test.jsonl   (natural/ holds only per-sample symlinks, no jsonl)
                             cot_cold / cot_warm / cot_pretended filled inline (backups in _pre_cgt/):
                             cot_cold = c_gt in the row's own wording (SoM rows: colour marks);
                             cot_warm / cot_pretended = c_1 / c_0 where the id has an accepted c_0 and
                             (SoM rows) the SoM question names the same objects (filters/som_cot_compatible_ids.txt)
Release ($FLOWBENCH_RELEASE_ROOT/sim), train and test, all three styles:
  cot_gt = recovered c_gt in that style's wording; cot_1 / cot_0 as above (test: null)
  dataset.jsonl rebuilt from sim/<split>/_build/pre_cot/<style>/dataset.jsonl; parquet columns replaced.

Run after integrate_c0_v11.py (which writes the filters).   python apply_cgt.py --workers 16
"""
from __future__ import annotations

import argparse, json, os, shutil, time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

V = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
SIM = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim")
CGT = Path("outputs/cgt")
STYLES = ("natural", "colour_som", "red_som")
FIELDS = ("cot_gt", "cot_1", "cot_0")


def read(p):
    return p.read_text().replace("<|endoftext|>", "").strip()


def load_tables():
    cgt = {"train": {}, "test": {}}
    for split in cgt:
        for line in open(CGT / f"cgt_{split}.jsonl"):
            r = json.loads(line)
            cgt[split][r["id"]] = {st: r.get(st) for st in STYLES}
    ok = set(open(V / "filters/c0_ids.txt").read().split())
    compat = set(open(V / "filters/som_cot_compatible_ids.txt").read().split())
    vlm = {rid: (read(V / "cot" / f"{rid}_warm.txt"), read(V / "cot" / f"{rid}_pretended.txt")) for rid in ok}
    return cgt, ok, compat, vlm


def fields_for(rid, style, split, T):
    cgt, ok, compat, vlm = T
    c = cgt[split].get(rid, {}).get(style)
    c1 = c0 = None
    if split == "train" and rid in ok and (style == "natural" or rid in compat):
        c1, c0 = vlm[rid]
    return {"cot_gt": c, "cot_1": c1, "cot_0": c0}


def rewrite_jsonl(src: Path, dst: Path, style: str, split: str, T, v11_names=False):
    n = {"cot_gt": 0, "cot_1": 0, "cot_0": 0, "rows": 0}
    tmp = dst.with_suffix(".jsonl.tmp")
    with open(src) as fi, open(tmp, "w") as fo:
        for line in fi:
            r = json.loads(line)
            f = fields_for(r["id"], style, split, T)
            if v11_names:
                r["cot_cold"], r["cot_warm"], r["cot_pretended"] = f["cot_gt"], f["cot_1"], f["cot_0"]
            else:
                r.update(f)
            for k in FIELDS:
                n[k] += f[k] is not None
            n["rows"] += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dst)
    return n


def backup(p: Path, root: Path):
    b = root / "_pre_cgt" / p.relative_to(root)
    b.parent.mkdir(parents=True, exist_ok=True)
    if not b.exists():
        shutil.copyfile(p, b)
    return b


_T = None


def patch_shard(job):
    path, style, split = job
    t = pq.read_table(path)
    t = t.drop_columns([c for c in FIELDS if c in t.column_names])
    ids = t.column("id").to_pylist()
    for f in FIELDS:
        t = t.append_column(pa.field(f, pa.string()), pa.array([fields_for(i, style, split, _T)[f] for i in ids], pa.string()))
    md = dict(t.schema.metadata or {})
    hf = json.loads(md[b"huggingface"])
    for f in FIELDS:
        hf["info"]["features"][f] = {"dtype": "string", "_type": "Value"}
    md[b"huggingface"] = json.dumps(hf).encode()
    t = t.replace_schema_metadata(md)
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, row_group_size=250, compression="zstd")
    os.replace(tmp, path)
    return f"{style}/{path.name}"


def main():
    global _T
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    T = load_tables()
    _T = T
    cgt = T[0]
    print(f"c_gt train {sum(1 for v in cgt['train'].values() if v['natural'])}, test {sum(1 for v in cgt['test'].values() if v['natural'])}", flush=True)
    # v11: cold files in natural wording
    # (ids without a consistent c_gt keep no _cold.txt; their original is in _orig_cold/). The _DONE marker keeps a
    # rerun from moving the new files into _orig_cold/.
    orig = V / "cot" / "_orig_cold"
    orig.mkdir(exist_ok=True)
    if (orig / "_DONE").exists():
        print("v11 cot/*_cold.txt: already rewritten (cot/_orig_cold/_DONE)", flush=True)
    else:
        n_cold = 0
        for rid, v in cgt["train"].items():
            dst = V / "cot" / f"{rid}_cold.txt"
            if dst.exists() and not (orig / dst.name).exists():
                shutil.move(dst, orig / dst.name)
            if v["natural"]:
                dst.write_text(v["natural"] + "\n"); n_cold += 1
        (orig / "_DONE").write_text(f"{n_cold} c_gt files written {time.strftime('%Y-%m-%d %H:%M')}\n")
        print(f"v11 cot/*_cold.txt written: {n_cold} (originals in {orig})", flush=True)
    # v11 jsonl
    for p, style in [(V / "som/dataset.jsonl", "colour_som"),
                     *[(V / f"{s}.jsonl", "colour_som") for s in ("train0", "train1", "train2", "val", "test")]]:
        if not p.exists() and not (V / "_pre_cgt" / p.relative_to(V)).exists():
            print(f"v11 {p.relative_to(V)}: missing, skipped", flush=True)
            continue
        b = backup(p, V)
        print(f"v11 {p.relative_to(V)}: {rewrite_jsonl(b, p, style, 'train', T, v11_names=True)}", flush=True)
    # release jsonl
    for split in ("train", "test"):
        for style in STYLES:
            src = SIM / split / "_build" / "pre_cot" / style / "dataset.jsonl"
            print(f"release {split}/{style}: {rewrite_jsonl(src, SIM / split / style / 'dataset.jsonl', style, split, T)}", flush=True)
    # release parquet
    jobs = [(p, st, p.name.split("-")[0]) for st in STYLES for p in sorted((SIM / "parquet" / st).glob("*.parquet"))]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(patch_shard, jobs)):
            if i % 50 == 0 or i == len(jobs) - 1:
                print(f"{time.strftime('%H:%M:%S')} [{i + 1}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
