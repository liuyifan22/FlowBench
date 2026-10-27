#!/usr/bin/env python3
"""Build the FlowBench *sim* release: ProcTHOR v11 (train) and iTHOR v11 (test), each with three image
styles (natural / colour SoM / red SoM) and two question types (question / static_question).

  $FLOWBENCH_RELEASE_ROOT/sim/<split>/
      natural/{dataset.jsonl, images/}        clean renders
      colour_som/{dataset.jsonl, images/}     text-free colour discs, "the bowl (marked blue)"
      red_som/{dataset.jsonl, images/}        red numbered / lettered badges, "the bowl (1)"
      samples/<sample_id>/                    per-sample GT (sample.json, flow / depth / seg / masks)

Sources (nothing is regenerated with AI2-THOR):
  colour SoM  = the v11 rows (reference text: all v11 filters and fixes, balanced answer letters)
  red SoM     = v9/v10 ``som/dataset_static_som.jsonl`` + the same text fixes + the colour row's
                ``choice_perm``; images hardlinked, except reshuffled object-choice rows (closest_three,
                iTHOR closer_to) whose lettered badges are redrawn with the new letters
  natural     = iTHOR: ithor_v11/natural. ProcTHOR: v9/v10 ``natural/dataset.jsonl`` + the v11 fixes,
                re-ordered into the SoM choice order when both exports ask about the same objects
                (``choices_match_som``), otherwise its own balanced letter shuffle
Natural images, red images and GT are hardlinks into the $CIT_DATA_ROOT sources (no extra space); colour images
are copied from procthor_v11 (``--stage swap-colour`` then turns the ProcTHOR source folder into a symlink).

  python flowbench_release/build_sim_release.py --split test  --stage text
  python flowbench_release/build_sim_release.py --split train --stage files --workers 48
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v11_colorsom_ithor import name_dangling_it  # noqa: E402
from cit_pipeline.color_som import (  # noqa: E402
    CHOICE_KEYS, OBJECT_CHOICE_CATS, TEXT_FIELDS, find_question, name_generic_mover, strip_move_prefix)
from reshuffle_v11_answer_letters import apply_perm, plan  # noqa: E402
from text_styles import colourize_red, letters_of, natural_row, red_row  # noqa: E402

A = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
B = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
OUT = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release"))
STYLES = ("natural", "colour_som", "red_som")
NATURAL_SEED = 0

SPLITS = {
    "train": {  # ProcTHOR v11
        "colour": A / "procthor_v11/som/dataset.jsonl",
        "colour_root": A / "procthor_v11/som",
        "internal_splits": {s: A / f"procthor_v11/{s}.jsonl" for s in ("train0", "train1", "train2", "val", "test")},
        "red_src": {v: B / v / "som/dataset_static_som.jsonl" for v in ("procthor_v9", "procthor_v10")},
        "natural_src": {v: B / v / "natural/dataset.jsonl" for v in ("procthor_v9", "procthor_v10")},
    },
    "test": {  # iTHOR v11 (ithor_v10 merge holds both v9 and v10 rows)
        "colour": A / "ithor_v11/som/dataset.jsonl",
        "colour_root": A / "ithor_v11/som",
        "red_src": {"ithor_v10": B / "ithor_v10/som/dataset_static_som.jsonl"},
        "natural_v11": A / "ithor_v11/natural/dataset.jsonl",
        "natural_root": B / "ithor_v10/natural",
    },
}

# Release row fields (style-specific extras are added per style).
COMMON = ("id", "sample_id", "scene", "view_id", "question_id", "category", "mcq_type", "n_choices",
          "image_before", "image_after", "motion_command", "question", "static_question",
          *CHOICE_KEYS, "answer", "source_version", "mover_object_id", "internal_split",
          "occludee_px", "occlusion_cover", "occlusion_relabelled")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_jsonl(path: Path, keep: set[str] | None = None) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep is None or r["id"] in keep:
                out[r["id"]] = {k: v for k, v in r.items() if not k.startswith("cot")}
    return out


def sample_dir(split: str, row: dict) -> Path:
    if split == "train":
        return B / row["source_version"] / "natural" / row["sample_id"]
    for v in ("ithor_v10", "ithor_v9"):
        d = B / v / "natural" / row["sample_id"]
        if (d / "sample.json").is_file():
            return d
    raise FileNotFoundError(row["sample_id"])


def stem(r: dict) -> str:
    return f"{r['sample_id']}_{r['view_id']}"


def release_row(r: dict, style: str, extra: dict) -> dict:
    s = stem(r)
    suffix = "" if style == "natural" else "_som"
    out = {k: r.get(k) for k in COMMON}
    out["image_before"] = f"images/{s}_rgb0{suffix}.png"
    out["image_after"] = f"images/{s}_rgb1{suffix}.png"
    out.update(extra)
    return out


# ----------------------------------------------------------------------------------------------- text

def process_sample(job: tuple) -> list[dict]:
    """All rows of one sample: red + natural rows, plus the file plan. Runs in a worker."""
    split, rows = job
    sdir = sample_dir(split, rows[0]["colour"])
    sample = json.loads((sdir / "sample.json").read_text())
    fix_static = (lambda r: strip_move_prefix(name_dangling_it(r)))
    out = []
    for item in rows:
        c, red_src, nat_src = item["colour"], item["red"], item.get("natural")
        view, q = find_question(sample, c)
        gm = q.get("gt_meta") or {}
        res = {"id": c["id"], "sample_dir": str(sdir), "view": {k: view[k] for k in ("rgb0_path", "rgb1_path")},
               "gm_marks": {i: gm.get(f"som_marks_rgb{i}") for i in (0, 1)}}
        red = red_row(red_src, c, gm)
        if red is not None and split == "test":  # iTHOR v11 also named the dangling "it" in SoM rows
            red["static_question"] = name_dangling_it(red)
        res["red"] = red
        if red is not None:  # the two SoM styles must say the same thing
            exp = colourize_red(red, c)
            for k in ("question", "static_question"):
                exp[k] = name_generic_mover(c, exp[k])
            res["red_mismatch"] = [k for k in TEXT_FIELDS + CHOICE_KEYS if exp.get(k) != c.get(k)] + (
                ["answer"] if red["answer"] != c["answer"] else [])
        if nat_src is not None:
            nat, status = natural_row(nat_src, c, gm, fix_static)
            res["natural"], res["natural_status"] = nat, status
        out.append(res)
    return out


def build_text(split: str, workers: int) -> None:
    cfg = SPLITS[split]
    root = OUT / "sim" / split
    colour = load_jsonl(cfg["colour"])
    ids = set(colour)
    log(f"{split}: {len(colour)} colour rows")
    internal = {}
    for name, p in cfg.get("internal_splits", {}).items():
        for line in open(p):
            internal[json.loads(line)["id"]] = name
    red_src = {}
    for v, p in cfg["red_src"].items():
        red_src.update(load_jsonl(p, ids))
    log(f"{split}: red source rows {len(red_src)}")
    nat_src = {}
    for v, p in cfg.get("natural_src", {}).items():
        nat_src.update(load_jsonl(p, ids))
    if "natural_v11" in cfg:  # iTHOR v11 natural rows already carry the v11 fixes (re-applying is a no-op)
        nat_src = load_jsonl(cfg["natural_v11"], ids)
    if nat_src:
        log(f"{split}: natural source rows {len(nat_src)}")

    by_sample = collections.defaultdict(list)
    for i, c in colour.items():
        c["internal_split"] = internal.get(i)
        by_sample[c["sample_id"]].append({"colour": c, "red": red_src[i], "natural": nat_src.get(i)})
    jobs = [(split, rows) for _, rows in sorted(by_sample.items())]
    results = {}
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(process_sample, jobs, chunksize=16)):
            for r in res:
                results[r["id"]] = r
            if k % 5000 == 0:
                log(f"{split}: text {k}/{len(jobs)} samples")

    stats = collections.Counter()
    bad = [r for r in results.values() if r["red"] is None or r.get("red_mismatch")]
    stats["red_mismatch"] = len(bad)
    if bad:
        (root / "_build").mkdir(parents=True, exist_ok=True)
        with open(root / "_build" / "red_mismatch.jsonl", "w") as f:
            for r in bad[:2000]:
                f.write(json.dumps({"id": r["id"], "fields": r.get("red_mismatch"), "red": r["red"],
                                    "colour": colour[r["id"]]}) + "\n")
    # natural rows
    if True:
        natural = {}
        for i, r in results.items():
            n = r.get("natural")
            st = r.get("natural_status")
            stats[f"natural:{colour[i]['category']}:{st}"] += 1
            if n is None:
                continue
            natural[i] = dict(n, choices_match_som=st in ("aligned", "aligned_by_answer"))
        # natural rows asking about other objects keep their own choices: balance their letters
        # (iTHOR v11 natural letters were already balanced by reshuffle_v11_answer_letters.py)
        own = [v for v in natural.values() if not v["choices_match_som"]]
        if split == "train":
            for i, p in plan(own, None, NATURAL_SEED).items():
                natural[i] = apply_perm(natural[i], p)
    keep = [i for i in sorted(colour) if results[i]["red"] is not None and not results[i].get("red_mismatch")
            and i in natural]
    stats["kept"] = len(keep)
    stats["dropped"] = len(colour) - len(keep)
    log(f"{split}: {dict(stats)}")

    # write datasets + file plan
    for st in STYLES:
        (root / st / "images").mkdir(parents=True, exist_ok=True)
    plan_rows = []
    with open(root / "natural/dataset.jsonl", "w") as fn, open(root / "colour_som/dataset.jsonl", "w") as fc, \
            open(root / "red_som/dataset.jsonl", "w") as fr:
        for i in keep:
            c, res, n = colour[i], results[i], natural[i]
            red = res["red"]
            perm = c.get("choice_perm") or {}
            relabel = c["category"] in OBJECT_CHOICE_CATS and any(k != v for k, v in perm.items())
            labels = {m["id"]: m["label"] for k in (0, 1) for m in (res["gm_marks"][k] or [])}
            if relabel:  # choice mark of source letter S now carries its new letter
                inv = {s: new for new, s in perm.items()}
                labels = {mid: inv.get(mid, lab) for mid, lab in labels.items()}
            base = {**c, "internal_split": c.get("internal_split")}
            fc.write(json.dumps(release_row(base, "colour_som", {"som_colors": c.get("som_colors")})) + "\n")
            fr.write(json.dumps(release_row({**base, **{k: red.get(k) for k in TEXT_FIELDS + CHOICE_KEYS + ("answer",)}},
                                            "red_som", {"som_labels": labels, "som_relabelled": relabel})) + "\n")
            fn.write(json.dumps(release_row({**base, **{k: n.get(k) for k in TEXT_FIELDS + CHOICE_KEYS + ("answer",)},
                                             "occlusion_relabelled": n.get("occlusion_relabelled", base.get("occlusion_relabelled"))},
                                            "natural", {"choices_match_som": n["choices_match_som"]})) + "\n")
            src_nat = (B / c["source_version"] / "natural" if split == "train" else cfg["natural_root"])
            src_red = (B / c["source_version"] / "som" if split == "train" else B / "ithor_v10/som")
            plan_rows.append({
                "id": i, "stem": stem(c), "sample_id": c["sample_id"], "view_id": c["view_id"],
                "sample_dir": res["sample_dir"],
                "natural": [str(src_nat / n[k]) for k in ("image_before", "image_after")],
                "colour": [str(cfg["colour_root"] / c[k]) for k in ("image_before", "image_after")],
                "red": [str(src_red / red_src[i][k]) for k in ("image_before", "image_after")],
                "relabel": relabel, "view": res["view"],
                "marks": [[dict(m, label=labels.get(m["id"], m["label"])) for m in (res["gm_marks"][k] or [])]
                          for k in (0, 1)],
            })
    (root / "_build").mkdir(parents=True, exist_ok=True)
    with open(root / "_build/file_plan.jsonl", "w") as f:
        for p in plan_rows:
            f.write(json.dumps(p) + "\n")
    (root / "_build/text_stats.json").write_text(json.dumps(
        {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **stats}, indent=1))
    log(f"{split}: wrote {len(keep)} rows x 3 styles")


# ---------------------------------------------------------------------------------------------- files

def _link(src: str, dst: Path) -> str:
    if dst.exists():
        return "exists"
    os.link(src, dst)
    return "linked"


def _redraw(view_path: str, marks: list[dict], dst: Path) -> str:
    """Red badges with the release letters; same style as the v9/v10 renders (radius 10, 13 px bold)."""
    if dst.exists():
        return "exists"
    from PIL import Image, ImageDraw, ImageFont
    from cit_pipeline.som import SOM_MARK_FILL, SOM_MARK_OUTLINE, SOM_TEXT_FILL
    img = Image.open(view_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    for m in marks:
        u, v = float(m["uv"][0]), float(m["uv"][1])
        d.ellipse([u - 10, v - 10, u + 10, v + 10], fill=tuple(SOM_MARK_FILL), outline=tuple(SOM_MARK_OUTLINE), width=2)
        lab = str(m.get("label") or m["id"])
        tw, th = d.textbbox((0, 0), lab, font=font)[2:]
        d.text((u - tw / 2, v - th / 2 + 0.5), lab, fill=tuple(SOM_TEXT_FILL), font=font)
    tmp = dst.with_suffix(".tmp.png")
    img.save(tmp)
    os.replace(tmp, dst)
    return "drawn"


def _copy(src: str, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size == os.path.getsize(src):
        return "exists"
    tmp = dst.with_suffix(".tmp" + dst.suffix)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def files_for_sample(job: tuple) -> collections.Counter:
    split, items = job
    root = OUT / "sim" / split
    st = collections.Counter()
    sdir = Path(items[0]["sample_dir"])
    gt_dst = root / "samples" / items[0]["sample_id"]
    gt_dst.mkdir(parents=True, exist_ok=True)
    st["gt_" + _link(str(sdir / "sample.json"), gt_dst / "sample.json")] += 1
    names = os.listdir(sdir)
    for it in items:
        s = it["stem"]
        for k, suf in ((0, "rgb0"), (1, "rgb1")):
            st["natural_" + _link(it["natural"][k], root / "natural/images" / f"{s}_{suf}.png")] += 1
            st["colour_" + _copy(it["colour"][k], root / "colour_som/images" / f"{s}_{suf}_som.png")] += 1
            dst = root / "red_som/images" / f"{s}_{suf}_som.png"
            if it["relabel"]:
                st["red_" + _redraw(it["view"][f"{suf}_path"], it["marks"][k], dst)] += 1
            else:
                st["red_" + _link(it["red"][k], dst)] += 1
        for n in names:  # GT files of this view
            if n.startswith(it["view_id"] + "_"):
                st["gt_" + _link(str(sdir / n), gt_dst / n)] += 1
    return st


def build_files(split: str, workers: int) -> None:
    root = OUT / "sim" / split
    (root / "samples").mkdir(parents=True, exist_ok=True)
    by_sample = collections.defaultdict(list)
    for line in open(root / "_build/file_plan.jsonl"):
        p = json.loads(line)
        by_sample[p["sample_id"]].append(p)
    jobs = [(split, v) for _, v in sorted(by_sample.items())]
    tot = collections.Counter()
    t0 = time.time()
    with Pool(workers) as pool:
        for k, st in enumerate(pool.imap_unordered(files_for_sample, jobs, chunksize=4)):
            tot.update(st)
            if k % 2000 == 0:
                log(f"{split}: files {k}/{len(jobs)} samples {dict(tot)} ({time.time() - t0:.0f}s)")
    log(f"{split}: files done {dict(tot)}")
    (root / "_build/file_stats.json").write_text(json.dumps(dict(tot), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--stage", choices=("text", "files"), required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    if args.stage == "text":
        build_text(args.split, args.workers)
    else:
        build_files(args.split, args.workers)


if __name__ == "__main__":
    main()
