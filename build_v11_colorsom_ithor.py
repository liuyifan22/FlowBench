#!/usr/bin/env python3
"""Build iTHOR v11 (colour SoM) from the ithor_v10 merge (v10 feasibility/contact_wall + v9 rest).

Rows come from ``ithor_v10/som/dataset_static_som.jsonl`` (SoM) and
``ithor_v10/natural/dataset.jsonl`` (natural, same ids). Kept rows: gt_complete +
letter answer, every mark reference drawn, and no structural mover (procthor_v11 rule
plus shelves, which cannot move in iTHOR). The same ids are kept in natural and som.

Layout (see README.md written next to it)::

  <out>/natural/dataset.jsonl   natural rows            natural/images/*_rgb{0,1}.png
  <out>/natural/samples/<sample_id>/   GT sample dirs (sample.json, flow/depth .npy, seg, masks)
  <out>/som/dataset.jsonl       colour-SoM rows         som/images/*_rgb{0,1}_som.png
  <out>/som/removed.jsonl       dropped SoM rows + reason
  <out>/build_meta.json, README.md, gallery.html

  python build_v11_colorsom_ithor.py
"""

from __future__ import annotations

import os
import argparse
import collections
import datetime
import html
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from build_v11_colorsom_procthor import STRUCTURAL_MOVER_TYPES, structural_mover, usable
from cit_pipeline.color_som import CHOICE_KEYS, colorize_row, find_question, fix_occlusion_row
from cit_pipeline.som import SOM_COLOR_PALETTE, SOM_COLOR_RADIUS, SOM_COLOR_RIM

SRC_ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
SRC = SRC_ROOT / "ithor_v10"
SAMPLE_SOURCES = ("ithor_v10", "ithor_v9")  # v9 rows' sample dirs live under ithor_v9/natural
OUT_ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11")
ITHOR_EXTRA_TYPES = {"shelf"}
# Several parts of one piece of furniture ("the shelf", "the shelf", ...) are indistinguishable
# by name, so questions naming the same part type twice or more are dropped.
REPEATED_PART_WORDS = ("shelf", "drawer")


# Natural near_object / count_near static questions refer to the mover as a bare "it"
# ("Is it within 0.50m of the cellphone?"); name it from the motion command instead.
_MOVER_FROM_MOTION = re.compile(r"^move (.+?) -?\d+(?:\.\d+)?\s?m\b", re.IGNORECASE)
DANGLING_IT_CATS = {"near_object", "count_near"}


def name_dangling_it(r: dict[str, Any]) -> str | None:
    q = r.get("static_question")
    if r.get("category") not in DANGLING_IT_CATS or not isinstance(q, str):
        return q
    m = _MOVER_FROM_MOTION.match(str(r.get("motion_command") or ""))
    if not m or len(re.findall(r"\bit\b", q, flags=re.IGNORECASE)) != 1:
        return q
    phrase = m[1]
    out = re.sub(r"\bit\b", phrase, q, count=1, flags=re.IGNORECASE)
    return out[:1].upper() + out[1:]


def repeated_part_ref(r: dict[str, Any]) -> bool:
    text = " ".join(str(r.get(k) or "") for k in ("question",) + CHOICE_KEYS).lower()
    return any(len(re.findall(rf"\bthe {w}\b", text)) >= 2 for w in REPEATED_PART_WORDS)


def sample_dir(sample_id: str) -> Path:
    for v in SAMPLE_SOURCES:
        d = SRC_ROOT / v / "natural" / sample_id
        if (d / "sample.json").is_file():
            return d
    raise FileNotFoundError(sample_id)


def load(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def render_gallery(out: Path, som: list[dict], nat: dict[str, dict], per_cat: int = 3) -> None:
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for r in som:
        by[r["category"]].append(r)
    sw = lambda c: f'<span class="sw" style="background:rgb{SOM_COLOR_PALETTE[c]}"></span>'  # noqa: E731

    def qa(d: dict) -> str:
        ch = "<br>".join(f"&nbsp;&nbsp;{k[-1]}. {html.escape(str(d[k]))}" for k in CHOICE_KEYS if d.get(k))
        return (f"<b>motion:</b> {html.escape(str(d.get('motion_command')))}<br>"
                f"<b>Q:</b> {html.escape(str(d.get('question')))}<br>"
                f"<b>static Q:</b> {html.escape(str(d.get('static_question')))}<br>{ch}"
                f"<br><b>answer:</b> {html.escape(str(d.get('answer')))}")

    cards = []
    for cat in sorted(by):
        for r in rng.sample(by[cat], min(per_cat, len(by[cat]))):
            n = nat[r["id"]]
            cols = " ".join(f"{sw(c)}{m}={c}" for m, c in r["som_colors"].items())
            cards.append(f"""<div class="card"><div class="hdr"><b>{cat}</b> · {html.escape(r['id'])} · {cols}</div>
<table><tr><th>natural before</th><th>natural after</th><th>colour SoM before</th><th>colour SoM after</th></tr>
<tr><td><img src="natural/{n['image_before']}"></td><td><img src="natural/{n['image_after']}"></td>
<td><img src="som/{r['image_before']}"></td><td><img src="som/{r['image_after']}"></td></tr>
<tr><td colspan=2 class="txt">{qa(n)}</td><td colspan=2 class="txt">{qa(r)}</td></tr></table></div>""")
    (out / "gallery.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>ithor_v11</title><style>
body{{font-family:sans-serif;margin:16px;background:#fafafa}} .card{{background:#fff;border:1px solid #ddd;margin:14px 0;padding:8px}}
img{{width:100%}} td{{width:25%;vertical-align:top}} table{{width:100%}} .txt{{font-size:13px;line-height:1.4}} .hdr{{font-size:13px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:50%;border:1px solid #999;margin:0 2px 0 6px;vertical-align:middle}}
</style></head><body><h2>ithor_v11 — natural vs colour SoM ({len(cards)} random questions, {per_cat} per category)</h2>{''.join(cards)}</body></html>""")


def write_readme(out: Path, m: dict[str, Any], som: list[dict], nat: list[dict]) -> None:
    cats = "\n".join(f"| {c} | {n} |" for c, n in m["rows_per_category"].items())
    src = ", ".join(f"{k}: {v}" for k, v in m["rows_per_source_version"].items())
    rt = ", ".join(f"{k} {v}" for k, v in m["removed_structural_by_type"].items())
    rm = m["removed"]
    same = sum(a["answer"] == b["answer"] for a, b in zip(som, nat))
    (out / "README.md").write_text(f"""# ithor_v11 — iTHOR CIT benchmark with colour SoM

Built {m['created_utc'][:10]} by `build_v11_colorsom_ithor.py` from the ithor_v10 merge
(`$CIT_DATA_ROOT/ithor_v10`): v10 `feasibility` / `contact_wall`
plus the other six categories from ithor_v9. Same colour-SoM recipe as `procthor_v11`.

**{m['n_rows']} questions** over **{m['n_samples']} scenes/samples** (rows per source: {src}).
`natural/` and `som/` hold the **same question ids in the same order**.

## Layout

```
ithor_v11/
  README.md
  build_meta.json          counts, filters, palette, sources
  gallery.html             natural vs colour SoM, 3 random questions per category
  benchmark_viz.html       benchmark viewer (viz_ithor_v11_benchmark.py): images, GT flow, highlighted answers
  natural/
    dataset.jsonl          natural questions (no marks), images relative to natural/
    images/                <sample_id>_<view_id>_rgb{{0,1}}.png   (before / after)
    samples/<sample_id>/   GT per sample: sample.json, <view_id>_gt_flow.npy (H,W,2 px),
                           <view_id>_rgb{{0,1}}_depth.npy (m), _seg / _seg_ids / _mask{{0,1}}, flow previews
  som/
    dataset.jsonl          colour-SoM questions, images relative to som/
    images/                <sample_id>_<view_id>_rgb{{0,1}}_som.png
    removed.jsonl          source rows dropped, with removed_reason
```

`samples/*/sample.json` still stores the original absolute generation-time paths in its `*_path` fields; the files
themselves are copied next to it.

## Categories

| category | rows |
|---|---|
{cats}

Answer letters are `A`–`F` (`closest_three` has three letters, e.g. `ABD`).

## Colour SoM

- Each mark is a solid disc (radius {m['mark_radius']} px, {m['mark_rim']} px white rim), no number.
- Palette (10): {", ".join(m['palette'])}. Colours are sampled without replacement per question,
  seeded by the row id, so colour carries no information about role, position or the answer.
  Same object = same colour in before and after images; `som_colors` maps mark id → colour
  (`1` mover, `2` landmark/occludee, `A`–`F` choice objects).
- Text refers to marks as `the <object> (marked <colour>)` in `motion_command`, `question`,
  `static_question`, and in object choices (`closer_to`, `closest_three`, and the object list of `count_near`).

## Row fields (both folders)

`id` (= `<sample_id>__<question_id>`), `sample_id`, `view_id`, `question_id`, `scene`, `category`,
`image_before`, `image_after`, `motion_command`, `question` (after the move), `static_question`
(asked on the after image), `choice_A`…, `answer`, `n_choices`, `mcq_type`, `gt_complete`,
`mover_object_id`, `source_version` (ithor_v9 / ithor_v10). SoM rows add `som_colors`, `som_style` = "color".
Answer letters are balanced per category by `reshuffle_v11_answer_letters.py`
(the source lists closest_three choices nearest-first, so "A" was almost always correct): choices are
permuted so correct letters are uniform (closest_three over all 20 three-letter combos; yes/no questions
within each polarity). `choice_perm` maps each new letter to the source letter. Natural and SoM rows of the
same id now share the answer letter ({same} / {m['n_rows']} match).

## Filters applied (source: {m['n_source_rows']} rows)

- `gt_complete` and a letter answer (dropped {rm.get('no_gt_answer', 0)}).
- No structural mover (dropped {rm.get('structural_mover', 0)}: {rt}). Banned types:
  {", ".join(m['structural_mover_types'])}; plus mover phrases the cabinet / cupboard / drawer / sink / faucet.
- No question naming "the shelf" or "the drawer" two or more times across question + choices, since parts of
  one piece of furniture cannot be told apart by name (dropped {rm.get('repeated_shelf_or_drawer', 0)}).
- Every mark referenced in the text is drawn in the images (dropped {rm.get('unresolved_mark_ref', 0)}).
- Occlusion: occluded object at least 150 px in the before image (dropped
  {rm.get('occlusion_tiny_occludee', 0)}); any mover cover of the occludee counts as "occluded"
  (cover 0-30% relabelled from "visible"); choices "Partly or fully hidden (occluded)." /
  "Totally visible (not occluded).". Rows carry `occludee_px`, `occlusion_cover`, `occlusion_relabelled`.

## Text normalization

- `static_question` never refers to the move: the leading "After the move, " is removed (near_object,
  count_near; both folders).
- Natural near_object / count_near static questions name the mover instead of a bare "it"
  ("Is it within 0.50m of the cellphone?" → "Is the pot within 0.50m of the cellphone?"), using the
  mover phrase from `motion_command`.
- SoM occlusion questions name the occluding mover instead of "the object (marked C)"
  ("occluded by the toaster (marked black)"), in both `question` and `static_question`.

For evaluation benchmarks, `contact_wall` and `near_object` are usually skipped at selection time;
they are kept here so the data stays complete.

No CoT and no tokenized codes are included for iTHOR.
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    out = args.out
    for d in ("natural/images", "natural/samples", "som/images"):
        (out / d).mkdir(parents=True, exist_ok=True)

    som_src = load(SRC / "som" / "dataset_static_som.jsonl")
    nat_src = {r["id"]: r for r in load(SRC / "natural" / "dataset.jsonl")}
    samples: dict[str, dict] = {}
    kept, removed = [], []
    for r in som_src:
        if not usable(r):
            removed.append({**r, "removed_reason": "no_gt_answer"})
            continue
        sid = r["sample_id"]
        if sid not in samples:
            samples[sid] = json.loads((sample_dir(sid) / "sample.json").read_text())
        new = colorize_row(r, samples[sid], out / "som")
        leftover = new.pop("_leftover")
        mover_type = str(new.get("mover_object_id") or "").split("|")[0].lower()
        drop = new.pop("_drop", None)
        reason = ("unresolved_mark_ref" if leftover
                  else drop if drop
                  else "structural_mover" if structural_mover(new) or mover_type in ITHOR_EXTRA_TYPES
                  else "repeated_shelf_or_drawer" if repeated_part_ref(new)
                  else None)
        new["source_version"] = "ithor_v10" if "ithor_v10" in str(sample_dir(sid)) else "ithor_v9"
        if reason:
            removed.append({**new, "removed_reason": reason})
            for k in ("image_before", "image_after"):
                (out / "som" / new[k]).unlink(missing_ok=True)
        else:
            kept.append(new)

    kept.sort(key=lambda r: r["id"])
    nat_kept = []
    for r in kept:
        n = dict(nat_src[r["id"]])
        for k in ("image_before", "image_after"):
            shutil.copy2(SRC / "natural" / n[k], out / "natural" / n[k])
        n["static_question"] = name_dangling_it(n)
        if n["category"] == "occlusion":
            _, q = find_question(samples[n["sample_id"]], n)
            n = fix_occlusion_row(n, q.get("gt_meta") or {})
        n["mover_object_id"] = r["mover_object_id"]
        n["source_version"] = r["source_version"]
        nat_kept.append(n)
    for sid in sorted({r["sample_id"] for r in kept}):
        dst = out / "natural" / "samples" / sid
        if not dst.exists():
            shutil.copytree(sample_dir(sid), dst)

    dump(out / "som" / "dataset.jsonl", kept)
    dump(out / "som" / "removed.jsonl", removed)
    dump(out / "natural" / "dataset.jsonl", nat_kept)

    per_cat = collections.Counter(r["category"] for r in kept)
    rm = collections.Counter(r["removed_reason"] for r in removed)
    rm_types = collections.Counter(
        str(r.get("mover_object_id") or "").split("|")[0] for r in removed if r["removed_reason"] == "structural_mover")
    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {"som": str(SRC / "som" / "dataset_static_som.jsonl"),
                    "natural": str(SRC / "natural" / "dataset.jsonl"),
                    "sample_dirs": [str(SRC_ROOT / v / "natural") for v in SAMPLE_SOURCES]},
        "n_source_rows": len(som_src),
        "n_rows": len(kept),
        "rows_per_category": dict(sorted(per_cat.items())),
        "rows_per_source_version": dict(collections.Counter(r["source_version"] for r in kept)),
        "removed": dict(rm),
        "removed_structural_by_type": dict(rm_types),
        "structural_mover_types": sorted(STRUCTURAL_MOVER_TYPES | ITHOR_EXTRA_TYPES),
        "n_samples": len({r["sample_id"] for r in kept}),
        "som_style": "color",
        "palette": SOM_COLOR_PALETTE,
        "mark_radius": SOM_COLOR_RADIUS,
        "mark_rim": SOM_COLOR_RIM,
        "text_ref_format": "the <object> (marked <colour>)",
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    render_gallery(out, kept, {r["id"]: r for r in nat_kept})
    write_readme(out, meta, kept, nat_kept)
    print(json.dumps({k: meta[k] for k in ("n_source_rows", "n_rows", "rows_per_category", "removed",
                                            "removed_structural_by_type", "n_samples")}, indent=1))


if __name__ == "__main__":
    main()
