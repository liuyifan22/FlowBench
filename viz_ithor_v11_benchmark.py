#!/usr/bin/env python3
"""HTML viewer for the ithor_v11 evaluation benchmark: random items per category.

Each card shows natural before/after, colour-SoM before/after and the GT flow
preview, with the natural and SoM question / choices side by side and the correct
choice highlighted. Writes ``<ithor_v11>/benchmark_viz.html``.

  python viz_ithor_v11_benchmark.py --per-cat 5 --seed 0
"""

from __future__ import annotations

import os
import argparse
import collections
import html
import json
import random
from pathlib import Path

from cit_pipeline.color_som import CHOICE_KEYS
from cit_pipeline.som import SOM_COLOR_PALETTE

ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11")
# Selected out of eval benchmarks by default (data kept in ithor_v11).
EVAL_SKIPPED = {"contact_wall", "near_object"}


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def swatch(color: str) -> str:
    return f'<span class="sw" style="background:rgb{SOM_COLOR_PALETTE[color]}"></span>'


def qa_block(r: dict, title: str) -> str:
    answer = set(str(r.get("answer") or ""))
    choices = "".join(
        f'<li class="{"gt" if k[-1] in answer else ""}"><b>{k[-1]}.</b> {html.escape(str(r[k]))}</li>'
        for k in CHOICE_KEYS if r.get(k)
    )
    return f"""<div class="qa"><div class="qt">{title}</div>
<div><span class="lbl">motion</span> {html.escape(str(r.get('motion_command')))}</div>
<div><span class="lbl">question</span> {html.escape(str(r.get('question')))}</div>
<div><span class="lbl">static</span> {html.escape(str(r.get('static_question')))}</div>
<ul>{choices}</ul><div><span class="lbl">answer</span> <b>{html.escape(str(r.get('answer')))}</b></div></div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_viz.html")
    args = ap.parse_args()

    som = load(ROOT / "som" / "dataset.jsonl")
    nat = {r["id"]: r for r in load(ROOT / "natural" / "dataset.jsonl")}
    by_cat = collections.defaultdict(list)
    for r in som:
        by_cat[r["category"]].append(r)
    rng = random.Random(args.seed)

    toc, sections = [], []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        pick = rng.sample(rows, min(args.per_cat, len(rows)))
        tag = ' <span class="skip">skipped in eval selection</span>' if cat in EVAL_SKIPPED else ""
        toc.append(f'<a href="#{cat}">{cat} ({len(rows)})</a>')
        cards = []
        for r in pick:
            n = nat[r["id"]]
            flow = f"natural/samples/{r['sample_id']}/{r['view_id']}_gt_flow_arrow.png"
            if not (ROOT / flow).exists():  # a few views only have the plain colour preview
                flow = flow.replace("_gt_flow_arrow.png", "_gt_flow.png")
            colors = " ".join(f"{swatch(c)}{m}={c}" for m, c in r["som_colors"].items())
            mover = html.escape(str(r.get("mover_object_id") or "").split("|")[0])
            cards.append(f"""<div class="card">
<div class="hdr"><b>{html.escape(r['id'])}</b> · {r['source_version']} · mover {mover} · {colors}</div>
<div class="imgs">
 <figure><img src="natural/{n['image_before']}" loading="lazy"><figcaption>natural before</figcaption></figure>
 <figure><img src="natural/{n['image_after']}" loading="lazy"><figcaption>natural after</figcaption></figure>
 <figure><img src="{flow}" loading="lazy"><figcaption>GT flow (before → after)</figcaption></figure>
 <figure><img src="som/{r['image_before']}" loading="lazy"><figcaption>colour SoM before</figcaption></figure>
 <figure><img src="som/{r['image_after']}" loading="lazy"><figcaption>colour SoM after</figcaption></figure>
</div>
<div class="qas">{qa_block(n, 'natural')}{qa_block(r, 'colour SoM')}</div></div>""")
        sections.append(f'<h2 id="{cat}">{cat} — {len(pick)} of {len(rows)}{tag}</h2>{"".join(cards)}')

    palette = " ".join(f"{swatch(c)}{c}" for c in SOM_COLOR_PALETTE)
    args.out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>ithor_v11 benchmark</title><style>
body{{font-family:sans-serif;margin:16px;background:#f5f5f5;color:#222}}
.toc a{{margin-right:14px}} .card{{background:#fff;border:1px solid #ddd;border-radius:6px;margin:12px 0;padding:10px}}
.hdr{{font-size:13px;margin-bottom:6px}} .imgs{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} figure{{margin:0;min-width:0}}
figure img{{width:100%;border:1px solid #eee}} figcaption{{font-size:11px;color:#777;text-align:center}}
.qas{{display:flex;gap:10px;margin-top:6px}} .qa{{flex:1;font-size:13px;line-height:1.45;background:#fafafa;border:1px solid #eee;padding:6px}}
.qt{{font-weight:bold;color:#335;margin-bottom:3px}} .lbl{{display:inline-block;min-width:62px;color:#888;font-size:11px}}
ul{{margin:4px 0;padding-left:18px}} li.gt{{background:#d8f5d0;font-weight:bold}}
.skip{{font-size:12px;color:#b00;font-weight:normal}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:50%;border:1px solid #999;margin:0 2px 0 6px;vertical-align:middle}}
</style></head><body>
<h1>ithor_v11 evaluation benchmark — {len(som)} questions, {args.per_cat} random per category (seed {args.seed})</h1>
<div>Correct choices are highlighted green. Palette: {palette}</div>
<p class="toc">{''.join(toc)}</p>{''.join(sections)}</body></html>""")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
