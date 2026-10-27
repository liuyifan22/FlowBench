#!/usr/bin/env python3
"""Visualize Qwen VL SFT training inputs (image + prompt) and expected answers.

Uses the same builders as training (``data.build_sft_example`` / ``make_dataset``).

Example::

  python -m qwen_finetune.visualize_train_inputs --mode 2A --n-per-category 2
  python qwen_finetune/visualize_train_inputs.py --mode 1A --n 12 --split val
"""

from __future__ import annotations

import os
import argparse
import html
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_CIT = _ROOT.parent
for p in (_CIT, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import build_sft_example  # noqa: E402
from dataset_config import DatasetFilterConfig, load_filtered_records  # noqa: E402

DEFAULT_DATASET = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/qwen_training/"
    "procthor_v7_som_gt_splits"
)
DEFAULT_OUT = _CIT / "outputs" / "qwen_finetune_train_viz"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "eval"),
        help="Which split jsonl to sample from",
    )
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--mode", default="2A", choices=("1A", "2A", "both"))
    p.add_argument("--n", type=int, default=0, help="Total examples (0 = use n-per-category)")
    p.add_argument("--n-per-category", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated category allow-list",
    )
    return p.parse_args()


def _pick_records(
    rows: list[dict],
    *,
    n: int,
    n_per_category: int,
    seed: int,
) -> list[dict]:
    import random

    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category") or "?")].append(r)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    picked: list[dict] = []
    if n > 0:
        # Round-robin across categories for diversity.
        cats = sorted(by_cat.keys())
        idx = {c: 0 for c in cats}
        while len(picked) < n and cats:
            progressed = False
            for c in list(cats):
                i = idx[c]
                if i >= len(by_cat[c]):
                    cats.remove(c)
                    continue
                picked.append(by_cat[c][i])
                idx[c] = i + 1
                progressed = True
                if len(picked) >= n:
                    break
            if not progressed:
                break
        return picked

    for cat in sorted(by_cat.keys()):
        picked.extend(by_cat[cat][: max(0, n_per_category)])
    return picked


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def _card_html(
    *,
    idx: int,
    mode: str,
    ex_id: str,
    category: str,
    prompt: str,
    answer: str,
    img_rel: str,
    image_path: str,
    meta: dict,
) -> str:
    prompt_esc = html.escape(prompt)
    ans_esc = html.escape(answer)
    mid = html.escape(ex_id)
    cat = html.escape(category)
    extra = ""
    if meta.get("motion_command"):
        extra += (
            f'<div class="meta"><b>motion</b>: '
            f'{html.escape(str(meta["motion_command"]))}</div>'
        )
    if meta.get("static_question"):
        extra += (
            f'<div class="meta"><b>static_question</b>: '
            f'{html.escape(str(meta["static_question"]))}</div>'
        )
    if meta.get("question"):
        extra += (
            f'<div class="meta"><b>question (raw)</b>: '
            f'{html.escape(str(meta["question"]))}</div>'
        )
    return f"""
<article class="card" id="ex-{idx}">
  <header>
    <span class="badge mode">{html.escape(mode)}</span>
    <span class="badge cat">{cat}</span>
    <span class="badge ans">GT → {ans_esc}</span>
    <code class="id">{mid}</code>
  </header>
  <div class="body">
    <figure>
      <img src="{html.escape(img_rel)}" alt="training image for {mid}" loading="lazy"/>
      <figcaption>image fed to the model<br/><code>{html.escape(image_path)}</code></figcaption>
    </figure>
    <div class="text">
      <h3>User prompt (text input)</h3>
      <pre class="prompt">{prompt_esc}</pre>
      <h3>Expected assistant output</h3>
      <pre class="answer">{ans_esc}</pre>
      {extra}
    </div>
  </div>
</article>
"""


def _page(title: str, cards: str, summary: dict) -> str:
    summary_json = html.escape(json.dumps(summary, indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222c;
    --text: #e7ecf1;
    --muted: #9aa7b5;
    --accent: #3d9cf0;
    --ans: #3ecf8e;
    --border: #2a3542;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1b2a3a 0%, var(--bg) 55%);
    color: var(--text);
    line-height: 1.45;
  }}
  h1 {{ font-weight: 600; letter-spacing: -0.02em; margin: 0 0 8px; }}
  .sub {{ color: var(--muted); margin-bottom: 24px; }}
  details.summary {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 28px;
  }}
  details.summary pre {{ color: var(--muted); font-size: 12px; overflow: auto; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; margin-bottom: 28px; overflow: hidden;
  }}
  .card header {{
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
  }}
  .badge {{
    font-size: 12px; font-weight: 600; padding: 3px 8px;
    border-radius: 999px; background: #243041; color: var(--text);
  }}
  .badge.mode {{ background: #1e3a5f; color: #9cc9ff; }}
  .badge.cat {{ background: #3a2f1e; color: #f0d39c; }}
  .badge.ans {{ background: #1e3d30; color: var(--ans); }}
  .id {{ color: var(--muted); font-size: 12px; margin-left: auto; }}
  .body {{
    display: grid; grid-template-columns: minmax(280px, 42%) 1fr; gap: 0;
  }}
  @media (max-width: 900px) {{
    .body {{ grid-template-columns: 1fr; }}
  }}
  figure {{ margin: 0; padding: 16px; border-right: 1px solid var(--border); }}
  @media (max-width: 900px) {{
    figure {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
  figure img {{
    width: 100%; height: auto; border-radius: 8px;
    background: #000; display: block;
  }}
  figcaption {{ color: var(--muted); font-size: 11px; margin-top: 8px; word-break: break-all; }}
  .text {{ padding: 16px 18px 20px; }}
  h3 {{ margin: 0 0 8px; font-size: 14px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; }}
  pre.prompt, pre.answer {{
    white-space: pre-wrap; word-break: break-word;
    background: #121820; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px; font-size: 13px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
  }}
  pre.answer {{
    color: var(--ans); font-size: 22px; font-weight: 700;
    display: inline-block; min-width: 3ch; text-align: center;
  }}
  .meta {{ color: var(--muted); font-size: 12px; margin-top: 10px; }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="sub">Same inputs the Trainer feeds Qwen3-VL (mode image + chat prompt) and the supervised letter target.</p>
  <details class="summary"><summary>run summary</summary><pre>{summary_json}</pre></details>
  {cards}
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset)
    jsonl = Path(args.jsonl) if args.jsonl else dataset_root / f"{args.split}.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"Missing jsonl: {jsonl}")

    cats = None
    if args.categories:
        cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    # Images already verified at split time; skip existence scan for speed.
    cfg = DatasetFilterConfig(
        require_gt_complete=True,
        require_letter_answer=True,
        require_images_exist=False,
        categories=cats,
        seed=int(args.seed),
    )
    rows = load_filtered_records(jsonl, dataset_root=dataset_root, cfg=cfg, limit=0)
    picked = _pick_records(
        rows, n=int(args.n), n_per_category=int(args.n_per_category), seed=int(args.seed)
    )
    if not picked:
        raise SystemExit("No examples selected")

    modes = ("1A", "2A") if args.mode == "both" else (args.mode.upper(),)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{'_'.join(modes)}_{args.split}_{stamp}"
    assets = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    manifest: list[dict] = []
    idx = 0
    for mode in modes:
        for rec in picked:
            ex = build_sft_example(rec, mode=mode, dataset_root=dataset_root)
            stem = f"{idx:03d}_{mode}_{ex.category}_{ex.id}"
            # sanitize filename
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:180]
            ext = ex.image_path.suffix or ".png"
            dst = assets / f"{safe}{ext}"
            _copy_image(ex.image_path, dst)
            img_rel = f"assets/{dst.name}"
            cards.append(
                _card_html(
                    idx=idx,
                    mode=ex.mode,
                    ex_id=ex.id,
                    category=ex.category,
                    prompt=ex.prompt,
                    answer=ex.answer,
                    img_rel=img_rel,
                    image_path=str(ex.image_path),
                    meta={
                        "motion_command": rec.get("motion_command"),
                        "static_question": rec.get("static_question"),
                        "question": rec.get("question"),
                        "som": rec.get("som"),
                        "image_before": rec.get("image_before"),
                        "image_after": rec.get("image_after"),
                    },
                )
            )
            manifest.append(
                {
                    "idx": idx,
                    "mode": ex.mode,
                    "id": ex.id,
                    "category": ex.category,
                    "answer": ex.answer,
                    "prompt": ex.prompt,
                    "image_path": str(ex.image_path),
                    "asset": img_rel,
                }
            )
            idx += 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "jsonl": str(jsonl),
        "split": args.split,
        "modes": list(modes),
        "n_source_rows": len(rows),
        "n_picked_records": len(picked),
        "n_cards": len(manifest),
        "n_per_category": args.n_per_category if args.n <= 0 else None,
        "n": args.n or None,
        "seed": args.seed,
        "categories_in_page": sorted({m["category"] for m in manifest}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    title = f"Qwen SFT inputs — {', '.join(modes)} / {args.split} ({len(manifest)} examples)"
    index = out_dir / "index.html"
    index.write_text(_page(title, "\n".join(cards), summary), encoding="utf-8")

    # Convenience alias at parent level
    alias = Path(args.out_dir) / "latest.html"
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(index)
    except OSError:
        shutil.copy2(index, alias)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz] wrote {index}", flush=True)
    print(f"[viz] alias {alias}", flush=True)


if __name__ == "__main__":
    main()
