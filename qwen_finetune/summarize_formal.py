#!/usr/bin/env python3
"""Aggregate the formal-format runs: mean over answer-choice seeds per (benchmark, setting, model).

Reads <root>/<job>/<bench>_<cell>_seed<k>/{metrics,predictions}.json(l). For each benchmark / setting /
model: per-category accuracy (mean over seeds), Avg (per question) and Cat-avg (mean of categories) as
mean +- std over seeds, number of seeds, and the clean-letter rate (first reply is a bare option letter).
Writes <root>/summary.json and prints Markdown tables.

  python qwen_finetune/summarize_formal.py <root>
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_qwen3_vl_32b_server import strict_letters  # noqa: E402

ORDER = ["Qwen3-VL-8B-Instruct", "Qwen3-VL-32B-Instruct", "Qwen3.5-9B", "InternVL3_5-8B", "InternVL3_5-30B-A3B",
         "LLaVA-OneVision-1.5-8B", "LLaVA-OneVision-2-8B", "deepseek-vl2-small", "Kimi-VL-A3B-Instruct", "GLM-4.6V-Flash", "gemma-4-12B-it", "gemma-4-31B-it",
         "Qwen3-VL-8B-ft1A", "Qwen3-VL-8B-ft2A", "Qwen3-VL-8B-ft1B", "Qwen3-VL-8B-ft1B-c0", "Qwen3-VL-8B-ft1B-c0v2",
         "Qwen3.5-9B-ft1A", "LLaVA-OneVision-2-8B-ft1A"]


def collect(root: Path) -> dict:
    res: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # bench -> cell -> model -> [run]
    for mf in sorted(root.glob("*/*_seed*/metrics.json")):
        model = mf.parent.parent.name.split("__")[0]
        bench, cell, seed = mf.parent.name.rsplit("_", 2)
        m = json.loads(mf.read_text())
        preds = [json.loads(l) for l in open(mf.parent / "predictions.jsonl") if l.strip()]
        clean = sum(strict_letters(str(p.get("raw") or "").split("\n[user] ")[0]) is not None for p in preds)
        pc = m["per_category"]
        accs = {c: 100 * v["accuracy"] for c, v in pc.items()}
        n = sum(v["n"] for v in pc.values())
        res[bench][cell][model].append({
            "seed": int(seed.removeprefix("seed")), "per_category": accs,
            "avg": 100 * sum(v["n_correct"] for v in pc.values()) / n,
            "cat_avg": sum(accs.values()) / len(accs), "n": n,
            "clean_rate": 100 * clean / max(1, len(preds)), "unparsed": m.get("n_parse_fail", 0),
        })
    return res


def summarize(runs: list[dict]) -> dict:
    cats = sorted(runs[0]["per_category"])
    sd = (lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0)
    return {
        "n_seeds": len(runs), "seeds": sorted(r["seed"] for r in runs),
        "per_category": {c: statistics.mean(r["per_category"][c] for r in runs) for c in cats},
        "avg": statistics.mean(r["avg"] for r in runs), "avg_sd": sd([r["avg"] for r in runs]),
        "cat_avg": statistics.mean(r["cat_avg"] for r in runs), "cat_avg_sd": sd([r["cat_avg"] for r in runs]),
        "clean_rate": statistics.mean(r["clean_rate"] for r in runs),
        "unparsed": statistics.mean(r["unparsed"] for r in runs), "n": runs[0]["n"],
    }


def main() -> None:
    root = Path(sys.argv[1])
    res = collect(root)
    out = {b: {c: {m: summarize(r) for m, r in ms.items()} for c, ms in cs.items()} for b, cs in res.items()}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    for bench in sorted(out):
        for cell in sorted(out[bench]):
            ms = out[bench][cell]
            cats = sorted(next(iter(ms.values()))["per_category"])
            print(f"\n### {bench} {cell} (n={next(iter(ms.values()))['n']}, mean over seeds)")
            print("| Model | seeds | " + " | ".join(cats) + " | Avg | Cat-avg | clean% |")
            print("|" + "---|" * (len(cats) + 5))
            for m in sorted(ms, key=lambda k: ORDER.index(k) if k in ORDER else 99):
                s = ms[m]
                print(f"| {m} | {s['n_seeds']} | " + " | ".join(f"{s['per_category'][c]:.1f}" for c in cats)
                      + f" | {s['avg']:.1f}±{s['avg_sd']:.1f} | {s['cat_avg']:.1f}±{s['cat_avg_sd']:.1f} | {s['clean_rate']:.0f} |")


if __name__ == "__main__":
    main()
