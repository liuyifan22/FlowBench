#!/usr/bin/env python3
"""Build 1B (reasoning) Qwen splits for procthor_v11 from the balanced 1A/2A splits.

Keeps rows whose warm CoT is on the reasoning allowlist (filters/som_cot_ids.txt: CoT exists
and the natural answer agrees with the SoM answer) and attaches ``cot_target``:
the warm CoT with its trailing "Answer: ..." line removed, then ``<answer>X</answer>`` with
the SoM row's letter. Images symlink to procthor_v11/som/images like the 1A/2A splits.

  python qwen_finetune/prepare_procthor_v11_1B_splits.py
"""

from __future__ import annotations

import os
import collections
import json
import re
from pathlib import Path

V = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
SRC = V / "qwen_training/som_balanced_splits"
OUT = V / "qwen_training/som_balanced_splits_1B"
_ANSWER_TAIL = re.compile(r"\n\s*Answer\s*:.*\Z", re.IGNORECASE | re.DOTALL)


def cot_target(rec: dict) -> str | None:
    p = V / "cot" / f"{rec['id']}_warm.txt"
    if not p.is_file():
        return None
    text = _ANSWER_TAIL.sub("", p.read_text().replace("<|endoftext|>", "").strip()).strip()
    m = re.search(r"\bAnswer\s*:", text, flags=re.IGNORECASE)  # same-line "Answer:" variant
    if m:
        text = text[: m.start()].rstrip()
    return f"{text}\n\n<answer>{rec['answer']}</answer>" if text else None


def main() -> None:
    allow = set(open(V / "filters/som_cot_ids.txt").read().split())
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "images").exists():
        (OUT / "images").symlink_to(V / "som/images")
    meta = {}
    for split in ("train", "val", "test", "eval"):
        n_in = n_out = 0
        cats = collections.Counter()
        with open(SRC / f"{split}.jsonl") as fi, open(OUT / f"{split}.jsonl", "w") as fo:
            for line in fi:
                n_in += 1
                r = json.loads(line)
                if r["id"] not in allow:
                    continue
                t = cot_target(r)
                if t is None:
                    continue
                r["cot_target"] = t
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
                cats[r["category"]] += 1
        meta[split] = {"n_in": n_in, "n_out": n_out, "per_category": dict(sorted(cats.items()))}
        print(split, n_in, "->", n_out, flush=True)
    (OUT / "split_meta.json").write_text(json.dumps({"source": str(SRC), "allowlist": str(V / "filters/som_cot_ids.txt"),
                                                     "target": "warm CoT (Answer: tail removed) + <answer>X</answer>",
                                                     "splits": meta}, indent=2))


if __name__ == "__main__":
    main()
