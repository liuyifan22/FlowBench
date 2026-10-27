#!/usr/bin/env python3
"""Instruction-following check for the formal-format runs: does each model answer with just the
option letter(s)?

Per run (<root>/<job>/<bench>_<cell>_seed<k>/predictions.jsonl): n so far, % first replies that are a
clean letter answer (``strict_letters`` on the first reply, i.e. no follow-up needed), % that needed the
follow-up, % unparsed, accuracy so far. A low clean-letter rate means re-adding
"Please directly give the index of the answer." may be needed.

  python qwen_finetune/formal_compliance.py <root>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_qwen3_vl_32b_server import strict_letters  # noqa: E402


def first_reply(row: dict) -> str:
    return str(row.get("raw") or "").split("\n[user] ")[0]


def main() -> None:
    root = Path(sys.argv[1])
    agg = defaultdict(lambda: [0, 0, 0, 0])  # n, clean, rescued, unparsed
    print(f"{'run':<58}{'n':>6}{'clean%':>8}{'nudged%':>9}{'unparsed%':>10}{'acc%':>7}")
    for pf in sorted(root.glob("*/*/predictions.jsonl")):
        rows = [json.loads(l) for l in open(pf) if l.strip()]
        if not rows:
            continue
        n = len(rows)
        clean = sum(strict_letters(first_reply(r)) is not None for r in rows)
        rescued = sum(bool(r.get("direct_rescued")) for r in rows)
        unparsed = sum(not r.get("parse_ok") for r in rows)
        acc = sum(bool(r.get("correct")) for r in rows)
        name = f"{pf.parent.parent.name}/{pf.parent.name}"
        print(f"{name:<58}{n:>6}{100 * clean / n:>8.1f}{100 * rescued / n:>9.1f}{100 * unparsed / n:>10.1f}{100 * acc / n:>7.1f}")
        model = pf.parent.parent.name.split("__")[0]
        for i, v in enumerate((n, clean, rescued, unparsed)):
            agg[model][i] += v
    print("\nper model:")
    for m, (n, c, r, u) in sorted(agg.items()):
        print(f"  {m:<30} n={n:<6} clean {100 * c / n:5.1f}%  nudged {100 * r / n:5.1f}%  unparsed {100 * u / n:5.1f}%")


if __name__ == "__main__":
    main()
