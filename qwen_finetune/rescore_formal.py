#!/usr/bin/env python3
"""Re-score formal-format direct runs from their saved raw replies with the current parser.

For when the parser changes after a run (e.g. GLM's <|begin_of_box|>A<|end_of_box|> answer boxes):
decoding is greedy and the first reply is saved, so applying the fixed ``strict_letters`` to the
first reply gives what a rerun would. If the first reply is still not a clean letter, the saved
follow-up reply is used, as the eval does. Originals are kept as predictions_orig.jsonl /
metrics_orig.json (once).

  python qwen_finetune/rescore_formal.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_qwen3_vl_32b_server import strict_letters  # noqa: E402


def rescore(run: Path) -> None:
    pf, mf = run / "predictions.jsonl", run / "metrics.json"
    for f in (pf, mf):
        orig = f.with_name(f.stem + "_orig" + f.suffix)
        if not orig.exists():
            shutil.copy(f, orig)
    rows = [json.loads(l) for l in open(run / "predictions_orig.jsonl") if l.strip()]
    by_cat: dict = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        raw = str(r.get("raw") or "")
        first = raw.split("\n[user] ")[0]
        pred = strict_letters(first)
        if pred is None and "\n[assistant] " in raw:
            pred = strict_letters(raw.rsplit("\n[assistant] ", 1)[1])
        r.update(pred=pred, parse_ok=pred is not None, correct=bool(pred is not None and pred == r["gt"]),
                 direct_rescued=strict_letters(first) is None, rescored=True)
        c = by_cat[r["category"]]
        c[0] += r["correct"]
        c[1] += 1
        c[2] += not r["parse_ok"]
    with open(pf, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    m = json.loads((run / "metrics_orig.json").read_text())
    n_correct = sum(v[0] for v in by_cat.values())
    m.update(per_category={k: {"n": t, "n_correct": cor, "n_parse_fail": pf_, "accuracy": cor / t if t else 0.0}
                           for k, (cor, t, pf_) in sorted(by_cat.items())},
             n_correct=n_correct, accuracy=n_correct / len(rows), n_parse_fail=sum(v[2] for v in by_cat.values()),
             rescored_with="strict_letters (answer-box markers stripped)")
    mf.write_text(json.dumps(m, indent=2))
    print(f"{run}: acc {100 * m['accuracy']:.1f}  unparsed {m['n_parse_fail']}  "
          f"(was acc {100 * json.loads((run / 'metrics_orig.json').read_text())['accuracy']:.1f})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rescore(Path(d))
