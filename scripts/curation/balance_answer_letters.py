#!/usr/bin/env python3
"""Reshuffle answer choices of the v11 datasets so correct letters are uniform.

The source exports list closest_three choices nearest-first, so "A" is in ~93% of the
correct answers and "F" almost never. Permuting the choice order fixes this without
touching the questions: each choice keeps its text (and "(marked colour)" tag), only
its letter changes.

Targets are drawn from a balanced bag per category (seeded, deterministic):
  * closest_three (mcq6_select3): uniform over all C(6,3) = 20 letter combos
  * 4-choice (closer_to, count_near): uniform over A-D
  * 2-choice: uniform over A/B, separately within each yes/no polarity group
    (``yes_no_gt``), so a letter never stands for "yes"
Each row gets ``choice_perm`` ({new letter: old letter}) and updated
``correct_phrase(s)``. ``som_colors`` is unchanged (it is keyed by mark id).

The permutation is computed once per id and applied identically to every file that
contains that id (dataset.jsonl and the split files).

  python scripts/curation/balance_answer_letters.py --target ithor      # all categories, som + natural
  python scripts/curation/balance_answer_letters.py --target procthor   # closest_three, som + splits
  python scripts/curation/balance_answer_letters.py --target procthor --categories occlusion
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flowbench.answer_balance import apply_perm, plan

D = Path(os.environ.get("FLOWBENCH_DATA_ROOT", "data/build"))
TARGETS = {
    # name: (files sharing ids, categories to reshuffle or None for all)
    "ithor_som": ([D / "ithor_v11/som/dataset.jsonl"], None),
    "ithor_natural": ([D / "ithor_v11/natural/dataset.jsonl"], None),
    "procthor": ([D / "procthor_v11/som/dataset.jsonl"]
                 + [D / f"procthor_v11/{s}.jsonl" for s in ("test", "val", "train0", "train1", "train2")],
                 {"closest_three"}),
}


def rewrite(path: Path, perms: dict[str, dict[str, str]]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            r = json.loads(line)
            if r["id"] in perms:
                r = apply_perm(r, perms[r["id"]])
                n += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return n


def letter_report(path: Path, cats: set[str] | None) -> dict[str, dict[str, int]]:
    by: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in open(path):
        r = json.loads(line)
        if cats is None or r["category"] in cats:
            for L in r["answer"]:
                by[r["category"]][L] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("ithor", "procthor"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--categories", default=None,
                    help="comma list restricting which categories are reshuffled (default: the target's own set)")
    args = ap.parse_args()
    only = {c.strip() for c in args.categories.split(",")} if args.categories else None
    names = ["ithor_som", "ithor_natural"] if args.target == "ithor" else ["procthor"]
    for name in names:
        files, cats = TARGETS[name]
        if only is not None:
            cats = only if cats is None else (cats & only) or only
        main_rows = [json.loads(l) for l in open(files[0])]
        perms = plan(main_rows, cats, args.seed)
        before = letter_report(files[0], cats)
        for f in files:
            print(f"{name}: {f.relative_to(D)} rewrote {rewrite(f, perms)} rows", flush=True)
        after = letter_report(files[0], cats)
        for c in after:
            print(f"   {c:<22} letter counts before {before[c]}\n   {'':<22}               after  {after[c]}")
        log = files[0].parent / f"reshuffle_answer_letters{'_' + '_'.join(sorted(only)) if only else ''}.log.json"
        log.write_text(json.dumps({"seed": args.seed, "files": [str(f) for f in files],
                                   "categories": sorted(cats) if cats else "all",
                                   "n_rows": len(perms), "before": before, "after": after}, indent=1))


if __name__ == "__main__":
    main()
