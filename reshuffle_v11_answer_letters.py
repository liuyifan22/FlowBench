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

  python reshuffle_v11_answer_letters.py --target ithor      # all categories, som + natural
  python reshuffle_v11_answer_letters.py --target procthor   # closest_three, som + splits
  python reshuffle_v11_answer_letters.py --target procthor --categories occlusion
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
from pathlib import Path
from typing import Any

D = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
TARGETS = {
    # name: (files sharing ids, categories to reshuffle or None for all)
    "ithor_som": ([D / "ithor_v11/som/dataset.jsonl"], None),
    "ithor_natural": ([D / "ithor_v11/natural/dataset.jsonl"], None),
    "procthor": ([D / "procthor_v11/som/dataset.jsonl"]
                 + [D / f"procthor_v11/{s}.jsonl" for s in ("test", "val", "train0", "train1", "train2")],
                 {"closest_three"}),
}


def letters_of(r: dict[str, Any]) -> list[str]:
    return [L for L in "ABCDEF" if r.get(f"choice_{L}") is not None]


def balanced_bag(n: int, universe: list[str], rng: random.Random) -> list[str]:
    base, rem = divmod(n, len(universe))
    bag = [u for i, u in enumerate(universe) for _ in range(base + (i < rem))]
    rng.shuffle(bag)
    return bag


def make_perm(r: dict[str, Any], target: str, rng: random.Random) -> dict[str, str]:
    """{new letter: old letter} that moves the current answer onto ``target``."""
    letters = letters_of(r)
    old_ans = list(r["answer"])
    new_ans = list(target)
    rng.shuffle(old_ans)
    perm = dict(zip(new_ans, old_ans))
    rest_new = [L for L in letters if L not in new_ans]
    rest_old = [L for L in letters if L not in old_ans]
    rng.shuffle(rest_old)
    perm.update(zip(rest_new, rest_old))
    return perm


def plan(rows: list[dict[str, Any]], cats: set[str] | None, seed: int) -> dict[str, dict[str, str]]:
    """id -> permutation, balanced per category (and per yes/no polarity for 2-choice)."""
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        if cats is not None and r["category"] not in cats:
            continue
        n = len(letters_of(r))
        pol = r.get("yes_no_gt") if n == 2 else None
        groups[(r["category"], n, pol)].append(r)
    perms = {}
    for key in sorted(groups, key=str):
        grp = groups[key]
        rng = random.Random(f"{seed}:{key}")
        n = key[1]
        letters = "ABCDEF"[:n]
        if len(grp[0]["answer"]) == 3:
            universe = ["".join(c) for c in itertools.combinations(letters, 3)]
        else:
            universe = list(letters)
        for r, t in zip(grp, balanced_bag(len(grp), universe, rng)):
            perms[r["id"]] = make_perm(r, t, rng)
    return perms


def apply_perm(r: dict[str, Any], perm: dict[str, str]) -> dict[str, Any]:
    out = dict(r)
    old = {L: r.get(f"choice_{L}") for L in perm.values()}
    for new_L, old_L in perm.items():
        out[f"choice_{new_L}"] = old[old_L]
    inv = {o: n for n, o in perm.items()}
    out["answer"] = "".join(sorted(inv[L] for L in r["answer"]))
    if "correct_phrase" in r and len(out["answer"]) == 1:
        out["correct_phrase"] = out[f"choice_{out['answer']}"]
    if "correct_phrases" in r:
        out["correct_phrases"] = [out[f"choice_{L}"] for L in out["answer"]]
    # Compose with any earlier reshuffle so choice_perm always points at the source letters.
    prev = r.get("choice_perm") or {L: L for L in perm.values()}
    out["choice_perm"] = {n: prev[o] for n, o in sorted(perm.items())}
    return out


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
