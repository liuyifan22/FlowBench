"""Balanced answer letters: permute each row's choices so correct letters are uniform per category.

Used by ``scripts/curation/balance_answer_letters.py`` (dataset rewrite), the release builder, and the
evaluation scripts (``--choice-seed`` re-permutes the options with the same rule).
"""

from __future__ import annotations

import collections
import itertools
import random
from typing import Any


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
