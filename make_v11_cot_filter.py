#!/usr/bin/env python3
"""Reasoning allowlist for ProcTHOR v11 colour-SoM rows.

Warm CoTs live under ``<source>/natural/cot/{id}_warm.txt`` and were written for
the natural (unmarked) question, whose choices are shuffled independently of the
SoM row. A row may reuse its CoT only when the CoT exists and the natural GT
matches the SoM GT: yes/no polarity for binary paraphrases, else mark-stripped
choice phrases (``som_natural_cot_compatible`` below, same rule as the PSI tokenizer, with
``(marked <colour>)`` tags stripped as well as ``(1)`` badges).

Writes ``<v11>/filters/som_cot_ids.txt`` (all splits) + ``cot_filter_meta.json``.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

SRC_ROOT = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
V11 = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")

# --- SoM / natural GT comparison (same rule as the PSI tokenizer's cit_tokenize) ---
# Strip colour tags too, so the canonical GT compares object names only.
_BADGE_RE = re.compile(r"\s*\((?:\d+|marked \w+)\)")
_LETTER_RE = re.compile(r"[^A-J]")
# Yes/no templates used by VLM option paraphrases (feasibility, landmark, wall, near).
_NO_RE = re.compile(
    r"\bno\b|\bfalse\b|feasible:\s*no|would collide|infeasible|does not|"
    r"not hold|not correct|stays clear|still clearance|blocked|"
    r"wall contact after the move:\s*no|: no\.?\s*$",
    re.IGNORECASE,
)
_YES_RE = re.compile(
    r"\byes\b|\btrue\b|feasible:\s*yes|can succeed|touching a wall|"
    r"in contact|against / touching|wall contact after the move:\s*yes|"
    r": yes\.?\s*$",
    re.IGNORECASE,
)


def _badge_strip(text: str) -> str:
    return _BADGE_RE.sub("", text or "").strip().lower()


def _answer_letters(rec: dict[str, Any]) -> str:
    return _LETTER_RE.sub("", str(rec.get("answer") or "").upper())


def _choice_map(rec: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in "ABCDEFGHIJ":
        v = rec.get(f"choice_{k}")
        if v is not None:
            out[k] = str(v).strip()
    return out


def _gt_phrases(rec: dict[str, Any]) -> tuple[str, ...]:
    ch = _choice_map(rec)
    return tuple(ch.get(L, "") for L in _answer_letters(rec))


def _yn_polarity(phrase: str) -> Optional[str]:
    """Return 'YES'/'NO' for binary option paraphrases, else None (object/number)."""
    p = _badge_strip(phrase)
    if not p:
        return None
    no_hit = bool(_NO_RE.search(p))
    yes_hit = bool(_YES_RE.search(p))
    if "wall contact after the move: no" in p or p.endswith(": no") or p.endswith(": no."):
        return "NO"
    if "wall contact after the move: yes" in p or p.endswith(": yes") or p.endswith(": yes."):
        return "YES"
    if no_hit and yes_hit:
        if re.search(r"not correct", p):
            return "NO"
        if re.match(r"no\b", p):
            return "NO"
        if re.match(r"yes\b", p):
            return "YES"
    if no_hit:
        return "NO"
    if yes_hit:
        return "YES"
    return None


def _canon_gt_tuple(rec: dict[str, Any]) -> tuple[str, ...]:
    """Canonical GT for SoM/natural compatibility: polarity if binary, else badge-stripped names."""
    atoms = []
    for p in _gt_phrases(rec):
        pol = _yn_polarity(p)
        atoms.append(pol if pol is not None else _badge_strip(p))
    return tuple(sorted(atoms))


def som_natural_cot_compatible(nat_rec: dict[str, Any], som_rec: dict[str, Any]) -> bool:
    """True if natural warm CoT can be reused on the SoM question.

    Accepts exact phrase match after stripping SoM badges, and yes/no polarity
    match when the VLM paraphrased the option templates (feasibility, landmark,
    contact_wall, near_object). Rejects collapsed-object mismatches
    (drawer vs dresser, numbered chairs vs 'the chair').
    """
    return _canon_gt_tuple(nat_rec) == _canon_gt_tuple(som_rec)


def main() -> None:
    som = {}
    with open(V11 / "som" / "dataset.jsonl") as f:
        for line in f:
            r = json.loads(line)
            som[r["id"]] = r
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    keep: list[str] = []
    for src in ("procthor_v9", "procthor_v10"):
        cot_dir = SRC_ROOT / src / "natural" / "cot"
        cot_files = set(os.listdir(cot_dir))
        with open(SRC_ROOT / src / "natural" / "dataset.jsonl") as f:
            for line in f:
                nat = json.loads(line)
                s = som.get(nat["id"])
                if s is None or s["source_version"] != src:
                    continue
                c = stats[s["category"]]
                c["n"] += 1
                if f"{nat['id']}_warm.txt" not in cot_files:
                    c["no_cot"] += 1
                    continue
                if not som_natural_cot_compatible(nat, s):
                    c["gt_mismatch"] += 1
                    continue
                c["keep"] += 1
                keep.append(nat["id"])
    out = V11 / "filters"
    out.mkdir(exist_ok=True)
    keep.sort()
    (out / "som_cot_ids.txt").write_text("\n".join(keep) + "\n")
    n = sum(c["n"] for c in stats.values())
    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "som_jsonl": str(V11 / "som" / "dataset.jsonl"),
        "cot_dirs": [str(SRC_ROOT / s / "natural" / "cot") for s in ("procthor_v9", "procthor_v10")],
        "n_rows": len(som),
        "n_paired": n,
        "n_keep": len(keep),
        "rule": "warm CoT exists + canonical GT equal (polarity for yes/no, else mark-stripped phrases)",
        "by_category": {k: dict(v) for k, v in sorted(stats.items())},
    }
    (out / "cot_filter_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
