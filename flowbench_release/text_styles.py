"""Per-style question text for the FlowBench sim release (natural / colour SoM / red SoM).

The v11 colour-SoM rows are the reference: they carry every v11 fix (filters, occlusion rewording and
relabel, static-question cleanup, mover naming, balanced answer letters via ``choice_perm``).
Red-SoM and natural rows are rebuilt from their own source exports with the same fixes and then put
into the colour row's choice order, so all three styles of a question share choices and answer letter.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from cit_pipeline.color_som import (
    CHOICE_KEYS,
    OBJECT_CHOICE_CATS,
    TEXT_FIELDS,
    fix_occlusion_row,
    rewrite_text,
    strip_move_prefix,
)

LETTERS = "ABCDEF"
_RED_MOTION_MOVER = re.compile(r"^move (.+? \(1\))", re.IGNORECASE)


def letters_of(r: dict[str, Any]) -> list[str]:
    return [L for L in LETTERS if r.get(f"choice_{L}") is not None]


def permute_choices(r: dict[str, Any], perm: Optional[dict[str, str]]) -> dict[str, Any]:
    """Apply ``perm`` ({new letter: source letter}) to choices and answer."""
    if not perm or all(k == v for k, v in perm.items()):
        return r
    out = dict(r)
    inv = {src: new for new, src in perm.items()}
    for new, src in perm.items():
        out[f"choice_{new}"] = r.get(f"choice_{src}")
    out["answer"] = "".join(sorted(inv[c] for c in str(r["answer"])))
    return out


def red_name_generic_mover(row: dict[str, Any], text: Any) -> Any:
    """Numbered twin of ``color_som.name_generic_mover``: ``the object (1)`` → ``the toaster (1)``."""
    if not isinstance(text, str):
        return text
    m = _RED_MOTION_MOVER.match(str(row.get("motion_command") or ""))
    return text.replace("the object (1)", m[1]) if m else text


def red_row(src: dict[str, Any], colour: dict[str, Any], gt_meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Red (numbered) SoM row from the v9/v10 ``dataset_static_som`` row, with the v11 text fixes."""
    r = dict(src)
    r["static_question"] = strip_move_prefix(r.get("static_question"))
    for k in ("question", "static_question"):
        r[k] = red_name_generic_mover(r, r.get(k))
    if r.get("category") == "occlusion":
        r = fix_occlusion_row(r, gt_meta)
        if r is None:
            return None
    return permute_choices(r, colour.get("choice_perm"))


def colourize_red(red: dict[str, Any], colour: dict[str, Any]) -> dict[str, Any]:
    """What the colour text should be, given the red row: used to check the two styles agree."""
    cmap = colour.get("som_colors") or {}
    out = {k: rewrite_text(red.get(k), cmap) for k in TEXT_FIELDS}
    perm = colour.get("choice_perm") or {}
    for k in CHOICE_KEYS:
        v = red.get(k)
        if v is not None and red.get("category") in OBJECT_CHOICE_CATS:
            src_letter = perm.get(k[-1], k[-1])
            v = f"{v} (marked {cmap[src_letter]})" if src_letter in cmap else v
        out[k] = v
    return out


_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b|\bfrom the (left|right)\b")


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z]", "", str(s or "").lower())


def _choice_keys(r: dict[str, Any], gt_meta: dict[str, Any], *, natural: bool,
                 perm: Optional[dict[str, str]] = None) -> Optional[dict[str, str]]:
    """{letter: key} identifying what each choice *is*, comparable across styles; None if ambiguous."""
    cat = r.get("category")
    letters = letters_of(r)
    if cat in OBJECT_CHOICE_CATS:
        if not natural:  # SoM: choice X is the object carrying mark perm[X]
            marks = {m["id"]: m["object_id"] for m in (gt_meta.get("som_marks_rgb0") or [])
                     + (gt_meta.get("som_marks_rgb1") or [])}
            keys = {L: marks.get((perm or {}).get(L, L)) for L in letters}
        else:
            pool = dict(zip(gt_meta.get("candidate_pool") or [], gt_meta.get("candidate_ids") or []))
            keys = {L: pool.get(r[f"choice_{L}"]) for L in letters}
        return keys if all(keys.values()) and len(set(keys.values())) == len(keys) else None
    texts = {L: _norm(r[f"choice_{L}"]) for L in letters}
    return texts if len(set(texts.values())) == len(texts) else None


def _type_keys(r: dict[str, Any], som_keys: dict[str, str]) -> Optional[dict[str, str]]:
    """closer_to has no candidate pool: match natural phrases to SoM objects by object type."""
    types = {oid: _norm(oid.split("|")[0]) for oid in som_keys.values()}
    keys = {}
    for L in letters_of(r):
        phrase = _norm(_ORDINAL.sub(" ", str(r[f"choice_{L}"])))
        hits = [oid for oid, t in types.items() if t and t in phrase]
        if len(hits) != 1:
            return None
        keys[L] = hits[0]
    return keys if len(set(keys.values())) == len(keys) else None


def natural_row(src: dict[str, Any], colour: dict[str, Any], gt_meta: dict[str, Any],
                fix_static) -> tuple[Optional[dict[str, Any]], str]:
    """Natural row with the v11 fixes, in the colour row's choice order. Returns (row, status)."""
    r = dict(src)
    r["static_question"] = fix_static(r)
    if r.get("category") == "occlusion":
        r = fix_occlusion_row(r, gt_meta)
        if r is None:
            return None, "occlusion_dropped"
    letters = letters_of(r)
    if letters != letters_of(colour):
        return r, "n_choices_differ"
    som_keys = _choice_keys(colour, gt_meta, natural=False, perm=colour.get("choice_perm"))
    if colour.get("category") in OBJECT_CHOICE_CATS:
        nat_keys = _choice_keys(r, gt_meta, natural=True)
        if nat_keys is None and som_keys is not None and colour.get("category") == "closer_to":
            nat_keys = _type_keys(r, som_keys)
    else:
        nat_keys = _choice_keys(r, gt_meta, natural=True)
        # 2-choice / count choices are the same strings in every export
        som_keys = _choice_keys({**colour, **{f"choice_{L}": colour[f"choice_{L}"] for L in letters}},
                                gt_meta, natural=True)
    if nat_keys is None or som_keys is None or set(nat_keys.values()) != set(som_keys.values()):
        if len(letters) == 2:  # yes/no worded differently: align by correctness
            na, ca = str(r["answer"]), str(colour["answer"])
            perm = {ca: na, ({"A", "B"} - {ca}).pop(): ({"A", "B"} - {na}).pop()}
            out = permute_choices(r, perm)
            return out, "aligned_by_answer"
        return r, "unaligned"
    by_key = {k: L for L, k in nat_keys.items()}
    perm = {L: by_key[som_keys[L]] for L in letters}  # colour letter -> natural source letter
    out = permute_choices(r, perm)
    return out, "aligned" if out["answer"] == colour["answer"] else "answer_disagrees"
