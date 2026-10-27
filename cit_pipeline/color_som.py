"""Convert numbered-SoM MCQ rows into colour-SoM rows.

Marks come from ``gt_meta.som_marks_rgb{0,1}`` in ``natural/<sample>/sample.json``
and are re-drawn on the clean ``rgb{0,1}`` frames as text-free coloured discs
(``cit_pipeline.som.draw_color_som_on_image``). Mark references in text,
``the bowl (1)`` / ``(A) the dresser``, become ``the bowl (marked blue)`` /
``the dresser (marked green)``. Colours are sampled per question, seeded by the
row id, so colour is independent of role, position, and answer letter.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from cit_pipeline.som import (
    assign_mark_colors,
    draw_color_som_on_image,
    marks_from_meta,
    som_color_tag,
)

TEXT_FIELDS = ("motion_command", "question", "static_question")
CHOICE_KEYS = tuple(f"choice_{c}" for c in "ABCDEF")
# Categories whose choice_X is the object carrying mark X.
OBJECT_CHOICE_CATS = {"closer_to", "closest_three"}

# Static questions are asked on the after image alone; drop move-referring lead-ins.
_MOVE_PREFIX = re.compile(r"^\s*After\s+the\s+move[,.]?\s*", re.IGNORECASE)


def strip_move_prefix(question: Any) -> Any:
    """``After the move, is the vase ...`` → ``Is the vase ...``."""
    if not isinstance(question, str):
        return question
    q = _MOVE_PREFIX.sub("", question, count=1)
    return q[:1].upper() + q[1:] if q else q


# Occlusion questions call the mover "the object (marked C)"; name it as in the motion command.
_MOTION_MOVER = re.compile(r"^move (.+? \(marked \w+\))", re.IGNORECASE)


def name_generic_mover(row: dict[str, Any], text: Any) -> Any:
    """``occluded by the object (marked black)`` → ``occluded by the toaster (marked black)``."""
    if not isinstance(text, str):
        return text
    m = _MOTION_MOVER.match(str(row.get("motion_command") or ""))
    color = (row.get("som_colors") or {}).get("1")
    if not m or not color:
        return text
    return text.replace(f"the object {som_color_tag(color)}", m[1])


_LIST_REF = re.compile(r"\(([A-F])\)\s+([^;?.()]+?)(?=\s*[;?.]|\s*\(|\s*$)")
_INLINE_REF = re.compile(r"\((\w)\)")


def rewrite_text(text: Any, cmap: dict[str, str]) -> Any:
    if not isinstance(text, str):
        return text
    text = _LIST_REF.sub(
        lambda m: f"{m[2]} {som_color_tag(cmap[m[1]])}" if m[1] in cmap else m[0], text
    )
    return _INLINE_REF.sub(lambda m: som_color_tag(cmap[m[1]]) if m[1] in cmap else m[0], text)


def leftover_refs(text: Any) -> list[str]:
    return _INLINE_REF.findall(text) if isinstance(text, str) else []


# Occlusion questions: the label is "mover covers >= 30% of the occludee", but most occludees
# are tiny (smaller than a colour mark) and the question asks whether "a part" is invisible.
# Drop tiny occludees, call any mover cover "occluded", and allow "fully hidden" in the wording.
OCCL_MIN_OCCLUDEE_PX = 150
_OCCL_OLD_CHOICE = "A part is invisible (occluded)."
_OCCL_NEW_CHOICE = "Partly or fully hidden (occluded)."
_OCCL_OLD_Q = "so that a part of it is invisible, or is it totally visible?"
_OCCL_NEW_Q = "so that it is partly or fully hidden, or is it totally visible?"


def fix_occlusion_row(row: dict[str, Any], gt_meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the fixed occlusion row, or None if it must be dropped (tiny occludee)."""
    px = int(gt_meta.get("occludee_mask_pixels_rgb0") or 0)
    cover = float(gt_meta.get("occlusion_rate") or 0.0)
    if px < OCCL_MIN_OCCLUDEE_PX:
        return None
    out = dict(row)
    occl_letter = next(L for L in "AB" if "(occluded)" in str(row.get(f"choice_{L}")))
    relabel = 0.0 < cover < float(gt_meta.get("threshold") or 0.3) and out["answer"] != occl_letter
    if relabel:
        out["answer"] = occl_letter
        out["yes_no_gt"] = out["answer_after"] = True
        out["answer_flipped"] = out.get("answer_before") is not True
    for L in "AB":
        if out.get(f"choice_{L}") == _OCCL_OLD_CHOICE:
            out[f"choice_{L}"] = _OCCL_NEW_CHOICE
    for k in ("question", "static_question"):
        if isinstance(out.get(k), str):
            out[k] = out[k].replace(_OCCL_OLD_Q, _OCCL_NEW_Q)
    out["occludee_px"] = px
    out["occlusion_cover"] = cover
    out["occlusion_relabelled"] = relabel
    return out


def find_question(sample: dict[str, Any], row: dict[str, Any]) -> tuple[dict, dict]:
    for view in sample["views"]:
        if view["view_id"] != row["view_id"]:
            continue
        for q in view["questions"]:
            if q["question_id"] == row["question_id"]:
                return view, q
    raise KeyError(row["id"])


def colorize_row(
    row: dict[str, Any],
    sample: dict[str, Any],
    out_root: Path,
    *,
    image_suffix: str = "_som",
    save_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw colour-SoM images under ``out_root/images`` and return the rewritten row.

    The returned row carries ``_leftover``: mark ids still referenced in text
    but absent from both frames (callers decide whether to drop the row).
    """
    view, q = find_question(sample, row)
    gm = q.get("gt_meta") or {}
    marks0 = marks_from_meta(gm.get("som_marks_rgb0"))
    marks1 = marks_from_meta(gm.get("som_marks_rgb1"))
    cmap = assign_mark_colors([m.id for m in marks0 + marks1], random.Random(row["id"]))

    stem = f"{row['sample_id']}_{row['view_id']}"
    new = dict(row)
    for i, (marks, key) in enumerate(((marks0, "rgb0_path"), (marks1, "rgb1_path"))):
        img = draw_color_som_on_image(Image.open(view[key]), marks, cmap)
        rel = f"images/{stem}_rgb{i}{image_suffix}.png"
        img.save(out_root / rel, **(save_kwargs or {}))
        new["image_before" if i == 0 else "image_after"] = rel

    for k in TEXT_FIELDS:
        new[k] = rewrite_text(row.get(k), cmap)
    new["static_question"] = strip_move_prefix(new.get("static_question"))
    if row.get("category") in OBJECT_CHOICE_CATS:
        for k in CHOICE_KEYS:
            letter = k[-1]
            if row.get(k) and letter in cmap:
                new[k] = f"{row[k]} {som_color_tag(cmap[letter])}"
        letters = [c for c in str(row.get("answer") or "") if f"choice_{c}" in new]
        if "correct_phrase" in row and len(letters) == 1:
            new["correct_phrase"] = new[f"choice_{letters[0]}"]
        if "correct_phrases" in row and letters:
            new["correct_phrases"] = [new[f"choice_{c}"] for c in letters]
    mover = next((m.object_id for m in marks0 + marks1 if m.role == "mover"), None)
    new["mover_object_id"] = mover or (view.get("move") or {}).get("object_id")
    new["som_colors"] = cmap
    new["som_style"] = "color"
    for k in ("question", "static_question"):
        new[k] = name_generic_mover(new, new.get(k))
    if row.get("category") == "occlusion":
        fixed = fix_occlusion_row(new, gm)
        if fixed is None:
            new["_drop"] = "occlusion_tiny_occludee"
        else:
            new = fixed
    new["_leftover"] = sorted(
        {r for k in TEXT_FIELDS + CHOICE_KEYS for r in leftover_refs(new.get(k))}
    )
    return new
