"""MCQ export with Mod-20 occlusion prompt variants.

Loads the legacy bytecode implementation and rewrites occlusion stems/choices
to match ``OCCLUDE_PROMPT_VARIANT`` (ablation winner: partial_or_totally_visible).

Also rewrites ``motion_command`` for SoM exports to the same mark style as
SoM questions (e.g. ``move the bowl (1) 0.5m left``).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Optional

from cit_pipeline.config import OCCLUDE_PROMPT_VARIANT, OCCLUDE_THRESHOLD

_LEGACY_PY = Path(__file__).with_name("_mcq_export_legacy.py")
_LEGACY_PYC = Path(__file__).with_name("_mcq_export_legacy.pyc")
_LEGACY_PATH = _LEGACY_PY if _LEGACY_PY.is_file() else _LEGACY_PYC
_spec = importlib.util.spec_from_file_location("_mcq_export_legacy", _LEGACY_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load legacy mcq_export from {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

_legacy_question_to_mcq = _legacy.question_to_mcq
_legacy_build_static_question = _legacy.build_static_question

AnswerFlipQuota = _legacy.AnswerFlipQuota
LETTERS4 = getattr(_legacy, "LETTERS4", ("A", "B", "C", "D"))
LETTERS6 = getattr(_legacy, "LETTERS6", ("A", "B", "C", "D", "E", "F"))
_legacy_export_sample_to_mcq_records = _legacy.export_sample_to_mcq_records
derive_static_question = _legacy.derive_static_question
derive_static_question_fallback = _legacy.derive_static_question_fallback


def build_som_motion_command(
    move: dict[str, Any],
    *,
    mark_id: str = "1",
) -> str:
    """Natural ``move.text`` → SoM style matching question marks.

    ``move the 1st box from the left 0.73m left`` →
    ``move the box (1) 0.73m left``
    """
    em = dict(move.get("edit_meta") or {})
    if em.get("text_som"):
        return str(em["text_som"]).strip()

    from cit_pipeline.som import som_marked_phrase

    motion = str(move.get("text") or "").strip()
    otype = move.get("object_type") or "object"
    som_p = som_marked_phrase(otype, str(mark_id))
    if not motion:
        return f"move {som_p}"
    # Displacement starts at the first ``X.YYm`` / ``Xm`` token (not ordinals like ``1st``).
    m = re.match(r"(?is)^move\s+(.+?)\s+(\d+\.\d+m(?:\s+.*)?)$", motion)
    if m:
        return f"move {som_p} {m.group(2).strip()}"
    m = re.match(r"(?is)^move\s+(.+?)\s+(\d+m(?:\s+.*)?)$", motion)
    if m:
        return f"move {som_p} {m.group(2).strip()}"
    if motion.lower().startswith("move "):
        return f"move {som_p}"
    return f"move {som_p} {motion}".strip()


def _mover_mark_id(q: Optional[dict[str, Any]]) -> str:
    meta = dict((q or {}).get("gt_meta") or {})
    for key in ("som_marks", "som_marks_rgb0", "som_marks_rgb1"):
        for m in meta.get(key) or []:
            if not isinstance(m, dict):
                continue
            if str(m.get("role") or "") == "mover" or str(m.get("id") or "") == "1":
                return str(m.get("id") or "1")
    return "1"


def export_sample_to_mcq_records(*args, **kwargs):
    """Wrap legacy export; restore incomplete-GT rows as ``answer='no answer'``.

    Also surface VLM usage onto each jsonl row (``vlm_used`` / ``vlm_worded``),
    and for SoM exports rewrite ``motion_command`` to mark style.
    """
    recs = _legacy_export_sample_to_mcq_records(*args, **kwargs)
    sample = args[0] if args else kwargs.get("sample")
    som = bool(kwargs.get("som", False))
    if not isinstance(sample, dict):
        return recs
    sample_meta = dict(sample.get("meta") or {})
    sample_vlm_used = sample_meta.get("vlm_used")
    by_qid: dict[str, dict] = {}
    move_by_view: dict[str, dict] = {}
    for view in sample.get("views") or []:
        vid = str(view.get("view_id") or "")
        if vid:
            move_by_view[vid] = dict(view.get("move") or {})
        for q in view.get("questions") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                by_qid[qid] = q
    out = []
    for rec in recs or []:
        rec = dict(rec)
        rid = str(rec.get("id") or "")
        qid = rid.split("__")[-1] if "__" in rid else str(rec.get("question_id") or "")
        q = by_qid.get(qid)
        gm = dict((q or {}).get("gt_meta") or {}) if q is not None else {}
        if sample_vlm_used is not None:
            rec["vlm_used"] = bool(sample_vlm_used)
        elif "vlm_worded" in gm:
            rec["vlm_used"] = bool(gm.get("vlm_worded"))
        if "vlm_worded" in gm:
            rec["vlm_worded"] = bool(gm.get("vlm_worded"))
        skip = gm.get("vlm_skip_reason") or sample_meta.get("vlm_skip_reason")
        if skip:
            rec["vlm_skip_reason"] = skip
        elif rec.get("vlm_used") is False:
            rec["vlm_skip_reason"] = "vlm_bypassed"
        if q is not None and not bool(q.get("gt_complete", True)):
            rec["gt_complete"] = False
            rec["answer"] = "no answer"
            rec["answer_after"] = None
            rec["needs_manual_annotation"] = True
            rec["answer_status"] = "no_answer"
            for k in ("cot_cold", "cot_warm", "cot_pretended"):
                if k in rec:
                    rec[k] = None
        if som:
            move = move_by_view.get(str(rec.get("view_id") or "")) or {}
            rec["motion_command"] = build_som_motion_command(
                move, mark_id=_mover_mark_id(q)
            )
        out.append(rec)
    return out


_legacy.export_sample_to_mcq_records = export_sample_to_mcq_records


def _occlusion_choice_pair(variant: str) -> tuple[str, str]:
    if variant == "visible_or_occluded":
        return "Occluded.", "Visible."
    if variant == "any_part_hidden":
        return "Yes — some part is hidden.", "No — it is fully visible."
    if variant == "blocked_from_view":
        return "Yes, it is blocked.", "No, it is not blocked."
    if variant == "baseline":
        return (
            "Yes — the moved object covers it enough to count as occluded.",
            "No — it remains sufficiently visible (not meaningfully occluded).",
        )
    return (
        "A part is invisible (occluded).",
        "Totally visible (not occluded).",
    )


def _occlusion_static_stem(
    *,
    occludee: str,
    mover: str,
    threshold: float,
    variant: str,
) -> str:
    if variant == "baseline":
        core = (
            f"Is {occludee} substantially occluded by {mover} "
            f"(at least {int(threshold * 100)}% of its visible pixels)"
        )
    elif variant == "visible_or_occluded":
        core = f"Is {occludee} visible, or is it occluded by {mover}"
    elif variant == "any_part_hidden":
        core = f"Is any part of {occludee} hidden behind {mover}"
    elif variant == "blocked_from_view":
        core = f"Is {occludee} blocked from view by {mover}"
    else:
        core = (
            f"Is {occludee} occluded by {mover} so that a part of it is "
            f"invisible, or is it totally visible"
        )
    return core + "?"


def build_static_question(
    *,
    category: str,
    question: str,
    gt_meta: Optional[dict[str, Any]] = None,
    som: bool = False,
) -> str:
    if str(category) != "occlusion":
        return _legacy_build_static_question(
            category=category, question=question, gt_meta=gt_meta, som=som
        )

    meta = dict(gt_meta or {})
    variant = str(meta.get("occlusion_prompt_variant") or OCCLUDE_PROMPT_VARIANT)
    thr = float(meta.get("threshold") or OCCLUDE_THRESHOLD)
    if som:
        ot = str(meta.get("occludee_type") or "object").lower()
        mt = str(meta.get("mover_type") or "object").lower()
        # Prefer explicit mover type from referring meta when present.
        ref = meta.get("mover_referring") or {}
        if isinstance(ref, dict) and ref.get("object_type"):
            mt = str(ref["object_type"]).lower()
        occ = f"the {ot} (2)"
        mov = f"the {mt} (1)"
    else:
        occ = str(meta.get("occludee_phrase") or meta.get("occludee_type") or "the object")
        mov = str(meta.get("mover_phrase") or "the moved object")
    return _occlusion_static_stem(
        occludee=occ, mover=mov, threshold=thr, variant=variant
    )


def question_to_mcq(q: dict[str, Any], *, rng, som: bool = False, **kwargs: Any):
    # Incomplete GT → keep a "no answer" MCQ row for manual annotation.
    if not bool((q or {}).get("gt_complete", True)):
        meta = dict((q or {}).get("gt_meta") or {})
        text = str((q or {}).get("text_som") if som and (q or {}).get("text_som") else (q or {}).get("text") or "")
        static = str((q or {}).get("text") or text)
        if som and str((q or {}).get("category") or "") == "occlusion":
            try:
                static = build_static_question(
                    category="occlusion",
                    question=text,
                    gt_meta=meta,
                    som=True,
                )
                text = f"In the after image, {static[0].lower() + static[1:]}"
            except Exception:  # noqa: BLE001
                pass
        return {
            "category": (q or {}).get("category"),
            "question": text,
            "static_question": static,
            "answer": "no answer",
            "answer_after": None,
            "gt_complete": False,
            "needs_manual_annotation": True,
            "answer_status": "no_answer",
            "choice_A": "no answer",
            "choice_B": "(reserved for manual annotation)",
            "mcq_type": "no_answer",
            "n_choices": 2,
            "gt_meta": meta,
            "som": bool(som),
        }

    rec = _legacy_question_to_mcq(q, rng=rng, som=som, **kwargs)
    if rec is None and str((q or {}).get("category") or "") == "count_near":
        # Mod-30: 4-way numeric MCQ (2/3/4/5) — not in legacy exporter.
        meta = dict((q or {}).get("gt_meta") or {})
        choices = [str(x) for x in (meta.get("choices") or ["2", "3", "4", "5"])]
        ans_val = str((q or {}).get("answer") or meta.get("answer_after") or "").strip()
        if ans_val not in choices:
            return None
        letters = list(LETTERS4)[: len(choices)]
        # Shuffle choice↔letter mapping.
        order = list(range(len(choices)))
        rng.shuffle(order)
        letter_of = {}
        rec = {
            "category": "count_near",
            "question": str(
                (q or {}).get("text_som")
                if som and (q or {}).get("text_som")
                else (q or {}).get("text")
                or ""
            ),
            "static_question": str((q or {}).get("text") or ""),
            "gt_complete": True,
            "gt_meta": meta,
            "som": bool(som),
            "mcq_type": "count_near",
            "n_choices": len(choices),
            "thresh_m": meta.get("thresh_m"),
            "count_before": meta.get("count_before"),
            "count_after": meta.get("count_after"),
            "answer_flipped": meta.get("answer_flipped"),
        }
        for i, idx in enumerate(order):
            let = letters[i]
            val = choices[idx]
            rec[f"choice_{let}"] = val
            letter_of[val] = let
        rec["answer"] = letter_of[ans_val]
        rec["answer_after"] = ans_val
        if meta.get("count_before") is not None:
            rec["answer_before"] = str(meta.get("count_before"))
        return rec
    if rec is None:
        return None
    # Ensure category is stamped (legacy closest_three sometimes omits it).
    if not rec.get("category"):
        rec["category"] = (q or {}).get("category")
    if str((q or {}).get("category") or "") != "occlusion":
        return rec

    meta = dict((q or {}).get("gt_meta") or {})
    variant = str(meta.get("occlusion_prompt_variant") or OCCLUDE_PROMPT_VARIANT)
    thr = float(meta.get("threshold") or OCCLUDE_THRESHOLD)
    yes_t, no_t = _occlusion_choice_pair(variant)

    natural_static = build_static_question(
        category="occlusion",
        question=str((q or {}).get("text") or ""),
        gt_meta=meta,
        som=False,
    )
    som_static = build_static_question(
        category="occlusion",
        question=str((q or {}).get("text_som") or (q or {}).get("text") or ""),
        gt_meta=meta,
        som=True,
    )
    rec["static_question"] = som_static if som else natural_static
    stem = som_static if som else natural_static
    rec["question"] = f"In the after image, {stem[0].lower() + stem[1:]}"

    ans = str(rec.get("answer") or "A").strip().upper()
    other = "B" if ans == "A" else "A"
    occluded = bool((q or {}).get("answer"))
    if "answer_after" in rec and rec.get("answer_after") is not None:
        occluded = bool(rec.get("answer_after"))
    if occluded:
        rec[f"choice_{ans}"] = yes_t
        rec[f"choice_{other}"] = no_t
    else:
        rec[f"choice_{ans}"] = no_t
        rec[f"choice_{other}"] = yes_t
    rec["occlusion_prompt_variant"] = variant
    rec["occlusion_threshold"] = thr
    return rec


_legacy.question_to_mcq = question_to_mcq
_legacy.build_static_question = build_static_question
_legacy.export_sample_to_mcq_records = export_sample_to_mcq_records
