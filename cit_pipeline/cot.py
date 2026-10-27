"""Chain-of-thought (CoT) annotations for CIT QA (Modification 7 + pretended).

Three layers
------------
1. **Cold** — deterministic templates filled from ``Question.gt_meta`` / move
   metadata. Precise and faithful to the GT rule for each category.
2. **Warm** — Qwen3-VL rewrite of the cold text, conditioned on before+after
   images, written as if an agent inferred everything visually (no metadata).
3. **Pretended** — Qwen3-VL rewrite that may peek at the warm CoT for hints,
   but is shown **only the before image** + move command. The after frame is
   imagination only (never shown to the student agent).

Answers are never changed; CoT only explains them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from cit_pipeline.vlm_client import VlmClient

WARM_SYSTEM = (
    "You write short visual chain-of-thought for spatial VQA. "
    "You must keep the same final answer as the given cold analysis. "
    "Write as an agent that only sees the images and the move command — "
    "never mention metadata, masks, teleports, AABB, bearings in exact "
    "degrees, occlusion rates, or that you were given numbers. "
    "You may use qualitative visual cues (left/right in the frame, "
    "overlap, floating, near a wall, closer/farther). "
    "Return ONLY the reasoning paragraphs, ending with a clear Answer line."
)

PRETENDED_SYSTEM = (
    "You write short counterfactual chain-of-thought for a student agent that "
    "sees ONLY the before image and a motion command. The after image is "
    "imagination — never claim you saw it, never refer to 'the second image' "
    "or 'after photo'. You may use a reference warm CoT as private hints about "
    "what happens, but rewrite as if you imagined the outcome from the before "
    "frame + move alone. Keep the same final answer. "
    "No metadata jargon (masks, AABB, teleports, exact degrees). "
    "Return ONLY the reasoning paragraphs, ending with a clear Answer line."
)


def _fmt_answer(answer: Any) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, list):
        return ", ".join(str(x) for x in answer)
    if answer is None:
        return "unknown"
    return str(answer)


def _f(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _mover_phrase(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    ref = meta.get("mover_referring") or {}
    if ref.get("phrase"):
        return str(ref["phrase"])
    if meta.get("mover_phrase"):
        return str(meta["mover_phrase"])
    return f"the {str(move.get('object_type', 'object')).lower()}"


def _yes_no_conclusion(answer: Any) -> str:
    return "yes" if bool(answer) else "no"


# ---------------------------------------------------------------------------
# Cold templates (one per category)
# ---------------------------------------------------------------------------
def cold_occlusion(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    b = meta.get("occludee_phrase") or meta.get("occludee_type") or "the occludee"
    rate = meta.get("occlusion_rate")
    thr = meta.get("threshold", 0.3)
    rem = meta.get("occlusion_remaining_b")
    metric = meta.get("occlusion_metric", "cover of B's original pixels by A")
    lines = [
        f"Question type: occlusion. We ask whether {b} is substantially occluded by {a} "
        f"after the move ({move.get('text', '')}).",
        f"Rule: compute the fraction of {b}'s before-image region that is covered by {a} "
        f"in the after image ({metric}). Threshold = {_f(thr, 2)}.",
        f"Measured cover rate = {_f(rate, 3)}"
        + (f"; remaining visible fraction of {b} ≈ {_f(rem, 3)}." if rem is not None else "."),
        f"Since cover {'≥' if bool(q.get('answer')) else '<'} {_f(thr, 2)}, "
        f"the answer is {_yes_no_conclusion(q.get('answer'))}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_closer_to(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    pa, pb = meta.get("phrase_a"), meta.get("phrase_b")
    da, db = meta.get("distance_a"), meta.get("distance_b")
    closer = meta.get("closer_phrase")
    far = meta.get("far_phrase")
    absent = meta.get("absent_distractor")
    choices = meta.get("choices") or []
    letter = meta.get("answer_letter")
    lines = [
        f"Question type: closer_to (4-way MCQ). After ({move.get('text', '')}), "
        f"which listed landmark is {a} closer to?",
        f"Compare 3D centroid distances from {a}: "
        f"{pa} at {_f(da, 3)} m vs {pb} at {_f(db, 3)} m.",
        f"The nearer of the two is {closer}.",
        f"MCQ also includes a far in-room distractor ({far}) and an absent type "
        f"({absent}). Choices: {choices}.",
        f"Correct choice is {closer}"
        + (f" (letter {letter})." if letter else "."),
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_contact_wall(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    edit = move.get("edit_meta") or {}
    a = _mover_phrase(q, move)
    before = meta.get("min_edge_clearance_before_m")
    after = meta.get("min_edge_clearance_m")
    thr = meta.get("threshold_m", 0.1)
    blocked = meta.get("wall_blocked_next_step")
    steps = meta.get("wall_steps_ok")
    toward = meta.get("wall_toward", edit.get("wall_toward"))
    toward_txt = (
        "toward the nearest wall"
        if toward is True
        else ("away from the nearest wall" if toward is False else "near a wall")
    )
    lines = [
        f"Question type: contact_wall. After ({move.get('text', '')}), "
        f"is {a} in contact with a wall?",
        f"The edit slides {toward_txt}.",
        f"Rule: yes if (for toward-wall moves) the next step is blocked, "
        f"or if min edge clearance < {_f(thr, 2)} m.",
        f"Clearance before ≈ {_f(before, 3)} m; after ≈ {_f(after, 3)} m; "
        f"successful wall steps = {steps}; next step blocked = {blocked}.",
        f"Therefore the answer is {_yes_no_conclusion(q.get('answer'))}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_affordance(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    mode = meta.get("affordance_mode") or (move.get("edit_meta") or {}).get(
        "affordance_mode"
    )
    tgt = meta.get("target_phrase") or "the landmark"
    ang = meta.get("angle_deg")
    half = meta.get("fov_half_angle_deg", 45.0)
    if mode == "observer_chair":
        seat = meta.get("seat_phrase") or "the chair"
        mover = _mover_phrase(q, move)
        lines = [
            f"Question type: affordance (observer chair). When sitting on {seat}, "
            f"is {mover} within ±{_f(half, 0)}° of the seat's forward heading "
            f"after {mover} moved?",
            f"Rule: use the seat yaw as facing; measure the horizontal angle from "
            f"that forward axis to the moved object ({mover}).",
            f"Measured angle = {_f(ang, 2)}° (half-FOV = {_f(half, 1)}°).",
            f"Since {_f(ang, 2)}° {'≤' if bool(q.get('answer')) else '>'} {_f(half, 1)}°, "
            f"the answer is {_yes_no_conclusion(q.get('answer'))}.",
            f"Answer: {_fmt_answer(q.get('answer'))}",
        ]
        return "\n".join(lines)

    a = meta.get("seat_phrase") or _mover_phrase(q, move)
    lines = [
        f"Question type: affordance (sitting forward FOV). When sitting on {a}, "
        f"is {tgt} within ±{_f(half, 0)}° of the seat's forward heading?",
        f"Rule: use the seat yaw as facing; measure the horizontal angle from "
        f"that forward axis to {tgt}.",
        f"Measured angle = {_f(ang, 2)}° (half-FOV = {_f(half, 1)}°).",
        f"Since {_f(ang, 2)}° {'≤' if bool(q.get('answer')) else '>'} {_f(half, 1)}°, "
        f"the answer is {_yes_no_conclusion(q.get('answer'))}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_left_right_landmark(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    land = meta.get("landmark_phrase") or "the landmark"
    asked = meta.get("asked", "left")
    b_a = meta.get("bearing_after_deg")
    b_l = meta.get("landmark_bearing_deg")
    # More negative bearing ⇒ farther left in the image.
    if asked == "left":
        relation = "to the left of" if bool(q.get("answer")) else "not to the left of"
        cmp = f"{_f(b_a, 2)}° {'<' if bool(q.get('answer')) else '≥'} {_f(b_l, 2)}°"
    else:
        relation = "to the right of" if bool(q.get("answer")) else "not to the right of"
        cmp = f"{_f(b_a, 2)}° {'>' if bool(q.get('answer')) else '≤'} {_f(b_l, 2)}°"
    lines = [
        f"Question type: left_right_landmark. In the after image, is {a} {asked} of {land} "
        f"from the camera's viewpoint?",
        "Rule: compare camera bearings atan2(x_cam, z_cam). More negative bearing means "
        "farther left on screen (image-space, not raw lateral meters).",
        f"After-move bearing of {a} = {_f(b_a, 2)}°; bearing of {land} = {_f(b_l, 2)}°.",
        f"Comparison for asked='{asked}': {cmp}. So {a} is {relation} {land}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_feasibility(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    d = meta.get("probe_direction")
    m = meta.get("probe_meters")
    ok = meta.get("teleport_success")
    ortho = meta.get("orthogonal_to_original")
    lines = [
        f"Question type: feasibility. Starting from the after pose, would moving {a} "
        f"{_f(m, 2)} m {d} be feasible without colliding?",
        f"Rule: attempt a collision-checked translation (forceAction=False) of "
        f"{_f(m, 2)} m {d} from the after pose; restore afterward. "
        f"Probe orthogonal to the original move = {ortho}.",
        f"Simulator lastActionSuccess = {ok}.",
        f"Therefore the answer is {_yes_no_conclusion(q.get('answer'))}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_support(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    before = meta.get("parent_receptacles_before") or []
    after = meta.get("parent_receptacles_after") or []
    on_furn = meta.get("on_furniture")
    on_f = meta.get("on_floor")
    dropped = meta.get("dropped_to_floor")
    yb, ya = meta.get("y_before"), meta.get("y_after")
    hb = meta.get("height_before", yb)
    ha = meta.get("height_after", ya)
    drop = meta.get("height_drop_m")
    thresh = meta.get("support_drop_height_m")
    lines = [
        f"Question type: support. After ({move.get('text', '')}), will {a} drop "
        f"onto the floor instead of staying well-supported by furniture?",
        "Rule: apply the collision-checked slide, then release kinematics and "
        "settle under gravity; compare height before vs after (drop ≥ "
        f"{_f(thresh, 2)} m ⇒ floor drop). Receptacles corroborate.",
        f"Receptacles before = {before}; after = {after}. "
        f"on_furniture={on_furn}, on_floor={on_f}, dropped_to_floor={dropped}. "
        f"Height: {_f(hb, 3)} → {_f(ha, 3)} m (Δ={_f(drop, 3)} m). "
        f"settle_gravity={meta.get('settle_gravity')}.",
        f"Therefore the answer is {_yes_no_conclusion(q.get('answer'))}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


def cold_closest_three(q: dict[str, Any], move: dict[str, Any]) -> str:
    meta = q.get("gt_meta") or {}
    a = _mover_phrase(q, move)
    pool = meta.get("candidate_pool") or []
    dists = meta.get("candidate_distances") or []
    ans = meta.get("answer_phrases") or q.get("answer") or []
    ranked = []
    for i, phrase in enumerate(pool):
        d = dists[i] if i < len(dists) else "?"
        ranked.append(f"{phrase} ({_f(d, 3)} m)")
    lines = [
        f"Question type: closest_three. After ({move.get('text', '')}), which three "
        f"objects are closest to {a}?",
        f"Rule: rank a pool of {meta.get('pool_size', len(pool))} candidates by "
        f"3D centroid distance; take the top {meta.get('answer_size', 3)}.",
        "Ranked pool: " + "; ".join(ranked) + ".",
        f"Top three: {ans}.",
        f"Answer: {_fmt_answer(q.get('answer'))}",
    ]
    return "\n".join(lines)


COLD_BUILDERS = {
    "occlusion": cold_occlusion,
    "closer_to": cold_closer_to,
    "contact_wall": cold_contact_wall,
    "affordance": cold_affordance,
    "left_right_landmark": cold_left_right_landmark,
    "feasibility": cold_feasibility,
    "support": cold_support,
    "closest_three": cold_closest_three,
}


def build_cold_cot(q: dict[str, Any], move: dict[str, Any]) -> str:
    cat = q.get("category")
    builder = COLD_BUILDERS.get(str(cat))
    if builder is None:
        return (
            f"Question type: {cat}. Move: {move.get('text')}. "
            f"Answer: {_fmt_answer(q.get('answer'))}"
        )
    return builder(q, move)


def polish_warm_cot(
    vlm: VlmClient,
    *,
    question_text: str,
    move_text: str,
    answer: Any,
    cold: str,
    rgb1_path: Path,
    rgb0_path: Optional[Path] = None,
) -> str:
    """Rewrite cold CoT as image-grounded agent reasoning (warm)."""
    prompt = (
        "Rewrite the cold analysis below into a warm visual chain-of-thought.\n"
        "Constraints:\n"
        "- Keep the SAME final answer.\n"
        "- Do not contradict the cold facts; soften numbers into qualitative cues.\n"
        "- Pretend you only looked at the images + move command.\n"
        "- 4–8 short sentences, then a final line: Answer: <same answer>\n\n"
        f"Move command: {move_text}\n"
        f"Question: {question_text}\n"
        f"Ground-truth answer (must match): {_fmt_answer(answer)}\n\n"
        f"Cold analysis:\n{cold}\n"
    )
    images: list[Path] = []
    if rgb0_path is not None and rgb0_path.is_file():
        images.append(rgb0_path)
    if rgb1_path.is_file():
        images.append(rgb1_path)
    raw = vlm.chat(
        prompt,
        images=images or None,
        system=WARM_SYSTEM,
        max_tokens=700,
        temperature=0.3,
    )
    warm = raw.strip()
    # Ensure an Answer line exists with the GT answer.
    ans = _fmt_answer(answer)
    if not re.search(r"(?i)\banswer\s*:", warm):
        warm = warm.rstrip() + f"\nAnswer: {ans}"
    return warm


def polish_pretended_cot(
    vlm: VlmClient,
    *,
    question_text: str,
    move_text: str,
    answer: Any,
    warm: Optional[str],
    cold: Optional[str],
    rgb0_path: Path,
) -> str:
    """CoT as if imagining rgb1 from rgb0 + move only (pretended).

    The VLM may read the warm CoT as private hints, but is shown **only** the
    before image. The after image must not be provided.
    """
    hint = warm or cold or ""
    prompt = (
        "Write a pretended visual chain-of-thought for a student agent.\n"
        "The agent will only ever see the BEFORE image and the move command "
        "(the after image is counterfactual imagination, not shown).\n"
        "Constraints:\n"
        "- Keep the SAME final answer.\n"
        "- Reason from the before image + move: imagine where the object goes "
        "and what that implies for the question.\n"
        "- You may use the reference warm CoT below as private hints, but do "
        "NOT say you looked at an after image or at a reference CoT.\n"
        "- Never mention 'second image', 'after photo', or that an after frame "
        "was shown to you.\n"
        "- 4–8 short sentences, then a final line: Answer: <same answer>\n\n"
        f"Move command: {move_text}\n"
        f"Question: {question_text}\n"
        f"Ground-truth answer (must match): {_fmt_answer(answer)}\n\n"
        f"Reference warm CoT (private hints only):\n{hint}\n"
    )
    images = [rgb0_path] if rgb0_path.is_file() else None
    raw = vlm.chat(
        prompt,
        images=images,
        system=PRETENDED_SYSTEM,
        max_tokens=700,
        temperature=0.3,
    )
    text = raw.strip()
    ans = _fmt_answer(answer)
    if not re.search(r"(?i)\banswer\s*:", text):
        text = text.rstrip() + f"\nAnswer: {ans}"
    return text


def attach_cot_to_question(
    q: dict[str, Any],
    move: dict[str, Any],
    *,
    rgb0: Path,
    rgb1: Path,
    vlm: Optional[VlmClient] = None,
    polish: bool = True,
    polish_pretended: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``q`` with ``cot`` = {cold, warm, pretended, ...}.

    Pretended CoT is off by default (see ``add_pretended_cot.py`` / pass
    ``polish_pretended=True``).
    """
    out = dict(q)
    cold = build_cold_cot(q, move)
    cot: dict[str, Any] = {
        "version": "v9_pretended",
        "cold": cold,
        "warm": None,
        "warm_model": None,
        "warm_error": None,
        "pretended": None,
        "pretended_model": None,
        "pretended_error": None,
    }
    if polish and vlm is not None and q.get("gt_complete", True):
        try:
            cot["warm"] = polish_warm_cot(
                vlm,
                question_text=str(q.get("text") or ""),
                move_text=str(move.get("text") or ""),
                answer=q.get("answer"),
                cold=cold,
                rgb1_path=rgb1,
                rgb0_path=rgb0,
            )
            cot["warm_model"] = vlm.model
        except Exception as exc:  # noqa: BLE001
            cot["warm_error"] = str(exc)
    if polish_pretended and vlm is not None and q.get("gt_complete", True):
        try:
            cot["pretended"] = polish_pretended_cot(
                vlm,
                question_text=str(q.get("text") or ""),
                move_text=str(move.get("text") or ""),
                answer=q.get("answer"),
                warm=cot.get("warm"),
                cold=cold,
                rgb0_path=rgb0,
            )
            cot["pretended_model"] = vlm.model
        except Exception as exc:  # noqa: BLE001
            cot["pretended_error"] = str(exc)
    out["cot"] = cot
    return out


def attach_pretended_cot_to_question(
    q: dict[str, Any],
    move: dict[str, Any],
    *,
    rgb0: Path,
    vlm: VlmClient,
    force: bool = False,
) -> dict[str, Any]:
    """Add/refresh ``cot.pretended`` only (keep existing cold/warm)."""
    out = dict(q)
    cot = dict(out.get("cot") or {})
    if cot.get("pretended") and not force:
        out["cot"] = cot
        return out
    if not q.get("gt_complete", True):
        cot["pretended_error"] = "gt_incomplete"
        out["cot"] = cot
        return out
    cold = cot.get("cold") or build_cold_cot(q, move)
    cot.setdefault("cold", cold)
    cot.setdefault("version", "v9_pretended")
    try:
        cot["pretended"] = polish_pretended_cot(
            vlm,
            question_text=str(q.get("text") or ""),
            move_text=str(move.get("text") or ""),
            answer=q.get("answer"),
            warm=cot.get("warm"),
            cold=cold,
            rgb0_path=rgb0,
        )
        cot["pretended_model"] = vlm.model
        cot.pop("pretended_error", None)
    except Exception as exc:  # noqa: BLE001
        cot["pretended_error"] = str(exc)
    out["cot"] = cot
    return out


def process_sample_dict(
    sample: dict[str, Any],
    sample_dir: Path,
    *,
    vlm: Optional[VlmClient] = None,
    polish: bool = True,
    polish_pretended: bool = False,
) -> dict[str, Any]:
    """Attach CoT to every question in an on-disk sample (no THOR)."""
    out = dict(sample)
    views_out = []
    for view in sample.get("views") or []:
        v = dict(view)
        move = v.get("move") or {}
        view_id = v.get("view_id") or "view"
        rgb0 = sample_dir / f"{view_id}_rgb0.png"
        rgb1 = sample_dir / f"{view_id}_rgb1.png"
        # Fall back to stored paths if local names missing.
        if not rgb0.is_file() and v.get("rgb0_path"):
            rgb0 = Path(v["rgb0_path"])
        if not rgb1.is_file() and v.get("rgb1_path"):
            rgb1 = Path(v["rgb1_path"])
        qs = [
            attach_cot_to_question(
                q,
                move,
                rgb0=rgb0,
                rgb1=rgb1,
                vlm=vlm,
                polish=polish,
                polish_pretended=polish_pretended,
            )
            for q in (v.get("questions") or [])
        ]
        v["questions"] = qs
        views_out.append(v)
    out["views"] = views_out
    meta = dict(out.get("meta") or {})
    if polish_pretended:
        meta["modification"] = "mod7_cot_cold_warm_pretended"
    else:
        meta["modification"] = "mod7_cot_cold_warm"
    meta["cot"] = {
        "cold": True,
        "warm": bool(polish and vlm is not None),
        "pretended": bool(polish_pretended and vlm is not None),
    }
    out["meta"] = meta
    return out


def process_sample_pretended_only(
    sample: dict[str, Any],
    sample_dir: Path,
    *,
    vlm: VlmClient,
    force: bool = False,
) -> dict[str, Any]:
    """Fill pretended CoT on an existing sample without regenerating warm/cold."""
    out = dict(sample)
    views_out = []
    for view in sample.get("views") or []:
        v = dict(view)
        move = v.get("move") or {}
        view_id = v.get("view_id") or "view"
        rgb0 = sample_dir / f"{view_id}_rgb0.png"
        if not rgb0.is_file() and v.get("rgb0_path"):
            rgb0 = Path(v["rgb0_path"])
        qs = [
            attach_pretended_cot_to_question(
                q, move, rgb0=rgb0, vlm=vlm, force=force
            )
            for q in (v.get("questions") or [])
        ]
        v["questions"] = qs
        views_out.append(v)
    out["views"] = views_out
    meta = dict(out.get("meta") or {})
    mods = str(meta.get("modification") or "")
    if "pretended" not in mods:
        meta["modification"] = (mods + "+pretended_cot").lstrip("+")
    cot_meta = dict(meta.get("cot") or {})
    cot_meta["pretended"] = True
    meta["cot"] = cot_meta
    out["meta"] = meta
    return out
