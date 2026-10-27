"""Spatial question generation for CIT eval views.

Flow
----
1. Modification 4: ``make_question_for_category`` builds one GT question for a
   category-matched move (pipeline selects the mover per category).
2. Legacy ``sample_two_questions`` still samples two eligible categories.
3. ``word_questions_with_vlm`` (optional) rephrases ``text`` only; GT is unchanged.

Categories (Modification 2–4)
-----------------------------
occlusion, closer_to (MCQ), contact_wall, affordance, left_right_landmark,
feasibility, support, closest_three

Review tip: trust answers only when ``Question.gt_complete`` is True.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import uuid4

from cit_pipeline.config import (
    ABSENT_DISTRACTOR_TYPES,
    AFFORDANCE_CHAIR_TYPES,
    ANCHOR_TYPE_CAP_ONCE,
    CHOICE_MASK_MIN_PIXELS,
    CLOSEST_THREE_ANSWER,
    CLOSEST_THREE_POOL,
    CLOSER_TO_GAP_M,
    FEASIBILITY_COMPOUND_METERS,
    FEASIBILITY_FAST_MAX_ATTEMPTS,
    FEASIBILITY_FAST_NO_METERS,
    FEASIBILITY_FAST_SHARED_METERS,
    FEASIBILITY_FAST_YES_METERS,
    FEASIBILITY_BALANCE_DIRECTION_POLARITY,
    FEASIBILITY_ORTHOGONAL_PROB,
    FEASIBILITY_PROBE_CONTINUOUS,
    FEASIBILITY_PROBE_DIRECTIONS,
    FEASIBILITY_PROBE_METERS,
    FEASIBILITY_PROBE_M_MAX,
    FEASIBILITY_PROBE_M_MIN,
    FEASIBILITY_PROBE_NO_POWER,
    FEASIBILITY_PROBE_YES_POWER,
    FEASIBILITY_TARGET_NO_FRAC,
    FEASIBILITY_VERTICAL_METERS,
    FORBIDDEN_MOVE_TYPES,
    MOVE_DIRECTIONS,
    OCCLUDE_THRESHOLD,
    OCCLUDEE_MASK_MIN_PIXELS,
    OCCLUDE_PROMPT_VARIANT,
    ORDINAL_MAX_SAME_TYPE,
    QA_CATEGORIES,
    SITTABLE_TYPES,
    SMALL_OCCLUDEE_TYPES,
    SOM_OCCLUDEE_MASK_MIN_PIXELS,
    SUPPORT_DROP_HEIGHT_M,
    SUPPORT_QA_FLOOR_BOTTOM_Y_M,
    SUPPORT_QA_MIN_BOTTOM_Y_M,
    WALL_CONTACT_M,
    WALL_QA_NEAR_M,
    WALL_STOP_MARGIN_M,
)
from cit_pipeline.edit_search import camera_delta_to_world
from cit_pipeline.move_text import format_direction_phrase
from cit_pipeline.occlusion import OcclusionPair, select_occlusion_pairs
from cit_pipeline.referring import (
    extreme_instance_phrase,
    instance_phrase,
    ordinal_from_left,
    referring_allowed,
)
from cit_pipeline.schemas import CameraPose, MoveAction, Question
from cit_pipeline.scene_integrity import object_on_screen
from cit_pipeline.size_filter import (
    aabb_bottom_y,
    aabb_xz_extent,
    is_visible_mask_large_enough,
    object_mask_pixel_count,
    object_resting_on_floor,
    support_qa_eligible,
)
from cit_pipeline.thor_env import teleport_object


def _qid(category: str) -> str:
    return f"{category}_{uuid4().hex[:8]}"


def _dist(a: dict[str, float], b: dict[str, float]) -> float:
    return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))


def _closer_obj_dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    """v7.2 closer_to distance: AABB surface when available, else centers."""
    from cit_pipeline.spatial_metrics import closer_dist_objects

    d, _ = closer_dist_objects(a, b)
    return float(d)


def _find_obj(event: Any, object_id: str) -> Optional[dict[str, Any]]:
    for o in event.metadata["objects"]:
        if o["objectId"] == object_id:
            return o
    return None


def _visible_others(event: Any, exclude_id: str) -> list[dict[str, Any]]:
    skip = {"Floor", "Wall", "Ceiling", "Window", "Doorway", "Doorframe"} | set(
        FORBIDDEN_MOVE_TYPES
    )
    out = []
    for o in event.metadata["objects"]:
        if o["objectId"] == exclude_id:
            continue
        if not o.get("visible"):
            continue
        if o["objectType"] in skip:
            continue
        # iTHOR has no room label; use instance-mask pixels as on-screen proxy.
        if not object_on_screen(event, o, min_pixels=int(CHOICE_MASK_MIN_PIXELS)):
            continue
        out.append(o)
    return out


def _phrase(event0: Any, object_id: str, object_type: str) -> str:
    """Ordinal-aware referring phrase from **rgb0** (Modification 2 / 8)."""
    return instance_phrase(event0, object_id, object_type)


def _phrase_with_receptacle(
    event0: Any,
    object_id: str,
    object_type: str,
) -> str:
    """For small items, mention the supporting receptacle (Modification 11).

    e.g. \"the bowl on the countertop\" — helps locate tiny occludees.
    """
    base = _phrase(event0, object_id, object_type)
    if object_type not in SMALL_OCCLUDEE_TYPES:
        return base
    obj = _find_obj(event0, object_id)
    if not obj:
        return base
    for pid in obj.get("parentReceptacles") or []:
        if not pid or not isinstance(pid, str):
            continue
        if pid == "Floor" or pid.startswith("Floor"):
            continue
        parent = _find_obj(event0, pid)
        if parent is None:
            continue
        parent_p = _phrase(event0, parent["objectId"], parent["objectType"])
        return f"{base} on {parent_p}"
    return base


def _referable_others(event0: Any, event1: Any, exclude_id: str) -> list[dict[str, Any]]:
    """Visible others whose type is within the ordinal cap (Modification 4)."""
    return [
        o for o in _visible_others(event1, exclude_id)
        if referring_allowed(event0, o["objectType"])
    ]


def take_with_anchor_type_caps(
    objs: Sequence[dict[str, Any]],
    *,
    n: Optional[int] = None,
    cap_types: frozenset[str] = ANCHOR_TYPE_CAP_ONCE,
    max_per_type: int = 1,
) -> list[dict[str, Any]]:
    """Keep order; drop extras of look-alike types (≤1 ShelvingUnit, ≤1 Drawer).

    Used for closer_to ABCD anchors and closest_three / count_near pools of 6.
    """
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for o in objs:
        t = str(o.get("objectType") or "")
        if t in cap_types:
            if counts.get(t, 0) >= max_per_type:
                continue
            counts[t] = counts.get(t, 0) + 1
        out.append(o)
        if n is not None and len(out) >= n:
            break
    return out


def filter_scored_anchor_type_caps(
    scored: Sequence[tuple[float, dict[str, Any]]],
    *,
    cap_types: frozenset[str] = ANCHOR_TYPE_CAP_ONCE,
    max_per_type: int = 1,
) -> list[tuple[float, dict[str, Any]]]:
    """Same type-cap as ``take_with_anchor_type_caps`` for (dist, obj) lists."""
    counts: dict[str, int] = {}
    out: list[tuple[float, dict[str, Any]]] = []
    for dist, o in scored:
        t = str(o.get("objectType") or "")
        if t in cap_types:
            if counts.get(t, 0) >= max_per_type:
                continue
            counts[t] = counts.get(t, 0) + 1
        out.append((dist, o))
    return out


def anchor_type_caps_ok(
    objs: Sequence[dict[str, Any]],
    *,
    cap_types: frozenset[str] = ANCHOR_TYPE_CAP_ONCE,
    max_per_type: int = 1,
) -> bool:
    """True iff no capped type appears more than ``max_per_type`` times."""
    counts: dict[str, int] = {}
    for o in objs:
        t = str(o.get("objectType") or "")
        if t not in cap_types:
            continue
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > max_per_type:
            return False
    return True


def _mover_receptacle_ids(event: Any, mover_id: str) -> set[str]:
    """Parent receptacle IDs of the mover (to ban as closer_to landmarks)."""
    obj = _find_obj(event, mover_id)
    if not obj:
        return set()
    out: set[str] = set()
    for pid in obj.get("parentReceptacles") or []:
        if pid and isinstance(pid, str) and not (
            pid == "Floor" or pid.startswith("Floor")
        ):
            out.add(pid)
    return out


def _mover_phrase(event0: Any, move: MoveAction) -> str:
    return _phrase(event0, move.object_id, move.object_type)


def _fov_angle_deg(
    seat: dict[str, Any],
    target: dict[str, Any],
) -> Optional[float]:
    """Horizontal angle from seat yaw to target (degrees)."""
    import numpy as np

    yaw = math.radians(float((seat.get("rotation") or {}).get("y", 0.0)))
    forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    to_tgt = np.array(
        [
            target["position"]["x"] - seat["position"]["x"],
            0.0,
            target["position"]["z"] - seat["position"]["z"],
        ]
    )
    nrm = float(np.linalg.norm(to_tgt))
    if nrm < 0.05:
        return None
    to_tgt /= nrm
    return math.degrees(math.acos(float(np.clip(np.dot(forward, to_tgt), -1.0, 1.0))))


def _move_axis(direction: str) -> str:
    """Lateral (left/right) vs depth (forward/back) vs vertical (up/down)."""
    if direction in ("left", "right"):
        return "lateral"
    if direction in ("up", "down"):
        return "vertical"
    return "depth"


def _orthogonal_directions(direction: str) -> tuple[str, ...]:
    axis = _move_axis(direction)
    # Modification 10: no upward orthogonal probes.
    if axis == "lateral":
        return ("forward", "back", "down")
    if axis == "depth":
        return ("left", "right", "down")
    # Vertical original move → any horizontal axis is orthogonal.
    return ("left", "right", "forward", "back")


# ---------------------------------------------------------------------------
# Occlusion — always name a concrete occludee (50/50 yes/no when possible)
# ---------------------------------------------------------------------------
def _occlusion_question_text(occludee_p: str, mover_p: str, threshold: float) -> str:
    """Natural-image occlusion stem (Mod 20 Round-2 prompt ablation winner)."""
    variant = str(OCCLUDE_PROMPT_VARIANT or "partial_or_totally_visible")
    if variant == "baseline":
        return (
            f"In the after image, is {occludee_p} substantially occluded by "
            f"{mover_p} (at least {int(threshold * 100)}% of its visible pixels)?"
        )
    if variant == "visible_or_occluded":
        return (
            f"In the after image, is {occludee_p} visible, or is it occluded by "
            f"{mover_p}?"
        )
    if variant == "any_part_hidden":
        return (
            f"In the after image, is any part of {occludee_p} hidden behind "
            f"{mover_p}?"
        )
    if variant == "blocked_from_view":
        return (
            f"In the after image, is {occludee_p} blocked from view by {mover_p}?"
        )
    # Default / partial_or_totally_visible
    return (
        f"In the after image, is {occludee_p} occluded by {mover_p} so that a "
        f"part of it is invisible, or is it totally visible?"
    )


def make_occlusion_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    threshold: float = OCCLUDE_THRESHOLD,
    prefer_yes: Optional[bool] = None,
    rng: random.Random,
) -> Question:
    """Ask whether a specific B is occluded by moved A (Modification 5)."""
    mover_p = _mover_phrase(event0, move)
    meta = dict(move.edit_meta or {})

    # Prefer the planned occludee from the cover-edit path.
    planned_id = meta.get("occludee_id")
    planned_type = meta.get("occludee_type")
    if planned_id and planned_type:
        # Modification 9/12: reject absolute speck occludees; SoM relaxes the floor.
        n_px = object_mask_pixel_count(event0, str(planned_id))
        if n_px is not None and n_px < SOM_OCCLUDEE_MASK_MIN_PIXELS:
            planned_id = None
            planned_type = None
    if planned_id and planned_type:
        rate = float(meta.get("occlusion_rate_rgb1", 0.0))
        method = meta.get("occlusion_mask_method_rgb1") or "unknown"
        remaining = meta.get("occlusion_remaining_b")
        # Recompute cover of B's rgb0 region by A in rgb1 (robust to full hide).
        try:
            from cit_pipeline.occlusion import occlusion_cover_rate

            cover, rem, method2, _ = occlusion_cover_rate(
                event0, event1, move.object_id, planned_id
            )
            rate, method, remaining = float(cover), method2, float(rem)
        except Exception:
            pass
        occludee_p = _phrase_with_receptacle(event0, planned_id, str(planned_type))
        is_occ = rate >= threshold
        text = _occlusion_question_text(occludee_p, mover_p, threshold)
        return Question(
            category="occlusion",
            question_id=_qid("occlusion"),
            text=text,
            answer=bool(is_occ),
            gt_meta={
                "occludee_id": planned_id,
                "occludee_type": planned_type,
                "occludee_phrase": occludee_p,
                "mover_phrase": mover_p,
                "occlusion_rate": rate,
                "occlusion_remaining_b": remaining,
                "occlusion_metric": meta.get("occlusion_metric", "cover_A1_of_B0"),
                "threshold": threshold,
                "mask_method": method,
                "occlusion_prompt_variant": OCCLUDE_PROMPT_VARIANT,
                "occludee_mask_pixels_rgb0": object_mask_pixel_count(
                    event0, str(planned_id)
                ),
                "cover_plan": {
                    k: meta[k]
                    for k in (
                        "planned_direction",
                        "planned_meters",
                        "pixel_du",
                        "pixel_dv",
                        "centroid_a_uv",
                        "centroid_b_uv",
                        "meters_per_pixel",
                    )
                    if k in meta
                },
            },
            gt_complete=True,
        )

    yes, no, method = select_occlusion_pairs(
        event1,
        move.object_id,
        threshold=threshold,
        balance=True,
        event0=event0,
    )
    chosen: Optional[OcclusionPair] = None
    if prefer_yes is True:
        # Strict: never emit a "no" sample when the caller asked for yes.
        if yes:
            chosen = yes[0]
    elif prefer_yes is False:
        # Strict: never emit a "yes" sample when the caller asked for no.
        if no:
            chosen = no[0]
    elif yes and no:
        chosen = rng.choice([yes[0], no[0]])
    elif yes or no:
        # No preference: single-sided pool is fine.
        chosen = (yes or no)[0]

    if chosen is None:
        note = "no intersecting occludee found"
        if prefer_yes is True:
            note = "no_occluded_pair_for_prefer_yes"
        elif prefer_yes is False:
            note = "no_visible_pair_for_prefer_no"
        return Question(
            category="occlusion",
            question_id=_qid("occlusion"),
            text=(
                f"In the after image, is any nearby object occluded by {mover_p}?"
            ),
            answer=None,
            gt_meta={"mask_method": method, "note": note, "prefer_yes": prefer_yes},
            gt_complete=False,
        )

    occludee_p = _phrase_with_receptacle(
        event0, chosen.occludee_id, chosen.occludee_type
    )
    return Question(
        category="occlusion",
        question_id=_qid("occlusion"),
        text=_occlusion_question_text(occludee_p, mover_p, threshold),
        answer=bool(chosen.is_occluded),
        gt_meta={
            "occludee_id": chosen.occludee_id,
            "occludee_type": chosen.occludee_type,
            "occludee_phrase": occludee_p,
            "mover_phrase": mover_p,
            "occlusion_rate": chosen.occlusion_rate,
            "threshold": threshold,
            "mask_method": chosen.mask_method,
            "occlusion_prompt_variant": OCCLUDE_PROMPT_VARIANT,
        },
        gt_complete=True,
    )


# ---------------------------------------------------------------------------
# Closer-to
# ---------------------------------------------------------------------------
def make_closer_to_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
    *,
    dataset: str = "ithor",
) -> Question:
    """Modification 4 / 9 / 12 / 15: always a 4-way MCQ (never 'closer than the camera').

    Modification 15: prefer the flip plan stored on ``move.edit_meta``:
    B closer before → C closer after; two distractors farther than B/C in both
    frames (ideally near A in rgb0). GT distances always come from executed
    simulator metadata.

    ProcTHOR: always prefer an executed flip MCQ when the planned (or chance)
    B→C flip holds. If that cannot be completed, use a random-4 after-nearest
    MCQ whenever the room has more than 5 referable objects.
    """
    preferred_landmarks = {
        "Sofa", "ArmChair", "Television", "DiningTable", "CoffeeTable", "SideTable",
        "Desk", "Painting", "FloorLamp", "HousePlant", "ShelvingUnit", "Dresser",
        "Bed", "CounterTop", "Refrigerator", "Stove", "Toaster", "Mug", "Drawer",
        "Chair", "Stool", "Ottoman", "GarbageCan", "Box", "Laptop", "Pillow",
    }
    # Modification 11: never ask whether a houseplant is closer to the table it sits on.
    ban_ids = _mover_receptacle_ids(event0, move.object_id) | _mover_receptacle_ids(
        event1, move.object_id
    )
    others = [
        o
        for o in _referable_others(event0, event1, move.object_id)
        if o["objectType"] != move.object_type and o["objectId"] not in ban_ids
    ]
    landmarks = [o for o in others if o["objectType"] in preferred_landmarks] or list(others)
    mover_p = _mover_phrase(event0, move)
    is_procthor = str(dataset).lower() == "procthor"

    def _incomplete(note: str) -> Question:
        return Question(
            category="closer_to",
            question_id=_qid("closer_to"),
            text=(
                f"After the move, is {mover_p} closer to which of the following? "
                "(A) … (B) … (C) … (D) …"
            ),
            answer=None,
            answer_type="multi_choice",
            gt_meta={"note": note, "banned_receptacle_ids": sorted(ban_ids)},
            gt_complete=False,
        )

    moved = _find_obj(event1, move.object_id)
    before = _find_obj(event0, move.object_id)
    if not moved or not before:
        return _incomplete("mover missing in rgb0/rgb1")

    edit_meta = dict(move.edit_meta or {})
    planned_b = edit_meta.get("closer_landmark_b_id")
    planned_c = edit_meta.get("closer_landmark_c_id")
    planned_d_ids = list(edit_meta.get("closer_distractor_ids") or [])
    gap = float(edit_meta.get("closer_gap_m") or CLOSER_TO_GAP_M)

    def _assemble(
        landmark_b: dict[str, Any],
        landmark_c: dict[str, Any],
        db0: float,
        dc0: float,
        db1: float,
        dc1: float,
        planned_d_ids_local: list[Any],
        *,
        random_fallback: bool,
        unique_min_gap: float,
    ) -> Optional[Question]:
        """Build a complete closer_to MCQ, or None if filters fail."""
        if not anchor_type_caps_ok([landmark_b, landmark_c]):
            return None
        pa = _phrase(event0, landmark_b["objectId"], landmark_b["objectType"])
        pc = _phrase(event0, landmark_c["objectId"], landmark_c["objectType"])
        used_ids = {
            landmark_b["objectId"],
            landmark_c["objectId"],
            move.object_id,
        }
        near_thresh0 = min(db0, dc0)
        answer_after_d = float(dc1)
        distractors: list[dict[str, Any]] = []

        def _distractor_ok(o: dict[str, Any]) -> bool:
            if random_fallback:
                return True
            d0 = _closer_obj_dist(before, o)
            d1 = _closer_obj_dist(moved, o)
            return (
                d0 >= near_thresh0 - 1e-6
                and d1 >= answer_after_d + float(gap) - 1e-9
            )

        def _type_ok(o: dict[str, Any]) -> bool:
            """≤1 ShelvingUnit and ≤1 Drawer among B/C + distractors."""
            return anchor_type_caps_ok(
                [landmark_b, landmark_c, *distractors, o]
            )

        for oid in planned_d_ids_local:
            o = _find_obj(event0, oid) or _find_obj(event1, oid)
            if o is None or o["objectId"] in used_ids:
                continue
            if not _type_ok(o):
                continue
            if _distractor_ok(o):
                distractors.append(o)
                used_ids.add(o["objectId"])
            if len(distractors) >= 2:
                break
        if len(distractors) < 2 and not random_fallback:
            ranked: list[tuple[float, dict[str, Any]]] = []
            for o in others:
                if o["objectId"] in used_ids:
                    continue
                if not _type_ok(o):
                    continue
                if not _distractor_ok(o):
                    continue
                d0 = _closer_obj_dist(before, o)
                ranked.append((d0 - near_thresh0, o))
            ranked.sort(key=lambda t: t[0])
            for _, o in ranked:
                if not _type_ok(o):
                    continue
                distractors.append(o)
                used_ids.add(o["objectId"])
                if len(distractors) >= 2:
                    break
        if len(distractors) < 2 and not random_fallback:
            ranked_far: list[tuple[float, dict[str, Any]]] = []
            for o in others:
                if o["objectId"] in used_ids:
                    continue
                if not _type_ok(o):
                    continue
                d1 = _closer_obj_dist(moved, o)
                if d1 + 1e-9 < answer_after_d + float(gap):
                    continue
                ranked_far.append((d1, o))
            ranked_far.sort(key=lambda t: -t[0])
            for _, o in ranked_far:
                if not _type_ok(o):
                    continue
                distractors.append(o)
                used_ids.add(o["objectId"])
                if len(distractors) >= 2:
                    break
        if len(distractors) < 2:
            return None

        far, fourth = distractors[0], distractors[1]
        if not anchor_type_caps_ok([landmark_b, landmark_c, far, fourth]):
            return None
        pf = _phrase(event0, far["objectId"], far["objectType"])
        p4 = _phrase(event0, fourth["objectId"], fourth["objectType"])
        far_d = _closer_obj_dist(moved, far)
        fourth_d = _closer_obj_dist(moved, fourth)

        choice_objs = [landmark_b, landmark_c, far, fourth]
        choice_dists = [
            (_closer_obj_dist(moved, o), o) for o in choice_objs
        ]
        choice_dists.sort(key=lambda t: t[0])
        closest_d, closest_obj = choice_dists[0]
        if closest_obj["objectId"] != landmark_c["objectId"]:
            landmark_c = closest_obj
            dc1_local = float(closest_d)
            pc = _phrase(event0, landmark_c["objectId"], landmark_c["objectType"])
            dc0 = _closer_obj_dist(before, landmark_c)
        else:
            dc1_local = float(dc1)
        closer_p = pc
        answer_after_d = float(dc1_local)

        if len(choice_dists) >= 2 and (choice_dists[1][0] - closest_d) < unique_min_gap:
            return None

        before_choice_dists = sorted(
            ((_closer_obj_dist(before, o), o) for o in choice_objs),
            key=lambda t: t[0],
        )
        closest_before_final = before_choice_dists[0][1]
        answer_flipped = closest_before_final["objectId"] != landmark_c["objectId"]
        closer_before_p = _phrase(
            event0,
            closest_before_final["objectId"],
            closest_before_final["objectType"],
        )
        pa = _phrase(event0, landmark_b["objectId"], landmark_b["objectType"])

        present_types = {o["objectType"] for o in event1.metadata["objects"]}
        absent_pool = [t for t in ABSENT_DISTRACTOR_TYPES if t not in present_types]
        if not absent_pool:
            absent_pool = ["Tablet", "Microwave", "Toilet", "Laptop"]
        rng.shuffle(absent_pool)

        if random_fallback:
            choices = [pa, pc, pf, p4]
            if len(set(choices)) < 4:
                choices = [
                    _phrase(event0, o["objectId"], o["objectType"]) for o in choice_objs
                ]
        else:
            choices = [pa, pc, pf]
            while len(choices) < 4 and absent_pool:
                choices.append(f"the {absent_pool.pop().lower()}")
            pad_i = 0
            while len(choices) < 4:
                choices.append(f"the missing object {pad_i}")
                pad_i += 1

        rng.shuffle(choices)
        letters = ["A", "B", "C", "D"]
        labeled = [f"({L}) {c}" for L, c in zip(letters, choices)]
        if closer_p not in choices:
            choices[0] = closer_p
            labeled = [f"({L}) {c}" for L, c in zip(letters, choices)]
        answer_letter = letters[choices.index(closer_p)]
        text = (
            f"After the move, is {mover_p} closer to which of the following? "
            + " ".join(labeled)
        )

        som_objs: list[dict[str, Any]] = [landmark_b, landmark_c, far, fourth]
        seen_som: set[str] = set()
        som_objs_u: list[dict[str, Any]] = []
        for o in som_objs:
            if o["objectId"] in seen_som:
                continue
            seen_som.add(o["objectId"])
            som_objs_u.append(o)
        pad_pool = list(others)
        if random_fallback:
            for o in _visible_others(event1, move.object_id):
                if o["objectId"] in ban_ids or o["objectType"] == move.object_type:
                    continue
                if o["objectId"] not in {x["objectId"] for x in pad_pool}:
                    pad_pool.append(o)
        while len(som_objs_u) < 4:
            filled = False
            for o in pad_pool:
                if o["objectId"] in seen_som:
                    continue
                if (
                    not random_fallback
                    and _closer_obj_dist(moved, o) + 1e-9
                    < answer_after_d + float(gap)
                ):
                    continue
                som_objs_u.append(o)
                seen_som.add(o["objectId"])
                filled = True
                break
            if not filled:
                break
        som_objs = som_objs_u
        som_complete = len(som_objs) >= 4
        if not som_complete:
            return None
        som_order = list(som_objs)
        rng.shuffle(som_order)
        som_choice_ids: dict[str, str] = {}
        som_choice_phrases: dict[str, str] = {}
        choices_som: list[str] = []
        for L, o in zip(letters, som_order):
            phrase = _phrase(event0, o["objectId"], o["objectType"])
            som_choice_ids[L] = o["objectId"]
            som_choice_phrases[L] = phrase
            choices_som.append(phrase)
        answer_letter_som = None
        for L, oid in som_choice_ids.items():
            if oid == landmark_c["objectId"]:
                answer_letter_som = L
                break
        som_dist: dict[str, float] = {}
        for L, oid in som_choice_ids.items():
            obj_l = _find_obj(event1, oid) or _find_obj(event0, oid)
            if obj_l is None:
                continue
            som_dist[L] = _closer_obj_dist(moved, obj_l)
        if som_dist:
            best_L = min(som_dist, key=som_dist.get)
            if best_L != answer_letter_som:
                answer_letter_som = best_L
                closer_p = som_choice_phrases[best_L]
                landmark_c = _find_obj(event1, som_choice_ids[best_L]) or _find_obj(
                    event0, som_choice_ids[best_L]
                )
                pc = closer_p
                dc1_local = float(som_dist[best_L])
                if closer_p in choices:
                    answer_letter = letters[choices.index(closer_p)]
                    text = (
                        f"After the move, is {mover_p} closer to which of the following? "
                        + " ".join(labeled)
                    )

        return Question(
            category="closer_to",
            question_id=_qid("closer_to"),
            text=text,
            answer=closer_p,
            answer_type="multi_choice",
            gt_meta={
                "mover_phrase": mover_p,
                "banned_receptacle_ids": sorted(ban_ids),
                "landmark_a_id": landmark_b["objectId"],
                "landmark_b_id": landmark_c["objectId"],
                "closer_before_id": landmark_b["objectId"],
                "closer_after_id": landmark_c["objectId"],
                "phrase_a": pa,
                "phrase_b": pc,
                "distance_a": db1,
                "distance_b": dc1_local,
                "distance_before_b": db0,
                "distance_before_c": dc0,
                "distance_after_b": db1,
                "distance_after_c": dc1_local,
                "closer_phrase": closer_p,
                "far_id": far["objectId"],
                "far_phrase": pf,
                "far_distance": far_d,
                "fourth_id": fourth["objectId"],
                "fourth_phrase": p4,
                "fourth_distance": fourth_d,
                "choice_distances_after": {
                    _phrase(event0, o["objectId"], o["objectType"]): float(d)
                    for d, o in choice_dists
                },
                "absent_distractor": next(
                    (c for c in choices if c not in {pa, pc, pf, p4}), None
                ),
                "choices": choices,
                "answer_letter": answer_letter,
                "choices_som": choices_som,
                "som_choice_ids": som_choice_ids,
                "som_choice_phrases": som_choice_phrases,
                "answer_letter_som": answer_letter_som,
                "som_four_real_objects": som_complete,
                "metric": (
                    "aabb_surface_distance_mcq4_random_fallback"
                    if random_fallback
                    else "aabb_surface_distance_mcq4_flip"
                ),
                "answer_before": closer_before_p,
                "answer_after": closer_p,
                "answer_flipped": bool(answer_flipped),
                "allow_nonflip_closer": bool(
                    random_fallback or edit_meta.get("allow_nonflip_closer")
                ),
                "flip_planned": bool(edit_meta.get("closer_to_flip_plan"))
                and not random_fallback,
                "nonflip_fallback": bool(
                    random_fallback or edit_meta.get("closer_nonflip_fallback")
                ),
                "closer_random_fallback": bool(random_fallback),
                "gap_m": gap,
                "answer_is_true_closest_after": True,
            },
            gt_complete=True,
        )

    def _procthor_random_fallback() -> Optional[Question]:
        """Sample 4 room objects; answer = nearest after. Needs >5 room objects."""
        # Count / sample from on-screen room objects (not only ordinal-referable).
        visible_pool = [
            o
            for o in _visible_others(event1, move.object_id)
            if o["objectId"] not in ban_ids and o["objectType"] != move.object_type
        ]
        # Prefer referable phrases when the room is rich enough; else all visible.
        referable_pool = [
            o
            for o in visible_pool
            if referring_allowed(event0, o["objectType"])
        ]
        if len(visible_pool) <= 5:
            return None
        pool = list(referable_pool) if len(referable_pool) > 5 else list(visible_pool)
        if len(pool) < 4:
            return None
        best: Optional[tuple[float, list[dict[str, Any]]]] = None
        # Many draws: maximize after-move 1st→2nd gap; accept any strict nearest.
        for _ in range(48):
            chosen = rng.sample(pool, 4)
            after_ranked = sorted(
                ((_closer_obj_dist(moved, o), o) for o in chosen),
                key=lambda t: t[0],
            )
            gap_after = float(after_ranked[1][0] - after_ranked[0][0])
            if gap_after <= 1e-6:
                continue
            if best is None or gap_after > best[0]:
                best = (gap_after, chosen)
                # Good enough early exit.
                if gap_after >= max(0.02, 0.25 * float(gap)):
                    break
        if best is None:
            # Last resort: take any 4 with a strict nearest (even tiny gap).
            for _ in range(64):
                chosen = rng.sample(pool, 4)
                after_ranked = sorted(
                    ((_closer_obj_dist(moved, o), o) for o in chosen),
                    key=lambda t: t[0],
                )
                if after_ranked[1][0] - after_ranked[0][0] > 1e-9:
                    best = (float(after_ranked[1][0] - after_ranked[0][0]), chosen)
                    break
        if best is None:
            return None
        chosen = best[1]
        before_ranked = sorted(
            ((_closer_obj_dist(before, o), o) for o in chosen),
            key=lambda t: t[0],
        )
        after_ranked = sorted(
            ((_closer_obj_dist(moved, o), o) for o in chosen),
            key=lambda t: t[0],
        )
        landmark_c = after_ranked[0][1]
        # Keep B and C distinct so the 4-choice / SoM set never collapses.
        if before_ranked[0][1]["objectId"] != landmark_c["objectId"]:
            landmark_b = before_ranked[0][1]
        else:
            landmark_b = before_ranked[1][1]
        db0 = _closer_obj_dist(before, landmark_b)
        dc0 = _closer_obj_dist(before, landmark_c)
        db1 = _closer_obj_dist(moved, landmark_b)
        dc1 = float(after_ranked[0][0])
        rest = [
            o
            for o in chosen
            if o["objectId"]
            not in {landmark_b["objectId"], landmark_c["objectId"]}
        ]
        if len(rest) < 2:
            # Should not happen with 4 distinct chosen + distinct B/C.
            extras = [
                o
                for o in pool
                if o["objectId"]
                not in {
                    landmark_b["objectId"],
                    landmark_c["objectId"],
                    *(r["objectId"] for r in rest),
                }
            ]
            rest = list(rest) + extras
            rest = rest[:2]
        if len(rest) < 2:
            return None
        return _assemble(
            landmark_b,
            landmark_c,
            db0,
            dc0,
            db1,
            dc1,
            [rest[0]["objectId"], rest[1]["objectId"]],
            random_fallback=True,
            # Soft uniqueness: already enforced by sampling; only reject true ties.
            unique_min_gap=1e-6,
        )

    # --- Prefer executed flip MCQ ---
    landmark_b = (
        (_find_obj(event0, planned_b) or _find_obj(event1, planned_b))
        if planned_b
        else None
    )
    landmark_c = (
        (_find_obj(event0, planned_c) or _find_obj(event1, planned_c))
        if planned_c
        else None
    )
    flip_ok = False
    db0 = dc0 = db1 = dc1 = 0.0
    if landmark_b is not None and landmark_c is not None:
        db0 = _closer_obj_dist(before, landmark_b)
        dc0 = _closer_obj_dist(before, landmark_c)
        db1 = _closer_obj_dist(moved, landmark_b)
        dc1 = _closer_obj_dist(moved, landmark_c)
        flip_ok = (db0 + gap < dc0) and (dc1 + gap < db1)

    hard_unique = max(0.02, 0.25 * float(gap))

    if flip_ok and landmark_b is not None and landmark_c is not None:
        q = _assemble(
            landmark_b,
            landmark_c,
            db0,
            dc0,
            db1,
            dc1,
            planned_d_ids,
            random_fallback=False,
            unique_min_gap=hard_unique,
        )
        if q is None:
            # Salvage near-tie flips before dropping to random.
            q = _assemble(
                landmark_b,
                landmark_c,
                db0,
                dc0,
                db1,
                dc1,
                planned_d_ids,
                random_fallback=False,
                unique_min_gap=1e-6,
            )
        if q is not None:
            return q
        if is_procthor:
            q = _procthor_random_fallback()
            if q is not None:
                return q
            return _incomplete(
                "flip MCQ failed and random closer_to needs >5 room objects "
                "with a unique nearest"
            )

    # Chance B/C flip search on the executed motion (ProcTHOR + iTHOR).
    best = None
    if len(landmarks) >= 2:
        pool_search = list(landmarks)
        rng.shuffle(pool_search)
        for i in range(len(pool_search)):
            for j in range(i + 1, len(pool_search)):
                lb, lc = pool_search[i], pool_search[j]
                for landmark_b_try, landmark_c_try in ((lb, lc), (lc, lb)):
                    db0_t = _closer_obj_dist(before, landmark_b_try)
                    dc0_t = _closer_obj_dist(before, landmark_c_try)
                    db1_t = _closer_obj_dist(moved, landmark_b_try)
                    dc1_t = _closer_obj_dist(moved, landmark_c_try)
                    if not (db0_t + gap < dc0_t and dc1_t + gap < db1_t):
                        continue
                    score = (dc0_t - db0_t) + (db1_t - dc1_t)
                    if best is None or score > best[0]:
                        best = (
                            score,
                            landmark_b_try,
                            landmark_c_try,
                            db0_t,
                            dc0_t,
                            db1_t,
                            dc1_t,
                        )
    if best is not None:
        _, landmark_b, landmark_c, db0, dc0, db1, dc1 = best
        q = _assemble(
            landmark_b,
            landmark_c,
            db0,
            dc0,
            db1,
            dc1,
            planned_d_ids,
            random_fallback=False,
            unique_min_gap=hard_unique,
        )
        if q is not None:
            return q

    if is_procthor:
        q = _procthor_random_fallback()
        if q is not None:
            return q
        if len(others) <= 5:
            return _incomplete("random closer_to needs >5 room objects")
        return _incomplete("random closer_to could not find unique nearest among 4")

    # iTHOR: optional nonflip when edit meta allows it.
    if len(landmarks) < 2:
        return _incomplete("need >=2 landmarks for MCQ (after receptacle ban)")
    if not bool(edit_meta.get("allow_nonflip_closer")):
        return _incomplete("no B/C pair with executed before→after flip")
    ranked_after: list[tuple[float, dict[str, Any]]] = []
    for o in landmarks:
        ranked_after.append((_closer_obj_dist(moved, o), o))
    ranked_after.sort(key=lambda t: t[0])
    if len(ranked_after) < 4:
        return _incomplete("nonflip closer_to needs >=4 landmarks")
    landmark_c = ranked_after[0][1]
    landmark_b = ranked_after[1][1]
    db0 = _closer_obj_dist(before, landmark_b)
    dc0 = _closer_obj_dist(before, landmark_c)
    db1 = float(ranked_after[1][0])
    dc1 = float(ranked_after[0][0])
    planned_d_ids = [
        ranked_after[2][1]["objectId"],
        ranked_after[3][1]["objectId"],
    ]
    q = _assemble(
        landmark_b,
        landmark_c,
        db0,
        dc0,
        db1,
        dc1,
        planned_d_ids,
        random_fallback=False,
        unique_min_gap=hard_unique,
    )
    return q or _incomplete("after-move closest not unique among choices")


# ---------------------------------------------------------------------------
# Contact wall — edge clearance; only when near a wall (Mod 2 §7, Mod 3 §1)
# ---------------------------------------------------------------------------
def _min_edge_wall_clearance(event: Any, obj: dict[str, Any]) -> tuple[float, str]:
    """Distance from object's closest horizontal edge to the nearest wall/bound.

    Prefer AABB↔sceneBounds clearance (edge-to-wall). Fall back to Wall
    metadata centroid distance, then object-extent proxy.
    """
    extent = aabb_xz_extent(obj)
    bounds = event.metadata.get("sceneBounds") or {}
    center = bounds.get("center")
    size = bounds.get("size")
    if extent and center and size:
        min_x, max_x, min_z, max_z = extent
        half_x = float(size["x"]) / 2.0
        half_z = float(size["z"]) / 2.0
        cx, cz = float(center["x"]), float(center["z"])
        scene_min_x, scene_max_x = cx - half_x, cx + half_x
        scene_min_z, scene_max_z = cz - half_z, cz + half_z
        clearance = min(
            min_x - scene_min_x,
            scene_max_x - max_x,
            min_z - scene_min_z,
            scene_max_z - max_z,
        )
        return max(0.0, float(clearance)), "aabb_sceneBounds"

    pos = obj.get("position") or {}
    walls = [
        o for o in event.metadata["objects"]
        if o.get("objectType") == "Wall" or str(o.get("objectType", "")).startswith("Wall")
    ]
    if walls and pos:
        return min(_dist(pos, w["position"]) for w in walls), "wall_metadata"

    if not pos:
        return 999.0, "none"

    xs, zs = [], []
    for o in event.metadata["objects"]:
        p = o.get("position") or {}
        if "x" in p and "z" in p:
            xs.append(float(p["x"]))
            zs.append(float(p["z"]))
    if not xs:
        return 999.0, "none"
    min_d = min(
        abs(float(pos["x"]) - min(xs)),
        abs(float(pos["x"]) - max(xs)),
        abs(float(pos["z"]) - min(zs)),
        abs(float(pos["z"]) - max(zs)),
    )
    return min_d, "object_extent_proxy"


def contact_wall_eligible(event: Any, obj: dict[str, Any]) -> bool:
    """Modification 3: only ask contact_wall when edge clearance < WALL_QA_NEAR_M."""
    clearance, method = _min_edge_wall_clearance(event, obj)
    return method != "none" and clearance < WALL_QA_NEAR_M


def make_contact_wall_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
) -> Question:
    after = _find_obj(event1, move.object_id)
    mover_p = _mover_phrase(event0, move)
    if not after:
        return Question(
            category="contact_wall",
            question_id=_qid("contact_wall"),
            text=f"After the move, is {mover_p} in contact with a wall?",
            answer=None,
            gt_complete=False,
            gt_meta={},
        )

    min_d, method = _min_edge_wall_clearance(event1, after)
    edit = dict(move.edit_meta or {})
    # Modification 5/20: yes only when edge clearance < WALL_CONTACT_M.
    # ``wall_blocked_next_step`` alone is NOT contact (furniture can block first).
    if "wall_contact_gt" in edit:
        contact = bool(edit["wall_contact_gt"])
    else:
        contact = min_d < WALL_CONTACT_M
    blocked = bool(edit.get("wall_blocked_next_step"))
    before = _find_obj(event0, move.object_id)
    meta: dict[str, Any] = {
        "method": method,
        "min_edge_clearance_m": min_d,
        "near_wall_eligibility_m": WALL_QA_NEAR_M,
        "threshold_m": WALL_CONTACT_M,
        "wall_blocked_next_step": blocked,
        "wall_contact_reason": edit.get("wall_contact_reason"),
        "wall_multistep": edit.get("wall_multistep", False),
        "wall_steps_ok": edit.get("wall_steps_ok"),
        "wall_total_planned_m": edit.get("wall_total_planned_m"),
        "wall_toward": edit.get("wall_toward"),
        "wall_axis": edit.get("wall_axis"),
        "wall_nearest_axis": edit.get("wall_nearest_axis"),
    }
    if before:
        d0, _ = _min_edge_wall_clearance(event0, before)
        meta["min_edge_clearance_before_m"] = d0
    meta["mover_phrase"] = mover_p

    return Question(
        category="contact_wall",
        question_id=_qid("contact_wall"),
        text=f"After the move, is {mover_p} in contact with a wall?",
        answer=bool(contact),
        gt_meta=meta,
        gt_complete=method != "none" or blocked,
    )


# ---------------------------------------------------------------------------
# Affordance — chairs only; sit-on-mover or observer-chair (Mod 8)
# ---------------------------------------------------------------------------
def make_affordance_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
) -> Question:
    meta = dict(move.edit_meta or {})
    mode = str(meta.get("affordance_mode") or "sit_on_mover")
    mover_p = _mover_phrase(event0, move)

    if mode == "observer_chair":
        seat_id = meta.get("seat_id")
        seat_type = str(meta.get("seat_type") or "Chair")
        seat0 = _find_obj(event0, seat_id) if seat_id else None
        seat1 = _find_obj(event1, seat_id) if seat_id else None
        seat = seat1 or seat0
        target = _find_obj(event1, move.object_id)
        if seat is None or target is None:
            return Question(
                category="affordance",
                question_id=_qid("affordance"),
                text=(
                    f"When sitting on the chair, is {mover_p} within your forward view "
                    f"(within ±45° of straight ahead) after it moved?"
                ),
                answer=None,
                gt_complete=False,
                gt_meta={"affordance_mode": mode, "note": "missing seat or mover"},
            )
        # Modification 11: always leftmost/rightmost wording for affordance seats.
        seat_p = extreme_instance_phrase(
            event0, seat["objectId"], seat["objectType"]
        )
        angle = _fov_angle_deg(seat, target)
        if angle is None:
            return Question(
                category="affordance",
                question_id=_qid("affordance"),
                text=(
                    f"When sitting on {seat_p}, is {mover_p} within your forward view "
                    f"(within ±45° of straight ahead) after it moved?"
                ),
                answer=None,
                gt_complete=False,
                gt_meta={"affordance_mode": mode, "seat_phrase": seat_p},
            )
        return Question(
            category="affordance",
            question_id=_qid("affordance"),
            text=(
                f"When sitting on {seat_p}, is {mover_p} within your forward view "
                f"(within ±45° of straight ahead) after it moved?"
            ),
            answer=bool(angle <= 45.0),
            gt_meta={
                "proxy": "sitting_forward_fov",
                "affordance_mode": mode,
                "seat_id": seat["objectId"],
                "seat_type": seat_type,
                "seat_phrase": seat_p,
                "seat_extreme": meta.get("seat_extreme"),
                "target_id": move.object_id,
                "target_phrase": mover_p,
                "angle_deg": angle,
                "fov_half_angle_deg": 45.0,
            },
            gt_complete=True,
        )

    # sit_on_mover: seat is the moved chair
    if move.object_type not in AFFORDANCE_CHAIR_TYPES:
        return Question(
            category="affordance",
            question_id=_qid("affordance"),
            text=f"Is {mover_p} a chair one could sit on?",
            answer=False,
            gt_meta={"note": "non-chair mover; category should be resampled"},
            gt_complete=False,
        )

    # Modification 11: phrase the seat as leftmost/rightmost, not \"Nth from left\".
    seat_p = extreme_instance_phrase(event0, move.object_id, move.object_type)
    after = _find_obj(event1, move.object_id)
    others = _referable_others(event0, event1, move.object_id)
    if not after or not others:
        return Question(
            category="affordance",
            question_id=_qid("affordance"),
            text=(
                f"When sitting on {seat_p}, is any landmark within ±45° ahead?"
            ),
            answer=None,
            gt_complete=False,
            gt_meta={"affordance_mode": "sit_on_mover", "seat_phrase": seat_p},
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    for tgt in others:
        angle = _fov_angle_deg(after, tgt)
        if angle is None:
            continue
        scored.append((angle, tgt))

    if not scored:
        return Question(
            category="affordance",
            question_id=_qid("affordance"),
            text=f"When sitting on {seat_p}, is any landmark within ±45° ahead?",
            answer=False,
            gt_complete=False,
            gt_meta={"proxy": "sitting_forward_fov", "affordance_mode": "sit_on_mover"},
        )

    in_fov = [t for a, t in scored if a <= 45.0]
    out_fov = [t for a, t in scored if a > 45.0]
    if in_fov and out_fov:
        target = rng.choice(in_fov if rng.random() < 0.5 else out_fov)
    else:
        target = rng.choice(in_fov or out_fov)

    angle = next(a for a, t in scored if t["objectId"] == target["objectId"])
    tgt_p = _phrase(event0, target["objectId"], target["objectType"])
    return Question(
        category="affordance",
        question_id=_qid("affordance"),
        text=(
            f"When sitting on {seat_p}, is {tgt_p} within your forward view "
            f"(within ±45° of straight ahead)?"
        ),
        answer=bool(angle <= 45.0),
        gt_meta={
            "proxy": "sitting_forward_fov",
            "affordance_mode": "sit_on_mover",
            "seat_id": move.object_id,
            "seat_type": move.object_type,
            "seat_phrase": seat_p,
            "seat_extreme": meta.get("seat_extreme"),
            "target_id": target["objectId"],
            "target_phrase": tgt_p,
            "angle_deg": angle,
            "fov_half_angle_deg": 45.0,
        },
        gt_complete=True,
    )


# ---------------------------------------------------------------------------
# Left / right — bearing span + ordinal landmark phrases
# ---------------------------------------------------------------------------
def make_left_right_landmark_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
    *,
    prefer_yes: Optional[bool] = None,
) -> Question:
    from cit_pipeline.config import (
        LEFT_RIGHT_MIN_SPAN_PX,
        LEFT_RIGHT_REQUIRE_FLIP,
    )
    from cit_pipeline.spatial_metrics import object_screen_u

    # Modification 12: ban mover's parent receptacles (same idea as closer_to Mod 11).
    ban_ids = _mover_receptacle_ids(event0, move.object_id) | _mover_receptacle_ids(
        event1, move.object_id
    )
    others = [
        o
        for o in _referable_others(event0, event1, move.object_id)
        if o["objectId"] not in ban_ids
    ]
    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)
    mover_p = _mover_phrase(event0, move)
    if not others or not before or not after:
        return Question(
            category="left_right_landmark",
            question_id=_qid("left_right_landmark"),
            text=f"Is {mover_p} to the left of any landmark?",
            answer=None,
            gt_complete=False,
            gt_meta={"banned_receptacle_ids": sorted(ban_ids)},
        )

    # v7.2: left/right from on-screen object-center *u* (mask centroid), not bearing.
    u0, src0 = object_screen_u(event0, before, prefer_mask=True)
    u1, src1 = object_screen_u(event1, after, prefer_mask=True)
    if u0 is None or u1 is None:
        return Question(
            category="left_right_landmark",
            question_id=_qid("left_right_landmark"),
            text=f"Is {mover_p} to the left of a nearby landmark?",
            answer=None,
            gt_complete=False,
            gt_meta={
                "note": "screen_u undefined for mover",
                "banned_receptacle_ids": sorted(ban_ids),
            },
        )

    lo, hi = (u0, u1) if u0 <= u1 else (u1, u0)
    span = abs(u1 - u0)
    margin_px = 24.0
    lo_e, hi_e = lo - margin_px, hi + margin_px
    edit_meta = dict(move.edit_meta or {})
    planned_land_id = edit_meta.get("left_right_landmark_id")

    scored: list[tuple[float, dict[str, Any], float, bool, str]] = []
    for obj in others:
        ul, src_l = object_screen_u(event1, obj, prefer_mask=True)
        if ul is None:
            continue
        if ul < lo_e:
            dist_to_span = lo_e - ul
        elif ul > hi_e:
            dist_to_span = ul - hi_e
        else:
            dist_to_span = 0.0
        left_before = u0 < ul
        left_after = u1 < ul
        flips = left_before != left_after
        # Prefer the planned landmark strongly (must match the flip edit).
        bonus = 0.0
        if planned_land_id and obj["objectId"] == planned_land_id:
            bonus = -50.0 if flips else -10.0
        scored.append((dist_to_span + bonus, obj, ul, flips, src_l))

    if not scored:
        return Question(
            category="left_right_landmark",
            question_id=_qid("left_right_landmark"),
            text=f"Is {mover_p} to the left of a landmark?",
            answer=None,
            gt_complete=False,
            gt_meta={"banned_receptacle_ids": sorted(ban_ids)},
        )

    scored.sort(key=lambda t: (t[0], abs(t[2] - 0.5 * (lo + hi))))
    # Prefer a flipping landmark (and decent span) whenever possible.
    flip_any = [t for t in scored if t[3]]
    if LEFT_RIGHT_REQUIRE_FLIP and not flip_any:
        return Question(
            category="left_right_landmark",
            question_id=_qid("left_right_landmark"),
            text=f"Is {mover_p} to the left of a landmark?",
            answer=None,
            gt_complete=False,
            gt_meta={
                "note": "no landmark with left/right flip after move",
                "u_span_px": [lo, hi],
                "span_px": span,
                "banned_receptacle_ids": sorted(ban_ids),
            },
        )
    if LEFT_RIGHT_REQUIRE_FLIP and span < LEFT_RIGHT_MIN_SPAN_PX * 0.5:
        return Question(
            category="left_right_landmark",
            question_id=_qid("left_right_landmark"),
            text=f"Is {mover_p} to the left of a landmark?",
            answer=None,
            gt_complete=False,
            gt_meta={
                "note": "pixel span too small for reliable flip",
                "span_px": span,
                "banned_receptacle_ids": sorted(ban_ids),
            },
        )

    # Prefer planned landmark if it flips.
    planned_hits = [
        t for t in scored if planned_land_id and t[1]["objectId"] == planned_land_id
    ]
    pool = [t for t in scored if t[0] <= margin_px + 5.0][:8] or scored[:5]
    flip_pool = [t for t in pool if t[3]] or flip_any[:8]
    if planned_hits and planned_hits[0][3]:
        _, landmark, land_u, flipped_lr, land_src = planned_hits[0]
    elif flip_pool:
        _, landmark, land_u, flipped_lr, land_src = rng.choice(flip_pool)
    else:
        _, landmark, land_u, flipped_lr, land_src = rng.choice(pool)
    land_p = _phrase(event0, landmark["objectId"], landmark["objectType"])

    left_before = bool(u0 < land_u)
    left_after = bool(u1 < land_u)
    # prefer_yes: choose left vs right question so Yes/No hits the balance target.
    if prefer_yes is True:
        ask_left = left_after
    elif prefer_yes is False:
        ask_left = not left_after
    elif flipped_lr:
        ask_left = left_after
        if rng.random() < 0.5:
            ask_left = not left_after
    else:
        ask_left = rng.random() < 0.5
    if ask_left:
        text = (
            f"In the after image, is {mover_p} to the left of {land_p} "
            f"from the camera's viewpoint?"
        )
        answer = left_after
        answer_before = left_before
    else:
        text = (
            f"In the after image, is {mover_p} to the right of {land_p} "
            f"from the camera's viewpoint?"
        )
        answer = bool(u1 > land_u)
        answer_before = bool(u0 > land_u)

    return Question(
        category="left_right_landmark",
        question_id=_qid("left_right_landmark"),
        text=text,
        answer=bool(answer),
        gt_meta={
            "mover_phrase": mover_p,
            "landmark_id": landmark["objectId"],
            "landmark_phrase": land_p,
            "moved_screen_u_before": u0,
            "moved_screen_u_after": u1,
            "landmark_screen_u": land_u,
            "u_span_px": [lo, hi],
            "span_px": span,
            "u_metric_sources": {
                "before": src0,
                "after": src1,
                "landmark": land_src,
            },
            "asked": "left" if ask_left else "right",
            "metric": "mask_centroid_u",
            "answer_metric": "screen_u",
            "banned_receptacle_ids": sorted(ban_ids),
            "preferred_flip": bool(flipped_lr),
            "flip_planned": bool(edit_meta.get("left_right_flip_plan")),
            "planned_landmark_id": planned_land_id,
            "answer_before": bool(answer_before),
            "answer_after": bool(answer),
            "answer_flipped": bool(answer_before) != bool(answer),
        },
        gt_complete=True,
    )


# ---------------------------------------------------------------------------
# Feasibility — prefer orthogonal axis vs applied move (Mod 2 §3, Mod 3 §3)
# ---------------------------------------------------------------------------
def _sample_feasibility_probe_meters(
    rng: random.Random,
    *,
    prefer_feasible: bool,
) -> float:
    """Sample probe length for feasibility QA.

    Same support ``[M_MIN, M_MAX]`` for Yes and No. Optional power skew
    counters physics (long probes fail more) so accepted Yes/No means stay
    close: Yes-intent uses power<1 (longer), No-intent power>1 (shorter).
    """
    lo = float(FEASIBILITY_PROBE_M_MIN)
    hi = float(FEASIBILITY_PROBE_M_MAX)
    if hi <= lo:
        return round(lo, 2)
    u = float(rng.random())
    power = (
        float(FEASIBILITY_PROBE_YES_POWER)
        if prefer_feasible
        else float(FEASIBILITY_PROBE_NO_POWER)
    )
    power = max(1e-3, power)
    u = u**power
    return round(lo + (hi - lo) * u, 2)


def make_feasibility_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
    *,
    controller: Any = None,
    cam: Optional[CameraPose] = None,
    prefer_feasible: Optional[bool] = None,
    fast_mode: bool = False,
    fast_max_attempts: Optional[int] = None,
) -> Question:
    """Probe an additional single-axis translation from rgb1.

    Modification 9: search **compound** (two-axis) collisions first. When a
    compound teleport fails, decompose into the two single-axis components and
    harvest failing singles as hard "no" probes. Target ~50% collision answers
    via ``prefer_feasible`` / ``FEASIBILITY_TARGET_NO_FRAC``.

    Mod 28 ``fast_mode``: endpoint (+ optional midpoint) probes, wall closed-form
    Nos, no compound search, ≤``FEASIBILITY_FAST_MAX_ATTEMPTS`` attempts
    (override with ``fast_max_attempts``).
    """
    from cit_pipeline.camera_search import camera_basis

    feas_fast_cap = int(
        FEASIBILITY_FAST_MAX_ATTEMPTS
        if fast_max_attempts is None
        else fast_max_attempts
    )
    mover_p = _mover_phrase(event0, move)
    if controller is None or cam is None:
        return Question(
            category="feasibility",
            question_id=_qid("feasibility"),
            text=f"Would an additional move of {mover_p} be feasible without colliding or passing through any surfaces or objects?",
            answer=None,
            gt_complete=False,
            gt_meta={"note": "controller not provided"},
        )

    agent = event1.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    cur = _find_obj(event1, move.object_id)
    if not cur:
        return Question(
            category="feasibility",
            question_id=_qid("feasibility"),
            text=f"Is a further move of {mover_p} feasible?",
            answer=None,
            gt_complete=False,
            gt_meta={},
        )

    old_pos = dict(cur["position"])
    old_rot = dict(cur["rotation"])

    def _penetrates_wall(ev: Any) -> bool:
        """True if the mover AABB crosses the scene bound (wall proxy)."""
        from cit_pipeline.category_edits import _signed_edge_wall_clearance
        from cit_pipeline.config import WALL_PENETRATION_TOL_M

        obj = _find_obj(ev, move.object_id)
        if obj is None:
            return True
        clr, method, _ = _signed_edge_wall_clearance(ev, obj)
        if method == "none":
            return False
        return float(clr) < -float(WALL_PENETRATION_TOL_M)

    def _try_delta(delta_xyz, *, swept: Optional[bool] = None) -> bool:
        """Collision-checked probe. Mod 24: non-kinematic + wall clip.

        Full quality: swept waypoints along the path.
        Fast mode: endpoint only; add midpoint when displacement > 0.4 m.
        """
        dx, dy, dz = float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2])
        mag = max(abs(dx), abs(dy), abs(dz))
        use_swept = (not fast_mode) if swept is None else bool(swept)
        if use_swept:
            n_steps = 4 if mag > 0.35 else 2
            fracs = [float(i) / float(n_steps) for i in range(1, n_steps + 1)]
        else:
            fracs = [0.5, 1.0] if mag > 0.4 else [1.0]
        for frac in fracs:
            new_pos = {
                "x": cur["position"]["x"] + dx * frac,
                "y": cur["position"]["y"] + dy * frac,
                "z": cur["position"]["z"] + dz * frac,
            }
            ev = teleport_object(
                controller,
                move.object_id,
                new_pos,
                old_rot,
                force=False,
                kinematic=False,
            )
            ok = bool(ev.metadata["lastActionSuccess"])
            if ok and _penetrates_wall(ev):
                ok = False
            teleport_object(
                controller, move.object_id, old_pos, old_rot, force=True, kinematic=True
            )
            if not ok:
                return False
        return True

    def _wall_blocks_probe(direction: str, meters: float) -> bool:
        """Closed-form No: probe toward nearest wall farther than clearance."""
        from cit_pipeline.category_edits import (
            _signed_edge_wall_clearance,
            wall_axis_to_world_delta,
            world_delta_to_camera_direction,
        )

        clr, method, axis = _signed_edge_wall_clearance(event1, cur)
        if method == "none" or axis is None:
            return False
        if float(clr) > float(meters) + 0.05:
            return False
        wall_dir = world_delta_to_camera_direction(
            wall_axis_to_world_delta(axis), basis
        )
        if wall_dir != direction:
            return False
        # Moving at least to (clearance - margin) reaches / passes the wall.
        return float(meters) >= max(0.05, float(clr) - float(WALL_STOP_MARGIN_M))

    ortho = set(_orthogonal_directions(move.direction))
    Probe = tuple[str, float, bool, str, Optional[str]]
    probes: list[Probe] = []
    seen_single: set[tuple[str, float]] = set()
    compound_hits = 0
    compound_decomp_nos = 0

    def _add_single(
        direction: str,
        meters: float,
        ok: bool,
        *,
        source: str = "single",
        partner: Optional[str] = None,
    ) -> None:
        key = (direction, float(meters))
        if key in seen_single and source == "single":
            return
        if key in seen_single:
            for i, p in enumerate(probes):
                if (p[0], p[1]) == key:
                    if source == "compound_decomp" and not ok and p[2]:
                        probes[i] = (direction, meters, ok, source, partner)
                    return
        seen_single.add(key)
        probes.append((direction, meters, ok, source, partner))

    if prefer_feasible is None:
        prefer_feasible = rng.random() >= FEASIBILITY_TARGET_NO_FRAC

    if fast_mode:
        # Meter-first continuous sampling (v10.1+): draw m from a shared-range
        # distribution (optional mild Yes/No power skew), then search directions
        # for the desired polarity. Avoids picking from polarity pools that are
        # naturally short-Yes / long-No under physics.
        dirs = [d for d in ("left", "right", "forward", "back") if d in ortho] or list(
            ("left", "right", "forward", "back")
        )
        chosen_fast: Optional[Probe] = None
        attempts = 0
        continuous = bool(FEASIBILITY_PROBE_CONTINUOUS)

        if continuous:
            # Each outer try commits to one meters draw, then probes all dirs.
            max_meter_draws = max(4, (feas_fast_cap + len(dirs) - 1) // max(len(dirs), 1))
            for _ in range(max_meter_draws):
                if attempts >= feas_fast_cap or chosen_fast is not None:
                    break
                meters = _sample_feasibility_probe_meters(
                    rng, prefer_feasible=bool(prefer_feasible)
                )
                dir_order = list(dirs)
                rng.shuffle(dir_order)
                at_m: list[Probe] = []
                for direction in dir_order:
                    if attempts >= feas_fast_cap:
                        break
                    if _wall_blocks_probe(direction, float(meters)):
                        _add_single(
                            direction, float(meters), False, source="wall_clearance"
                        )
                        at_m.append(
                            (direction, float(meters), False, "wall_clearance", None)
                        )
                        continue
                    attempts += 1
                    delta = camera_delta_to_world(direction, meters, basis)
                    ok = _try_delta(delta, swept=False)
                    _add_single(direction, float(meters), ok, source="fast_single")
                    at_m.append((direction, float(meters), ok, "fast_single", None))
                want = bool(prefer_feasible)
                matched = [p for p in at_m if bool(p[2]) == want]
                if not matched:
                    continue
                use = matched
                if FEASIBILITY_BALANCE_DIRECTION_POLARITY:
                    opp_dirs = {p[0] for p in at_m if bool(p[2]) != want}
                    mixed = [p for p in matched if p[0] in opp_dirs]
                    if mixed:
                        use = mixed
                chosen_fast = rng.choice(use)
        else:
            # Legacy discrete shared grid + polarity-pool pick.
            rng.shuffle(dirs)
            meters_list = list(FEASIBILITY_FAST_SHARED_METERS)
            rng.shuffle(meters_list)
            for direction in dirs:
                if attempts >= feas_fast_cap:
                    break
                for meters in meters_list:
                    if attempts >= feas_fast_cap:
                        break
                    if _wall_blocks_probe(direction, float(meters)):
                        _add_single(
                            direction, float(meters), False, source="wall_clearance"
                        )
                        continue
                    attempts += 1
                    delta = camera_delta_to_world(direction, meters, basis)
                    ok = _try_delta(delta, swept=False)
                    _add_single(direction, float(meters), ok, source="fast_single")

            feas = [p for p in probes if p[2]]
            infeas = [p for p in probes if not p[2]]
            pool = feas if prefer_feasible else infeas
            if pool:
                use_pool = pool
                if FEASIBILITY_BALANCE_DIRECTION_POLARITY:
                    opp = infeas if prefer_feasible else feas
                    opp_dirs = {p[0] for p in opp}
                    mixed = [p for p in pool if p[0] in opp_dirs]
                    if mixed:
                        use_pool = mixed
                    opp_meters = {float(p[1]) for p in opp}
                    both_m = [p for p in use_pool if float(p[1]) in opp_meters]
                    if both_m:
                        use_pool = both_m
                by_dir: dict[str, list[Probe]] = {}
                for p in use_pool:
                    by_dir.setdefault(str(p[0]), []).append(p)
                dir_key = rng.choice(sorted(by_dir.keys()))
                chosen_fast = rng.choice(by_dir[dir_key])

        feas = [p for p in probes if p[2]]
        infeas = [p for p in probes if not p[2]]

        if chosen_fast is None:
            note = "no_feasible_probe" if prefer_feasible else "no_infeasible_probe"
            return Question(
                category="feasibility",
                question_id=_qid("feasibility"),
                text=(
                    f"Starting from the after image, would moving {mover_p} "
                    f"further be feasible without colliding?"
                ),
                answer=None,
                gt_complete=False,
                gt_meta={
                    "note": note,
                    "prefer_feasible": prefer_feasible,
                    "fast_mode": True,
                    "n_feas_probes": len(feas),
                    "n_infeas_probes": len(infeas),
                    "fast_attempts": attempts,
                    "shared_meter_pool": True,
                    "continuous_meters": continuous,
                },
            )
        direction, meters, ok, source, partner = chosen_fast
        dir_phrase = format_direction_phrase(str(direction))
        return Question(
            category="feasibility",
            question_id=_qid("feasibility"),
            text=(
                f"Starting from the after image, would moving {mover_p} "
                f"{meters:.2f}m {dir_phrase} be feasible without colliding or "
                f"passing through any surfaces or objects?"
            ),
            answer=bool(ok),
            gt_complete=True,
            gt_meta={
                "mover_phrase": mover_p,
                "probe_direction": direction,
                "probe_direction_phrase": dir_phrase,
                "probe_meters": meters,
                "teleport_success": ok,
                "forceAction": False,
                "original_move_direction": move.direction,
                "original_move_meters": move.meters,
                "probe_axis": _move_axis(direction),
                "original_axis": _move_axis(move.direction),
                "probe_source": source,
                "compound_partner": partner,
                "n_feas_probes": len(feas),
                "n_infeas_probes": len(infeas),
                "shared_meter_pool": True,
                "continuous_meters": continuous,
                "probe_m_range": [float(FEASIBILITY_PROBE_M_MIN), float(FEASIBILITY_PROBE_M_MAX)],
                "probe_yes_power": float(FEASIBILITY_PROBE_YES_POWER),
                "probe_no_power": float(FEASIBILITY_PROBE_NO_POWER),
                "prefer_feasible": prefer_feasible,
                "fast_mode": True,
                "fast_attempts": attempts,
            },
        )

    # ---- Full-quality path (iTHOR) ----
    for direction in FEASIBILITY_PROBE_DIRECTIONS:
        meters_list = (
            FEASIBILITY_VERTICAL_METERS
            if direction in ("up", "down")
            else FEASIBILITY_PROBE_METERS
        )
        for meters in meters_list:
            delta = camera_delta_to_world(direction, meters, basis)
            ok = _try_delta(delta, swept=True)
            _add_single(direction, meters, ok, source="single")

    vert_down = ("down",)
    compound_pairs: list[tuple[str, str]] = []
    for a in ("left", "right"):
        for b in ("forward", "back") + vert_down:
            compound_pairs.append((a, b))
    for a in ("forward", "back"):
        for b in vert_down:
            compound_pairs.append((a, b))
    rng.shuffle(compound_pairs)

    max_compound_tries = 18
    tries = 0
    for d1, d2 in compound_pairs:
        stop = False
        for meters in FEASIBILITY_COMPOUND_METERS:
            if tries >= max_compound_tries or compound_decomp_nos >= 4:
                stop = True
                break
            tries += 1
            delta = camera_delta_to_world(d1, meters, basis) + camera_delta_to_world(
                d2, meters, basis
            )
            compound_ok = _try_delta(delta, swept=True)
            if compound_ok:
                continue
            compound_hits += 1
            for d, partner in ((d1, d2), (d2, d1)):
                ok = _try_delta(camera_delta_to_world(d, meters, basis), swept=True)
                _add_single(
                    d, meters, ok, source="compound_decomp", partner=partner
                )
                if not ok:
                    compound_decomp_nos += 1
        if stop:
            break

    def _pick(pool: list[Probe]) -> Optional[Probe]:
        if not pool:
            return None
        if FEASIBILITY_BALANCE_DIRECTION_POLARITY:
            opp = [p for p in probes if p[2] != pool[0][2]]
            opp_dirs = {p[0] for p in opp}
            mixed = [p for p in pool if p[0] in opp_dirs]
            if mixed:
                pool = mixed
            opp_meters = {float(p[1]) for p in opp}
            both_m = [p for p in pool if float(p[1]) in opp_meters]
            if both_m:
                pool = both_m
        ortho_pool = [p for p in pool if p[0] in ortho]
        compound_no = [
            p for p in pool if (not p[2]) and p[3] == "compound_decomp"
        ]
        if prefer_feasible is False and compound_no:
            ortho_c = [p for p in compound_no if p[0] in ortho]
            if ortho_c and rng.random() < FEASIBILITY_ORTHOGONAL_PROB:
                return rng.choice(ortho_c)
            return rng.choice(compound_no)
        prefer_ortho = rng.random() < FEASIBILITY_ORTHOGONAL_PROB
        cand = ortho_pool if (prefer_ortho and ortho_pool) else pool
        by_dir: dict[str, list[Probe]] = {}
        for p in cand:
            by_dir.setdefault(str(p[0]), []).append(p)
        return rng.choice(by_dir[rng.choice(sorted(by_dir.keys()))])

    feas = [p for p in probes if p[2]]
    infeas = [p for p in probes if not p[2]]

    chosen: Optional[Probe] = None
    if prefer_feasible is True:
        chosen = _pick(feas)
        if chosen is None:
            return Question(
                category="feasibility",
                question_id=_qid("feasibility"),
                text=(
                    f"Starting from the after image, would moving {mover_p} "
                    f"further be feasible without colliding?"
                ),
                answer=None,
                gt_complete=False,
                gt_meta={
                    "note": "no_feasible_probe",
                    "prefer_feasible": True,
                    "n_infeas_probes": len(infeas),
                    "compound_hits": compound_hits,
                },
            )
    elif prefer_feasible is False:
        chosen = _pick(infeas)
        if chosen is None and feas:
            for direction in FEASIBILITY_PROBE_DIRECTIONS:
                if chosen is not None:
                    break
                for meters in (2.0, 1.75, 1.5):
                    delta = camera_delta_to_world(direction, meters, basis)
                    ok = _try_delta(delta, swept=True)
                    if not ok:
                        chosen = (direction, float(meters), False, "extended_no", None)
                        break
        if chosen is None:
            return Question(
                category="feasibility",
                question_id=_qid("feasibility"),
                text=(
                    f"Starting from the after image, would moving {mover_p} "
                    f"further be feasible without colliding?"
                ),
                answer=None,
                gt_complete=False,
                gt_meta={
                    "note": "no_infeasible_probe",
                    "prefer_feasible": False,
                    "n_feas_probes": len(feas),
                    "compound_hits": compound_hits,
                },
            )
    elif feas and infeas:
        chosen = _pick(feas if rng.random() < 0.5 else infeas)
    else:
        chosen = _pick(feas or infeas)

    if chosen is None:
        return Question(
            category="feasibility",
            question_id=_qid("feasibility"),
            text=f"Is any further translation of {mover_p} feasible?",
            answer=None,
            gt_complete=False,
            gt_meta={"note": "no probes", "compound_hits": compound_hits},
        )

    direction, meters, ok, source, partner = chosen
    dir_phrase = format_direction_phrase(str(direction))
    return Question(
        category="feasibility",
        question_id=_qid("feasibility"),
        text=(
            f"Starting from the after image, would moving {mover_p} "
            f"{meters:.2f}m {dir_phrase} be feasible without colliding or "
            f"passing through any surfaces or objects?"
        ),
        answer=bool(ok),
        gt_complete=True,
        gt_meta={
            "mover_phrase": mover_p,
            "probe_direction": direction,
            "probe_direction_phrase": dir_phrase,
            "probe_meters": meters,
            "teleport_success": ok,
            "forceAction": False,
            "original_move_direction": move.direction,
            "original_move_meters": move.meters,
            "probe_axis": _move_axis(direction),
            "original_axis": _move_axis(move.direction),
            "orthogonal_to_original": direction in ortho,
            "probe_source": source,
            "compound_partner": partner,
            "compound_hits": compound_hits,
            "n_feas_probes": len(feas),
            "n_infeas_probes": len(infeas),
            "collision_includes_surfaces": True,
            "fast_mode": False,
            "shared_meter_pool": True,
        },
    )


# ---------------------------------------------------------------------------
# Support — elevated, non-floor pickupables (Mod 2 §6, Mod 3 §4, Mod 10)
# ---------------------------------------------------------------------------
def _is_floor_receptacle_id(parent_id: Any) -> bool:
    if not parent_id or not isinstance(parent_id, str):
        return False
    return parent_id == "Floor" or parent_id.startswith("Floor")


def _height_for_support(obj: dict[str, Any]) -> Optional[float]:
    """Prefer AABB bottom y; fall back to pose center y."""
    bottom = aabb_bottom_y(obj)
    if bottom is not None:
        return float(bottom)
    pos = obj.get("position") or {}
    if "y" in pos:
        return float(pos["y"])
    return None


def make_support_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
) -> Question:
    """Modification 10: after gravity settle, floor-drop vs furniture-supported.

    Primary GT signal is height drop (AABB bottom / pose y before vs after).
    Receptacle labels corroborate when present.
    """
    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)
    mover_p = _mover_phrase(event0, move)
    q_text = (
        f"After the move, will {mover_p} drop onto the floor "
        f"instead of staying well-supported by furniture?"
    )
    if not before or not after:
        return Question(
            category="support",
            question_id=_qid("support"),
            text=q_text,
            answer=None,
            answer_type="yes_no",
            gt_complete=False,
            gt_meta={},
        )

    # Pre-move must already be an elevated, non-floor support candidate.
    if not support_qa_eligible(before):
        return Question(
            category="support",
            question_id=_qid("support"),
            text=q_text,
            answer=None,
            answer_type="yes_no",
            gt_complete=False,
            gt_meta={
                "note": "mover not eligible for support QA (floor / not pickupable)",
                "settle_gravity": bool((move.edit_meta or {}).get("settle_gravity")),
                "resting_on_floor": object_resting_on_floor(before),
            },
        )

    parents_after = after.get("parentReceptacles") or []
    parents_before = before.get("parentReceptacles") or []
    y = float(after["position"]["y"])
    y0 = float(before["position"]["y"])
    bottom0 = aabb_bottom_y(before)
    bottom1 = aabb_bottom_y(after)
    h0 = _height_for_support(before)
    h1 = _height_for_support(after)
    height_drop = (
        float(h0) - float(h1) if h0 is not None and h1 is not None else None
    )

    on_furniture = any(
        p and not _is_floor_receptacle_id(p) for p in parents_after
    )
    on_floor = any(_is_floor_receptacle_id(p) for p in parents_after) or (
        h1 is not None and h1 < SUPPORT_QA_FLOOR_BOTTOM_Y_M
    )
    dropped_by_height = (
        height_drop is not None and height_drop >= SUPPORT_DROP_HEIGHT_M
    )
    # Height drop is the main floor-drop signal when receptacles are empty/lagging.
    # Still-on-furniture wins over a partial height change (settled onto lower furniture).
    if on_furniture and not (on_floor and dropped_by_height):
        dropped = False
        furniture_supported = True
    elif on_floor and not on_furniture:
        dropped = True
        furniture_supported = False
    elif dropped_by_height and not on_furniture:
        dropped = True
        furniture_supported = False
    else:
        # e.g. empty parents + still elevated ⇒ gravity did not ground the object.
        dropped = False
        furniture_supported = False

    base_meta = {
        "mover_phrase": mover_p,
        "parent_receptacles_before": parents_before,
        "parent_receptacles_after": parents_after,
        "y_before": y0,
        "y_after": y,
        "bottom_y_before": bottom0,
        "bottom_y_after": bottom1,
        "height_before": h0,
        "height_after": h1,
        "height_drop_m": height_drop,
        "support_drop_height_m": SUPPORT_DROP_HEIGHT_M,
        "support_min_bottom_y_m": SUPPORT_QA_MIN_BOTTOM_Y_M,
        "support_floor_bottom_y_m": SUPPORT_QA_FLOOR_BOTTOM_Y_M,
        "on_furniture": on_furniture,
        "on_floor": on_floor,
        "dropped_by_height": dropped_by_height,
        "settle_gravity": bool((move.edit_meta or {}).get("settle_gravity")),
        "metric": "height_drop_after_gravity",
    }

    if not dropped and not furniture_supported:
        return Question(
            category="support",
            question_id=_qid("support"),
            text=q_text,
            answer=None,
            answer_type="yes_no",
            gt_complete=False,
            gt_meta={
                **base_meta,
                "note": "ambiguous support after gravity settle",
            },
        )

    return Question(
        category="support",
        question_id=_qid("support"),
        text=q_text,
        answer=bool(dropped),
        answer_type="yes_no",
        gt_meta={
            **base_meta,
            "dropped_to_floor": bool(dropped),
            "furniture_supported": bool(furniture_supported),
        },
        gt_complete=True,
    )


# ---------------------------------------------------------------------------
# Closest-three — name 3 nearest objects; GT pool of 6 (Modification 4)
# ---------------------------------------------------------------------------
def make_closest_three_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
) -> Question:
    mover_p = _mover_phrase(event0, move)
    moved = _find_obj(event1, move.object_id)
    if not moved:
        return Question(
            category="closest_three",
            question_id=_qid("closest_three"),
            text=f"Name the three objects closest to {mover_p} after the move.",
            answer=None,
            answer_type="set",
            gt_complete=False,
            gt_meta={},
        )

    others = _referable_others(event0, event1, move.object_id)
    scored: list[tuple[float, dict[str, Any]]] = []
    for o in others:
        scored.append((_dist(moved["position"], o["position"]), o))
    scored.sort(key=lambda t: t[0])
    scored = filter_scored_anchor_type_caps(scored)
    if len(scored) < CLOSEST_THREE_ANSWER:
        return Question(
            category="closest_three",
            question_id=_qid("closest_three"),
            text=f"Name the three objects closest to {mover_p} after the move.",
            answer=None,
            answer_type="set",
            gt_meta={"note": "too few neighbors"},
            gt_complete=False,
        )

    top = scored[:CLOSEST_THREE_POOL]
    # If fewer than pool, pad is unnecessary; require at least ANSWER.
    pool_objs = [o for _, o in top]
    # Ensure pool size: if we have more scored, already truncated; if fewer, use all.
    while len(pool_objs) < CLOSEST_THREE_POOL and len(scored) > len(pool_objs):
        pool_objs.append(scored[len(pool_objs)][1])
    pool_objs = take_with_anchor_type_caps(pool_objs, n=CLOSEST_THREE_POOL)
    # If still short of 6, keep what we have but mark incomplete if < 6 preferred.
    answer_objs = pool_objs[:CLOSEST_THREE_ANSWER]
    pool_phrases = [_phrase(event0, o["objectId"], o["objectType"]) for o in pool_objs]
    answer_phrases = pool_phrases[:CLOSEST_THREE_ANSWER]
    # Shuffle display order of the candidate pool.
    display = list(pool_phrases)
    rng.shuffle(display)
    text = (
        f"After the move, which three objects are closest to {mover_p}? "
        f"Choose three from: " + "; ".join(display) + "."
    )
    return Question(
        category="closest_three",
        question_id=_qid("closest_three"),
        text=text,
        answer=answer_phrases,
        answer_type="set",
        gt_meta={
            "mover_phrase": mover_p,
            "candidate_pool": pool_phrases,
            "candidate_ids": [o["objectId"] for o in pool_objs],
            "candidate_distances": [_dist(moved["position"], o["position"]) for o in pool_objs],
            "answer_phrases": answer_phrases,
            "answer_ids": [o["objectId"] for o in answer_objs],
            "pool_size": len(pool_objs),
            "answer_size": CLOSEST_THREE_ANSWER,
            "metric": "3d_centroid_distance_top3_of_6",
        },
        gt_complete=len(pool_objs) >= CLOSEST_THREE_POOL,
    )


# ---------------------------------------------------------------------------
# Eligibility + sampling (Modification 3 §5)
# ---------------------------------------------------------------------------
def annotate_answer_flip(
    q: Question,
    event0: Any,
    event1: Any,
    move: MoveAction,
) -> Question:
    """Fill ``answer_before`` / ``answer_after`` / ``answer_flipped`` in gt_meta.

    Modification 12: used for train (~50% identical) vs bench (≤10% identical)
    quotas at export time. Categories that already set these fields are left as-is.
    """
    if not q.gt_complete:
        return q
    meta = dict(q.gt_meta or {})
    if "answer_flipped" in meta and "answer_after" in meta:
        q.gt_meta = meta
        return q

    cat = str(q.category)
    answer_after = q.answer
    answer_before: Any = None

    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)

    if cat == "closer_to" and before is not None:
        a_id, b_id = meta.get("landmark_a_id"), meta.get("landmark_b_id")
        a = _find_obj(event0, a_id) if a_id else None
        b = _find_obj(event0, b_id) if b_id else None
        # Landmarks are static; positions from either event are fine.
        a = a or (_find_obj(event1, a_id) if a_id else None)
        b = b or (_find_obj(event1, b_id) if b_id else None)
        if a and b:
            da0 = _dist(before["position"], a["position"])
            db0 = _dist(before["position"], b["position"])
            pa, pb = meta.get("phrase_a"), meta.get("phrase_b")
            answer_before = pa if da0 <= db0 else pb

    elif cat == "occlusion":
        oid = meta.get("occludee_id")
        thr = float(meta.get("threshold") or OCCLUDE_THRESHOLD)
        if oid:
            try:
                from cit_pipeline.occlusion import occlusion_cover_rate

                # Cover of B0 by A0 (before move) — usually near zero.
                cover0, _, _, _ = occlusion_cover_rate(
                    event0, event0, move.object_id, str(oid)
                )
                answer_before = bool(float(cover0) >= thr)
            except Exception:  # noqa: BLE001
                answer_before = False

    elif cat == "contact_wall" and before is not None:
        d0, method0 = _min_edge_wall_clearance(event0, before)
        if method0 != "none":
            answer_before = bool(d0 < WALL_CONTACT_M)
        else:
            answer_before = False

    elif cat == "affordance" and before is not None:
        mode = str(meta.get("affordance_mode") or "sit_on_mover")
        if mode == "observer_chair":
            seat = _find_obj(event0, meta.get("seat_id"))
            target = before
            if seat and target:
                ang = _fov_angle_deg(seat, target)
                if ang is not None:
                    answer_before = bool(ang <= 45.0)
        else:
            tgt = _find_obj(event0, meta.get("target_id")) or _find_obj(
                event1, meta.get("target_id")
            )
            if tgt:
                ang = _fov_angle_deg(before, tgt)
                if ang is not None:
                    answer_before = bool(ang <= 45.0)

    elif cat == "closest_three" and before is not None:
        cids = list(meta.get("candidate_ids") or [])
        if cids:
            scored = []
            for oid in cids:
                o = _find_obj(event0, oid) or _find_obj(event1, oid)
                if o is None:
                    continue
                scored.append((_dist(before["position"], o["position"]), oid))
            scored.sort(key=lambda t: t[0])
            before_ids = [oid for _, oid in scored[:CLOSEST_THREE_ANSWER]]
            after_ids = list(meta.get("answer_ids") or [])
            answer_before = before_ids
            # Flip if the top-3 set changes.
            answer_after_ids = after_ids
            flipped = set(before_ids) != set(answer_after_ids)
            meta["answer_before"] = before_ids
            meta["answer_after"] = after_ids
            meta["answer_flipped"] = flipped
            q.gt_meta = meta
            return q

    elif cat == "support" and before is not None and after is not None:
        # Before the move the object is (by eligibility) furniture-supported →
        # "will it drop?" is False before the edit is applied.
        answer_before = False

    elif cat == "feasibility":
        # Probe is defined from the after state; no meaningful before answer.
        answer_before = None

    # left_right already sets flip fields in its builder.

    if answer_before is not None and "answer_flipped" not in meta:
        meta["answer_before"] = answer_before
        meta["answer_after"] = answer_after
        if isinstance(answer_before, list) or isinstance(answer_after, list):
            meta["answer_flipped"] = set(answer_before or []) != set(answer_after or [])
        else:
            meta["answer_flipped"] = answer_before != answer_after
    elif "answer_flipped" not in meta:
        meta["answer_after"] = answer_after
        meta["answer_before"] = answer_before
        meta["answer_flipped"] = None if answer_before is None else (
            answer_before != answer_after
        )

    q.gt_meta = meta
    return q


def eligible_qa_categories(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    categories: Sequence[str] = QA_CATEGORIES,
) -> list[str]:
    """Drop categories that do not apply to this mover / view."""
    cats = list(categories)
    after = _find_obj(event1, move.object_id)
    before = _find_obj(event0, move.object_id)

    if "affordance" in cats:
        mode = (move.edit_meta or {}).get("affordance_mode")
        if mode == "observer_chair":
            pass
        elif move.object_type not in AFFORDANCE_CHAIR_TYPES:
            cats = [c for c in cats if c != "affordance"]

    if "contact_wall" in cats:
        if after is None or not contact_wall_eligible(event1, after):
            cats = [c for c in cats if c != "contact_wall"]

    if "support" in cats:
        ref = before or after
        if ref is None or not support_qa_eligible(ref):
            cats = [c for c in cats if c != "support"]

    if "occlusion" in cats:
        yes, no, _ = select_occlusion_pairs(
            event1, move.object_id, balance=True, event0=event0
        )
        if not yes and not no:
            cats = [c for c in cats if c != "occlusion"]

    if "closest_three" in cats:
        others = _referable_others(event0, event1, move.object_id)
        if len(others) < CLOSEST_THREE_POOL:
            cats = [c for c in cats if c != "closest_three"]

    if "count_near" in cats:
        others = _referable_others(event0, event1, move.object_id)
        if len(others) < CLOSEST_THREE_POOL:
            cats = [c for c in cats if c != "count_near"]

    if "near_object" in cats:
        others = _referable_others(event0, event1, move.object_id)
        if len(others) < 2:
            cats = [c for c in cats if c != "near_object"]

    return cats


def make_question_for_category(
    category: str,
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    rng: Optional[random.Random] = None,
    controller: Any = None,
    cam: Optional[CameraPose] = None,
    prefer_occ: Optional[bool] = None,
    prefer_feas: Optional[bool] = None,
    prefer_lr: Optional[bool] = None,
    prefer_near: Optional[bool] = None,
    prefer_v30_flip: Optional[bool] = None,
    fast_mode: bool = False,
    fast_feasibility_max_attempts: Optional[int] = None,
    dataset: str = "ithor",
) -> Question:
    """Build exactly one GT question for ``category`` (Modification 4)."""
    rng = rng or random.Random()
    if category == "occlusion":
        q = make_occlusion_question(
            event0, event1, move, rng=rng, prefer_yes=prefer_occ
        )
    elif category == "closer_to":
        q = make_closer_to_question(
            event0, event1, move, rng, dataset=str(dataset or "ithor")
        )
    elif category == "contact_wall":
        q = make_contact_wall_question(event0, event1, move)
    elif category == "affordance":
        q = make_affordance_question(event0, event1, move, rng)
    elif category == "left_right_landmark":
        q = make_left_right_landmark_question(
            event0, event1, move, rng, prefer_yes=prefer_lr
        )
    elif category == "feasibility":
        q = make_feasibility_question(
            event0,
            event1,
            move,
            rng,
            controller=controller,
            cam=cam,
            prefer_feasible=prefer_feas,
            fast_mode=bool(fast_mode),
            fast_max_attempts=fast_feasibility_max_attempts,
        )
    elif category == "support":
        q = make_support_question(event0, event1, move)
    elif category == "closest_three":
        from cit_pipeline.v30_categories import make_closest_three_flip_question

        q = make_closest_three_flip_question(event0, event1, move, rng)
    elif category == "count_near":
        from cit_pipeline.v30_categories import make_count_near_question

        q = make_count_near_question(event0, event1, move, rng)
    elif category == "near_object":
        from cit_pipeline.v30_categories import make_near_object_question

        q = make_near_object_question(
            event0,
            event1,
            move,
            rng,
            prefer_yes=prefer_near,
            prefer_flip=prefer_v30_flip,
        )
    else:
        raise ValueError(category)
    return annotate_answer_flip(q, event0, event1, move)


def sample_two_questions(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    rng: Optional[random.Random] = None,
    categories: Sequence[str] = QA_CATEGORIES,
    occlude_yes_counter: Optional[list[int]] = None,
    feasible_counter: Optional[list[int]] = None,
    controller: Any = None,
    cam: Optional[CameraPose] = None,
) -> list[Question]:
    """Pick two distinct *eligible* QA categories; build GT questions."""
    rng = rng or random.Random()
    cats = eligible_qa_categories(event0, event1, move, categories=categories)
    if len(cats) < 2:
        for fallback in ("closer_to", "left_right_landmark", "feasibility"):
            if fallback not in cats and fallback in categories:
                cats.append(fallback)
            if len(cats) >= 2:
                break
    if len(cats) < 2:
        raise ValueError(
            f"need at least 2 QA categories after filtering; got {cats}"
        )

    n = min(2, len(cats))
    chosen = rng.sample(cats, n)

    prefer_occ: Optional[bool] = None
    if occlude_yes_counter is not None and "occlusion" in chosen:
        y, n = occlude_yes_counter[0], occlude_yes_counter[1]
        if y < n:
            prefer_occ = True
        elif y > n:
            prefer_occ = False
        else:
            prefer_occ = rng.random() >= 0.50

    prefer_feas: Optional[bool] = None
    if feasible_counter is not None and "feasibility" in chosen:
        y, n = feasible_counter[0], feasible_counter[1]
        if y < n:
            prefer_feas = True
        elif y > n:
            prefer_feas = False
        else:
            prefer_feas = rng.random() >= FEASIBILITY_TARGET_NO_FRAC

    questions: list[Question] = []
    for cat in chosen:
        q = make_question_for_category(
            cat,
            event0,
            event1,
            move,
            rng=rng,
            controller=controller,
            cam=cam,
            prefer_occ=prefer_occ,
            prefer_feas=prefer_feas,
        )
        questions.append(q)
        if cat == "occlusion" and occlude_yes_counter is not None and q.gt_complete:
            if q.answer:
                occlude_yes_counter[0] += 1
            else:
                occlude_yes_counter[1] += 1
        if cat == "feasibility" and feasible_counter is not None and q.gt_complete:
            if q.answer:
                feasible_counter[0] += 1
            else:
                feasible_counter[1] += 1
    return questions


_CATEGORY_HINTS = {
    "occlusion": (
        "Ask whether a SPECIFIC named small object B is occluded by "
        "the moved large object A so that a part of B is invisible, or totally "
        "visible (yes/no). Keep referring phrases. Do NOT ask the vague "
        "'is anything occluded?' form."
    ),
    "closer_to": (
        "Ask a 4-way multiple-choice: which listed object is the mover closer to. "
        "Keep the four choice phrases unchanged."
    ),
    "contact_wall": (
        "Ask whether the moved object touches / is against a wall (yes/no). "
        "The object was stepped toward a nearby wall."
    ),
    "affordance": (
        "ONLY for chairs: ask whether a named landmark (or the moved object) is "
        "within ±45° ahead when sitting on the named chair. Keep leftmost/rightmost "
        "or ordinal phrases unchanged."
    ),
    "left_right_landmark": (
        "Ask left/right of a named landmark from the camera. Keep ordinal phrases."
    ),
    "feasibility": (
        "Ask whether a stated further translation would be feasible without colliding "
        "or passing through any surfaces or objects (tables/counters count). "
        "Prefer a different axis than the original move."
    ),
    "support": (
        "Ask whether the moved object will DROP ONTO THE FLOOR after the move "
        "(with gravity), rather than stay well-supported by furniture. yes = "
        "drops to floor; no = remains on furniture. Keep referring phrases."
    ),
    "closest_three": (
        "Ask which three objects are closest to the mover after the move, from a "
        "six-candidate SoM list. The correct trio must differ from before the move."
    ),
    "count_near": (
        "Ask how many of six SoM-marked objects lie within a stated distance of the "
        "moved object after the move. Choices are 2/3/4/5; the count must change vs before."
    ),
    "near_object": (
        "Ask a yes/no whether the moved object is within a stated distance of a named "
        "SoM landmark after the move. Prefer answers that flip vs before the move."
    ),
}


def word_questions_with_vlm(
    vlm: Any,
    move: MoveAction,
    questions: list[Question],
    rgb0_path: Path,
    rgb1_path: Path,
    *,
    use_images: bool = False,
) -> list[Question]:
    """Rephrase question text only; never invent sitting on non-seats.

    ``use_images=False`` (Mod 27 / risky v7 default): text-only rewrite. Images
    rarely changed the draft and dominated wall time.
    """
    rgb0_path = Path(rgb0_path)
    rgb1_path = Path(rgb1_path)
    edit_meta = move.edit_meta or {}
    sittable = (
        move.object_type in AFFORDANCE_CHAIR_TYPES
        or edit_meta.get("affordance_mode") == "observer_chair"
        or edit_meta.get("seat_type") in AFFORDANCE_CHAIR_TYPES
    )
    for q in questions:
        template = q.text
        if not q.gt_complete:
            # Do not let the VLM invent specific objects when GT is incomplete.
            q.gt_meta = dict(q.gt_meta)
            q.gt_meta["vlm_worded"] = False
            q.gt_meta["vlm_skip_reason"] = "gt_incomplete"
            continue
        facts = json.dumps(q.gt_meta, default=str)
        image_line = (
            "Images: first BEFORE, second AFTER.\n"
            if use_images
            else "No images are provided; rewrite from the draft and facts only.\n"
        )
        prompt = (
            "You write one clear spatial question for a counterfactual vision eval.\n"
            f"Category: {q.category}\n"
            f"Category instruction: {_CATEGORY_HINTS.get(q.category, '')}\n"
            f"Move action: {move.text}\n"
            f"Mover type: {move.object_type}; sittable={sittable}\n"
            "Hard rule: NEVER ask anyone to sit on a non-seat object "
            "(bowl, plate, newspaper, book, box, …).\n"
            "Keep ordinal referring phrases unchanged when present "
            "(e.g. 'the 2nd stool from the left').\n"
            f"Ground-truth facts (do NOT reveal the answer): {facts}\n"
            f"Draft question (improve wording, same meaning): {template}\n"
            f"{image_line}"
            "Return ONLY the final question sentence."
        )
        try:
            text = vlm.chat(
                prompt,
                images=[rgb0_path, rgb1_path] if use_images else None,
                max_tokens=96,
                temperature=0.2,
            )
            line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            # Reject VLM wording that invents sitting on non-seats.
            if line and (sittable or "sitting on" not in line.lower()):
                if not line.endswith("?"):
                    line = line.rstrip(".") + "?"
                q.text = line
                q.gt_meta = dict(q.gt_meta)
                q.gt_meta["vlm_worded"] = True
                q.gt_meta["vlm_word_use_images"] = bool(use_images)
                q.gt_meta["template_text"] = template
            else:
                q.gt_meta = dict(q.gt_meta)
                q.gt_meta["vlm_worded"] = False
                q.gt_meta["vlm_reject_reason"] = "non-seat sitting wording"
        except Exception:
            q.gt_meta = dict(q.gt_meta)
            q.gt_meta["vlm_worded"] = False
    return questions
