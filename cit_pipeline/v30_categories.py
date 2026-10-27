"""Mod-30 / v30 distance categories: closest_three (flip), count_near, near_object.

All three:
  1) find an elegant translation edit
  2) derive answers from rgb0/rgb1 metadata distances (no extra sim queries)
  3) SoM on mover + referenced objects
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from ai2thor.controller import Controller

from cit_pipeline.camera_search import camera_basis
from cit_pipeline.category_edits import (
    filter_proposals_for_category,
    order_equal_by_mover,
    try_edit,
)
from cit_pipeline.config import (
    ANGULAR_SIZE_MIN,
    CLOSEST_THREE_ANSWER,
    CLOSEST_THREE_POOL,
    DIST_MAX,
    DIST_MIN,
)
from cit_pipeline.qa import (
    _dist,
    _find_obj,
    _mover_phrase,
    _phrase,
    _qid,
    _referable_others,
    filter_scored_anchor_type_caps,
    take_with_anchor_type_caps,
)
from cit_pipeline.schemas import CameraPose, MoveAction, Question
from cit_pipeline.vlm_client import VlmClient

# Distance thresholds (meters) tried for count_near / near_object.
V30_NEAR_THRESHOLDS_M = (0.35, 0.5, 0.65, 0.8, 1.0, 1.25)
COUNT_NEAR_ANSWERS = (2, 3, 4, 5)
# Leave ~20% unflipped so the model cannot shortcut on "always flips".
V30_TARGET_FLIP_FRAC = 0.80


def _simple_type_phrase(object_type: str) -> str:
    """SoM-friendly referring: type only (no '2nd … from the left')."""
    return f"the {str(object_type or 'object').lower()}"


def _sample_prefer_flip(
    rng: random.Random,
    prefer_flip: Optional[bool] = None,
    *,
    target_flip_frac: float = V30_TARGET_FLIP_FRAC,
) -> bool:
    if prefer_flip is not None:
        return bool(prefer_flip)
    return rng.random() < float(target_flip_frac)


def _neighbor_distances(
    event_ref: Any,
    event_pos: Any,
    mover_id: str,
    mover_pos: dict[str, float],
) -> list[tuple[float, dict[str, Any]]]:
    """Neighbors ordered by 3D distance to ``mover_pos`` (from event_pos objects).

    Caps ShelvingUnit / Drawer to ≤1 each so closest_three / count_near pools
    never list multiple look-alike furniture pieces.
    """
    others = _referable_others(event_ref, event_pos, mover_id)
    scored: list[tuple[float, dict[str, Any]]] = []
    for o in others:
        scored.append((_dist(mover_pos, o["position"]), o))
    scored.sort(key=lambda t: t[0])
    return filter_scored_anchor_type_caps(scored)


def _top_ids(scored: list[tuple[float, dict[str, Any]]], k: int) -> list[str]:
    return [o["objectId"] for _, o in scored[:k]]


def build_closest_three_flip_pool(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    require_flip: bool = True,
) -> Optional[dict[str, Any]]:
    """Pool of 6 from before/after top-3 (+pad). Default requires a top-3 set flip."""
    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)
    if not before or not after:
        return None
    scored0 = _neighbor_distances(event0, event0, move.object_id, before["position"])
    scored1 = _neighbor_distances(event0, event1, move.object_id, after["position"])
    if len(scored1) < CLOSEST_THREE_ANSWER or len(scored0) < CLOSEST_THREE_ANSWER:
        return None
    top0 = _top_ids(scored0, CLOSEST_THREE_ANSWER)
    top1 = _top_ids(scored1, CLOSEST_THREE_ANSWER)
    flipped = set(top0) != set(top1)
    if require_flip and not flipped:
        return None
    if (not require_flip) and flipped:
        return None

    # Prefer union of the two top-3 sets, then pad from after-nearest.
    # Re-apply type caps on the merged pool (each scored list is capped alone,
    # but union of before/after tops can still accumulate two Drawers etc.).
    ordered_ids: list[str] = []
    by_id = {o["objectId"]: o for _, o in scored1}
    for _, o in scored0:
        by_id.setdefault(o["objectId"], o)
    for oid in top0 + top1:
        if oid not in ordered_ids:
            ordered_ids.append(oid)
    for _, o in scored1:
        if o["objectId"] not in ordered_ids:
            ordered_ids.append(o["objectId"])
        if len(ordered_ids) >= CLOSEST_THREE_POOL * 2:
            break
    pool_objs_ordered = [by_id[i] for i in ordered_ids if i in by_id]
    pool_objs = take_with_anchor_type_caps(pool_objs_ordered, n=CLOSEST_THREE_POOL)
    if len(pool_objs) < CLOSEST_THREE_POOL:
        return None
    # Answer must be the after top-3, restricted to objects that remain in pool.
    pool_id_set = {o["objectId"] for o in pool_objs}
    answer_ids = [i for i in top1 if i in pool_id_set]
    if len(answer_ids) < CLOSEST_THREE_ANSWER:
        # Pad answer from after-nearest remaining in pool.
        for _, o in scored1:
            oid = o["objectId"]
            if oid in pool_id_set and oid not in answer_ids:
                answer_ids.append(oid)
            if len(answer_ids) >= CLOSEST_THREE_ANSWER:
                break
    if len(answer_ids) < CLOSEST_THREE_ANSWER:
        return None
    answer_ids = answer_ids[:CLOSEST_THREE_ANSWER]
    return {
        "pool_objs": pool_objs,
        "answer_ids": answer_ids,
        "before_ids": top0,
        "after_ids": top1,
        "flipped": flipped,
    }


def make_closest_three_flip_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
) -> Question:
    mover_p = _mover_phrase(event0, move)
    built = None
    meta_edit = dict(move.edit_meta or {})
    if meta_edit.get("before_ids") and meta_edit.get("after_ids"):
        # Rebuild pool preferring the planned flip mode from the edit finder.
        planned = meta_edit.get("answer_flipped_planned")
        require = True if planned is None else bool(planned)
        built = build_closest_three_flip_pool(
            event0, event1, move, require_flip=require
        )
        if built is None:
            built = build_closest_three_flip_pool(
                event0, event1, move, require_flip=not require
            )
    if built is None:
        want = _sample_prefer_flip(rng, None)
        built = build_closest_three_flip_pool(
            event0, event1, move, require_flip=want
        )
        if built is None:
            built = build_closest_three_flip_pool(
                event0, event1, move, require_flip=not want
            )
    if built is None:
        return Question(
            category="closest_three",
            question_id=_qid("closest_three"),
            text=f"After the move, which three objects are closest to {mover_p}?",
            answer=None,
            answer_type="set",
            gt_complete=False,
            gt_meta={"note": "no_flip_or_pool"},
        )

    pool_objs = built["pool_objs"]
    answer_ids = list(built["answer_ids"])
    pool_phrases = [_phrase(event0, o["objectId"], o["objectType"]) for o in pool_objs]
    id_to_phrase = {
        o["objectId"]: _phrase(event0, o["objectId"], o["objectType"]) for o in pool_objs
    }
    answer_phrases = [id_to_phrase[i] for i in answer_ids if i in id_to_phrase]
    display = list(pool_phrases)
    rng.shuffle(display)
    after = _find_obj(event1, move.object_id)
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
            "candidate_distances": [
                _dist(after["position"], o["position"]) for o in pool_objs
            ]
            if after
            else [],
            "answer_phrases": answer_phrases,
            "answer_ids": answer_ids,
            "answer_before": built["before_ids"],
            "answer_after": answer_ids,
            "answer_flipped": bool(built["flipped"]),
            "pool_size": len(pool_objs),
            "answer_size": CLOSEST_THREE_ANSWER,
            "metric": "3d_centroid_top3_flip_union6",
            "require_flip": bool(built["flipped"]),
        },
        gt_complete=True,
    )


def _count_within(
    mover_pos: dict[str, float],
    objs: list[dict[str, Any]],
    thresh: float,
) -> int:
    return sum(1 for o in objs if _dist(mover_pos, o["position"]) <= thresh)


def build_count_near(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    thresh: float,
    prefer_count: Optional[int] = None,
    require_count_change: bool = True,
) -> Optional[dict[str, Any]]:
    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)
    if not before or not after:
        return None
    scored1 = _neighbor_distances(event0, event1, move.object_id, after["position"])
    if len(scored1) < CLOSEST_THREE_POOL:
        return None
    pool_objs = [o for _, o in scored1[: CLOSEST_THREE_POOL * 2]]
    pool_objs = take_with_anchor_type_caps(pool_objs, n=CLOSEST_THREE_POOL)
    if len(pool_objs) < CLOSEST_THREE_POOL:
        return None
    # Same object identities in before (fallback to after pose if missing).
    before_objs = []
    for o in pool_objs:
        b = _find_obj(event0, o["objectId"]) or o
        before_objs.append(b)
    n0 = _count_within(before["position"], before_objs, thresh)
    n1 = _count_within(after["position"], pool_objs, thresh)
    if n1 not in COUNT_NEAR_ANSWERS:
        return None
    changed = int(n0) != int(n1)
    if require_count_change and not changed:
        return None
    if (not require_count_change) and changed:
        return None
    if prefer_count is not None and n1 != int(prefer_count):
        return None
    return {
        "pool_objs": pool_objs,
        "thresh": float(thresh),
        "count_before": int(n0),
        "count_after": int(n1),
        "flipped": changed,
    }


def make_count_near_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
    *,
    prefer_count: Optional[int] = None,
) -> Question:
    mover_p = _mover_phrase(event0, move)
    built = None
    thresh_order = list(V30_NEAR_THRESHOLDS_M)
    rng.shuffle(thresh_order)
    # Prefer meta stamped by the edit finder.
    meta_edit = dict(move.edit_meta or {})
    if meta_edit.get("count_near"):
        cn = meta_edit["count_near"]
        built = {
            "pool_objs": [
                _find_obj(event1, oid) or _find_obj(event0, oid)
                for oid in cn.get("candidate_ids") or []
            ],
            "thresh": float(cn["thresh"]),
            "count_before": int(cn["count_before"]),
            "count_after": int(cn["count_after"]),
            "flipped": bool(
                cn.get(
                    "flipped",
                    int(cn["count_before"]) != int(cn["count_after"]),
                )
            ),
        }
        built["pool_objs"] = [o for o in built["pool_objs"] if o is not None]
        built["pool_objs"] = take_with_anchor_type_caps(
            built["pool_objs"], n=CLOSEST_THREE_POOL
        )
        if len(built["pool_objs"]) < CLOSEST_THREE_POOL:
            built = None
    if built is None:
        want_change = _sample_prefer_flip(rng, None)
        for thresh in thresh_order:
            built = build_count_near(
                event0,
                event1,
                move,
                thresh=thresh,
                prefer_count=prefer_count,
                require_count_change=want_change,
            )
            if built is not None:
                break
        if built is None:
            for thresh in thresh_order:
                built = build_count_near(
                    event0,
                    event1,
                    move,
                    thresh=thresh,
                    prefer_count=None,
                    require_count_change=not want_change,
                )
                if built is not None:
                    break
                built = build_count_near(
                    event0,
                    event1,
                    move,
                    thresh=thresh,
                    prefer_count=None,
                    require_count_change=want_change,
                )
                if built is not None:
                    break

    if built is None or len(built["pool_objs"]) < CLOSEST_THREE_POOL:
        return Question(
            category="count_near",
            question_id=_qid("count_near"),
            text=f"How many of these objects are within 0.5m of {mover_p} after the move?",
            answer=None,
            answer_type="choice",
            gt_complete=False,
            gt_meta={"note": "no_count_near"},
        )

    pool_objs = built["pool_objs"][:CLOSEST_THREE_POOL]
    pool_phrases = [_phrase(event0, o["objectId"], o["objectType"]) for o in pool_objs]
    display = list(pool_phrases)
    rng.shuffle(display)
    thresh = float(built["thresh"])
    n1 = int(built["count_after"])
    # Shuffled for MCQ export; do NOT embed choice list in the question text.
    choices = [str(x) for x in COUNT_NEAR_ANSWERS]
    rng.shuffle(choices)
    text = (
        f"After the move, how many of these objects are within {thresh:.2f}m of "
        f"{mover_p}? Objects: " + "; ".join(display) + "."
    )
    return Question(
        category="count_near",
        question_id=_qid("count_near"),
        text=text,
        answer=str(n1),
        answer_type="choice",
        gt_meta={
            "mover_phrase": mover_p,
            "candidate_pool": pool_phrases,
            "candidate_ids": [o["objectId"] for o in pool_objs],
            "thresh_m": thresh,
            "count_before": int(built["count_before"]),
            "count_after": n1,
            "answer_before": str(built["count_before"]),
            "answer_after": str(n1),
            "answer_flipped": bool(built.get("flipped", int(built["count_before"]) != n1)),
            "choices": choices,
            "metric": "count_within_thresh_of_6",
        },
        gt_complete=True,
    )


def build_near_object(
    event0: Any,
    event1: Any,
    move: MoveAction,
    *,
    thresh: float,
    prefer_yes: Optional[bool] = None,
    prefer_flip: Optional[bool] = None,
    rng: Optional[random.Random] = None,
) -> Optional[dict[str, Any]]:
    rng = rng or random.Random(0)
    before = _find_obj(event0, move.object_id)
    after = _find_obj(event1, move.object_id)
    if not before or not after:
        return None
    others1 = _referable_others(event0, event1, move.object_id)
    if len(others1) < 2:
        return None

    def _within(ev_pos_obj: dict[str, Any], objs: list[dict[str, Any]]) -> set[str]:
        return {
            o["objectId"]
            for o in objs
            if _dist(ev_pos_obj["position"], o["position"]) <= thresh
        }

    # Use after-visible referable set; resolve before poses when present.
    objs_after = others1
    objs_before = []
    for o in objs_after:
        b = _find_obj(event0, o["objectId"])
        objs_before.append(b if b is not None else o)

    set0 = _within(before, objs_before)
    set1 = _within(after, objs_after)
    only0 = set0 - set1
    only1 = set1 - set0
    both = set0 & set1
    want_flip = _sample_prefer_flip(rng, prefer_flip)

    candidates: list[str] = []
    if want_flip:
        candidates = list(only0 | only1)
        if not candidates:
            return None
    else:
        # Unflipped: same membership before/after (intersection), or any
        # object when the within-sets are identical.
        candidates = list(both) if both else (list(set0) if set0 == set1 else [])
        if not candidates:
            return None
    rng.shuffle(candidates)

    for oid in candidates:
        o1 = _find_obj(event1, oid) or _find_obj(event0, oid)
        o0 = _find_obj(event0, oid) or o1
        if o1 is None or o0 is None:
            continue
        ans0 = _dist(before["position"], o0["position"]) <= thresh
        ans1 = _dist(after["position"], o1["position"]) <= thresh
        flipped = ans0 != ans1
        if want_flip and not flipped:
            continue
        if (not want_flip) and flipped:
            continue
        if prefer_yes is not None and bool(ans1) != bool(prefer_yes):
            continue
        return {
            "landmark_id": oid,
            "landmark_type": o1["objectType"],
            "thresh": float(thresh),
            "answer_before": bool(ans0),
            "answer_after": bool(ans1),
            "answer_flipped": flipped,
            "set0": sorted(set0),
            "set1": sorted(set1),
        }
    return None


def make_near_object_question(
    event0: Any,
    event1: Any,
    move: MoveAction,
    rng: random.Random,
    *,
    prefer_yes: Optional[bool] = None,
    prefer_flip: Optional[bool] = None,
) -> Question:
    mover_p = _mover_phrase(event0, move)
    built = None
    meta_edit = dict(move.edit_meta or {})
    if meta_edit.get("near_object"):
        built = dict(meta_edit["near_object"])
    if built is None:
        thresh_order = list(V30_NEAR_THRESHOLDS_M)
        rng.shuffle(thresh_order)
        for thresh in thresh_order:
            built = build_near_object(
                event0,
                event1,
                move,
                thresh=thresh,
                prefer_yes=prefer_yes,
                prefer_flip=prefer_flip,
                rng=rng,
            )
            if built is not None:
                break
            built = build_near_object(
                event0,
                event1,
                move,
                thresh=thresh,
                prefer_yes=None,
                prefer_flip=prefer_flip,
                rng=rng,
            )
            if built is not None:
                break

    if built is None:
        return Question(
            category="near_object",
            question_id=_qid("near_object"),
            text=f"After the move, is {mover_p} within 0.5m of another object?",
            answer=None,
            answer_type="yes_no",
            gt_complete=False,
            gt_meta={"note": "no_near_object"},
        )

    land = _find_obj(event0, built["landmark_id"]) or _find_obj(
        event1, built["landmark_id"]
    )
    land_p = (
        _phrase(event0, built["landmark_id"], built.get("landmark_type") or "object")
        if land
        else f"the {str(built.get('landmark_type') or 'object').lower()}"
    )
    thresh = float(built["thresh"])
    ans = bool(built["answer_after"])
    text = (
        f"After the move, is {mover_p} within {thresh:.2f}m of {land_p}?"
    )
    return Question(
        category="near_object",
        question_id=_qid("near_object"),
        text=text,
        answer=ans,
        answer_type="yes_no",
        gt_meta={
            "mover_phrase": mover_p,
            "landmark_id": built["landmark_id"],
            "landmark_phrase": land_p,
            "landmark_type": built.get("landmark_type"),
            "thresh_m": thresh,
            "answer_before": bool(built["answer_before"]),
            "answer_after": ans,
            "answer_flipped": bool(built["answer_flipped"]),
            "set0": list(built.get("set0") or []),
            "set1": list(built.get("set1") or []),
            "metric": "within_thresh_of_landmark",
        },
        gt_complete=True,
    )


def _generic_proposals(
    event: Any,
    cam: CameraPose,
    category: str,
    *,
    dist_min: float,
    dist_max: float,
    angular_size_min: float,
    som: bool,
):
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    return filter_proposals_for_category(
        event,
        basis,
        category,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    ), basis


def find_closest_three_flip_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    max_tries: int = 40,
    prefer_flip: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    som: bool = False,
    used_movers_in_room: Optional[set[str]] = None,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    rng = rng or random.Random(0)
    want_flip = _sample_prefer_flip(rng, prefer_flip)
    proposals, _basis = _generic_proposals(
        event,
        cam,
        "closest_three",
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    )
    if not proposals:
        return None
    order = order_equal_by_mover(
        proposals,
        object_id_fn=lambda p: p.object_id,
        rng=rng,
        penalize_ids=used_movers_in_room,
    )

    def _try(require_flip: bool) -> Optional[tuple[MoveAction, Any, Any, str]]:
        for proposal in order[:max_tries]:
            result = try_edit(controller, cam, proposal, settle_gravity=False)
            if result is None:
                continue
            move, event0, event1 = result
            built = build_closest_three_flip_pool(
                event0, event1, move, require_flip=require_flip
            )
            if built is None:
                continue
            move.edit_meta = {
                **dict(move.edit_meta or {}),
                "closest_three_flip": bool(built["flipped"]),
                "before_ids": built["before_ids"],
                "after_ids": built["after_ids"],
                "answer_flipped_planned": bool(built["flipped"]),
            }
            tag = "flip" if built["flipped"] else "same"
            return move, event0, event1, f"category:closest_three+{tag}"
        return None

    hit = _try(require_flip=want_flip)
    if hit is not None:
        return hit
    # Soft target: fall back to the other flip mode rather than fail the category.
    return _try(require_flip=not want_flip)


def find_count_near_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    max_tries: int = 40,
    prefer_count: Optional[int] = None,
    prefer_flip: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    som: bool = False,
    used_movers_in_room: Optional[set[str]] = None,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    rng = rng or random.Random(0)
    want_change = _sample_prefer_flip(rng, prefer_flip)
    proposals, _ = _generic_proposals(
        event,
        cam,
        "count_near",
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    )
    if not proposals:
        return None
    order = order_equal_by_mover(
        proposals,
        object_id_fn=lambda p: p.object_id,
        rng=rng,
        penalize_ids=used_movers_in_room,
    )
    thresh_order = list(V30_NEAR_THRESHOLDS_M)
    rng.shuffle(thresh_order)

    def _try(require_change: bool) -> Optional[tuple[MoveAction, Any, Any, str]]:
        attempts = 0
        for proposal in order:
            if attempts >= max_tries:
                break
            result = try_edit(controller, cam, proposal, settle_gravity=False)
            attempts += 1
            if result is None:
                continue
            move, event0, event1 = result
            built = None
            for thresh in thresh_order:
                built = build_count_near(
                    event0,
                    event1,
                    move,
                    thresh=thresh,
                    prefer_count=prefer_count,
                    require_count_change=require_change,
                )
                if built is not None:
                    break
            if built is None and prefer_count is not None:
                for thresh in thresh_order:
                    built = build_count_near(
                        event0,
                        event1,
                        move,
                        thresh=thresh,
                        prefer_count=None,
                        require_count_change=require_change,
                    )
                    if built is not None:
                        break
            if built is None:
                continue
            move.edit_meta = {
                **dict(move.edit_meta or {}),
                "count_near": {
                    "candidate_ids": [o["objectId"] for o in built["pool_objs"]],
                    "thresh": built["thresh"],
                    "count_before": built["count_before"],
                    "count_after": built["count_after"],
                    "flipped": built["flipped"],
                },
                "answer_flipped_planned": bool(built["flipped"]),
            }
            tag = "flip_count" if built["flipped"] else "same_count"
            return move, event0, event1, f"category:count_near+{tag}"
        return None

    hit = _try(require_change=want_change)
    if hit is not None:
        return hit
    return _try(require_change=not want_change)


def find_near_object_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    max_attempts: int = 8,
    prefer_yes: Optional[bool] = None,
    prefer_flip: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    som: bool = False,
    used_movers_in_room: Optional[set[str]] = None,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Iterate ≤8 successful move×distance tries for yes/no proximity."""
    rng = rng or random.Random(0)
    want_flip = _sample_prefer_flip(rng, prefer_flip)
    proposals, _ = _generic_proposals(
        event,
        cam,
        "near_object",
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    )
    if not proposals:
        return None
    order = order_equal_by_mover(
        proposals,
        object_id_fn=lambda p: p.object_id,
        rng=rng,
        penalize_ids=used_movers_in_room,
    )
    thresh_order = list(V30_NEAR_THRESHOLDS_M)
    rng.shuffle(thresh_order)
    success_attempts = 0
    proposal_budget = max(40, max_attempts * 8)

    def _search(prefer_f: Optional[bool], prefer_y: Optional[bool]) -> Optional[
        tuple[MoveAction, Any, Any, str]
    ]:
        nonlocal success_attempts
        for proposal in order[:proposal_budget]:
            if success_attempts >= max_attempts:
                break
            result = try_edit(controller, cam, proposal, settle_gravity=False)
            if result is None:
                continue
            success_attempts += 1
            move, event0, event1 = result
            for thresh in thresh_order:
                built = build_near_object(
                    event0,
                    event1,
                    move,
                    thresh=thresh,
                    prefer_yes=prefer_y,
                    prefer_flip=prefer_f,
                    rng=rng,
                )
                if built is None:
                    continue
                move.edit_meta = {
                    **dict(move.edit_meta or {}),
                    "near_object": built,
                    "answer_flipped_planned": bool(built["answer_flipped"]),
                }
                return move, event0, event1, "category:near_object"
        return None

    # Prefer yes/no + flip target; then relax flip; then relax yes.
    for pf, py in (
        (want_flip, prefer_yes),
        (not want_flip, prefer_yes),
        (want_flip, None),
        (not want_flip, None),
    ):
        # Reset attempt budget across relaxations so we can keep searching.
        success_attempts = 0
        hit = _search(pf, py)
        if hit is not None:
            return hit
    return None
