"""End-to-end builder for one CIT eval sample (iTHOR or ProcTHOR room).

Modification 4+
---------------
- Agent-eye camera(s); ProcTHOR v8 selects ``y`` cameras apartment-wide.
- For each QA category: pick a category-specific mover, apply one move, ask
  exactly one question of that type → one view per category.

ProcTHOR
--------
AI2-THOR loads the **full apartment**. Default ``procthor_camera_mode=
"apartment"``: propose 2–3+ views per room (FPS in polygon, 8 yaws, coin-flip
horizon, camera-xyz separation), then globally select with hard room coverage
+ diversity; emit one sample per camera and fall back across the apartment
camera pool when a category fails on the primary — ideally
``y × |categories|`` questions. Legacy ``per_room`` mode keeps one sample per
room.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from cit_pipeline.camera_search import (
    find_agent_eye_cameras,
    find_apartment_cameras,
    camera_basis,
)
from cit_pipeline.category_edits import find_edit_for_category
from cit_pipeline.config import (
    CONTACT_FAST_SINGLE_CAMERA,
    CONTACT_FEAS_FIRST,
    CONTACT_ROOM_WALL_MAX_M,
    CONTACT_ROOM_WALL_PREFILTER,
    CONTACT_FAST_MAX_CANDIDATES,
    CONTACT_TARGET_NO_FRAC,
    FEASIBILITY_TARGET_NO_FRAC,
    MODIFICATION_TAG,
    OCCLUDE_TARGET_NO_FRAC,
    QA_CATEGORIES,
    PipelineConfig,
    use_fast_mode,
    use_snapshot_restore,
)
from cit_pipeline.edit_search import find_edit_for_view
from cit_pipeline.move_text import format_move_displacement
from cit_pipeline.qa import make_question_for_category, sample_two_questions, word_questions_with_vlm
from cit_pipeline.referring import referring_phrase
from cit_pipeline.schemas import CameraPose, EvalSample, MoveAction, Question, ViewSample
from cit_pipeline.scene_integrity import scene_collateral_ok
from cit_pipeline.scene_snapshot import (
    SceneSnapshot,
    capture_scene_snapshot,
    restore_scene_snapshot,
)
from cit_pipeline.som import attach_som_to_question, save_som_rgb
from cit_pipeline.som_hooks import apply_som_hooks
from cit_pipeline.thor_env import (
    merge_seg_label_dicts,
    reset_scene,
    save_aux_maps_for_rgb,
    save_rgb,
    start_controller,
    teleport_full,
)
from cit_pipeline.gt_object_flow import attach_gt_flow_to_view
from cit_pipeline.vlm_client import VlmClient


def _move_text_with_camera(move: MoveAction, cam: CameraPose, phrase: str) -> str:
    basis = camera_basis(cam.rotation_y, cam.horizon)
    return (
        f"move {phrase} "
        f"{format_move_displacement(move.direction, move.meters, move.delta_world, basis=basis)}"
    )


def _stamp_move_text_som(move: MoveAction) -> None:
    """Attach SoM-style motion (``the bowl (1)``) on ``edit_meta['text_som']``."""
    import re

    from cit_pipeline.som import som_marked_phrase

    natural = str(move.text or "").strip()
    som_p = som_marked_phrase(move.object_type, "1")
    m = re.match(r"(?is)^move\s+(.+?)\s+(\d+\.\d+m(?:\s+.*)?)$", natural)
    if not m:
        m = re.match(r"(?is)^move\s+(.+?)\s+(\d+m(?:\s+.*)?)$", natural)
    text_som = f"move {som_p} {m.group(2).strip()}" if m else f"move {som_p}"
    move.edit_meta = {**(move.edit_meta or {}), "text_som": text_som}


def _attach_som_for_view(
    *,
    questions: list[Question],
    move: MoveAction,
    event0: Any,
    event1: Any,
    sample_dir: Path,
    view_id: str,
    enable_som: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Write SoM PNGs and fill ``text_som`` / mark metadata. Returns path strings."""
    if not enable_som or not questions:
        return None, None
    apply_som_hooks()
    from cit_pipeline.config import OCCLUDE_PROMPT_VARIANT, SOM_PIN_OCCLUDEE_TO_RGB0
    from cit_pipeline.som import (
        SomMark,
        build_mark_specs_for_question,
        marks_to_meta,
        resolve_marks_for_event,
    )

    for qq in questions:
        if "mover_phrase" not in (qq.gt_meta or {}):
            qq.gt_meta = dict(qq.gt_meta or {})
            ref = qq.gt_meta.get("mover_referring") or {}
            if isinstance(ref, dict) and ref.get("phrase"):
                qq.gt_meta["mover_phrase"] = ref["phrase"]
        attach_som_to_question(qq, move, event0, event1)
        if str(qq.category) == "occlusion":
            _rewrite_occlusion_text_som(qq, move, variant=OCCLUDE_PROMPT_VARIANT)
        elif str(qq.category) in ("closest_three", "count_near", "near_object"):
            _rewrite_v30_text_som(qq, move)

    # Draw marks per frame (occlusion: occludee on both rgb0 and rgb1).
    specs0: list[tuple[str, str, str]] = []
    specs1: list[tuple[str, str, str]] = []
    seen0: set[str] = set()
    seen1: set[str] = set()
    for qq in questions:
        for mid, oid, role in build_mark_specs_for_question(qq, move, frame="rgb0"):
            if oid in seen0:
                continue
            seen0.add(oid)
            specs0.append((mid, oid, role))
        for mid, oid, role in build_mark_specs_for_question(qq, move, frame="rgb1"):
            if oid in seen1:
                continue
            seen1.add(oid)
            specs1.append((mid, oid, role))
    frame = getattr(event1, "frame", None)
    h, w = (int(frame.shape[0]), int(frame.shape[1])) if frame is not None else (600, 800)
    marks0 = resolve_marks_for_event(event0, specs0, width=w, height=h)
    marks1 = resolve_marks_for_event(event1, specs1, width=w, height=h)

    # Pin occludee (2) to its rgb0 badge location so the mark does not jump to a
    # tiny remaining fragment (e.g. near a lamp) after the occluder covers it.
    if SOM_PIN_OCCLUDEE_TO_RGB0:
        marks1 = _pin_occludee_marks_to_rgb0(marks0, marks1)
        for qq in questions:
            if str(qq.category) != "occlusion":
                continue
            qq.gt_meta = dict(qq.gt_meta or {})
            qq.gt_meta["occludee_mark_on_after"] = False
            qq.gt_meta["occludee_mark_pinned_rgb0"] = True
            qq.gt_meta["som_marks"] = marks_to_meta(marks1 or marks0)
            qq.gt_meta["som_marks_rgb0"] = marks_to_meta(marks0)
            qq.gt_meta["som_marks_rgb1"] = marks_to_meta(marks1)

    rgb0_som = sample_dir / f"{view_id}_rgb0_som.png"
    rgb1_som = sample_dir / f"{view_id}_rgb1_som.png"
    save_som_rgb(event0, marks0, rgb0_som)
    save_som_rgb(event1, marks1, rgb1_som)
    return str(rgb0_som), str(rgb1_som)


def _pin_occludee_marks_to_rgb0(marks0: list, marks1: list) -> list:
    from cit_pipeline.som import SomMark

    by_id0 = {str(m.id): m for m in (marks0 or [])}
    out = []
    for m in marks1 or []:
        if str(getattr(m, "role", "")) == "occludee" and str(m.id) in by_id0:
            src = by_id0[str(m.id)]
            out.append(
                SomMark(
                    id=m.id,
                    object_id=m.object_id,
                    label=m.label,
                    uv=tuple(src.uv),
                    role=m.role,
                    uv_centroid=tuple(src.uv_centroid)
                    if src.uv_centroid is not None
                    else None,
                )
            )
        else:
            out.append(m)
    return out


def _rewrite_v30_text_som(q: Question, move: MoveAction) -> None:
    """SoM question text: mark ids + simple type names (no ordinals / no MCQ choices)."""
    from cit_pipeline.v30_categories import _simple_type_phrase

    meta = dict(q.gt_meta or {})
    mt = _simple_type_phrase(str(move.object_type or "object"))
    cat = str(q.category)
    if cat == "near_object":
        lt = _simple_type_phrase(str(meta.get("landmark_type") or "object"))
        thresh = float(meta.get("thresh_m") or 0.5)
        q.text_som = (
            f"After the move, is {mt} (1) within {thresh:.2f}m of {lt} (2)?"
        )
        return
    if cat == "count_near":
        thresh = float(meta.get("thresh_m") or 0.5)
        ids = list(meta.get("candidate_ids") or [])
        types = []
        # Prefer types from som mark specs / pool phrases fallback.
        pool = list(meta.get("candidate_pool") or [])
        letters = "ABCDEF"
        parts = []
        for i, oid in enumerate(ids[:6]):
            # Extract type from objectId prefix when possible.
            typ = str(oid).split("|", 1)[0] if oid else "object"
            parts.append(f"({letters[i]}) {_simple_type_phrase(typ)}")
        if not parts and pool:
            for i, phrase in enumerate(pool[:6]):
                parts.append(f"({letters[i]}) {phrase}")
        obj_list = "; ".join(parts)
        q.text_som = (
            f"After the move, how many of the following objects are within "
            f"{thresh:.2f}m of {mt} (1)? Objects: {obj_list}."
        )
        return
    if cat == "closest_three":
        ids = list(meta.get("candidate_ids") or [])
        letters = "ABCDEF"
        parts = []
        for i, oid in enumerate(ids[:6]):
            typ = str(oid).split("|", 1)[0] if oid else "object"
            parts.append(f"({letters[i]}) {_simple_type_phrase(typ)}")
        if parts:
            q.text_som = (
                f"After the move, which three objects are closest to {mt} (1)? "
                f"Choose three from: " + "; ".join(parts) + "."
            )


def _rewrite_occlusion_text_som(q: Question, move: MoveAction, *, variant: str) -> None:
    meta = dict(q.gt_meta or {})
    ot = str(meta.get("occludee_type") or "object").lower()
    mt = str(move.object_type or "object").lower()
    if variant == "baseline":
        thr = int(float(meta.get("threshold") or 0.3) * 100)
        q.text_som = (
            f"In the after image, is the {ot} (2) substantially occluded by "
            f"the {mt} (1) (at least {thr}% of its visible pixels)?"
        )
    elif variant == "visible_or_occluded":
        q.text_som = (
            f"In the after image, is the {ot} (2) visible, or is it occluded by "
            f"the {mt} (1)?"
        )
    elif variant == "any_part_hidden":
        q.text_som = (
            f"In the after image, is any part of the {ot} (2) hidden behind "
            f"the {mt} (1)?"
        )
    elif variant == "blocked_from_view":
        q.text_som = (
            f"In the after image, is the {ot} (2) blocked from view by the {mt} (1)?"
        )
    else:
        q.text_som = (
            f"In the after image, is the {ot} (2) occluded by the {mt} (1) so that "
            f"a part of it is invisible, or is it totally visible?"
        )
    meta["occlusion_prompt_variant"] = variant
    q.gt_meta = meta


def _save_view_aux_maps(
    *,
    event0: Any,
    event1: Any,
    sample_dir: Path,
    view_id: str,
    enable: bool,
) -> dict[str, Any]:
    """Save seg/depth for rgb0+rgb1; return ViewSample aux kwargs."""
    if not enable:
        return {}
    try:
        aux0 = save_aux_maps_for_rgb(event0, sample_dir, f"{view_id}_rgb0")
        aux1 = save_aux_maps_for_rgb(event1, sample_dir, f"{view_id}_rgb1")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] aux maps failed for {view_id}: {exc}", flush=True)
        return {}
    return {
        "seg0_path": aux0.get("seg_path"),
        "seg1_path": aux1.get("seg_path"),
        "seg0_ids_path": aux0.get("seg_ids_path"),
        "seg1_ids_path": aux1.get("seg_ids_path"),
        "depth0_path": aux0.get("depth_path"),
        "depth1_path": aux1.get("depth_path"),
        "seg_id_to_label": merge_seg_label_dicts(
            aux0.get("seg_id_to_label"),
            aux1.get("seg_id_to_label"),
        ),
    }


def _prefer_yes_for_balance(
    counts: list[int],
    rng: random.Random,
    *,
    target_no_frac: float = 0.50,
) -> bool:
    """Return whether to prefer a yes answer given running [yes, no] counts.

    When counts are tied (including 0/0), sample with ``P(prefer yes) =
    1 - target_no_frac`` so multi-process / single-Q-per-scene builds still
    hit ~50% collision for feasibility (Modification 9).
    """
    yes_n, no_n = int(counts[0]), int(counts[1])
    if yes_n < no_n:
        return True
    if yes_n > no_n:
        return False
    return rng.random() >= float(target_no_frac)


def _maybe_vlm(cfg: PipelineConfig) -> Optional[VlmClient]:
    """Connect to the VLM.

    When ``use_vlm_ranking`` is False, or ``vlm_connect_retries<=0`` with
    ``vlm_optional``, returns None so the pipeline keeps GT template questions.
    Otherwise connect failure is fatal (no silent skip).
    """
    if not cfg.use_vlm_ranking:
        print(
            "[info] VLM disabled (use_vlm_ranking=False); writing GT template questions",
            flush=True,
        )
        return None
    optional = bool(getattr(cfg, "vlm_optional", False))
    n_tries = int(getattr(cfg, "vlm_connect_retries", 8))
    if n_tries <= 0:
        msg = (
            f"[warn] VLM connect retries=0 ({cfg.vlm_base_url}); "
            "writing GT template questions"
        )
        if optional:
            print(msg, flush=True)
            return None
        raise RuntimeError(
            "vlm_connect_retries=0 but vlm_optional=False "
            "(pass --no-vlm / --vlm-optional, or set retries>0)"
        )
    last_exc: Optional[BaseException] = None
    delay_s = float(getattr(cfg, "vlm_connect_retry_delay_s", 5.0))
    for attempt in range(1, n_tries + 1):
        try:
            vlm = VlmClient(base_url=cfg.vlm_base_url, model=cfg.vlm_model)
            vlm.chat_text("Reply with OK.", max_tokens=8)
            return vlm
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(
                f"[warn] VLM connect attempt {attempt}/{n_tries} failed ({exc})",
                flush=True,
            )
            if attempt < n_tries:
                time.sleep(delay_s)
    if optional:
        print(
            f"[warn] VLM unavailable after {n_tries} tries ({cfg.vlm_base_url}): "
            f"{last_exc}; writing GT template questions",
            flush=True,
        )
        return None
    raise RuntimeError(
        f"VLM required but unavailable after {n_tries} tries "
        f"({cfg.vlm_base_url}): {last_exc}"
    )


def _require_vlm_cot(data: dict[str, Any], *, need_pretended: bool) -> None:
    """Raise if any complete question is missing warm (and pretended) CoT."""
    missing: list[str] = []
    for view in data.get("views") or []:
        vid = view.get("view_id") or "?"
        move = view.get("move") or {}
        for q in view.get("questions") or []:
            if not q.get("gt_complete", True):
                continue
            qid = q.get("question_id") or q.get("category") or "?"
            cot = q.get("cot") or {}
            if not cot.get("warm"):
                missing.append(f"{vid}/{qid}:warm ({cot.get('warm_error')})")
            if need_pretended and not cot.get("pretended"):
                missing.append(
                    f"{vid}/{qid}:pretended ({cot.get('pretended_error')})"
                )
    if missing:
        raise RuntimeError(
            "VLM CoT incomplete (refusing to keep sample): " + "; ".join(missing[:12])
        )


def _mark_template_wording(
    questions: list[Any],
    *,
    skip_reason: str = "vlm_bypassed",
) -> None:
    """Stamp questions kept as GT drafts when VLM wording is skipped."""
    for qq in questions:
        meta = dict(getattr(qq, "gt_meta", None) or {})
        if meta.get("vlm_worded") is True:
            continue
        meta["vlm_worded"] = False
        meta["vlm_skip_reason"] = skip_reason
        qq.gt_meta = meta


def _attach_cot_if_requested(
    cfg: PipelineConfig,
    sample: EvalSample,
    sample_dir: Path,
    vlm: Optional[VlmClient],
) -> EvalSample:
    if not cfg.attach_cot:
        return sample
    from cit_pipeline.cot import process_sample_dict

    if vlm is None:
        # Template / --no-vlm / vlm_optional mode: keep sample without CoT polish.
        print(
            "[warn] attach_cot requested but no VLM; skipping CoT",
            flush=True,
        )
        return sample

    polish_pretended = bool(cfg.attach_pretended_cot and vlm is not None)
    data = sample.to_dict()
    data = process_sample_dict(
        data,
        sample_dir,
        vlm=vlm,
        polish=bool(vlm is not None),
        polish_pretended=polish_pretended,
    )
    if vlm is not None:
        _require_vlm_cot(data, need_pretended=polish_pretended)
    (sample_dir / "sample.json").write_text(json.dumps(data, indent=2))
    # Re-hydrate is unnecessary for callers that only need the written JSON;
    # keep the in-memory sample and mirror CoT flag in meta.
    sample.meta = dict(data.get("meta") or sample.meta)
    return sample


def _cfg_bool(cfg: PipelineConfig, name: str, default: bool) -> bool:
    v = getattr(cfg, name, None)
    return default if v is None else bool(v)


def _cfg_float(cfg: PipelineConfig, name: str, default: float) -> float:
    v = getattr(cfg, name, None)
    return default if v is None else float(v)


def _cfg_int(cfg: PipelineConfig, name: str, default: int) -> int:
    v = getattr(cfg, name, None)
    return default if v is None else int(v)


def _ordered_categories(categories: tuple[str, ...], cfg: PipelineConfig) -> tuple[str, ...]:
    """When enabled, run feasibility before contact_wall (v10 fast path)."""
    if not _cfg_bool(cfg, "contact_feas_first", CONTACT_FEAS_FIRST):
        return categories
    cats = list(categories)
    if "feasibility" in cats and "contact_wall" in cats:
        cats.remove("feasibility")
        insert_at = cats.index("contact_wall")
        cats.insert(insert_at, "feasibility")
    return tuple(cats)


def build_category_driven_views(
    *,
    controller: Any,
    scene_spec: Any,
    cam: CameraPose,
    sample_dir: Path,
    cfg: PipelineConfig,
    rng: random.Random,
    vlm: Optional[VlmClient],
    view_id_prefix: str = "view",
    cameras: Optional[list[CameraPose]] = None,
    room_id: Optional[str] = None,
    used_movers_in_room: Optional[set[str]] = None,
) -> tuple[list[ViewSample], list[str], list[int], list[int]]:
    """Shared Mod-4 category loop used by iTHOR and ProcTHOR rooms.

    When ``cameras`` is provided, each category tries poses in order until an
    edit succeeds (helps sparse categories like affordance / contact_wall).

    ``used_movers_in_room`` (mutated): object ids already moved in this room;
    edit search tries unused movers first for diversity.

    When ``use_snapshot_restore(cfg)`` is True: one clean load + pose snapshot;
    between categories rewind changed objects to exact snapshot poses (full
    ``reset`` only on restore/verify failure). Kill switch:
    ``PipelineConfig.snapshot_restore=False``.
    """
    views: list[ViewSample] = []
    categories_done: list[str] = []
    occlude_balance = [0, 0]
    feasible_balance = [0, 0]
    contact_balance = [0, 0]
    left_right_balance = [0, 0]
    near_object_balance = [0, 0]  # [yes, no]
    used_movers = used_movers_in_room if used_movers_in_room is not None else set()
    _ = room_id  # documented for callers; penalty is keyed via used_movers set
    occ_no = float(
        OCCLUDE_TARGET_NO_FRAC
        if getattr(cfg, "occlude_target_no_frac", None) is None
        else cfg.occlude_target_no_frac
    )
    contact_no = float(
        CONTACT_TARGET_NO_FRAC
        if getattr(cfg, "contact_target_no_frac", None) is None
        else cfg.contact_target_no_frac
    )
    feas_no = float(
        FEASIBILITY_TARGET_NO_FRAC
        if getattr(cfg, "feasibility_target_no_frac", None) is None
        else cfg.feasibility_target_no_frac
    )
    lr_no = float(
        0.50
        if getattr(cfg, "left_right_target_no_frac", None) is None
        else cfg.left_right_target_no_frac
    )
    categories = _ordered_categories(
        tuple(cfg.categories) if cfg.categories else QA_CATEGORIES, cfg
    )
    cam_list = list(cameras) if cameras else [cam]
    contact_single_cam = _cfg_bool(cfg, "contact_fast_single_camera", CONTACT_FAST_SINGLE_CAMERA)
    snap_on = use_snapshot_restore(cfg)
    snapshot: Optional[SceneSnapshot] = None
    n_full_resets = 0
    n_snap_restores = 0
    n_snap_fallbacks = 0

    def _full_reset_and_snapshot(*, reason: str) -> None:
        nonlocal snapshot, n_full_resets
        reset_scene(controller, scene_spec)
        n_full_resets += 1
        if not snap_on:
            snapshot = None
            return
        # Capture from a known camera so agent pose is well-defined.
        ev = teleport_full(controller, **cam_list[0].teleport_kwargs())
        if not ev.metadata.get("lastActionSuccess"):
            ev = controller.last_event
        snapshot = capture_scene_snapshot(ev)
        print(
            f"  [snapshot] full reset ({reason}); "
            f"tracked={len(snapshot.objects)} objects",
            flush=True,
        )

    def _rewind(*, reason: str, event: Any = None) -> bool:
        """Restore snapshot poses; fall back to full reset on failure."""
        nonlocal snapshot, n_snap_restores, n_snap_fallbacks
        if not snap_on or snapshot is None:
            _full_reset_and_snapshot(reason=reason)
            return False
        ok, report = restore_scene_snapshot(
            controller, snapshot, event=event, restore_agent=True
        )
        if ok:
            n_snap_restores += 1
            print(
                f"  [snapshot] restore ok ({reason}): "
                f"changed={report.get('n_changed')} "
                f"restored={report.get('n_restored')}",
                flush=True,
            )
            return True
        n_snap_fallbacks += 1
        print(
            f"  [snapshot] restore FAILED ({reason}): "
            f"{report.get('reason')} → full reset",
            flush=True,
        )
        _full_reset_and_snapshot(reason=f"fallback_after_{reason}")
        return False

    # Initial clean state (one house load for the whole category loop when snap_on).
    _full_reset_and_snapshot(reason="category_loop_start")

    for cat in categories:
        result = None
        used_cam = cam
        q = None
        # Without snapshot restore: legacy per-category full reset.
        if not snap_on:
            reset_scene(controller, scene_spec)
            n_full_resets += 1
        # Sample polarity once per category — not per camera. Re-rolling inside
        # the camera loop lets hard Yes intents (full wall slides / collisions)
        # fail over into easy No intents and destroys the target mix (v10).
        prefer_occ_yes = _prefer_yes_for_balance(
            occlude_balance, rng, target_no_frac=occ_no
        )
        prefer_contact = _prefer_yes_for_balance(
            contact_balance, rng, target_no_frac=contact_no
        )
        prefer_lr_yes = _prefer_yes_for_balance(
            left_right_balance, rng, target_no_frac=lr_no
        )
        prefer_near_yes = _prefer_yes_for_balance(
            near_object_balance, rng, target_no_frac=0.50
        )
        from cit_pipeline.v30_categories import V30_TARGET_FLIP_FRAC

        prefer_v30_flip = rng.random() < float(V30_TARGET_FLIP_FRAC)
        prefer_feas = _prefer_yes_for_balance(
            feasible_balance, rng, target_no_frac=feas_no
        )
        fast_edit = use_fast_mode(cfg)
        if cat == "contact_wall" and contact_single_cam:
            cat_cam_list = [cam]
        else:
            cat_cam_list = cam_list
        for trial_cam in cat_cam_list:
            event = teleport_full(controller, **trial_cam.teleport_kwargs())
            if not event.metadata["lastActionSuccess"]:
                continue
            result = find_edit_for_category(
                controller,
                trial_cam,
                event,
                cat,
                vlm=vlm,
                angular_size_min=cfg.angular_size_min,
                dist_min=cfg.dist_min,
                dist_max=cfg.dist_max,
                prefer_occ_yes=prefer_occ_yes,
                prefer_contact=prefer_contact,
                rng=rng,
                som=bool(getattr(cfg, "enable_som", True)),
                fast_find_edit=fast_edit,
                fast_contact_max_tries=getattr(cfg, "fast_contact_max_tries", None),
                fast_occlusion_max_tries=getattr(cfg, "fast_occlusion_max_tries", None),
                wall_away_prob=getattr(cfg, "wall_away_prob", None),
                fast_wall_select_near_m=getattr(cfg, "fast_wall_select_near_m", None),
                contact_always_toward=getattr(cfg, "contact_always_toward", None),
                contact_half_motion_no_frac=getattr(
                    cfg, "contact_half_motion_no_frac", None
                ),
                procthor_rooms=getattr(cfg, "procthor_house_rooms", None),
                camera_room_id=room_id,
                contact_room_wall_prefilter=_cfg_bool(
                    cfg, "contact_room_wall_prefilter", CONTACT_ROOM_WALL_PREFILTER
                ),
                contact_room_wall_max_m=_cfg_float(
                    cfg, "contact_room_wall_max_m", CONTACT_ROOM_WALL_MAX_M
                ),
                contact_fast_max_candidates=_cfg_int(
                    cfg, "contact_fast_max_candidates", CONTACT_FAST_MAX_CANDIDATES
                ),
                prefer_near_yes=prefer_near_yes if cat == "near_object" else None,
                prefer_v30_flip=(
                    prefer_v30_flip
                    if cat in ("closest_three", "count_near", "near_object")
                    else None
                ),
                used_movers_in_room=used_movers,
            )
            if result is None:
                # Restore scene for the next camera.
                if snap_on:
                    _rewind(reason=f"no_edit_{cat}", event=controller.last_event)
                else:
                    reset_scene(controller, scene_spec)
                    n_full_resets += 1
                continue

            move, event0, event1, rank_notes = result
            used_cam = trial_cam

            # Reject edits that wrecked collateral furniture (force tips, etc.).
            ok_scene, scene_report = scene_collateral_ok(
                event0, event1, move.object_id
            )
            if not ok_scene:
                print(
                    f"  [warn] scene_collateral bad for {cat}: "
                    f"{scene_report} → rewind",
                    flush=True,
                )
                if snap_on:
                    _rewind(reason=f"collateral_{cat}", event=event1)
                else:
                    reset_scene(controller, scene_spec)
                    n_full_resets += 1
                result = None
                q = None
                continue

            prefer_occ = prefer_occ_yes
            q = make_question_for_category(
                cat,
                event0,
                event1,
                move,
                rng=rng,
                controller=controller,
                cam=used_cam,
                prefer_occ=prefer_occ if cat == "occlusion" else None,
                prefer_feas=prefer_feas if cat == "feasibility" else None,
                prefer_lr=prefer_lr_yes if cat == "left_right_landmark" else None,
                prefer_near=prefer_near_yes if cat == "near_object" else None,
                prefer_v30_flip=(
                    prefer_v30_flip
                    if cat in ("closest_three", "count_near", "near_object")
                    else None
                ),
                fast_mode=fast_edit,
                fast_feasibility_max_attempts=getattr(
                    cfg, "fast_feasibility_max_attempts", None
                ),
                dataset=str(getattr(cfg, "dataset", "ithor") or "ithor"),
            )
            intentional_nonflip = bool(
                (move.edit_meta or {}).get("answer_flipped_planned") is False
            ) or bool((q.gt_meta or {}).get("closer_random_fallback"))
            if q.gt_complete:
                if (
                    cat == "closer_to"
                    and (q.gt_meta or {}).get("answer_flipped") is False
                    and not intentional_nonflip
                    and not fast_edit
                ):
                    if snap_on:
                        _rewind(reason="closer_nonflip_reject", event=event1)
                    else:
                        reset_scene(controller, scene_spec)
                        n_full_resets += 1
                    result = None
                    q = None
                    continue
                if (
                    cat == "left_right_landmark"
                    and (q.gt_meta or {}).get("answer_flipped") is False
                    and not intentional_nonflip
                ):
                    if snap_on:
                        _rewind(reason="left_right_nonflip_reject", event=event1)
                    else:
                        reset_scene(controller, scene_spec)
                        n_full_resets += 1
                    result = None
                    q = None
                    continue
            else:
                q.gt_meta = {
                    **dict(q.gt_meta or {}),
                    "needs_manual_annotation": True,
                    "answer_status": "no_answer",
                }
                if q.answer is not None:
                    q.answer = None
            break

        if result is None or q is None:
            print(f"[warn] no edit for category={cat}")
            if snap_on:
                _rewind(reason=f"skip_{cat}", event=controller.last_event)
            continue

        move, event0, event1, rank_notes = result
        used_movers.add(str(move.object_id))
        view_id = f"{view_id_prefix}_{cat}"
        rgb0_path = sample_dir / f"{view_id}_rgb0.png"
        rgb1_path = sample_dir / f"{view_id}_rgb1.png"
        save_rgb(event0, rgb0_path)
        save_rgb(event1, rgb1_path)
        aux_kwargs = _save_view_aux_maps(
            event0=event0,
            event1=event1,
            sample_dir=sample_dir,
            view_id=view_id,
            enable=bool(getattr(cfg, "enable_aux_maps", True)),
        )

        if bool(getattr(cfg, "enable_gt_flow", True)):
            try:
                gt_meta = attach_gt_flow_to_view(
                    event0=event0,
                    event1=event1,
                    move=move,
                    cam=used_cam,
                    sample_dir=sample_dir,
                    view_id=view_id,
                    fov_deg=float(getattr(cfg, "fov", 90)),
                )
                move.edit_meta = {**(move.edit_meta or {}), "gt_flow": gt_meta}
                print(
                    f"  {view_id} gt_flow[{gt_meta.get('method')}]: "
                    f"mean|f|={gt_meta.get('mean_flow_mag_px', 0):.1f}px "
                    f"max={gt_meta.get('max_flow_mag_px', 0):.1f}px "
                    f"mask={gt_meta.get('mask_pixels_before')}→{gt_meta.get('mask_pixels_after')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] gt_flow failed for {view_id}: {exc}", flush=True)

        ref = referring_phrase(
            event0,
            move.object_id,
            move.object_type,
            vlm=vlm if getattr(cfg, "vlm_referring_polish", False) else None,
            rgb_path=rgb0_path,
        )
        move.text = _move_text_with_camera(move, used_cam, ref["phrase"])
        _stamp_move_text_som(move)

        questions = [q]
        if vlm is not None:
            questions = word_questions_with_vlm(
                vlm,
                move,
                questions,
                rgb0_path,
                rgb1_path,
                use_images=bool(getattr(cfg, "vlm_word_use_images", False)),
            )
        else:
            _mark_template_wording(questions)

        mask_method = None
        for qq in questions:
            if qq.category == "occlusion":
                mask_method = qq.gt_meta.get("mask_method")
            qq.gt_meta = dict(qq.gt_meta)
            qq.gt_meta["mover_referring"] = ref
            if "mover_phrase" not in qq.gt_meta and ref.get("phrase"):
                qq.gt_meta["mover_phrase"] = ref["phrase"]
            if qq.category == "occlusion" and qq.gt_complete:
                if qq.answer:
                    occlude_balance[0] += 1
                else:
                    occlude_balance[1] += 1
            if qq.category == "feasibility" and qq.gt_complete:
                if qq.answer:
                    feasible_balance[0] += 1
                else:
                    feasible_balance[1] += 1
            if qq.category == "contact_wall" and qq.gt_complete:
                if qq.answer:
                    contact_balance[0] += 1
                else:
                    contact_balance[1] += 1
            if qq.category == "left_right_landmark" and qq.gt_complete:
                if qq.answer:
                    left_right_balance[0] += 1
                else:
                    left_right_balance[1] += 1
            if qq.category == "near_object" and qq.gt_complete:
                if qq.answer:
                    near_object_balance[0] += 1
                else:
                    near_object_balance[1] += 1

        rgb0_som_path, rgb1_som_path = _attach_som_for_view(
            questions=questions,
            move=move,
            event0=event0,
            event1=event1,
            sample_dir=sample_dir,
            view_id=view_id,
            enable_som=bool(getattr(cfg, "enable_som", True)),
        )

        views.append(
            ViewSample(
                view_id=view_id,
                camera=used_cam,
                move=move,
                rgb0_path=str(rgb0_path),
                rgb1_path=str(rgb1_path),
                questions=questions,
                vlm_rank_notes=rank_notes,
                occlusion_method=mask_method,
                target_category=cat,
                rgb0_som_path=rgb0_som_path,
                rgb1_som_path=rgb1_som_path,
                **aux_kwargs,
            )
        )
        categories_done.append(cat)
        print(
            f"  {view_id}: {move.text} | "
            f"q={[qq.category for qq in questions]} "
            f"complete={[qq.gt_complete for qq in questions]}"
        )

        # After rgb/aux/SoM are saved: rewind to pristine poses for the next category.
        if snap_on:
            _rewind(reason=f"after_{cat}", event=event1)

    if snap_on:
        print(
            f"  [snapshot] summary: restores={n_snap_restores} "
            f"fallbacks={n_snap_fallbacks} full_resets={n_full_resets}",
            flush=True,
        )
    return views, categories_done, occlude_balance, feasible_balance


def build_one_eval_sample(cfg: Optional[PipelineConfig] = None) -> EvalSample:
    """Generate one eval sample (iTHOR FloorPlan, or one ProcTHOR room via helpers)."""
    cfg = cfg or PipelineConfig()
    if cfg.dataset == "procthor":
        raise RuntimeError(
            "For ProcTHOR use build_procthor_house_samples(cfg) — one sample per room."
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    sample_id = f"ithor_{cfg.scene}_{uuid4().hex[:8]}"
    sample_dir = cfg.out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    vlm = _maybe_vlm(cfg)
    controller = start_controller(
        scene=cfg.scene,
        width=cfg.width,
        height=cfg.height,
        fov=cfg.fov,
        visibility_distance=cfg.visibility_distance,
        render_instance_segmentation=True,
        render_depth=bool(
            getattr(cfg, "enable_gt_flow", True)
            or getattr(cfg, "enable_aux_maps", True)
        ),
    )

    views: list[ViewSample] = []
    categories_done: list[str] = []
    occlude_balance = [0, 0]
    feasible_balance = [0, 0]

    try:
        fast_cam = use_fast_mode(cfg)
        cam_max_pos = (
            int(cfg.fast_camera_max_positions)
            if fast_cam
            else int(cfg.max_camera_positions)
        )
        cam_n_yaws = (
            int(cfg.fast_camera_n_yaws) if fast_cam else int(cfg.camera_n_yaws)
        )
        cameras = find_agent_eye_cameras(
            controller,
            n_views=(
                max(12, len(cfg.categories or QA_CATEGORIES))
                if cfg.category_driven
                else max(1, cfg.n_views)
            ),
            max_positions=cam_max_pos,
            n_yaws=cam_n_yaws,
            dist_min=cfg.dist_min,
            dist_max=cfg.dist_max,
            rng=rng,
        )
        print(f"Found {len(cameras)} camera pose(s)")
        if not cameras:
            raise RuntimeError(f"No camera poses for {cfg.scene}")

        if cfg.category_driven:
            views, categories_done, occlude_balance, feasible_balance = build_category_driven_views(
                controller=controller,
                scene_spec=cfg.scene,
                cam=cameras[0],
                sample_dir=sample_dir,
                cfg=cfg,
                rng=rng,
                vlm=vlm,
                cameras=cameras,
            )
        else:
            for vi, cam in enumerate(cameras):
                reset_scene(controller, cfg.scene)
                event = teleport_full(controller, **cam.teleport_kwargs())
                if not event.metadata["lastActionSuccess"]:
                    print(f"[warn] teleport failed for view {vi}")
                    continue

                result = find_edit_for_view(
                    controller,
                    cam,
                    event,
                    vlm=vlm,
                    angular_size_min=cfg.angular_size_min,
                    dist_min=cfg.dist_min,
                    dist_max=cfg.dist_max,
                )
                if result is None:
                    print(f"[warn] no valid edit for view {vi}")
                    continue

                move, event0, event1, rank_notes = result
                view_id = f"view{vi}"
                rgb0_path = sample_dir / f"{view_id}_rgb0.png"
                rgb1_path = sample_dir / f"{view_id}_rgb1.png"
                save_rgb(event0, rgb0_path)
                save_rgb(event1, rgb1_path)
                aux_kwargs = _save_view_aux_maps(
                    event0=event0,
                    event1=event1,
                    sample_dir=sample_dir,
                    view_id=view_id,
                    enable=bool(getattr(cfg, "enable_aux_maps", True)),
                )

                if bool(getattr(cfg, "enable_gt_flow", True)):
                    try:
                        gt_meta = attach_gt_flow_to_view(
                            event0=event0,
                            event1=event1,
                            move=move,
                            cam=cam,
                            sample_dir=sample_dir,
                            view_id=view_id,
                            fov_deg=float(getattr(cfg, "fov", 90)),
                        )
                        move.edit_meta = {**(move.edit_meta or {}), "gt_flow": gt_meta}
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [warn] gt_flow failed for {view_id}: {exc}", flush=True)

                ref = referring_phrase(
                    event0,
                    move.object_id,
                    move.object_type,
                    vlm=vlm if getattr(cfg, "vlm_referring_polish", False) else None,
                    rgb_path=rgb0_path,
                )
                move.text = _move_text_with_camera(move, cam, ref["phrase"])
                _stamp_move_text_som(move)

                questions = sample_two_questions(
                    event0,
                    event1,
                    move,
                    rng=rng,
                    occlude_yes_counter=occlude_balance,
                    feasible_counter=feasible_balance,
                    controller=controller,
                    cam=cam,
                )
                if vlm is not None:
                    questions = word_questions_with_vlm(
                        vlm,
                        move,
                        questions,
                        rgb0_path,
                        rgb1_path,
                        use_images=bool(getattr(cfg, "vlm_word_use_images", False)),
                    )
                else:
                    _mark_template_wording(questions)

                mask_method = None
                for q in questions:
                    if q.category == "occlusion":
                        mask_method = q.gt_meta.get("mask_method")
                    q.gt_meta = dict(q.gt_meta)
                    q.gt_meta["mover_referring"] = ref
                    if "mover_phrase" not in q.gt_meta and ref.get("phrase"):
                        q.gt_meta["mover_phrase"] = ref["phrase"]

                rgb0_som_path, rgb1_som_path = _attach_som_for_view(
                    questions=questions,
                    move=move,
                    event0=event0,
                    event1=event1,
                    sample_dir=sample_dir,
                    view_id=view_id,
                    enable_som=bool(getattr(cfg, "enable_som", True)),
                )

                views.append(
                    ViewSample(
                        view_id=view_id,
                        camera=cam,
                        move=move,
                        rgb0_path=str(rgb0_path),
                        rgb1_path=str(rgb1_path),
                        questions=questions,
                        vlm_rank_notes=rank_notes,
                        occlusion_method=mask_method,
                        rgb0_som_path=rgb0_som_path,
                        rgb1_som_path=rgb1_som_path,
                        **aux_kwargs,
                    )
                )
                print(
                    f"  {view_id}: {move.text} | "
                    f"questions={[q.category for q in questions]}"
                )
                if len(views) >= cfg.n_views:
                    break
    finally:
        controller.stop()

    if not views:
        raise RuntimeError(
            f"Failed to generate any views for scene {cfg.scene}. "
            "Try another FloorPlan or relax distance constraints."
        )

    sample = EvalSample(
        sample_id=sample_id,
        dataset="ithor",
        scene=cfg.scene,
        views=views,
        meta={
            "visibility_distance": cfg.visibility_distance,
            "dist_band": [cfg.dist_min, cfg.dist_max],
            "angular_size_min": cfg.angular_size_min,
            "occlude_threshold": cfg.occlude_threshold,
            "occlude_yes_no_counts": list(occlude_balance),
            "feasible_yes_no_counts": list(feasible_balance),
            "image_size": {"width": cfg.width, "height": cfg.height},
            "fov": cfg.fov,
            "vlm_used": vlm is not None,
            "vlm_skip_reason": None if vlm is not None else "vlm_bypassed",
            "n_views_requested": (
                len(cfg.categories) if cfg.categories else cfg.n_views
            ),
            "n_views_got": len(views),
            "category_driven": cfg.category_driven,
            "categories_done": categories_done,
            "categories_requested": list(cfg.categories)
            if cfg.categories
            else list(QA_CATEGORIES),
            "modification": cfg.modification or MODIFICATION_TAG,
            "enable_som": bool(getattr(cfg, "enable_som", True)),
            "dataset_backend": "ithor",
        },
    )

    out_json = sample_dir / "sample.json"
    out_json.write_text(json.dumps(sample.to_dict(), indent=2))
    print(f"Wrote {out_json}")
    sample = _attach_cot_if_requested(cfg, sample, sample_dir, vlm)
    return sample


def build_procthor_house_samples(cfg: PipelineConfig) -> list[EvalSample]:
    """Load one ProcTHOR apartment; emit samples (apartment- or room-scoped)."""
    from cit_pipeline.procthor_rooms import (
        filter_positions_in_room,
        list_rooms,
        load_procthor_house,
        room_label,
    )

    if cfg.house_index is None:
        raise ValueError("PipelineConfig.house_index is required for ProcTHOR")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    house = load_procthor_house(
        int(cfg.house_index),
        split=cfg.house_split,
        revision=cfg.procthor_revision,
    )
    rooms = list_rooms(house)
    if cfg.max_rooms and cfg.max_rooms > 0:
        rooms = rooms[: cfg.max_rooms]

    scene_name = f"procthor-{cfg.house_split}-{cfg.house_index}"
    cam_mode = str(getattr(cfg, "procthor_camera_mode", "apartment") or "apartment")
    print(
        f"ProcTHOR house {cfg.house_index} ({cfg.house_split}): "
        f"{len(rooms)} room(s) mode={cam_mode} → {cfg.out_dir}",
        flush=True,
    )

    vlm = _maybe_vlm(cfg)
    controller = start_controller(
        scene=house,
        width=cfg.width,
        height=cfg.height,
        fov=cfg.fov,
        visibility_distance=cfg.visibility_distance,
        render_instance_segmentation=True,
        render_depth=bool(
            getattr(cfg, "enable_gt_flow", True)
            or getattr(cfg, "enable_aux_maps", True)
        ),
    )

    samples: list[EvalSample] = []
    try:
        all_positions = controller.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ] or []
        print(f"Reachable positions in apartment: {len(all_positions)}")

        if cam_mode == "apartment":
            samples.extend(
                _build_procthor_apartment_camera_samples(
                    cfg=cfg,
                    controller=controller,
                    house=house,
                    rooms=rooms,
                    all_positions=all_positions,
                    scene_name=scene_name,
                    rng=rng,
                    vlm=vlm,
                )
            )
        else:
            samples.extend(
                _build_procthor_per_room_samples(
                    cfg=cfg,
                    controller=controller,
                    house=house,
                    rooms=rooms,
                    all_positions=all_positions,
                    scene_name=scene_name,
                    rng=rng,
                    vlm=vlm,
                )
            )
    finally:
        controller.stop()

    if not samples:
        raise RuntimeError(
            f"No ProcTHOR samples for house {cfg.house_index}. "
            "Try another index or relax distance constraints."
        )
    return samples


def _build_procthor_apartment_camera_samples(
    *,
    cfg: PipelineConfig,
    controller: Any,
    house: dict[str, Any],
    rooms: list[dict[str, Any]],
    all_positions: list[dict[str, float]],
    scene_name: str,
    rng: random.Random,
    vlm: Any,
) -> list[EvalSample]:
    """y apartment-wide cameras → y samples; category fallbacks across the pool."""
    from cit_pipeline.procthor_rooms import room_label

    n_cats = len(cfg.categories or QA_CATEGORIES)
    y_req = max(1, int(getattr(cfg, "procthor_cameras_per_apartment", 10) or 10))
    # Never more than 3 viewpoints slots per room (even if y_req is larger).
    y = min(y_req, max(1, 3 * len(rooms)))
    room_bonus = float(getattr(cfg, "procthor_room_diversity_bonus", 1000.0) or 1000.0)
    min_sep = float(getattr(cfg, "procthor_camera_min_separation", 1.0) or 1.0)
    proposals_per_room = max(
        2, int(getattr(cfg, "procthor_proposals_per_room", 3) or 3)
    )
    max_pos_per_room = max(
        8, int(getattr(cfg, "procthor_max_positions_per_room", 10) or 10)
    )
    # Apartment search always uses 8 yaws (45°); horizon stays a coin flip.
    # (Fast-mode 4-yaw still applies to legacy single-room / per_room search.)
    cam_n_yaws = max(8, int(cfg.camera_n_yaws))

    picked: list[tuple[CameraPose, str, str]] = []
    camera_fallback = "strict_band"
    search_attempts: list[dict[str, Any]] = [
        {
            "name": "strict_band",
            "dist_min": cfg.dist_min,
            "dist_max": cfg.dist_max,
            "require_distance_band": True,
            "n_yaws": cam_n_yaws,
            "max_positions_per_room": max_pos_per_room,
            "proposals_per_room": proposals_per_room,
        },
        {
            "name": "relaxed_band",
            "dist_min": max(0.3, float(cfg.dist_min) * 0.4),
            "dist_max": max(float(cfg.dist_max), 6.0),
            "require_distance_band": True,
            "n_yaws": cam_n_yaws,
            "max_positions_per_room": max(max_pos_per_room, 20),
            "proposals_per_room": max(proposals_per_room, 3),
        },
        {
            "name": "any_teleport",
            "dist_min": cfg.dist_min,
            "dist_max": cfg.dist_max,
            "require_distance_band": False,
            "n_yaws": cam_n_yaws,
            "max_positions_per_room": max(max_pos_per_room, 24),
            "proposals_per_room": max(proposals_per_room, 3),
        },
    ]
    last_exc: Optional[BaseException] = None
    for attempt in search_attempts:
        try:
            picked = find_apartment_cameras(
                controller,
                n_cameras=y,
                rooms=rooms,
                positions=all_positions,
                max_positions_per_room=int(attempt["max_positions_per_room"]),
                proposals_per_room=int(attempt["proposals_per_room"]),
                n_yaws=int(attempt["n_yaws"]),
                dist_min=float(attempt["dist_min"]),
                dist_max=float(attempt["dist_max"]),
                min_separation=min_sep,
                room_diversity_bonus=room_bonus,
                require_distance_band=bool(attempt["require_distance_band"]),
                rng=rng,
            )
            camera_fallback = str(attempt["name"])
            break
        except RuntimeError as exc:
            last_exc = exc
            picked = []
    if not picked:
        raise RuntimeError(f"no apartment cameras: {last_exc}")

    cameras = [c for c, _rid, _rt in picked]
    print(
        f"Apartment mode: y={y} got={len(cameras)} "
        f"fallback_stage={camera_fallback} "
        f"target_questions≈{len(cameras) * n_cats}",
        flush=True,
    )

    cfg.procthor_house_rooms = rooms

    samples: list[EvalSample] = []
    used_movers_by_room: dict[str, set[str]] = {}
    for cam_i, (cam, rid, rtype) in enumerate(picked):
        label = room_label({"id": rid, "roomType": rtype})
        # Primary first; remaining apartment cameras fill failed categories.
        cam_pool = [cam] + [c for j, c in enumerate(cameras) if j != cam_i]
        sample_id = (
            f"procthor_{cfg.house_split}{cfg.house_index}_cam{cam_i}_{label}_"
            f"{uuid4().hex[:8]}"
        )
        sample_dir = cfg.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        room_used = used_movers_by_room.setdefault(str(rid), set())
        views, categories_done, occlude_balance, feasible_balance = (
            build_category_driven_views(
                controller=controller,
                scene_spec=house,
                cam=cam,
                sample_dir=sample_dir,
                cfg=cfg,
                rng=rng,
                vlm=vlm,
                view_id_prefix="view",
                cameras=cam_pool,
                room_id=str(rid),
                used_movers_in_room=room_used,
            )
        )
        if not views:
            print(f"  [warn] no views for apartment camera {cam_i} ({label})")
            continue

        sample = EvalSample(
            sample_id=sample_id,
            dataset="procthor-10k",
            scene=f"{scene_name}/cam{cam_i}_{label}",
            views=views,
            meta={
                "visibility_distance": cfg.visibility_distance,
                "dist_band": [cfg.dist_min, cfg.dist_max],
                "angular_size_min": cfg.angular_size_min,
                "occlude_threshold": cfg.occlude_threshold,
                "occlude_yes_no_counts": list(occlude_balance),
                "feasible_yes_no_counts": list(feasible_balance),
                "image_size": {"width": cfg.width, "height": cfg.height},
                "fov": cfg.fov,
                "vlm_used": vlm is not None,
                "vlm_skip_reason": None if vlm is not None else "vlm_bypassed",
                "n_views_requested": n_cats,
                "n_views_got": len(views),
                "category_driven": True,
                "categories_done": categories_done,
                "modification": (
                    str(getattr(cfg, "modification", None) or "mod_procthor_apartment")
                ),
                "dataset_backend": "procthor",
                "house_index": cfg.house_index,
                "house_split": cfg.house_split,
                "camera_mode": "apartment",
                "cameras_per_apartment": y,
                "camera_index": cam_i,
                "n_apartment_cameras": len(cameras),
                "room_id": rid,
                "room_type": rtype,
                "room_label": label,
                "rooms_in_house": len(rooms),
                "camera_fallback": camera_fallback,
                "camera_score": cam.score,
                "room_diversity_bonus": room_bonus,
                "camera_min_separation": min_sep,
                "proposals_per_room": proposals_per_room,
                "max_positions_per_room": max_pos_per_room,
                "camera_n_yaws": cam_n_yaws,
            },
        )
        out_json = sample_dir / "sample.json"
        out_json.write_text(json.dumps(sample.to_dict(), indent=2))
        print(
            f"Wrote {out_json} (cats={len(categories_done)}/{n_cats})",
            flush=True,
        )
        sample = _attach_cot_if_requested(cfg, sample, sample_dir, vlm)
        samples.append(sample)

    n_q = sum(len(v.questions) for s in samples for v in s.views)
    print(
        f"Apartment house done: samples={len(samples)}/{len(cameras)} "
        f"questions={n_q} (ideal={len(cameras) * n_cats})",
        flush=True,
    )
    return samples


def _build_procthor_per_room_samples(
    *,
    cfg: PipelineConfig,
    controller: Any,
    house: dict[str, Any],
    rooms: list[dict[str, Any]],
    all_positions: list[dict[str, float]],
    scene_name: str,
    rng: random.Random,
    vlm: Any,
) -> list[EvalSample]:
    """Legacy v7 path: one (or more) samples per room with an in-room camera pool."""
    from cit_pipeline.procthor_rooms import filter_positions_in_room, room_label

    samples: list[EvalSample] = []
    for room in rooms:
        label = room_label(room)
        room_pos = filter_positions_in_room(all_positions, room)
        print(
            f"Room {room.get('id')} ({room.get('roomType')}): "
            f"{len(room_pos)} reachable positions"
        )
        if len(room_pos) < 1:
            print(f"  [warn] skip room — no reachable positions")
            continue

        fast_cam = use_fast_mode(cfg)
        cam_max_pos = (
            int(cfg.fast_camera_max_positions)
            if fast_cam
            else int(cfg.max_camera_positions)
        )
        cam_n_yaws = (
            int(cfg.fast_camera_n_yaws) if fast_cam else int(cfg.camera_n_yaws)
        )
        n_cats = len(cfg.categories or QA_CATEGORIES)
        cfg_cams = int(getattr(cfg, "procthor_cameras_per_room", 0) or 0)
        n_samples_room = max(1, int(getattr(cfg, "procthor_samples_per_room", 1) or 1))
        n_cams = cfg_cams if cfg_cams > 0 else max(8, n_cats, n_samples_room)
        max_pos = min(cam_max_pos, max(8, len(room_pos)))

        cameras: list[CameraPose] = []
        camera_fallback = "strict_band"
        search_attempts: list[dict[str, Any]] = [
            {
                "name": "strict_band",
                "dist_min": cfg.dist_min,
                "dist_max": cfg.dist_max,
                "require_distance_band": True,
                "max_positions": max_pos,
                "n_yaws": cam_n_yaws,
            },
            {
                "name": "relaxed_band",
                "dist_min": max(0.3, float(cfg.dist_min) * 0.4),
                "dist_max": max(float(cfg.dist_max), 6.0),
                "require_distance_band": True,
                "max_positions": min(len(room_pos), max(max_pos, 30)),
                "n_yaws": max(cam_n_yaws, 8),
            },
            {
                "name": "any_teleport",
                "dist_min": cfg.dist_min,
                "dist_max": cfg.dist_max,
                "require_distance_band": False,
                "max_positions": min(len(room_pos), max(max_pos, 40)),
                "n_yaws": 8,
            },
        ]
        last_exc: Optional[BaseException] = None
        for attempt in search_attempts:
            try:
                cameras = find_agent_eye_cameras(
                    controller,
                    n_views=n_cams,
                    max_positions=int(attempt["max_positions"]),
                    n_yaws=int(attempt["n_yaws"]),
                    dist_min=float(attempt["dist_min"]),
                    dist_max=float(attempt["dist_max"]),
                    positions=room_pos,
                    rng=rng,
                    require_distance_band=bool(attempt["require_distance_band"]),
                )
                camera_fallback = str(attempt["name"])
                break
            except RuntimeError as exc:
                last_exc = exc
                cameras = []
        if not cameras:
            print(f"  [warn] no camera in room: {last_exc}")
            continue

        n_emit = min(n_samples_room, len(cameras))
        print(
            f"  camera pool={len(cameras)} "
            f"(want={n_cams}, samples_per_room={n_samples_room}, "
            f"fallback={camera_fallback}) → emit {n_emit} sample(s)",
            flush=True,
        )

        for cam_i in range(n_emit):
            cam = cameras[cam_i]
            room_cameras = cameras[cam_i:]
            sample_id = (
                f"procthor_{cfg.house_split}{cfg.house_index}_{label}_"
                f"c{cam_i}_{uuid4().hex[:8]}"
                if n_emit > 1
                else f"procthor_{cfg.house_split}{cfg.house_index}_{label}_{uuid4().hex[:8]}"
            )
            sample_dir = cfg.out_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            views, categories_done, occlude_balance, feasible_balance = (
                build_category_driven_views(
                    controller=controller,
                    scene_spec=house,
                    cam=cam,
                    sample_dir=sample_dir,
                    cfg=cfg,
                    rng=rng,
                    vlm=vlm,
                    view_id_prefix="view",
                    cameras=room_cameras,
                )
            )
            if not views:
                print(f"  [warn] no views for room {label} camera {cam_i}")
                continue

            sample = EvalSample(
                sample_id=sample_id,
                dataset="procthor-10k",
                scene=f"{scene_name}/{label}",
                views=views,
                meta={
                    "visibility_distance": cfg.visibility_distance,
                    "dist_band": [cfg.dist_min, cfg.dist_max],
                    "angular_size_min": cfg.angular_size_min,
                    "occlude_threshold": cfg.occlude_threshold,
                    "occlude_yes_no_counts": list(occlude_balance),
                    "feasible_yes_no_counts": list(feasible_balance),
                    "image_size": {"width": cfg.width, "height": cfg.height},
                    "fov": cfg.fov,
                    "vlm_used": vlm is not None,
                    "vlm_skip_reason": None if vlm is not None else "vlm_bypassed",
                    "n_views_requested": n_cats,
                    "n_views_got": len(views),
                    "category_driven": True,
                    "categories_done": categories_done,
                    "modification": (
                        str(getattr(cfg, "modification", None) or "mod_procthor_per_room")
                    ),
                    "dataset_backend": "procthor",
                    "house_index": cfg.house_index,
                    "house_split": cfg.house_split,
                    "camera_mode": "per_room",
                    "room_id": room.get("id"),
                    "room_type": room.get("roomType"),
                    "room_label": label,
                    "n_reachable_in_room": len(room_pos),
                    "cameras_per_room": n_cams,
                    "samples_per_room": n_samples_room,
                    "camera_index": cam_i,
                    "camera_fallback": camera_fallback,
                    "camera_score": cam.score,
                },
            )
            out_json = sample_dir / "sample.json"
            out_json.write_text(json.dumps(sample.to_dict(), indent=2))
            print(f"Wrote {out_json}")
            sample = _attach_cot_if_requested(cfg, sample, sample_dir, vlm)
            samples.append(sample)
    return samples


def write_manifest(out_dir: Path, sample: EvalSample) -> Path:
    """Append a one-line index entry for dataset bookkeeping."""
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a") as f:
        f.write(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "scene": sample.scene,
                    "dataset": sample.dataset,
                    "n_views": len(sample.views),
                    "categories": [
                        v.target_category or (v.questions[0].category if v.questions else None)
                        for v in sample.views
                    ],
                    "room_type": (sample.meta or {}).get("room_type"),
                    "house_index": (sample.meta or {}).get("house_index"),
                }
            )
            + "\n"
        )
    return manifest
