"""Category-specific move-target selection (Modification 4–5, 8).

Each QA category picks a mover with its own rules, then validates the
translation. Modification 5 adds specialized paths for ``contact_wall``
(multi-step toward-wall) and ``occlusion`` (pixel-planned cover of small B).

Modification 8: chair-only affordance (extreme leftmost/rightmost when many
chairs; optional observer-chair FOV) and 1.5 m wall selection with extreme
referring for crowded small-object types.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Optional, Sequence

import numpy as np
from ai2thor.controller import Controller

from cit_pipeline.camera_search import camera_basis, objects_in_distance_band
from cit_pipeline.config import (
    AFFORDANCE_CHAIR_TYPES,
    ANGULAR_SIZE_MIN,
    CLOSER_TO_FAST_MAX_TRIES,
    CLOSER_TO_GAP_M,
    CLOSER_TO_MAX_MOVE_M,
    CLOSER_TO_MIN_BC_SEP_M,
    CLOSER_TO_MIN_MOVE_M,
    DIST_MAX,
    DIST_MIN,
    FOV,
    FORBIDDEN_MOVE_TYPES,
    HEIGHT,
    LARGE_OCCLUDER_TYPES,
    LEFT_RIGHT_MIN_CROSS_DEG,
    LEFT_RIGHT_MIN_SPAN_DEG,
    LEFT_RIGHT_REQUIRE_FLIP,
    LEFT_RIGHT_NONFLIP_FRAC,
    FLIP_PLAN_FORCE_EDIT,
    CHOICE_MASK_MIN_PIXELS,
    MOVER_MASK_MIN_PIXELS_MOD20,
    MOVE_DIRECTIONS,
    MOVE_METERS_CANDIDATES,
    OCCLUDE_COVER_BACK_M,
    OCCLUDE_COVER_METERS_MAX,
    OCCLUDE_COVER_METERS_MIN,
    OCCLUDE_COVER_SCALE,
    OCCLUDE_PAIR_MAX_DIST_M,
    OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M,
    OCCLUDE_RAY_FRONT_MARGIN_M,
    OCCLUDE_RAY_NO_OFFSET_M,
    OCCLUDE_THRESHOLD,
    OCCLUDEE_MASK_MIN_PIXELS,
    SOM_OCCLUDEE_MASK_MIN_PIXELS,
    SOM_MOVER_MASK_MIN_PIXELS,
    ORDINAL_MAX_SAME_TYPE,
    SMALL_OCCLUDEE_TYPES,
    WALL_AFTER_MASK_MIN_PIXELS,
    WALL_AWAY_PROB,
    WALL_BANNED_MOVE_TYPES,
    WALL_CONTACT_M,
    WALL_MOVE_METERS,
    WALL_MOVE_STEPS,
    WALL_PENETRATION_TOL_M,
    WALL_SELECT_NEAR_M,
    WALL_SELECT_NEAR_M_FAST,
    WALL_SLIDE_STEP_M,
    WALL_STOP_MARGIN_M,
    CONTACT_ALWAYS_TOWARD,
    CONTACT_HALF_MIN_LEFTOVER_M,
    CONTACT_HALF_MOTION_NO_FRAC,
    CONTACT_HALF_MOVE_SUB_M,
    CONTACT_ROOM_WALL_MAX_M,
    CONTACT_ROOM_WALL_PREFILTER,
    CONTACT_FAST_MAX_CANDIDATES,
    FAST_CONTACT_MAX_TRIES,
    FAST_OCCLUSION_MAX_TRIES,
    WIDTH,
)
from cit_pipeline.edit_search import (
    EditProposal,
    camera_delta_to_world,
    enumerate_edit_proposals,
    try_edit,
)
from cit_pipeline.move_text import format_move_displacement, primary_camera_direction
from cit_pipeline.occlusion import get_object_mask, occlusion_cover_rate
from cit_pipeline.referring import (
    camera_bearing_deg,
    camera_frame_x,
    extreme_side,
    instance_referring_allowed,
    ordinal_from_left,
    referring_allowed,
)
from cit_pipeline.schemas import CameraPose, MoveAction, Vec3
from cit_pipeline.scene_integrity import object_on_screen, scene_collateral_ok
from cit_pipeline.size_filter import (
    aabb_bottom_y,
    aabb_footprint,
    aabb_xz_extent,
    is_small_enough_to_move,
    is_visible_mask_large_enough,
    object_mask_pixel_count,
    scene_table_footprint_limit,
    support_qa_eligible,
)
from cit_pipeline.thor_env import get_object, pin_object_pose, still_visible, teleport_full, teleport_object
from cit_pipeline.vlm_client import VlmClient

# Extend size_filter bans with Mod-20 forbidden types (Sink/Sofa already present;
# adds LightSwitch / CD / ArmChair tippers).
import cit_pipeline.size_filter as _sf

_sf.BLOCKED_MOVE_TYPES = tuple(
    sorted(set(_sf.BLOCKED_MOVE_TYPES) | set(FORBIDDEN_MOVE_TYPES) | set(WALL_BANNED_MOVE_TYPES))
)


def _mover_forbidden(obj: dict[str, Any]) -> bool:
    typ = obj.get("objectType")
    return typ in FORBIDDEN_MOVE_TYPES or typ in WALL_BANNED_MOVE_TYPES


def _mover_visually_ok(event: Any, obj: dict[str, Any], *, som: bool = False) -> bool:
    """Reject movers that are metadata-visible but essentially off-screen."""
    min_px = SOM_MOVER_MASK_MIN_PIXELS if som else MOVER_MASK_MIN_PIXELS_MOD20
    return object_on_screen(event, obj, min_pixels=int(min_px))


def _choice_visually_ok(event: Any, obj: dict[str, Any]) -> bool:
    return object_on_screen(event, obj, min_pixels=int(CHOICE_MASK_MIN_PIXELS))


def order_equal_by_mover(
    items: Sequence[Any],
    *,
    object_id_fn: Callable[[Any], str],
    rng: random.Random,
    penalize_ids: Optional[set[str]] = None,
) -> list[Any]:
    """Shuffle so each distinct mover is equally likely; previously-used last.

    Within a mover, items are also shuffled. Items whose ``object_id`` is in
    ``penalize_ids`` (same-room repeats) are tried only after unused movers.
    """
    by_id: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        by_id[str(object_id_fn(item))].append(item)
    if not by_id:
        return []
    pen = penalize_ids or set()
    unused = [oid for oid in by_id if oid not in pen]
    used = [oid for oid in by_id if oid in pen]
    rng.shuffle(unused)
    rng.shuffle(used)
    out: list[Any] = []
    for oid in unused + used:
        chunk = list(by_id[oid])
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def _relabel_move_camera_frame(
    move: MoveAction,
    basis: dict[str, np.ndarray],
) -> MoveAction:
    """Rewrite direction + text from world Δ in the camera frame (Mod 20)."""
    delta = np.array(
        [move.delta_world.x, move.delta_world.y, move.delta_world.z],
        dtype=np.float64,
    )
    direction = primary_camera_direction(delta, basis)
    move.direction = direction  # type: ignore[assignment]
    move.text = (
        f"move the {move.object_type.lower()} "
        f"{format_move_displacement(direction, move.meters, delta, basis=basis)}"
    )
    return move


def _signed_edge_wall_clearance(
    event: Any, obj: dict[str, Any]
) -> tuple[float, str, Optional[str]]:
    """Return (signed_clearance_m, method, nearest_wall_axis).

    Negative clearance means the AABB already penetrates past the scene bound
    (Modification 10: used to reject force-through-wall poses).
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
        clearances = {
            "-x": min_x - scene_min_x,
            "+x": scene_max_x - max_x,
            "-z": min_z - scene_min_z,
            "+z": scene_max_z - max_z,
        }
        axis = min(clearances, key=clearances.get)
        return float(clearances[axis]), "aabb_sceneBounds", axis
    return 999.0, "none", None


def _min_edge_wall_clearance(event: Any, obj: dict[str, Any]) -> tuple[float, str, Optional[str]]:
    """Return (non-negative clearance_m, method, nearest_wall_axis)."""
    clr, method, axis = _signed_edge_wall_clearance(event, obj)
    return max(0.0, float(clr)), method, axis


def wall_axis_to_world_delta(axis: str) -> np.ndarray:
    mapping = {
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+x": np.array([1.0, 0.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
    }
    return mapping[axis]


def opposite_wall_axis(axis: str) -> str:
    return {"-x": "+x", "+x": "-x", "-z": "+z", "+z": "-z"}[axis]


def world_delta_to_camera_direction(
    world_delta: np.ndarray, basis: dict[str, np.ndarray]
) -> str:
    delta = np.array([world_delta[0], 0.0, world_delta[2]], dtype=np.float64)
    n = float(np.linalg.norm(delta))
    if n < 1e-8:
        return "forward"
    delta /= n
    scores = {
        "left": float(np.dot(delta, -basis["right"])),
        "right": float(np.dot(delta, basis["right"])),
        "forward": float(np.dot(delta, basis["forward"])),
        "back": float(np.dot(delta, -basis["forward"])),
    }
    return max(scores, key=scores.get)


def _mask_centroid_uv(mask: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Return (u, v, area) for a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(len(xs))


def _meters_per_pixel_horizontal(depth_m: float, width: int = WIDTH, fov_deg: float = FOV) -> float:
    return float(2.0 * depth_m * math.tan(math.radians(fov_deg) / 2.0) / max(1, width))


def filter_proposals_for_category(
    event: Any,
    basis: dict[str, np.ndarray],
    category: str,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    som: bool = False,
) -> list[EditProposal]:
    """Build edit proposals tailored to ``category`` (generic categories)."""
    proposals = enumerate_edit_proposals(
        event,
        basis,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    )
    if category == "affordance":
        # Affordance uses find_affordance_edit; keep a soft filter for legacy paths.
        chairs = [
            p
            for p in proposals
            if p.object_type in AFFORDANCE_CHAIR_TYPES
            and instance_referring_allowed(event, p.object_id, p.object_type)
        ]
        return chairs

    proposals = [
        p
        for p in proposals
        if instance_referring_allowed(event, p.object_id, p.object_type)
        and p.object_type not in FORBIDDEN_MOVE_TYPES
        and p.object_type not in WALL_BANNED_MOVE_TYPES
    ]
    if not proposals:
        return []

    if category == "support":
        out = []
        for p in proposals:
            obj = next(
                (o for o in event.metadata["objects"] if o["objectId"] == p.object_id),
                None,
            )
            if obj is not None and support_qa_eligible(obj):
                out.append(p)
        return out

    return proposals


def _make_move(
    *,
    object_id: str,
    object_type: str,
    direction: str,
    meters: float,
    world_delta: np.ndarray,
    old_pos: dict[str, float],
    moved_pos: dict[str, float],
    old_rot: dict[str, float],
    distance_before: float,
    edit_meta: Optional[dict[str, Any]] = None,
    basis: Optional[dict[str, np.ndarray]] = None,
) -> MoveAction:
    if basis is not None:
        direction = primary_camera_direction(world_delta, basis)
    return MoveAction(
        object_id=object_id,
        object_type=object_type,
        direction=direction,  # type: ignore[arg-type]
        meters=float(meters),
        delta_world=Vec3(
            x=float(world_delta[0]),
            y=float(world_delta[1]),
            z=float(world_delta[2]),
        ),
        position_before=Vec3.from_dict(old_pos),
        position_after=Vec3.from_dict(moved_pos),
        rotation_before=dict(old_rot),
        distance_before=float(distance_before),
        angular_size_proxy=float(meters) / max(1e-3, float(distance_before)),
        still_visible_in_rgb1=True,
        text=(
            f"move the {object_type.lower()} "
            f"{format_move_displacement(direction, meters, world_delta, basis=basis)}"
        ),
        edit_meta=dict(edit_meta or {}),
    )


def try_wall_multistep_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    obj: dict[str, Any],
    *,
    total_meters: float,
    n_steps: int = WALL_MOVE_STEPS,
    toward_wall: bool = True,
    min_after_mask_pixels: int = WALL_AFTER_MASK_MIN_PIXELS,
    _mask_halve_depth: int = 0,
) -> Optional[tuple[MoveAction, Any, Any]]:
    """Slide an object toward (or away from) the nearest wall (Modification 5 / 8 / 10).

    rgb1 is the last *successful* collision-checked pose. Contact is recorded if a
    later step is blocked by the simulator and/or final edge clearance <
    ``WALL_CONTACT_M``.

    Modification 8: slide along the **world** wall axis (not a quantized camera
    direction). ``toward_wall=False`` flips the axis (intended "no").

    Modification 10: ``forceAction=False`` + small steps so objects cannot be
    forced into / through walls. Poses with signed clearance <
    ``-WALL_PENETRATION_TOL_M`` are rejected as penetration.

    Modification 12: if the after-move instance mask has fewer than
    ``WALL_AFTER_MASK_MIN_PIXELS`` pixels, **halve** ``total_meters`` and retry
    (binary refine) so harsh slides do not erase the object (e.g. towel).
    """
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    clearance0, method, nearest_axis = _signed_edge_wall_clearance(event, obj)
    if method == "none" or nearest_axis is None or clearance0 >= WALL_SELECT_NEAR_M:
        return None
    if clearance0 < -WALL_PENETRATION_TOL_M:
        return None

    # Slide toward nearest wall, or away from it for balanced "no" answers.
    axis = nearest_axis if toward_wall else opposite_wall_axis(nearest_axis)
    wall_unit = wall_axis_to_world_delta(axis)
    direction = world_delta_to_camera_direction(wall_unit, basis)
    old_pos = dict(obj["position"])
    old_rot = dict(obj["rotation"])
    # Modification 11: binary-refine along the wall axis instead of tiny fixed steps.
    from cit_pipeline.config import BINARY_SLIDE_ITERS
    from cit_pipeline.edit_search import binary_refine_slide

    cur_pos, applied_m, binary_trace = binary_refine_slide(
        controller,
        obj["objectId"],
        old_pos,
        old_rot,
        wall_unit,
        total_meters=float(total_meters),
        force=False,
        n_iters=BINARY_SLIDE_ITERS,
        hold_y=float(old_pos["y"]),
    )
    if applied_m < 1e-3:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    # Visibility / penetration check at refined pose.
    pin_object_pose(controller, obj["objectId"], cur_pos, old_rot)
    probe_ev = teleport_full(controller, **cam.teleport_kwargs())
    if not still_visible(probe_ev, obj["objectId"]):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    probe_obj = get_object(probe_ev, obj["objectId"])
    if probe_obj is None:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    clr_signed, _, _ = _signed_edge_wall_clearance(probe_ev, probe_obj)
    if clr_signed < -WALL_PENETRATION_TOL_M:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    sim_blocked = any(not t.get("ok") for t in binary_trace) and applied_m + 1e-3 < float(
        total_meters
    )
    step_clearances = [float(clr_signed)]
    steps_ok = sum(1 for t in binary_trace if t.get("ok"))
    step = float(WALL_SLIDE_STEP_M)
    moved_pos = {
        "x": float(probe_obj["position"]["x"]),
        "y": float(old_pos["y"]),
        "z": float(probe_obj["position"]["z"]),
    }
    # Render rgb0 / rgb1 with kinematic pins.
    pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, obj["objectId"], moved_pos, old_rot)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, obj["objectId"], moved_pos, old_rot)

    if not still_visible(event1, obj["objectId"]):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    after = get_object(event1, obj["objectId"])
    if after is None:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    final_signed, final_method, _ = _signed_edge_wall_clearance(event1, after)
    if final_signed < -WALL_PENETRATION_TOL_M:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    # Modification 12: reject near-invisible after poses; halve motion and retry.
    after_px = object_mask_pixel_count(event1, obj["objectId"])
    if (
        after_px is not None
        and after_px < int(min_after_mask_pixels)
        and float(total_meters) > 0.05
        and _mask_halve_depth < 4
    ):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return try_wall_multistep_edit(
            controller,
            cam,
            event,
            obj,
            total_meters=0.5 * float(total_meters),
            n_steps=n_steps,
            toward_wall=toward_wall,
            min_after_mask_pixels=min_after_mask_pixels,
            _mask_halve_depth=_mask_halve_depth + 1,
        )
    if after_px is not None and after_px < int(min_after_mask_pixels):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    final_clearance = max(0.0, float(final_signed))
    # Contact = near the *scene* wall edge (AABB vs sceneBounds).
    # Do NOT treat sim_blocked alone as contact: binary slides often stop on
    # furniture/counter while wall clearance is still ≫ WALL_CONTACT_M
    # (newspaper/lettuce false-positives in v4).
    contact = final_clearance < WALL_CONTACT_M
    if toward_wall and bool(sim_blocked) and not contact:
        # Hit a non-wall obstacle before reaching the wall — reject this try.
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    world_delta = np.array(
        [
            moved_pos["x"] - old_pos["x"],
            0.0,
            moved_pos["z"] - old_pos["z"],
        ],
        dtype=np.float64,
    )
    applied_m = float(np.linalg.norm(world_delta))
    move = _make_move(
        object_id=obj["objectId"],
        object_type=obj["objectType"],
        direction=direction,
        meters=applied_m if applied_m > 1e-3 else float(np.linalg.norm(world_delta)),
        world_delta=world_delta,
        old_pos=old_pos,
        moved_pos=moved_pos,
        old_rot=old_rot,
        distance_before=float(obj.get("distance") or 2.0),
        basis=basis,
        edit_meta={
            "wall_multistep": True,
            "wall_binary_slide": True,
            "wall_collision_checked": True,
            "wall_axis": axis,
            "wall_nearest_axis": nearest_axis,
            "wall_toward": bool(toward_wall),
            "wall_total_planned_m": float(total_meters),
            "wall_step_m": step,
            "wall_steps_ok": steps_ok,
            "wall_steps_planned": n_steps,
            "wall_blocked_next_step": sim_blocked,
            "wall_clearance_before_m": float(clearance0),
            "wall_clearance_after_m": final_clearance,
            "wall_clearance_after_signed_m": float(final_signed),
            "wall_step_clearances": step_clearances,
            "wall_binary_trace": binary_trace,
            "wall_contact_threshold_m": WALL_CONTACT_M,
            "wall_contact_gt": contact,
            "wall_contact_reason": ("clearance" if contact else "not_near_wall"),
            "wall_method": final_method,
            "wall_after_mask_pixels": after_px,
            "wall_mask_halve_depth": int(_mask_halve_depth),
            "wall_after_mask_min_pixels": int(min_after_mask_pixels),
        },
    )
    return move, event0, event1


def try_wall_direct_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    obj: dict[str, Any],
    *,
    toward_wall: bool = True,
    stop_margin: float = WALL_STOP_MARGIN_M,
    pure_cardinal: bool = False,
    motion_scale: float = 1.0,
    room_clearance_m: Optional[float] = None,
    push_xz: Optional[tuple[float, float]] = None,
) -> Optional[tuple[MoveAction, Any, Any]]:
    """One-shot wall slide from closed-form clearance (Mod 28 fast find_edit).

    Move ``clearance - stop_margin`` along the nearest sceneBounds axis (or away).
    Optionally lengthen along a pure camera cardinal by ``1/cos(theta)``.

    ``motion_scale`` (v10): ``1.0`` aims for contact; ``0.5`` is an intentional
    short toward-wall slide (``0.5 * clearance - CONTACT_HALF_MOVE_SUB_M``)
    that should leave clearance ≥ ``CONTACT_HALF_MIN_LEFTOVER_M``.
    """
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    clearance0, method, nearest_axis = _signed_edge_wall_clearance(event, obj)
    if (method == "none" or nearest_axis is None) and (
        push_xz is None or room_clearance_m is None
    ):
        return None

    eff_clear = (
        float(room_clearance_m)
        if room_clearance_m is not None
        else float(clearance0)
    )
    if toward_wall and eff_clear < -WALL_PENETRATION_TOL_M:
        return None

    axis = str(nearest_axis or "room_wall")
    if push_xz is not None:
        wall_unit = np.array([float(push_xz[0]), 0.0, float(push_xz[1])], dtype=np.float64)
        n = float(np.linalg.norm(wall_unit))
        if n < 1e-8:
            return None
        wall_unit /= n
        axis = "room_wall"
    else:
        axis = nearest_axis if toward_wall else opposite_wall_axis(nearest_axis)
        wall_unit = wall_axis_to_world_delta(axis)
    old_pos = dict(obj["position"])
    old_rot = dict(obj["rotation"])
    scale = float(motion_scale)
    if scale <= 0.0:
        return None

    if toward_wall:
        # Need room to stop short of the wall and still land in contact band
        # when scale≈1; for half-motion Nos, require expected leftover ≥
        # CONTACT_HALF_MIN_LEFTOVER_M after move ``0.5*clearance - sub``.
        if eff_clear <= float(stop_margin) + 1e-3:
            return None
        full_m = float(eff_clear) - float(stop_margin)
        if full_m < 0.05:
            return None
        if scale < 0.999:
            move_m = 0.5 * float(eff_clear) - float(CONTACT_HALF_MOVE_SUB_M)
            expected_clr = float(eff_clear) - float(move_m)
            if expected_clr < float(CONTACT_HALF_MIN_LEFTOVER_M):
                return None
        else:
            move_m = full_m * scale
        if move_m < 0.04:
            return None
    else:
        # Push away enough to leave contact; use at least ~0.35 m.
        move_m = max(0.35, float(abs(clearance0)) + 0.25) * scale

    delta = wall_unit * float(move_m)
    if pure_cardinal:
        # Reach same wall plane with a single camera-frame axis when aligned enough.
        scores = {
            "left": float(np.dot(wall_unit, -basis["right"])),
            "right": float(np.dot(wall_unit, basis["right"])),
            "forward": float(np.dot(wall_unit, basis["forward"])),
            "back": float(np.dot(wall_unit, -basis["forward"])),
        }
        best_dir = max(scores, key=scores.get)
        cos_t = float(scores[best_dir])
        if cos_t >= 0.55:
            pure = camera_delta_to_world(best_dir, 1.0, basis)
            pure_xz = np.array([pure[0], 0.0, pure[2]], dtype=np.float64)
            n = float(np.linalg.norm(pure_xz))
            if n > 1e-8:
                pure_xz /= n
                # Distance along pure axis to cover wall-normal displacement.
                move_m_pure = float(move_m) / max(cos_t, 0.55)
                delta = pure_xz * move_m_pure

    new_pos = {
        "x": float(old_pos["x"] + delta[0]),
        "y": float(old_pos["y"]),
        "z": float(old_pos["z"] + delta[2]),
    }
    ev = teleport_object(
        controller,
        obj["objectId"],
        new_pos,
        old_rot,
        force=False,
        allow_surface_place=False,
        kinematic=True,
    )
    if not ev.metadata["lastActionSuccess"]:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, obj["objectId"], new_pos, old_rot)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, obj["objectId"], new_pos, old_rot)

    if not still_visible(event1, obj["objectId"]):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    after = get_object(event1, obj["objectId"])
    if after is None:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    final_signed, final_method, _ = _signed_edge_wall_clearance(event1, after)
    if final_signed < -WALL_PENETRATION_TOL_M:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    after_px = object_mask_pixel_count(event1, obj["objectId"])
    if after_px is not None and after_px < int(WALL_AFTER_MASK_MIN_PIXELS):
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None

    final_clearance = max(0.0, float(final_signed))
    contact = final_clearance < WALL_CONTACT_M
    world_delta = np.array(
        [new_pos["x"] - old_pos["x"], 0.0, new_pos["z"] - old_pos["z"]],
        dtype=np.float64,
    )
    applied_m = float(np.linalg.norm(world_delta))
    if applied_m < 0.04:
        pin_object_pose(controller, obj["objectId"], old_pos, old_rot)
        return None
    direction = world_delta_to_camera_direction(world_delta, basis)
    move = _make_move(
        object_id=obj["objectId"],
        object_type=obj["objectType"],
        direction=direction,
        meters=applied_m,
        world_delta=world_delta,
        old_pos=old_pos,
        moved_pos=new_pos,
        old_rot=old_rot,
        distance_before=float(obj.get("distance") or 2.0),
        basis=basis,
        edit_meta={
            "wall_contact_gt": bool(contact),
            "wall_clearance_before": float(max(0.0, clearance0)),
            "wall_clearance_after": float(final_clearance),
            "wall_axis": axis,
            "wall_toward": bool(toward_wall),
            "wall_method": final_method or method,
            "wall_direct": True,
            "wall_stop_margin_m": float(stop_margin),
            "wall_pure_cardinal": bool(pure_cardinal),
            "wall_motion_scale": float(scale),
            "wall_half_motion_no": bool(scale < 0.999),
            "wall_full_planned_m": float(eff_clear) - float(stop_margin)
            if toward_wall
            else None,
            "wall_room_polygon": push_xz is not None,
        },
    )
    return move, event0, event1


def _contact_half_viable(clearance_m: float) -> bool:
    half_move = 0.5 * float(clearance_m) - float(CONTACT_HALF_MOVE_SUB_M)
    expected = float(clearance_m) - half_move
    return expected >= float(CONTACT_HALF_MIN_LEFTOVER_M) and float(clearance_m) > float(
        WALL_CONTACT_M
    )


def _contact_rank_score(
    event: Any,
    obj: dict[str, Any],
    clearance_m: float,
    *,
    want_half: bool,
) -> float:
    """Higher = try first. Soft size/visibility; clearance sweet-spot by polarity.

    Not a hard filter: large or low-pixel objects can still be chosen if they
    sit in a good wall-clearance band. Full Yes prefers objects already near
    the wall; half-No prefers a bit more leftover so the label stays No.
    """
    fp = aabb_footprint(obj)
    fp_m = float(fp) if fp is not None and fp == fp else 0.6
    px = object_mask_pixel_count(event, obj["objectId"])
    px_n = float(px) if px is not None else 0.0
    target = 0.85 if want_half else 0.40
    clr_w = math.exp(-((float(clearance_m) - target) ** 2) / (2.0 * 0.50**2))
    size_w = 1.0 / (0.08 + max(0.02, fp_m))
    vis_w = math.log1p(px_n)
    return 3.0 * clr_w + 1.2 * size_w + 0.35 * vis_w


def _rank_contact_entries(
    event: Any,
    entries: list[tuple[float, dict[str, Any], tuple[float, float]]],
    *,
    want_half: bool,
    max_n: int,
) -> list[tuple[float, dict[str, Any], tuple[float, float]]]:
    scored = sorted(
        entries,
        key=lambda e: _contact_rank_score(event, e[1], e[0], want_half=want_half),
        reverse=True,
    )
    return scored[: max(1, int(max_n))]


def _room_wall_contact_candidates(
    event: Any,
    rooms: list[dict[str, Any]],
    room_id: str,
    *,
    max_near_m: float,
) -> list[tuple[float, dict[str, Any], tuple[float, float]]]:
    """Visible, referable movers inside the room polygon near a wall.

    Contact-wall only: no distance-band / size / visual-pixel hard cuts.
    Bathtubs and other structural types stay banned via referring + type bans.
    """
    from cit_pipeline.procthor_rooms import (
        object_room_wall_clearance,
        point_in_floor_polygon,
        room_by_id,
        room_floor_wall_segments,
    )

    room = room_by_id(rooms, room_id)
    if room is None:
        return []
    poly = room.get("floorPolygon") or []
    segs = room_floor_wall_segments(room)
    if not poly or not segs:
        return []
    out: list[tuple[float, dict[str, Any], tuple[float, float]]] = []
    for obj in event.metadata["objects"]:
        if not obj.get("visible"):
            continue
        if _mover_forbidden(obj):
            continue
        if not instance_referring_allowed(event, obj["objectId"], obj["objectType"]):
            continue
        pos = obj.get("position") or {}
        if not point_in_floor_polygon(float(pos.get("x") or 0), float(pos.get("z") or 0), poly):
            continue
        clr, push = object_room_wall_clearance(obj, segs)
        if clr >= float(max_near_m) or clr <= float(WALL_CONTACT_M):
            continue
        out.append((float(clr), obj, push))
    return out


def _try_room_wall_direct(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    obj: dict[str, Any],
    clearance: float,
    push: tuple[float, float],
    *,
    motion_scale: float,
) -> Optional[tuple[MoveAction, Any, Any]]:
    for pure in (False, True):
        result = try_wall_direct_edit(
            controller,
            cam,
            event,
            obj,
            toward_wall=True,
            pure_cardinal=pure,
            motion_scale=motion_scale,
            room_clearance_m=clearance,
            push_xz=push,
        )
        if result is not None:
            return result
        pin_object_pose(
            controller,
            obj["objectId"],
            dict(obj["position"]),
            dict(obj["rotation"]),
        )
    return None


def find_wall_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    max_tries: int = 40,
    rng: Optional[random.Random] = None,
    som: bool = False,
    prefer_contact: Optional[bool] = None,
    fast_find_edit: bool = False,
    wall_away_prob: Optional[float] = None,
    select_near_m: Optional[float] = None,
    always_toward: Optional[bool] = None,
    half_motion_no_frac: Optional[float] = None,
    procthor_rooms: Optional[list[dict[str, Any]]] = None,
    camera_room_id: Optional[str] = None,
    contact_room_wall_prefilter: Optional[bool] = None,
    contact_room_wall_max_m: Optional[float] = None,
    contact_fast_max_candidates: Optional[int] = None,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Near-wall objects; multi-step slide toward or away from wall (Mod 5 / 8 / 24).

    Modification 8: select within ``WALL_SELECT_NEAR_M`` (1.5 m). Crowded types
    remain eligible via leftmost/rightmost referring.

    Mod 24: ``prefer_contact`` (from running yes/no balance) overrides the
    random ``WALL_AWAY_PROB`` coin-flip so we intentionally hunt for "yes"
    pairs most of the time. Toward-wall search is the positive-pair algorithm.

    Mod 28 ``fast_find_edit``: closed-form one-shot slide (clearance − margin).

    Mod 29: ``wall_away_prob`` / ``select_near_m`` override globals (iTHOR balance).

    v10: ``always_toward`` (default ``CONTACT_ALWAYS_TOWARD``) disables away
    slides. Intentional Nos use ``motion_scale=0.5`` (~``half_motion_no_frac``);
    full slides aim for contact but may fail → No (helps balance).
    """
    rng = rng or random.Random()
    use_always_toward = bool(
        CONTACT_ALWAYS_TOWARD if always_toward is None else always_toward
    )
    half_frac = float(
        CONTACT_HALF_MOTION_NO_FRAC
        if half_motion_no_frac is None
        else half_motion_no_frac
    )

    if use_always_toward:
        toward_wall = True
        # Coin flip once per call; applied after a viable full slide is found so
        # half-Nos are not over-represented via survivor bias (full teleports
        # collide more often than half).
        want_half = rng.random() < half_frac
        want_full = not want_half
        motion_scale = 1.0  # search with full; may downscale below
    else:
        away_prob = float(WALL_AWAY_PROB if wall_away_prob is None else wall_away_prob)
        if prefer_contact is None:
            toward_wall = rng.random() >= away_prob
        else:
            toward_wall = bool(prefer_contact)
        want_half = False
        want_full = True
        motion_scale = 1.0

    prefer_contact_flag = toward_wall and want_full

    if select_near_m is not None:
        select_near = float(select_near_m)
    else:
        select_near = float(WALL_SELECT_NEAR_M_FAST if fast_find_edit else WALL_SELECT_NEAR_M)
    select_near = max(
        select_near,
        float(
            CONTACT_ROOM_WALL_MAX_M
            if contact_room_wall_max_m is None
            else contact_room_wall_max_m
        ),
    )

    use_room = bool(
        fast_find_edit
        and use_always_toward
        and procthor_rooms
        and camera_room_id
        and (
            CONTACT_ROOM_WALL_PREFILTER
            if contact_room_wall_prefilter is None
            else contact_room_wall_prefilter
        )
    )
    if use_room:
        max_near = float(
            CONTACT_ROOM_WALL_MAX_M
            if contact_room_wall_max_m is None
            else contact_room_wall_max_m
        )
        max_cand = int(
            CONTACT_FAST_MAX_CANDIDATES
            if contact_fast_max_candidates is None
            else contact_fast_max_candidates
        )
        room_entries = _room_wall_contact_candidates(
            event,
            procthor_rooms,
            str(camera_room_id),
            max_near_m=max_near,
        )
        if room_entries:
            pool = _rank_contact_entries(
                event, room_entries, want_half=want_half, max_n=max_cand
            )
            if want_half:
                for clearance, obj, push in pool:
                    got = _try_room_wall_direct(
                        controller,
                        cam,
                        event,
                        obj,
                        clearance,
                        push,
                        motion_scale=0.5,
                    )
                    if got is None:
                        continue
                    move, event0, event1 = got
                    if not bool(move.edit_meta.get("wall_contact_gt")):
                        return (
                            move,
                            event0,
                            event1,
                            "category:contact_wall+room+toward+half_no",
                        )
            else:
                for clearance, obj, push in pool:
                    got = _try_room_wall_direct(
                        controller,
                        cam,
                        event,
                        obj,
                        clearance,
                        push,
                        motion_scale=1.0,
                    )
                    if got is None:
                        continue
                    move, event0, event1 = got
                    got_contact = bool(move.edit_meta.get("wall_contact_gt"))
                    note = (
                        "category:contact_wall+room+toward+yes"
                        if got_contact
                        else "category:contact_wall+room+toward+missed"
                    )
                    return move, event0, event1, note
            # Ranked room-polygon tries exhausted → fall through to sceneBounds.

    cands: list[tuple[float, dict[str, Any]]] = []
    for obj in event.metadata["objects"]:
        if not obj.get("visible"):
            continue
        if _mover_forbidden(obj):
            continue
        if not instance_referring_allowed(event, obj["objectId"], obj["objectType"]):
            continue
        clearance, method, axis = _min_edge_wall_clearance(event, obj)
        if method == "none" or axis is None or clearance >= select_near:
            continue
        # Toward-wall Yes: prefer not already contacting so the move is meaningful.
        if toward_wall and clearance <= WALL_CONTACT_M:
            continue
        cands.append((clearance, obj))
    # Rank by polarity sweet-spot + soft size/visibility (not closest-2 only).
    max_cand_fb = int(
        CONTACT_FAST_MAX_CANDIDATES
        if contact_fast_max_candidates is None
        else contact_fast_max_candidates
    )
    cands.sort(
        key=lambda t: _contact_rank_score(event, t[1], t[0], want_half=want_half),
        reverse=True,
    )
    cands = cands[: max(1, max_cand_fb)]

    if fast_find_edit:
        tries = 0
        for clearance, obj in cands:
            if tries >= max_tries:
                break
            tries += 1
            # Prefer compound perpendicular; fall back to pure cardinal once.
            for pure in (False, True):
                if use_always_toward:
                    # Always validate a full toward slide first (same object /
                    # geometry). Then with want_half, re-execute at 0.5 so the
                    # intentional-No rate tracks half_frac instead of teleport
                    # failure rates.
                    full = try_wall_direct_edit(
                        controller,
                        cam,
                        event,
                        obj,
                        toward_wall=True,
                        pure_cardinal=pure,
                        motion_scale=1.0,
                    )
                    if full is None:
                        pin_object_pose(
                            controller,
                            obj["objectId"],
                            dict(obj["position"]),
                            dict(obj["rotation"]),
                        )
                        continue
                    move_f, event0_f, event1_f = full
                    if want_half:
                        pin_object_pose(
                            controller,
                            obj["objectId"],
                            dict(obj["position"]),
                            dict(obj["rotation"]),
                        )
                        # Enough clearance that half stays clearly non-contact.
                        half_move = 0.5 * float(clearance) - float(
                            CONTACT_HALF_MOVE_SUB_M
                        )
                        expected = float(clearance) - half_move
                        if expected < float(CONTACT_HALF_MIN_LEFTOVER_M):
                            # Cannot safely make a bound-No on this object —
                            # try the next candidate (do NOT demote to full Yes).
                            continue
                        half = try_wall_direct_edit(
                            controller,
                            cam,
                            event,
                            obj,
                            toward_wall=True,
                            pure_cardinal=pure,
                            motion_scale=0.5,
                        )
                        if half is not None:
                            move_h, event0_h, event1_h = half
                            if not bool(move_h.edit_meta.get("wall_contact_gt")):
                                return (
                                    move_h,
                                    event0_h,
                                    event1_h,
                                    "category:contact_wall+direct+toward+half_no",
                                )
                            pin_object_pose(
                                controller,
                                obj["objectId"],
                                dict(obj["position"]),
                                dict(obj["rotation"]),
                            )
                        # Half failed or accidentally contacted → next object.
                        continue
                    got_contact = bool(move_f.edit_meta.get("wall_contact_gt"))
                    note = (
                        "category:contact_wall+direct+toward+yes"
                        if got_contact
                        else "category:contact_wall+direct+toward+missed"
                    )
                    return move_f, event0_f, event1_f, note

                result = try_wall_direct_edit(
                    controller,
                    cam,
                    event,
                    obj,
                    toward_wall=toward_wall,
                    pure_cardinal=pure,
                    motion_scale=motion_scale,
                )
                if result is None:
                    pin_object_pose(
                        controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
                    )
                    continue
                move, event0, event1 = result
                got_contact = bool(move.edit_meta.get("wall_contact_gt"))
                if prefer_contact_flag == got_contact:
                    note = (
                        "category:contact_wall+direct+toward"
                        if toward_wall
                        else "category:contact_wall+direct+away"
                    )
                    return move, event0, event1, note
                pin_object_pose(
                    controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
                )
        return None

    def _try_one(obj: dict[str, Any], total: float) -> Optional[tuple[MoveAction, Any, Any]]:
        return try_wall_multistep_edit(
            controller,
            cam,
            event,
            obj,
            total_meters=float(total) * float(motion_scale),
            toward_wall=toward_wall,
        )

    tries = 0
    for clearance, obj in cands:
        if toward_wall:
            totals = sorted(
                WALL_MOVE_METERS,
                key=lambda t: (0 if float(t) >= float(clearance) - 0.05 else 1, -float(t)),
            )
        else:
            # Away: longer slides more reliably leave contact / stay non-contact.
            totals = sorted(WALL_MOVE_METERS, reverse=True)
        for total in totals:
            if tries >= max_tries:
                return None
            tries += 1
            result = _try_one(obj, float(total))
            if result is None:
                pin_object_pose(
                    controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
                )
                continue
            move, event0, event1 = result
            got_contact = bool(move.edit_meta.get("wall_contact_gt"))
            move.edit_meta["wall_motion_scale"] = float(motion_scale)
            move.edit_meta["wall_half_motion_no"] = bool(motion_scale < 0.999)
            if use_always_toward:
                if motion_scale < 0.999:
                    if got_contact:
                        pin_object_pose(
                            controller,
                            obj["objectId"],
                            dict(obj["position"]),
                            dict(obj["rotation"]),
                        )
                        continue
                    return move, event0, event1, "category:contact_wall+toward+half_no"
                note = (
                    "category:contact_wall+toward+yes"
                    if got_contact
                    else "category:contact_wall+toward+missed"
                )
                return move, event0, event1, note
            if prefer_contact_flag and got_contact:
                note = "category:contact_wall+toward"
                return move, event0, event1, note
            if (not prefer_contact_flag) and (not got_contact):
                note = "category:contact_wall+away"
                return move, event0, event1, note
            pin_object_pose(
                controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
            )

    # Second pass: only accept edits matching the intended polarity.
    # Never keep a polarity_miss (toward-wall intent with non-contact GT, etc.).
    # Skipped under always_toward (first pass already accepts Yes/No outcomes).
    if use_always_toward:
        return None
    tries = 0
    for clearance, obj in cands:
        for total in sorted(WALL_MOVE_METERS, reverse=True):
            if tries >= max_tries:
                return None
            tries += 1
            result = _try_one(obj, float(total))
            if result is None:
                pin_object_pose(
                    controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
                )
                continue
            move, event0, event1 = result
            got_contact = bool(move.edit_meta.get("wall_contact_gt"))
            if prefer_contact_flag != got_contact:
                pin_object_pose(
                    controller, obj["objectId"], dict(obj["position"]), dict(obj["rotation"])
                )
                continue
            note = (
                "category:contact_wall+toward"
                if toward_wall
                else "category:contact_wall+away"
            )
            return move, event0, event1, note
    return None


def plan_cover_move_from_masks(
    event: Any,
    mover: dict[str, Any],
    occludee: dict[str, Any],
) -> Optional[tuple[str, float, dict[str, Any]]]:
    """Plan a camera-frame slide from rgb0 masks so A's centroid covers B.

    Pixel Δu between mask centroids → meters at A's depth. Also records how much
    A should be pulled toward the camera so it ends up in front of B.
    """
    mover_mask, method_m = get_object_mask(event, mover["objectId"])
    occ_mask, method_o = get_object_mask(event, occludee["objectId"])
    if mover_mask is None or occ_mask is None:
        return None
    ca = _mask_centroid_uv(mover_mask)
    cb = _mask_centroid_uv(occ_mask)
    if ca is None or cb is None:
        return None
    au, av, a_area = ca
    bu, bv, b_area = cb
    if a_area < b_area * 0.5:
        return None

    depth_a = float(mover.get("distance") or 2.0)
    depth_b = float(occludee.get("distance") or depth_a)
    mpp = _meters_per_pixel_horizontal(depth_a)
    du = float(bu - au)  # + → B is right of A → move A right
    dv = float(bv - av)
    # Need enough lateral shift that A's body covers B (use centroid gap).
    lateral_m = abs(du) * mpp
    # If already nearly aligned horizontally, still allow a small nudge + back pull.
    if lateral_m < 0.05 and abs(du) < 8.0:
        lateral_m = max(lateral_m, OCCLUDE_COVER_METERS_MIN * 0.5)

    if abs(du) >= 6.0 or lateral_m >= 0.08:
        direction = "right" if du > 0 else "left"
        meters = float(
            np.clip(max(lateral_m, OCCLUDE_COVER_METERS_MIN), OCCLUDE_COVER_METERS_MIN, OCCLUDE_COVER_METERS_MAX)
        )
    else:
        # Mostly depth / vertical: slide A toward camera ("back") to cover B.
        direction = "back"
        meters = float(
            np.clip(
                max(0.15, depth_a - depth_b + 0.2, abs(dv) * mpp),
                OCCLUDE_COVER_METERS_MIN,
                OCCLUDE_COVER_METERS_MAX,
            )
        )

    # Match B height when A is much higher (may lower); never raise A (Mod 10).
    y_a = float(mover["position"]["y"])
    y_b = float(occludee["position"]["y"])
    height_diff = abs(y_b - y_a)
    match_height = height_diff > 0.15
    target_y = min(y_a, y_b) if match_height else y_a

    meta = {
        "mask_method_rgb0": method_m if method_m == method_o else "mixed",
        "centroid_a_uv": [au, av],
        "centroid_b_uv": [bu, bv],
        "pixel_du": du,
        "pixel_dv": dv,
        "meters_per_pixel": mpp,
        "planned_direction": direction,
        "planned_meters": meters,
        "mover_mask_area": a_area,
        "occludee_mask_area": b_area,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "match_height": match_height,
        "target_y": target_y,
        "plan": "pixel_centroid_cover",
        "no_upward": True,
    }
    return direction, meters, meta


def plan_occlusion_ray_target(
    event: Any,
    mover: dict[str, Any],
    occludee: dict[str, Any],
    *,
    prefer_yes: bool = True,
    front_margin: float = OCCLUDE_RAY_FRONT_MARGIN_M,
    no_offset_m: float = OCCLUDE_RAY_NO_OFFSET_M,
) -> Optional[tuple[dict[str, float], dict[str, Any]]]:
    """Place A on the camera→B ray in front of B (Mod 28 closed-form occlusion).

    Foot of perpendicular from A onto the ray, then pull in front of B if needed.
    For prefer_no, offset laterally off the ray so cover stays a near-miss.
    """
    agent = event.metadata["agent"]
    cam = event.metadata.get("cameraPosition") or agent["position"]
    C = np.array([float(cam["x"]), float(cam["y"]), float(cam["z"])], dtype=np.float64)
    A = np.array(
        [
            float(mover["position"]["x"]),
            float(mover["position"]["y"]),
            float(mover["position"]["z"]),
        ],
        dtype=np.float64,
    )
    B = np.array(
        [
            float(occludee["position"]["x"]),
            float(occludee["position"]["y"]),
            float(occludee["position"]["z"]),
        ],
        dtype=np.float64,
    )
    ray = B - C
    dist_b = float(np.linalg.norm(ray))
    if dist_b < 0.25:
        return None
    dhat = ray / dist_b
    t_a = float(np.dot(A - C, dhat))
    t_front = dist_b - float(front_margin)
    if t_front < 0.35:
        return None
    # On-ray depth: prefer staying near A's depth when already in front of B.
    if t_a > 0.35 and t_a < t_front:
        t_target = t_a
    else:
        t_target = t_front
    A_new = C + t_target * dhat
    if not prefer_yes:
        # Lateral offset in world XZ perpendicular to ray XZ.
        right = np.array([-dhat[2], 0.0, dhat[0]], dtype=np.float64)
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= rn
        A_new = A_new + right * float(no_offset_m)
    # Never raise the occluder.
    new_pos = {
        "x": float(A_new[0]),
        "y": float(min(float(mover["position"]["y"]), float(A_new[1]), float(occludee["position"]["y"]))),
        "z": float(A_new[2]),
    }
    meta = {
        "plan": "camera_ray_cover",
        "ray_t_a": t_a,
        "ray_t_target": t_target,
        "ray_dist_b": dist_b,
        "front_margin_m": float(front_margin),
        "prefer_yes": bool(prefer_yes),
        "no_offset_m": float(no_offset_m) if not prefer_yes else 0.0,
        "camera_xyz": [float(C[0]), float(C[1]), float(C[2])],
        "target_xyz": [new_pos["x"], new_pos["y"], new_pos["z"]],
        "no_upward": True,
    }
    return new_pos, meta


def try_occlusion_ray_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    mover: dict[str, Any],
    occludee: dict[str, Any],
    *,
    prefer_yes: bool = True,
) -> Optional[tuple[MoveAction, Any, Any]]:
    """Single teleport to ray-planned cover pose; measure cover once."""
    planned = plan_occlusion_ray_target(event, mover, occludee, prefer_yes=prefer_yes)
    if planned is None:
        return None
    new_pos, plan_meta = planned
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    old_pos = dict(mover["position"])
    old_rot = dict(mover["rotation"])
    world_delta = np.array(
        [
            new_pos["x"] - old_pos["x"],
            new_pos["y"] - old_pos["y"],
            new_pos["z"] - old_pos["z"],
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(world_delta[[0, 2]])) < 0.04:
        return None

    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)

    ev = teleport_object(
        controller,
        mover["objectId"],
        new_pos,
        old_rot,
        force=False,
        allow_surface_place=False,
        kinematic=True,
    )
    if not ev.metadata["lastActionSuccess"]:
        pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
        return None

    pin_object_pose(controller, mover["objectId"], new_pos, old_rot)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, mover["objectId"], new_pos, old_rot)

    if not still_visible(event1, mover["objectId"]):
        pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
        return None

    cover, remaining, method, intersects = occlusion_cover_rate(
        event0, event1, mover["objectId"], occludee["objectId"]
    )
    m_xz = float(math.hypot(world_delta[0], world_delta[2]))
    direction = world_delta_to_camera_direction(world_delta, basis)
    meters = float(max(m_xz, OCCLUDE_COVER_METERS_MIN))
    is_occ = bool(cover >= OCCLUDE_THRESHOLD)
    move = _make_move(
        object_id=mover["objectId"],
        object_type=mover["objectType"],
        direction=direction,
        meters=meters,
        world_delta=world_delta,
        old_pos=old_pos,
        moved_pos=new_pos,
        old_rot=old_rot,
        distance_before=float(mover.get("distance") or 2.0),
        basis=basis,
        edit_meta={
            "occlusion_cover": True,
            "occlusion_ray": True,
            "occludee_id": occludee["objectId"],
            "occludee_type": occludee["objectType"],
            "occlusion_rate_rgb1": cover,
            "occlusion_remaining_b": remaining,
            "occlusion_intersects_rgb1": intersects,
            "occlusion_mask_method_rgb1": method,
            "occlusion_threshold": OCCLUDE_THRESHOLD,
            "occlusion_is_occluded": is_occ,
            "occlusion_metric": "cover_A1_of_B0",
            **plan_meta,
        },
    )
    return move, event0, event1


def try_occlusion_cover_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    mover: dict[str, Any],
    occludee: dict[str, Any],
    direction: str,
    meters: float,
    plan_meta: dict[str, Any],
) -> Optional[tuple[MoveAction, Any, Any]]:
    """Search scale × back-pull; keep the pose with highest cover of B_rgb0 by A_rgb1."""
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    old_pos = dict(mover["position"])
    old_rot = dict(mover["rotation"])
    target_y = float(plan_meta.get("target_y", old_pos["y"]))
    base_dir = str(plan_meta.get("planned_direction") or direction)
    base_m = float(plan_meta.get("planned_meters") or meters)

    # Freeze rgb0 once for cover metric (B's original mask region).
    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)

    best: Optional[tuple[float, float, dict[str, float], np.ndarray, str, bool]] = None
    # (cover, remaining, new_pos, world_delta, method, intersects)

    for scale in OCCLUDE_COVER_SCALE:
        lat_m = float(np.clip(base_m * float(scale), OCCLUDE_COVER_METERS_MIN, OCCLUDE_COVER_METERS_MAX))
        for back_m in OCCLUDE_COVER_BACK_M:
            if base_dir in ("left", "right"):
                delta = camera_delta_to_world(base_dir, lat_m, basis)
                if back_m > 1e-6:
                    delta = delta + camera_delta_to_world("back", float(back_m), basis)
            elif base_dir == "back":
                delta = camera_delta_to_world("back", lat_m + float(back_m), basis)
            else:
                delta = camera_delta_to_world(base_dir, lat_m, basis)
                if back_m > 1e-6:
                    delta = delta + camera_delta_to_world("back", float(back_m), basis)

            new_pos = {
                "x": float(old_pos["x"] + delta[0]),
                # Modification 10: never raise the occluder (no upward moves).
                # Lateral / back slides only; may lower to match B but not lift.
                "y": float(min(float(old_pos["y"]), float(target_y))),
                "z": float(old_pos["z"] + delta[2]),
            }
            # Keep lateral/back delta y from camera basis out; height is explicit.
            world_delta = np.array(
                [new_pos["x"] - old_pos["x"], new_pos["y"] - old_pos["y"], new_pos["z"] - old_pos["z"]],
                dtype=np.float64,
            )
            # Modification 10: collision-checked cover (no force through walls).
            ev = teleport_object(
                controller,
                mover["objectId"],
                new_pos,
                old_rot,
                force=False,
                allow_surface_place=False,
                kinematic=True,
            )
            if not ev.metadata["lastActionSuccess"]:
                continue
            pin_object_pose(controller, mover["objectId"], new_pos, old_rot)
            ev = teleport_full(controller, **cam.teleport_kwargs())
            pin_object_pose(controller, mover["objectId"], new_pos, old_rot)
            cover, remaining, method, intersects = occlusion_cover_rate(
                event0, ev, mover["objectId"], occludee["objectId"]
            )
            if best is None or cover > best[0] or (cover == best[0] and remaining < best[1]):
                best = (float(cover), float(remaining), dict(new_pos), world_delta, method, bool(intersects))

    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
    if best is None:
        return None
    cover, remaining, new_pos, world_delta, method, intersects = best
    m_xz = float(math.hypot(world_delta[0], world_delta[2]))
    if m_xz < 0.05 and cover < OCCLUDE_THRESHOLD:
        return None

    direction = world_delta_to_camera_direction(world_delta, basis)
    meters = float(max(m_xz, OCCLUDE_COVER_METERS_MIN))

    pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, mover["objectId"], new_pos, old_rot)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    pin_object_pose(controller, mover["objectId"], new_pos, old_rot)
    cover, remaining, method, intersects = occlusion_cover_rate(
        event0, event1, mover["objectId"], occludee["objectId"]
    )

    if not still_visible(event1, mover["objectId"]):
        pin_object_pose(controller, mover["objectId"], old_pos, old_rot)
        return None

    is_occ = bool(cover >= OCCLUDE_THRESHOLD)
    move = _make_move(
        object_id=mover["objectId"],
        object_type=mover["objectType"],
        direction=direction,
        meters=meters,
        world_delta=world_delta,
        old_pos=old_pos,
        moved_pos=new_pos,
        old_rot=old_rot,
        distance_before=float(mover.get("distance") or 2.0),
        basis=basis,
        edit_meta={
            "occlusion_cover": True,
            "occludee_id": occludee["objectId"],
            "occludee_type": occludee["objectType"],
            "occlusion_rate_rgb1": cover,
            "occlusion_remaining_b": remaining,
            "occlusion_intersects_rgb1": intersects,
            "occlusion_mask_method_rgb1": method,
            "occlusion_threshold": OCCLUDE_THRESHOLD,
            "occlusion_is_occluded": is_occ,
            "occlusion_metric": "cover_A1_of_B0",
            **plan_meta,
            "cover_target_xyz": [new_pos["x"], new_pos["y"], new_pos["z"]],
            "cover_scales": list(OCCLUDE_COVER_SCALE),
            "cover_back_m": list(OCCLUDE_COVER_BACK_M),
        },
    )
    return move, event0, event1


def find_occlusion_cover_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    max_tries: int = 48,
    prefer_yes: bool = True,
    som: bool = False,
    fast_find_edit: bool = False,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Large A + nearby small B; pixel-planned cover move (Modification 5).

    Mod 28 ``fast_find_edit``: one camera-ray teleport per pair (no scale×back grid).
    """
    max_fp = scene_table_footprint_limit(event)
    occ_min_px = SOM_OCCLUDEE_MASK_MIN_PIXELS if som else OCCLUDEE_MASK_MIN_PIXELS
    movers = []
    for obj in event.metadata["objects"]:
        if not obj.get("visible"):
            continue
        d = obj.get("distance")
        if d is None or not (dist_min < float(d) < dist_max):
            continue
        if obj["objectType"] not in LARGE_OCCLUDER_TYPES:
            continue
        if _mover_forbidden(obj):
            continue
        if not referring_allowed(event, obj["objectType"]):
            continue
        if not is_small_enough_to_move(obj, max_fp, event, som=som):
            continue
        if not _mover_visually_ok(event, obj, som=som):
            continue
        fp = aabb_footprint(obj) or 0.0
        movers.append((fp, obj))
    movers.sort(key=lambda t: -t[0])

    smalls = [
        o
        for o in event.metadata["objects"]
        if o.get("visible")
        and o["objectType"] in SMALL_OCCLUDEE_TYPES
        and referring_allowed(event, o["objectType"])
        # Modification 9 / 12: speck floor; SoM uses a lower pixel floor.
        and is_visible_mask_large_enough(event, o, min_pixels=occ_min_px)
    ]

    if fast_find_edit:
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for _, mover in movers:
            d_a = float(mover.get("distance") or 99.0)
            y_a = float(mover["position"]["y"])
            for occ in smalls:
                if occ["objectId"] == mover["objectId"]:
                    continue
                d_b = float(occ.get("distance") or 99.0)
                y_b = float(occ["position"]["y"])
                dist = math.dist(
                    (mover["position"]["x"], mover["position"]["y"], mover["position"]["z"]),
                    (occ["position"]["x"], occ["position"]["y"], occ["position"]["z"]),
                )
                if dist > OCCLUDE_PAIR_MAX_DIST_M:
                    continue
                if abs(y_b - y_a) > OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M:
                    continue
                score = (
                    (OCCLUDE_PAIR_MAX_DIST_M - dist)
                    + (OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M - abs(y_b - y_a)) * 0.8
                    + max(0.0, d_b - d_a) * 0.5
                )
                candidates.append((score, mover, occ))
        candidates.sort(key=lambda t: -t[0])
        yes_fallback = None
        no_fallback = None
        tries = 0
        for _, mover, occ in candidates:
            if tries >= max_tries:
                break
            tries += 1
            result = try_occlusion_ray_edit(
                controller, cam, event, mover, occ, prefer_yes=prefer_yes
            )
            if result is None:
                pin_object_pose(
                    controller, mover["objectId"], dict(mover["position"]), dict(mover["rotation"])
                )
                continue
            move, event0, event1 = result
            note = "category:occlusion+ray_plan"
            if move.edit_meta.get("occlusion_is_occluded"):
                if prefer_yes:
                    return move, event0, event1, note
                yes_fallback = (move, event0, event1, note)
            else:
                if not prefer_yes:
                    return move, event0, event1, note
                if no_fallback is None or float(
                    move.edit_meta.get("occlusion_rate_rgb1") or 0
                ) > float(no_fallback[0].edit_meta.get("occlusion_rate_rgb1") or 0):
                    no_fallback = (move, event0, event1, note)
            pin_object_pose(
                controller, mover["objectId"], dict(mover["position"]), dict(mover["rotation"])
            )
        if prefer_yes:
            return yes_fallback
        return no_fallback

    candidates_grid: list[
        tuple[float, dict[str, Any], dict[str, Any], str, float, dict[str, Any]]
    ] = []
    for _, mover in movers:
        d_a = float(mover.get("distance") or 99.0)
        y_a = float(mover["position"]["y"])
        for occ in smalls:
            if occ["objectId"] == mover["objectId"]:
                continue
            d_b = float(occ.get("distance") or 99.0)
            y_b = float(occ["position"]["y"])
            dist = math.dist(
                (mover["position"]["x"], mover["position"]["y"], mover["position"]["z"]),
                (occ["position"]["x"], occ["position"]["y"], occ["position"]["z"]),
            )
            if dist > OCCLUDE_PAIR_MAX_DIST_M:
                continue
            if abs(y_b - y_a) > OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M:
                continue
            planned = plan_cover_move_from_masks(event, mover, occ)
            if planned is None:
                continue
            direction, meters, meta = planned
            meta = dict(meta)
            pixel_gap = abs(float(meta.get("pixel_du") or 0.0))
            # Prefer nearby pairs, similar height, A already closer, moderate pixel gap.
            score = (
                (OCCLUDE_PAIR_MAX_DIST_M - dist)
                + (OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M - abs(y_b - y_a)) * 0.8
                + max(0.0, d_b - d_a) * 0.5
                - abs(pixel_gap - 80.0) * 0.01
                + float(meta.get("mover_mask_area") or 0.0) / max(float(meta.get("occludee_mask_area") or 1.0), 1.0) * 0.05
            )
            candidates_grid.append((score, mover, occ, direction, meters, meta))
    candidates_grid.sort(key=lambda t: -t[0])

    yes_fallback: Optional[tuple[MoveAction, Any, Any, str]] = None
    no_fallback: Optional[tuple[MoveAction, Any, Any, str]] = None
    tries = 0
    for _, mover, occ, direction, meters, meta in candidates_grid:
        if tries >= max_tries:
            break
        tries += 1
        result = try_occlusion_cover_edit(
            controller, cam, event, mover, occ, direction, meters, meta
        )
        if result is None:
            pin_object_pose(
                controller, mover["objectId"], dict(mover["position"]), dict(mover["rotation"])
            )
            continue
        move, event0, event1 = result
        note = "category:occlusion+cover_plan"
        if move.edit_meta.get("occlusion_is_occluded"):
            if prefer_yes:
                return move, event0, event1, note
            yes_fallback = (move, event0, event1, note)
        else:
            if not prefer_yes:
                return move, event0, event1, note
            if no_fallback is None or float(move.edit_meta.get("occlusion_rate_rgb1") or 0) > float(
                no_fallback[0].edit_meta.get("occlusion_rate_rgb1") or 0
            ):
                no_fallback = (move, event0, event1, note)
        pin_object_pose(
            controller, mover["objectId"], dict(mover["position"]), dict(mover["rotation"])
        )

    if prefer_yes:
        return yes_fallback
    return no_fallback


def _visible_moveable_chairs(
    event: Any,
    *,
    dist_min: float,
    dist_max: float,
    max_fp: float,
    som: bool = False,
) -> list[dict[str, Any]]:
    """Visible chairs in the distance band.

    Modification 8: do **not** apply the table-footprint size filter — chairs are
    often larger than a tiny SideTable, which would zero out affordance.

    Modification 9: still require on-screen mask ≥ ``MOVER_MASK_MIN_PIXELS``.
    Modification 12: SoM uses ``SOM_MOVER_MASK_MIN_PIXELS``.
    """
    del max_fp  # retained for call-site compatibility
    min_px = SOM_MOVER_MASK_MIN_PIXELS if som else None
    out: list[dict[str, Any]] = []
    for obj in event.metadata["objects"]:
        if obj.get("objectType") not in AFFORDANCE_CHAIR_TYPES:
            continue
        if not obj.get("visible"):
            continue
        d = obj.get("distance")
        if d is None or not (dist_min < float(d) < dist_max):
            continue
        if min_px is None:
            if not is_visible_mask_large_enough(event, obj):
                continue
        elif not is_visible_mask_large_enough(event, obj, min_pixels=min_px):
            continue
        out.append(obj)
    return out


def _affordance_seat_chairs(event: Any, chairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chairs we may sit on / move (Modification 11).

    Always keep only the leftmost and rightmost of each type when there are 2+
    chairs, so questions never say \"the 4th chair from the left\".
    """
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chairs:
        by_type[str(c["objectType"])].append(c)
    seats: list[dict[str, Any]] = []
    for objs in by_type.values():
        objs_sorted = sorted(objs, key=lambda o: camera_frame_x(event, o["position"]))
        n = len(objs_sorted)
        if n <= 1:
            seats.extend(objs_sorted)
        else:
            seats.append(objs_sorted[0])
            if objs_sorted[-1]["objectId"] != objs_sorted[0]["objectId"]:
                seats.append(objs_sorted[-1])
    return seats


def _seat_extreme_label(event: Any, seat: dict[str, Any]) -> str:
    idx, n, _ = ordinal_from_left(event, seat["objectId"], seat["objectType"])
    return extreme_side(idx, n) or "only"


def _proposals_for_objects(
    event: Any,
    basis: dict[str, np.ndarray],
    objects: list[dict[str, Any]],
    *,
    angular_size_min: float = ANGULAR_SIZE_MIN,
) -> list[EditProposal]:
    """Build slide proposals for an explicit object list (no size-band filter)."""
    proposals: list[EditProposal] = []
    for obj in objects:
        distance = float(obj.get("distance") or 2.0)
        old_pos = dict(obj["position"])
        old_rot = dict(obj["rotation"])
        for direction in MOVE_DIRECTIONS:
            for meters in MOVE_METERS_CANDIDATES:
                if (meters / max(1e-3, distance)) <= angular_size_min:
                    continue
                world_delta = camera_delta_to_world(direction, meters, basis)
                new_pos = {
                    "x": old_pos["x"] + float(world_delta[0]),
                    "y": old_pos["y"],
                    "z": old_pos["z"] + float(world_delta[2]),
                }
                proposals.append(
                    EditProposal(
                        object_id=obj["objectId"],
                        object_type=obj["objectType"],
                        direction=direction,  # type: ignore[arg-type]
                        meters=float(meters),
                        distance=distance,
                        angular_size_proxy=float(meters) / distance,
                        old_pos=old_pos,
                        old_rot=old_rot,
                        new_pos=new_pos,
                        world_delta=world_delta,
                    )
                )
    proposals.sort(key=lambda p: (-p.angular_size_proxy, p.object_type))
    return proposals


def find_affordance_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    vlm: Optional[VlmClient] = None,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    max_tries: int = 50,
    rng: Optional[random.Random] = None,
    som: bool = False,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Chair affordance edits (Modification 8).

    Modes (tried in random order):
    - ``sit_on_mover``: move a chair (extreme when many), sit on it, ask FOV to landmark.
    - ``observer_chair``: move another object A; sit on leftmost/rightmost chair; ask if A
      is in ±45° after the move.
    """
    rng = rng or random.Random()
    # Chairs are often closer than the default 1.5 m band in small rooms.
    dist_min = min(float(dist_min), 1.0)
    max_fp = scene_table_footprint_limit(event)
    chairs = _visible_moveable_chairs(
        event, dist_min=dist_min, dist_max=dist_max, max_fp=max_fp, som=som
    )
    seats = _affordance_seat_chairs(event, chairs)
    if not seats:
        return None

    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    modes = ["sit_on_mover", "observer_chair"]
    rng.shuffle(modes)

    for mode in modes:
        if mode == "sit_on_mover":
            proposals = _proposals_for_objects(
                event, basis, seats, angular_size_min=angular_size_min
            )
            if not proposals:
                continue
            order = proposals
            if vlm is not None:
                compact = [
                    {
                        "i": i,
                        "objectType": p.object_type,
                        "direction": p.direction,
                        "meters": p.meters,
                        "distance": round(p.distance, 2),
                        "angular": round(p.angular_size_proxy, 3),
                    }
                    for i, p in enumerate(proposals[:24])
                ]
                ranked = vlm.rank_proposals(compact, top_k=min(12, len(compact)))
                idxs = [int(r["i"]) for r in ranked if "i" in r]
                if idxs:
                    order = [proposals[i] for i in idxs] + [
                        p for j, p in enumerate(proposals) if j not in set(idxs)
                    ]
            for proposal in order[:max_tries]:
                result = try_edit(controller, cam, proposal, force=True)
                if result is None:
                    continue
                move, event0, event1 = result
                side = _seat_extreme_label(
                    event, next(s for s in seats if s["objectId"] == move.object_id)
                )
                move.edit_meta = {
                    **dict(move.edit_meta or {}),
                    "affordance_mode": "sit_on_mover",
                    "seat_id": move.object_id,
                    "seat_type": move.object_type,
                    "seat_extreme": side,
                }
                return move, event0, event1, "category:affordance+sit_on_mover"

        else:  # observer_chair
            # Prefer extreme seats when multiple chairs of a type are visible.
            extreme_seats = []
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in chairs:
                by_type[str(c["objectType"])].append(c)
            for objs in by_type.values():
                objs_sorted = sorted(
                    objs, key=lambda o: camera_frame_x(event, o["position"])
                )
                extreme_seats.append(objs_sorted[0])
                if len(objs_sorted) > 1:
                    extreme_seats.append(objs_sorted[-1])
            if not extreme_seats:
                continue
            rng.shuffle(extreme_seats)

            mover_proposals = [
                p
                for p in enumerate_edit_proposals(
                    event,
                    basis,
                    dist_min=dist_min,
                    dist_max=dist_max,
                    angular_size_min=angular_size_min,
                    som=som,
                )
                if p.object_type not in AFFORDANCE_CHAIR_TYPES
                and instance_referring_allowed(event, p.object_id, p.object_type)
            ]
            if not mover_proposals:
                continue
            rng.shuffle(mover_proposals)

            tries = 0
            for seat in extreme_seats:
                for proposal in mover_proposals:
                    if tries >= max_tries:
                        break
                    if proposal.object_id == seat["objectId"]:
                        continue
                    tries += 1
                    result = try_edit(controller, cam, proposal)
                    if result is None:
                        continue
                    move, event0, event1 = result
                    side = _seat_extreme_label(event, seat)
                    move.edit_meta = {
                        **dict(move.edit_meta or {}),
                        "affordance_mode": "observer_chair",
                        "seat_id": seat["objectId"],
                        "seat_type": seat["objectType"],
                        "seat_extreme": side,
                    }
                    return (
                        move,
                        event0,
                        event1,
                        "category:affordance+observer_chair",
                    )
                if tries >= max_tries:
                    break
    return None


# ---------------------------------------------------------------------------
# Modification 15 — metadata-planned flips for closer_to / left_right
# ---------------------------------------------------------------------------
_CLOSER_LANDMARK_TYPES = frozenset(
    {
        "Sofa",
        "ArmChair",
        "Television",
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "Desk",
        "Painting",
        "FloorLamp",
        "HousePlant",
        "ShelvingUnit",
        "Dresser",
        "Bed",
        "CounterTop",
        "Refrigerator",
        "Stove",
        "Toaster",
        "Mug",
        "Drawer",
        "Chair",
        "Stool",
        "Ottoman",
        "GarbageCan",
        "Box",
        "Laptop",
        "Pillow",
        "Bowl",
        "Plate",
        "Cup",
        "Pot",
        "Pan",
        "Apple",
        "Tomato",
        "Bread",
        "Bottle",
        "Statue",
        "Vase",
    }
)


def _xz(pos: dict[str, float]) -> np.ndarray:
    return np.array([float(pos["x"]), float(pos["z"])], dtype=np.float64)


def _dist3(a: dict[str, float], b: dict[str, float]) -> float:
    return float(
        math.dist(
            (a["x"], a["y"], a["z"]),
            (b["x"], b["y"], b["z"]),
        )
    )


def _plan_closer_to_target(
    pos_a: dict[str, float],
    pos_b: dict[str, float],
    pos_c: dict[str, float],
    *,
    gap: float = CLOSER_TO_GAP_M,
    min_move: float = CLOSER_TO_MIN_MOVE_M,
    max_move: float = CLOSER_TO_MAX_MOVE_M,
) -> Optional[dict[str, float]]:
    """Plan A' on the C side of the B–C bisector (XZ), keep A's height.

    Before: B must already be obviously closer. After: C obviously closer.
    Target is not extreme — typically past the midpoint toward C by ~0.25–0.4·|BC|.
    """
    a = _xz(pos_a)
    b = _xz(pos_b)
    c = _xz(pos_c)
    bc = c - b
    bc_len = float(np.linalg.norm(bc))
    if bc_len < CLOSER_TO_MIN_BC_SEP_M:
        return None
    # Before: B obviously closer.
    if _dist3(pos_a, pos_b) + gap >= _dist3(pos_a, pos_c):
        return None
    u = bc / bc_len
    mid = 0.5 * (b + c)
    # Candidate offsets past the bisector toward C (not too extreme).
    for frac in (0.30, 0.22, 0.38, 0.18, 0.45):
        ideal = mid + frac * bc_len * u
        delta = ideal - a
        move_m = float(np.linalg.norm(delta))
        if move_m < 1e-4:
            continue
        scale = 1.0
        if move_m > max_move:
            scale = max_move / move_m
        elif move_m < min_move:
            # Push further along the same direction if short.
            scale = min_move / move_m
            if scale * move_m > max_move:
                continue
        tgt_xz = a + scale * delta
        tgt = {"x": float(tgt_xz[0]), "y": float(pos_a["y"]), "z": float(tgt_xz[1])}
        if _dist3(tgt, pos_c) + gap >= _dist3(tgt, pos_b):
            continue
        applied = float(np.linalg.norm(tgt_xz - a))
        if applied < min_move * 0.8 or applied > max_move + 1e-3:
            continue
        return tgt
    return None


def _score_closer_to_plan(
    event: Any,
    mover: dict[str, Any],
    landmark_b: dict[str, Any],
    landmark_c: dict[str, Any],
    target: dict[str, float],
    others: list[dict[str, Any]],
    *,
    gap: float = CLOSER_TO_GAP_M,
) -> Optional[tuple[float, list[dict[str, Any]]]]:
    """Score a B→C flip plan; optionally attach near-but-farther distractors."""
    a0 = mover["position"]
    db0 = _dist3(a0, landmark_b["position"])
    dc0 = _dist3(a0, landmark_c["position"])
    if db0 + gap >= dc0:
        return None
    db1 = _dist3(target, landmark_b["position"])
    dc1 = _dist3(target, landmark_c["position"])
    if dc1 + gap >= db1:
        return None
    used = {mover["objectId"], landmark_b["objectId"], landmark_c["objectId"]}
    near_thresh0 = min(db0, dc0)
    near_thresh1 = min(db1, dc1)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for o in others:
        if o["objectId"] in used:
            continue
        d0 = _dist3(a0, o["position"])
        d1 = _dist3(target, o["position"])
        # Soft: keep distractors from winning after the move; allow near-A in rgb0.
        if d1 + 1e-6 < near_thresh1:
            continue
        if d0 + 1e-6 < near_thresh0 * 0.85:
            continue
        proximity = max(0.0, d0 - near_thresh0)
        candidates.append((proximity, o))
    candidates.sort(key=lambda t: t[0])
    distractors = [c[1] for c in candidates[:2]]
    move_m = math.dist((a0["x"], a0["z"]), (target["x"], target["z"]))
    score = (dc0 - db0) + (db1 - dc1) - 0.15 * move_m
    if len(distractors) >= 2:
        score += 0.2 - 0.05 * (candidates[0][0] + candidates[1][0])
    return score, distractors


def plan_closer_to_flip_candidates(
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    som: bool = False,
    max_plans: int = 40,
) -> list[dict[str, Any]]:
    """Enumerate (mover, B, C, target) flip plans from metadata only (no sim)."""
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    band = objects_in_distance_band(event, dist_min=dist_min, dist_max=dist_max)
    max_fp = scene_table_footprint_limit(event)
    min_px = SOM_MOVER_MASK_MIN_PIXELS if som else None
    movers: list[dict[str, Any]] = []
    for obj in band:
        if _mover_forbidden(obj):
            continue
        if not instance_referring_allowed(event, obj["objectId"], obj["objectType"]):
            continue
        if not is_small_enough_to_move(obj, max_fp, event, som=som):
            continue
        if min_px is not None and not is_visible_mask_large_enough(
            event, obj, min_pixels=min_px
        ):
            continue
        if not _mover_visually_ok(event, obj, som=som):
            continue
        movers.append(obj)
    if not movers:
        return []

    skip = {"Floor", "Wall", "Ceiling", "Window", "Doorway", "Doorframe"}
    landmarks = [
        o
        for o in event.metadata["objects"]
        if o.get("visible")
        and o["objectType"] not in skip
        and referring_allowed(event, o["objectType"])
        and _choice_visually_ok(event, o)
    ]
    preferred = [o for o in landmarks if o["objectType"] in _CLOSER_LANDMARK_TYPES]
    land_pool = list(preferred) + [o for o in landmarks if o not in preferred]

    plans: list[dict[str, Any]] = []
    for mover in movers:
        ban = set()
        for pid in mover.get("parentReceptacles") or []:
            if pid and isinstance(pid, str) and not (
                pid == "Floor" or pid.startswith("Floor")
            ):
                ban.add(pid)
        others = [
            o
            for o in land_pool
            if o["objectId"] != mover["objectId"] and o["objectId"] not in ban
        ]
        if len(others) < 2:
            continue
        others_ranked = sorted(
            others, key=lambda o: _dist3(mover["position"], o["position"])
        )
        pool = others_ranked[:16]
        for i, b in enumerate(pool):
            for c in pool[i + 1 :]:
                for landmark_b, landmark_c in ((b, c), (c, b)):
                    tgt = _plan_closer_to_target(
                        mover["position"],
                        landmark_b["position"],
                        landmark_c["position"],
                    )
                    if tgt is None:
                        continue
                    scored = _score_closer_to_plan(
                        event, mover, landmark_b, landmark_c, tgt, others
                    )
                    if scored is None:
                        continue
                    score, distractors = scored
                    delta = np.array(
                        [
                            tgt["x"] - mover["position"]["x"],
                            0.0,
                            tgt["z"] - mover["position"]["z"],
                        ],
                        dtype=np.float64,
                    )
                    meters = float(np.linalg.norm(delta[[0, 2]]))
                    if meters < CLOSER_TO_MIN_MOVE_M * 0.75:
                        continue
                    direction = world_delta_to_camera_direction(delta, basis)
                    if mover.get("pickupable"):
                        score += 0.15
                    plans.append(
                        {
                            "score": score,
                            "mover": mover,
                            "landmark_b": landmark_b,
                            "landmark_c": landmark_c,
                            "distractors": distractors,
                            "target": tgt,
                            "direction": direction,
                            "meters": meters,
                            "world_delta": delta,
                        }
                    )
    plans.sort(key=lambda p: -float(p["score"]))
    return plans[:max_plans]


def find_closer_to_flip_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    max_tries: int = 24,
    rng: Optional[random.Random] = None,
    som: bool = False,
    fast_find_edit: bool = False,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Modification 15: plan B→C closer flip from metadata, then validate in sim.

    Mod 28 ``fast_find_edit``: at most ``CLOSER_TO_FAST_MAX_TRIES`` plans, no force
    retry; if none flip, accept a random successful teleport (non-flip allowed).
    Always enumerate at least 3 motion plans so flip search is not a single shot.
    """
    rng = rng or random.Random(0)
    if fast_find_edit:
        max_tries = min(int(max_tries), int(CLOSER_TO_FAST_MAX_TRIES))
    max_tries = max(int(max_tries), 3)
    plans = plan_closer_to_flip_candidates(
        event,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
        max_plans=max(max_tries * 2, 40),
    )
    if not plans:
        return None
    # Mild shuffle among top plans so scenes don't always pick the same pair.
    top = plans[: min(len(plans), max_tries)]
    order = list(top)
    rng.shuffle(order)
    order = sorted(order, key=lambda p: -float(p["score"]))[:max_tries]

    nonflip_hits: list[tuple[MoveAction, Any, Any]] = []
    for plan in order:
        mover = plan["mover"]
        proposal = EditProposal(
            object_id=mover["objectId"],
            object_type=mover["objectType"],
            direction=plan["direction"],  # type: ignore[arg-type]
            meters=float(plan["meters"]),
            distance=float(mover.get("distance") or 2.0),
            angular_size_proxy=float(plan["meters"])
            / max(1e-3, float(mover.get("distance") or 2.0)),
            old_pos=dict(mover["position"]),
            old_rot=dict(mover["rotation"]),
            new_pos=dict(plan["target"]),
            world_delta=np.asarray(plan["world_delta"], dtype=np.float64),
        )
        # Prefer collision-checked teleport; only force if needed, then reject tips.
        used_force = False
        result = try_edit(
            controller,
            cam,
            proposal,
            force=False,
            settle_gravity=False,
        )
        if result is None and FLIP_PLAN_FORCE_EDIT and not fast_find_edit:
            result = try_edit(
                controller,
                cam,
                proposal,
                force=True,
                settle_gravity=False,
            )
            used_force = result is not None
        if result is None:
            continue
        move, event0, event1 = result
        ok_scene, scene_report = scene_collateral_ok(event0, event1, move.object_id)
        if not ok_scene:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        after = get_object(event1, move.object_id)
        b = get_object(event0, plan["landmark_b"]["objectId"]) or get_object(
            event1, plan["landmark_b"]["objectId"]
        )
        c = get_object(event0, plan["landmark_c"]["objectId"]) or get_object(
            event1, plan["landmark_c"]["objectId"]
        )
        before = get_object(event0, move.object_id)
        if after is None or b is None or c is None or before is None:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        gap = CLOSER_TO_GAP_M
        db0 = _dist3(before["position"], b["position"])
        dc0 = _dist3(before["position"], c["position"])
        db1 = _dist3(after["position"], b["position"])
        dc1 = _dist3(after["position"], c["position"])
        flip_ok = bool(db0 + gap < dc0 and dc1 + gap < db1)
        agent = event0.metadata["agent"]
        basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
        move = _relabel_move_camera_frame(move, basis)
        move.edit_meta = {
            **dict(move.edit_meta or {}),
            "closer_to_flip_plan": True,
            "closer_landmark_b_id": b["objectId"],
            "closer_landmark_c_id": c["objectId"],
            "closer_distractor_ids": [d["objectId"] for d in plan["distractors"]],
            "closer_gap_m": gap,
            "closer_dist_before": {"b": db0, "c": dc0},
            "closer_dist_after": {"b": db1, "c": dc1},
            "planned_target": dict(plan["target"]),
            "force_edit": bool(used_force),
            "scene_integrity": scene_report,
            "answer_flipped_planned": bool(flip_ok),
            "allow_nonflip_closer": bool(fast_find_edit and not flip_ok),
        }
        if flip_ok:
            return move, event0, event1, "category:closer_to+flip_plan"
        if fast_find_edit:
            nonflip_hits.append((move, event0, event1))
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        pin_object_pose(
            controller, move.object_id, move.position_before.to_dict(), move.rotation_before
        )
    if fast_find_edit and nonflip_hits:
        move, event0, event1 = rng.choice(nonflip_hits)
        # Re-materialize before/after frames for the chosen fallback.
        pin_object_pose(
            controller,
            move.object_id,
            move.position_before.to_dict(),
            move.rotation_before,
        )
        event0 = teleport_full(controller, **cam.teleport_kwargs())
        pin_object_pose(
            controller,
            move.object_id,
            move.position_after.to_dict(),
            move.rotation_before,
        )
        event1 = teleport_full(controller, **cam.teleport_kwargs())
        pin_object_pose(
            controller,
            move.object_id,
            move.position_after.to_dict(),
            move.rotation_before,
        )
        move.edit_meta = {
            **dict(move.edit_meta or {}),
            "closer_to_flip_plan": False,
            "allow_nonflip_closer": True,
            "answer_flipped_planned": False,
            "closer_nonflip_fallback": True,
        }
        return move, event0, event1, "category:closer_to+nonflip_fallback"
    return None


def _bearing_from_cam(
    cam_pos: dict[str, float],
    basis: dict[str, np.ndarray],
    position: dict[str, float],
) -> Optional[float]:
    p = np.array(
        [
            position["x"] - cam_pos["x"],
            position["y"] - cam_pos["y"],
            position["z"] - cam_pos["z"],
        ],
        dtype=np.float64,
    )
    x_cam = float(np.dot(p, basis["right"]))
    z_cam = float(np.dot(p, basis["forward"]))
    if z_cam <= 1e-4:
        return None
    return math.degrees(math.atan2(x_cam, z_cam))


def plan_left_right_flip_candidates(
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    som: bool = False,
    max_plans: int = 40,
) -> list[dict[str, Any]]:
    """Enumerate moves that cross a landmark bearing (metadata-only)."""
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    cam_pos = event.metadata.get("cameraPosition") or agent["position"]
    band = objects_in_distance_band(event, dist_min=dist_min, dist_max=dist_max)
    max_fp = scene_table_footprint_limit(event)
    min_px = SOM_MOVER_MASK_MIN_PIXELS if som else None
    movers: list[dict[str, Any]] = []
    for obj in band:
        if _mover_forbidden(obj):
            continue
        if not instance_referring_allowed(event, obj["objectId"], obj["objectType"]):
            continue
        if not is_small_enough_to_move(obj, max_fp, event, som=som):
            continue
        if min_px is not None and not is_visible_mask_large_enough(
            event, obj, min_pixels=min_px
        ):
            continue
        if not _mover_visually_ok(event, obj, som=som):
            continue
        movers.append(obj)
    if not movers:
        return []

    skip = {"Floor", "Wall", "Ceiling", "Window", "Doorway", "Doorframe"}
    landmarks = [
        o
        for o in event.metadata["objects"]
        if o.get("visible")
        and o["objectType"] not in skip
        and referring_allowed(event, o["objectType"])
        and _choice_visually_ok(event, o)
    ]

    meter_opts = (0.80, 0.65, 0.50, 0.40, 0.30, 0.25, 0.20)
    plans: list[dict[str, Any]] = []
    for mover in movers:
        ban = set()
        for pid in mover.get("parentReceptacles") or []:
            if pid and isinstance(pid, str) and not (
                pid == "Floor" or pid.startswith("Floor")
            ):
                ban.add(pid)
        b0 = _bearing_from_cam(cam_pos, basis, mover["position"])
        if b0 is None:
            continue
        for land in landmarks:
            if land["objectId"] == mover["objectId"] or land["objectId"] in ban:
                continue
            bl = _bearing_from_cam(cam_pos, basis, land["position"])
            if bl is None:
                continue
            # Landmark near the mover's bearing so a modest slide can cross.
            if abs(b0 - bl) > 40.0:
                continue
            for direction in ("left", "right"):
                for meters in meter_opts:
                    delta = camera_delta_to_world(direction, meters, basis)
                    tgt = {
                        "x": float(mover["position"]["x"] + delta[0]),
                        "y": float(mover["position"]["y"]),
                        "z": float(mover["position"]["z"] + delta[2]),
                    }
                    b1 = _bearing_from_cam(cam_pos, basis, tgt)
                    if b1 is None:
                        continue
                    left0 = b0 < bl
                    left1 = b1 < bl
                    flipped = left0 != left1
                    span = abs(b1 - b0)
                    if flipped:
                        if span < LEFT_RIGHT_MIN_SPAN_DEG:
                            continue
                        if abs(b0 - bl) < 1.0 or abs(b1 - bl) < 1.0:
                            continue
                        score = span + 0.5 * min(abs(b0 - bl), abs(b1 - bl)) - 0.1 * meters
                    else:
                        # Same-side (non-flip): still a clear lateral slide near landmark.
                        if span < max(3.0, LEFT_RIGHT_MIN_SPAN_DEG * 0.5):
                            continue
                        if abs(b0 - bl) > 25.0:
                            continue
                        # Stay clearly on one side of the landmark.
                        if min(abs(b0 - bl), abs(b1 - bl)) < 2.0:
                            continue
                        score = 0.35 * span + 0.15 * min(abs(b0 - bl), abs(b1 - bl)) - 0.1 * meters
                    if mover.get("pickupable"):
                        score += 2.0
                    if flipped:
                        score += 1.0
                    plans.append(
                        {
                            "score": score,
                            "mover": mover,
                            "landmark": land,
                            "direction": direction,
                            "meters": meters,
                            "world_delta": delta,
                            "target": tgt,
                            "bearing_before": b0,
                            "bearing_after": b1,
                            "landmark_bearing": bl,
                            "span_deg": span,
                            "flipped_planned": flipped,
                        }
                    )
    plans.sort(key=lambda p: -float(p["score"]))
    return plans[:max_plans]


def find_left_right_flip_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    max_tries: int = 24,
    rng: Optional[random.Random] = None,
    som: bool = False,
    require_flip: bool = LEFT_RIGHT_REQUIRE_FLIP,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Modification 15/20: plan a bearing slide; ~80% flip / ~20% same-side."""
    rng = rng or random.Random(0)
    # Per-call target: mostly flips, keep a non-flip fraction for anti-shortcut.
    if require_flip:
        want_flip = True
    else:
        want_flip = rng.random() >= float(LEFT_RIGHT_NONFLIP_FRAC)
    plans = plan_left_right_flip_candidates(
        event,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
        max_plans=max(max_tries * 2, 40),
    )
    flip_plans = [p for p in plans if p.get("flipped_planned", True)]
    nonflip_plans = [p for p in plans if not p.get("flipped_planned", True)]
    preferred = flip_plans if want_flip else nonflip_plans
    # Strict: do not silently swap flip↔nonflip when the requested pool is empty.
    if not preferred:
        return None
    order = list(preferred[:max_tries])
    rng.shuffle(order)
    order = sorted(order, key=lambda p: -float(p["score"]))[:max_tries]

    for plan in order:
        mover = plan["mover"]
        proposal = EditProposal(
            object_id=mover["objectId"],
            object_type=mover["objectType"],
            direction=plan["direction"],  # type: ignore[arg-type]
            meters=float(plan["meters"]),
            distance=float(mover.get("distance") or 2.0),
            angular_size_proxy=float(plan["meters"])
            / max(1e-3, float(mover.get("distance") or 2.0)),
            old_pos=dict(mover["position"]),
            old_rot=dict(mover["rotation"]),
            new_pos=dict(plan["target"]),
            world_delta=np.asarray(plan["world_delta"], dtype=np.float64),
        )
        used_force = False
        result = try_edit(
            controller,
            cam,
            proposal,
            force=False,
            settle_gravity=False,
        )
        if result is None and FLIP_PLAN_FORCE_EDIT:
            result = try_edit(
                controller,
                cam,
                proposal,
                force=True,
                settle_gravity=False,
            )
            used_force = result is not None
        if result is None:
            continue
        move, event0, event1 = result
        ok_scene, scene_report = scene_collateral_ok(event0, event1, move.object_id)
        if not ok_scene:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        before = get_object(event0, move.object_id)
        after = get_object(event1, move.object_id)
        land = get_object(event0, plan["landmark"]["objectId"]) or get_object(
            event1, plan["landmark"]["objectId"]
        )
        if before is None or after is None or land is None:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        b0 = camera_bearing_deg(event0, before["position"])
        b1 = camera_bearing_deg(event1, after["position"])
        bl = camera_bearing_deg(event1, land["position"])
        if b0 is None or b1 is None or bl is None:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        left0 = b0 < bl
        left1 = b1 < bl
        flipped = left0 != left1
        span = abs(b1 - b0)
        plan_wants_flip = bool(plan.get("flipped_planned", True))
        if plan_wants_flip and not flipped:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        if (not plan_wants_flip) and flipped:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        if plan_wants_flip and span < LEFT_RIGHT_MIN_SPAN_DEG * 0.5:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        if require_flip and not flipped:
            pin_object_pose(
                controller, move.object_id, move.position_before.to_dict(), move.rotation_before
            )
            continue
        agent = event0.metadata["agent"]
        basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
        move = _relabel_move_camera_frame(move, basis)
        move.edit_meta = {
            **dict(move.edit_meta or {}),
            "left_right_flip_plan": True,
            "left_right_landmark_id": land["objectId"],
            "bearing_before_deg": b0,
            "bearing_after_deg": b1,
            "landmark_bearing_deg": bl,
            "bearing_span_deg": span,
            "answer_flipped_planned": flipped,
            "force_edit": bool(used_force),
            "scene_integrity": scene_report,
        }
        note = (
            "category:left_right+flip_plan"
            if flipped
            else "category:left_right+nonflip_plan"
        )
        return move, event0, event1, note
    return None


def find_edit_for_category(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    category: str,
    *,
    vlm: Optional[VlmClient] = None,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    max_tries: int = 50,
    prefer_occ_yes: bool = True,
    prefer_contact: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    som: bool = False,
    fast_find_edit: bool = False,
    fast_contact_max_tries: Optional[int] = None,
    fast_occlusion_max_tries: Optional[int] = None,
    wall_away_prob: Optional[float] = None,
    fast_wall_select_near_m: Optional[float] = None,
    contact_always_toward: Optional[bool] = None,
    contact_half_motion_no_frac: Optional[float] = None,
    procthor_rooms: Optional[list] = None,
    camera_room_id: Optional[str] = None,
    contact_room_wall_prefilter: Optional[bool] = None,
    contact_room_wall_max_m: Optional[float] = None,
    contact_fast_max_candidates: Optional[int] = None,
    **_unused: Any,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Pick a category-appropriate edit for this camera (Modification 4–5, 8, 15, 24, 28)."""
    del _unused  # forward-compat only
    if category == "contact_wall":
        contact_cap = int(
            FAST_CONTACT_MAX_TRIES
            if fast_contact_max_tries is None
            else fast_contact_max_tries
        )
        return find_wall_edit(
            controller,
            cam,
            event,
            dist_min=dist_min,
            dist_max=dist_max,
            max_tries=max_tries if not fast_find_edit else min(max_tries, contact_cap),
            rng=rng,
            som=som,
            prefer_contact=prefer_contact,
            fast_find_edit=fast_find_edit,
            wall_away_prob=wall_away_prob,
            select_near_m=(
                float(fast_wall_select_near_m)
                if (fast_find_edit and fast_wall_select_near_m is not None)
                else None
            ),
            always_toward=contact_always_toward,
            half_motion_no_frac=contact_half_motion_no_frac,
            procthor_rooms=procthor_rooms,
            camera_room_id=camera_room_id,
            contact_room_wall_prefilter=contact_room_wall_prefilter,
            contact_room_wall_max_m=contact_room_wall_max_m,
            contact_fast_max_candidates=contact_fast_max_candidates,
        )
    if category == "occlusion":
        occ_cap = int(
            FAST_OCCLUSION_MAX_TRIES
            if fast_occlusion_max_tries is None
            else fast_occlusion_max_tries
        )
        return find_occlusion_cover_edit(
            controller,
            cam,
            event,
            dist_min=dist_min,
            dist_max=dist_max,
            max_tries=max_tries if not fast_find_edit else min(max_tries, occ_cap),
            prefer_yes=prefer_occ_yes,
            som=som,
            fast_find_edit=fast_find_edit,
        )
    if category == "affordance":
        return find_affordance_edit(
            controller,
            cam,
            event,
            vlm=vlm,
            angular_size_min=angular_size_min,
            dist_min=dist_min,
            dist_max=dist_max,
            max_tries=max_tries,
            rng=rng,
            som=som,
        )
    if category == "closer_to":
        hit = find_closer_to_flip_edit(
            controller,
            cam,
            event,
            dist_min=dist_min,
            dist_max=dist_max,
            angular_size_min=angular_size_min,
            max_tries=min(max_tries, 24),
            rng=rng,
            som=som,
            fast_find_edit=fast_find_edit,
        )
        if hit is not None:
            return hit
        # Fall through to generic proposals only if flip plan failed entirely.
        # Stamp nonflip / random-fallback meta so ProcTHOR QA can complete.
    if category == "left_right_landmark":
        hit = find_left_right_flip_edit(
            controller,
            cam,
            event,
            dist_min=dist_min,
            dist_max=dist_max,
            angular_size_min=angular_size_min,
            max_tries=min(max_tries, 24),
            rng=rng,
            som=som,
            require_flip=LEFT_RIGHT_REQUIRE_FLIP,
        )
        if hit is not None:
            return hit
        if LEFT_RIGHT_REQUIRE_FLIP:
            return None

    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    proposals = filter_proposals_for_category(
        event,
        basis,
        category,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
        som=som,
    )
    if not proposals:
        return None

    order = proposals
    rank_notes = f"category:{category}"
    if vlm is not None:
        compact = [
            {
                "i": i,
                "objectType": p.object_type,
                "direction": p.direction,
                "meters": p.meters,
                "distance": round(p.distance, 2),
                "angular": round(p.angular_size_proxy, 3),
            }
            for i, p in enumerate(proposals[:24])
        ]
        ranked = vlm.rank_proposals(compact, top_k=min(12, len(compact)))
        idxs = [int(r["i"]) for r in ranked if "i" in r]
        if idxs:
            order = [proposals[i] for i in idxs] + [
                p for j, p in enumerate(proposals) if j not in set(idxs)
            ]
            rank_notes = f"category:{category}+vlm_ranked"

    from cit_pipeline.config import GRAVITY_SETTLE_CATEGORIES

    for proposal in order[:max_tries]:
        # Modification 11: settle gravity for support + feasibility (+ other
        # pickupable-friendly categories) so plants/mugs do not float in rgb1.
        settle = category in GRAVITY_SETTLE_CATEGORIES
        if settle and category != "support":
            # Only settle when the mover is pickupable (heavy furniture stays kinematic).
            src = next(
                (
                    o
                    for o in event.metadata["objects"]
                    if o["objectId"] == proposal.object_id
                ),
                None,
            )
            if src is None or not src.get("pickupable"):
                settle = False
        result = try_edit(controller, cam, proposal, settle_gravity=settle)
        if result is not None:
            move, event0, event1 = result
            if category == "closer_to":
                move.edit_meta = {
                    **dict(move.edit_meta or {}),
                    "closer_to_flip_plan": False,
                    "allow_nonflip_closer": True,
                    "answer_flipped_planned": False,
                    "closer_nonflip_fallback": True,
                    "closer_generic_fallthrough": True,
                }
            return move, event0, event1, rank_notes
    return None
