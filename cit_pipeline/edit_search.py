"""Propose and validate camera-frame object translations.

Filters (all must pass)
-----------------------
1. Object visible with distance in ``(dist_min, dist_max)``.
2. ``meters / distance > angular_size_min`` (obviousness proxy).
3. ``TeleportObject`` (or PlaceObjectAtPoint) returns success.
4. After the move, the object is still ``visible`` from the **same** camera.

Optional VLM ranking reorders proposals but never rejects them (role b, not c).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from ai2thor.controller import Controller

from cit_pipeline.camera_search import camera_basis, objects_in_distance_band
from cit_pipeline.config import (
    ANGULAR_SIZE_MIN,
    BINARY_SLIDE_ITERS,
    DIST_MAX,
    DIST_MIN,
    MOVE_DIRECTIONS,
    MOVE_METERS_CANDIDATES,
    MOVE_TEXT_VERTICAL_MIN_M,
    SUPPORT_GRAVITY_MAX_STEPS,
    SUPPORT_GRAVITY_MIN_STEPS,
    SUPPORT_GRAVITY_TIMESTEP,
    SUPPORT_QA_FLOOR_BOTTOM_Y_M,
)
from cit_pipeline.schemas import CameraPose, Direction, MoveAction, Vec3
from cit_pipeline.thor_env import (
    get_object,
    pin_object_pose,
    settle_object_with_gravity,
    still_visible,
    teleport_full,
    teleport_object,
)
from cit_pipeline.vlm_client import VlmClient
from cit_pipeline.move_text import format_direction_phrase, format_move_displacement


@dataclass
class EditProposal:
    """Candidate translation before simulator validation."""

    object_id: str
    object_type: str
    direction: Direction
    meters: float
    distance: float
    angular_size_proxy: float
    old_pos: dict[str, float]
    old_rot: dict[str, float]
    new_pos: dict[str, float]
    world_delta: np.ndarray


def camera_delta_to_world(
    direction: str, meters: float, basis: dict[str, np.ndarray]
) -> np.ndarray:
    """Map a cardinal move to a world-space XYZ delta (meters).

    ``left`` / ``right`` / ``forward`` / ``back`` are camera-frame.
    ``up`` / ``down`` are world +Y / −Y (lift / drop).
    """
    mapping = {
        "left": -basis["right"],
        "right": basis["right"],
        "forward": basis["forward"],  # into the scene
        "back": -basis["forward"],  # toward the camera plane
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "down": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }
    if direction not in mapping:
        raise ValueError(f"Unknown direction {direction!r}")
    return np.asarray(mapping[direction], dtype=np.float64) * float(meters)


def format_move_displacement(
    direction: str,
    meters: float,
    delta_world: Any,
    *,
    dy_min: float = MOVE_TEXT_VERTICAL_MIN_M,
) -> str:
    """Format horizontal (+ optional downward) move, e.g. ``0.54m right and 0.80m down``.

    Modification 10: never emit an *upward* clause (against gravity). Positive Δy
    is ignored in the command text; only drops are mentioned.
    """
    if hasattr(delta_world, "y"):
        dy = float(delta_world.y)
    elif isinstance(delta_world, dict):
        dy = float(delta_world.get("y", 0.0))
    else:
        dy = float(delta_world[1])
    horiz = f"{float(meters):.2f}m {direction}"
    # Only describe downward displacement (never "up").
    if dy > -float(dy_min):
        return horiz
    return f"{horiz} and {abs(dy):.2f}m down"


def _meets_obviousness(distance: float, meters: float, angular_min: float) -> bool:
    if not (DIST_MIN < distance < DIST_MAX):
        return False
    return (meters / distance) > angular_min


def enumerate_edit_proposals(
    event: Any,
    basis: dict[str, np.ndarray],
    *,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    directions: Sequence[str] = MOVE_DIRECTIONS,
    meters_list: Sequence[float] = MOVE_METERS_CANDIDATES,
    som: bool = False,
) -> list[EditProposal]:
    """Cartesian product of band objects × directions × distances (pre-filter)."""
    candidates = objects_in_distance_band(
        event, dist_min=dist_min, dist_max=dist_max, som=som
    )
    proposals: list[EditProposal] = []
    for obj in candidates:
        distance = float(obj["distance"])
        old_pos = dict(obj["position"])
        old_rot = dict(obj["rotation"])
        for direction in directions:
            for meters in meters_list:
                if not _meets_obviousness(distance, meters, angular_size_min):
                    continue
                world_delta = camera_delta_to_world(direction, meters, basis)
                # Keep height; slide on the floor plane.
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
    mid = 0.5 * (dist_min + dist_max)
    proposals.sort(
        key=lambda p: (-p.angular_size_proxy, abs(p.distance - mid), p.object_type)
    )
    return proposals


def binary_refine_slide(
    controller: Controller,
    object_id: str,
    old_pos: dict[str, float],
    old_rot: dict[str, float],
    world_delta: np.ndarray,
    *,
    total_meters: float,
    force: bool = False,
    n_iters: int = BINARY_SLIDE_ITERS,
    hold_y: Optional[float] = None,
) -> tuple[dict[str, float], float, list[dict[str, Any]]]:
    """Slide along ``world_delta`` with binary step refinement (Modification 11).

    Start with step = ``total_meters``. Each iteration: try advancing by ``step``
    from the last successful pose; on failure keep pose and always halve ``step``.
    Caps at ``total_meters``. Returns ``(final_pos, applied_meters, trace)``.
    """
    delta = np.asarray(world_delta, dtype=np.float64).reshape(3)
    horiz = np.array([delta[0], 0.0, delta[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horiz))
    if norm < 1e-6:
        return dict(old_pos), 0.0, []
    unit = horiz / norm
    y_hold = float(old_pos["y"] if hold_y is None else hold_y)
    total = max(0.0, float(total_meters))
    step = total
    applied = 0.0
    cur = {
        "x": float(old_pos["x"]),
        "y": y_hold,
        "z": float(old_pos["z"]),
    }
    trace: list[dict[str, Any]] = []
    pin_object_pose(controller, object_id, cur, old_rot)

    for _ in range(max(1, int(n_iters))):
        if step < 1e-4:
            break
        nxt_applied = applied + step
        if nxt_applied > total + 1e-6:
            step *= 0.5
            continue
        trial = {
            "x": float(old_pos["x"]) + float(unit[0]) * nxt_applied,
            "y": y_hold,
            "z": float(old_pos["z"]) + float(unit[2]) * nxt_applied,
        }
        ev = teleport_object(
            controller,
            object_id,
            trial,
            old_rot,
            force=force,
            allow_surface_place=False,
            kinematic=True,
        )
        ok = bool(ev.metadata["lastActionSuccess"])
        trace.append(
            {
                "attempt_m": float(nxt_applied),
                "step_m": float(step),
                "ok": ok,
            }
        )
        if ok:
            pin_object_pose(controller, object_id, trial, old_rot)
            cur = dict(trial)
            applied = float(nxt_applied)
        else:
            pin_object_pose(controller, object_id, cur, old_rot)
        step *= 0.5

    return cur, float(applied), trace


def try_edit(
    controller: Controller,
    cam: CameraPose,
    proposal: EditProposal,
    *,
    force: bool = False,
    settle_gravity: bool = False,
) -> Optional[tuple[MoveAction, Any, Any]]:
    """Apply proposal without forcing through collisions; return events or None.

    Modification 2: default ``force=False`` on the trial teleport so counter/wall
    crashes register as failures. Restores always use ``force=True``.

    Modification 3: default keeps the intended floating height (``proposal.new_pos``).

    Modification 8: ``force=True`` allowed for chair affordance slides.

    Modification 10: ``settle_gravity=True`` (support QA) releases kinematics after
    the collision-checked slide so the object can fall onto furniture or the floor.
    Upward displacements are rejected.

    Modification 11: if the full slide is blocked, binary-halve the step along the
    ray (``BINARY_SLIDE_ITERS``) so applied meters match a feasible pose.
    """
    intended_rot = dict(proposal.old_rot)
    y_hold = float(proposal.old_pos["y"])
    # Modification 10: never move upward against gravity.
    target_y = min(float(proposal.new_pos["y"]), y_hold)
    planned_delta = np.array(
        [
            float(proposal.new_pos["x"]) - float(proposal.old_pos["x"]),
            0.0,
            float(proposal.new_pos["z"]) - float(proposal.old_pos["z"]),
        ],
        dtype=np.float64,
    )
    planned_m = float(np.linalg.norm(planned_delta[[0, 2]]))
    if planned_m < 1e-4:
        planned_m = float(proposal.meters)
        # Reconstruct horizontal delta from camera-frame proposal when possible.
        planned_delta = np.array(
            [
                float(proposal.world_delta[0]),
                0.0,
                float(proposal.world_delta[2]),
            ],
            dtype=np.float64,
        )

    intended_pos, applied_m, binary_trace = binary_refine_slide(
        controller,
        proposal.object_id,
        proposal.old_pos,
        intended_rot,
        planned_delta,
        total_meters=max(planned_m, float(proposal.meters)),
        force=force,
        hold_y=y_hold,
    )
    intended_pos["y"] = target_y
    if applied_m < 1e-3:
        pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
        return None

    # Re-pin at refined pose before visibility / gravity checks.
    pin_object_pose(controller, proposal.object_id, intended_pos, intended_rot)

    if settle_gravity:
        settle_object_with_gravity(
            controller,
            proposal.object_id,
            intended_pos,
            intended_rot,
            max_steps=SUPPORT_GRAVITY_MAX_STEPS,
            time_step=SUPPORT_GRAVITY_TIMESTEP,
            min_steps=SUPPORT_GRAVITY_MIN_STEPS,
            force_place=True,
        )
        event1_probe = teleport_full(controller, **cam.teleport_kwargs())
        probe_obj = get_object(event1_probe, proposal.object_id)
        if probe_obj is None or not still_visible(event1_probe, proposal.object_id):
            pin_object_pose(
                controller, proposal.object_id, proposal.old_pos, proposal.old_rot
            )
            return None
        moved_pos = dict(probe_obj["position"])
        moved_rot = dict(probe_obj.get("rotation") or intended_rot)
        # Reject mid-air floats: gravity did not ground the object (empty
        # parentReceptacles and height still near the intended slide pose).
        parents_probe = probe_obj.get("parentReceptacles") or []
        y_now = float(moved_pos["y"])
        y_intended = float(intended_pos["y"])
        bottom_now = None
        aabb = probe_obj.get("axisAlignedBoundingBox") or {}
        center = aabb.get("center") or {}
        size = aabb.get("size") or {}
        if "y" in center and "y" in size:
            bottom_now = float(center["y"]) - 0.5 * float(size["y"])
        near_floor = (bottom_now is not None and bottom_now < SUPPORT_QA_FLOOR_BOTTOM_Y_M) or (
            y_now < SUPPORT_QA_FLOOR_BOTTOM_Y_M
        )
        if (not parents_probe) and (not near_floor) and abs(y_now - y_intended) < 0.08:
            pin_object_pose(
                controller, proposal.object_id, proposal.old_pos, proposal.old_rot
            )
            return None
        # Require a real horizontal displacement from the original pose.
        dx = moved_pos["x"] - proposal.old_pos["x"]
        dz = moved_pos["z"] - proposal.old_pos["z"]
        if (dx * dx + dz * dz) ** 0.5 < 0.15 * max(applied_m, 1e-3):
            # Still accept if it clearly fell (large Δy) after a short slide.
            if float(moved_pos["y"]) > float(proposal.old_pos["y"]) - 0.05:
                pin_object_pose(
                    controller, proposal.object_id, proposal.old_pos, proposal.old_rot
                )
                return None

        pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
        event0 = teleport_full(controller, **cam.teleport_kwargs())
        # Re-apply slide + gravity for rgb1 (deterministic enough for GT).
        teleport_object(
            controller,
            proposal.object_id,
            intended_pos,
            intended_rot,
            force=force,
            kinematic=True,
        )
        settle_object_with_gravity(
            controller,
            proposal.object_id,
            intended_pos,
            intended_rot,
            max_steps=SUPPORT_GRAVITY_MAX_STEPS,
            time_step=SUPPORT_GRAVITY_TIMESTEP,
            min_steps=SUPPORT_GRAVITY_MIN_STEPS,
            force_place=True,
        )
        event1 = teleport_full(controller, **cam.teleport_kwargs())
        settled = get_object(event1, proposal.object_id)
        if settled is not None:
            moved_pos = dict(settled["position"])
            moved_rot = dict(settled.get("rotation") or moved_rot)
        world_delta = np.array(
            [
                moved_pos["x"] - proposal.old_pos["x"],
                moved_pos["y"] - proposal.old_pos["y"],
                moved_pos["z"] - proposal.old_pos["z"],
            ],
            dtype=np.float64,
        )
        applied_final = float(np.linalg.norm(world_delta[[0, 2]]))
        move = MoveAction(
            object_id=proposal.object_id,
            object_type=proposal.object_type,
            direction=proposal.direction,
            meters=applied_final if applied_final > 1e-3 else applied_m,
            delta_world=Vec3(
                x=float(world_delta[0]),
                y=float(world_delta[1]),
                z=float(world_delta[2]),
            ),
            position_before=Vec3.from_dict(proposal.old_pos),
            position_after=Vec3.from_dict(moved_pos),
            rotation_before=dict(proposal.old_rot),
            distance_before=proposal.distance,
            angular_size_proxy=proposal.angular_size_proxy,
            still_visible_in_rgb1=True,
            text=(
                f"move the {proposal.object_type.lower()} "
                f"{format_move_displacement(proposal.direction, applied_m, world_delta)}"
            ),
            edit_meta={
                "settle_gravity": True,
                "intended_pos": intended_pos,
                "height_before": float(proposal.old_pos["y"]),
                "height_after": float(moved_pos["y"]),
                "binary_slide": True,
                "binary_slide_trace": binary_trace,
                "planned_meters": float(proposal.meters),
                "applied_meters": float(applied_m),
            },
        )
        return move, event0, event1

    def _render_pinned() -> Any:
        pin_object_pose(controller, proposal.object_id, intended_pos, intended_rot)
        return teleport_full(controller, **cam.teleport_kwargs())

    event1_probe = _render_pinned()
    # If autosim nudged height, pin once more and re-render.
    probe_obj = get_object(event1_probe, proposal.object_id)
    if probe_obj is not None and abs(
        float(probe_obj["position"]["y"]) - float(intended_pos["y"])
    ) > 0.05:
        event1_probe = _render_pinned()
        probe_obj = get_object(event1_probe, proposal.object_id)

    if probe_obj is None or not still_visible(event1_probe, proposal.object_id):
        pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
        return None

    # Use intended floating pose for GT (ignore residual physics drift).
    moved_pos = dict(intended_pos)
    moved_rot = dict(intended_rot)

    # Require a real horizontal displacement (reject no-ops / clipped teleports).
    probe_pos = dict(probe_obj["position"])
    dx = probe_pos["x"] - proposal.old_pos["x"]
    dz = probe_pos["z"] - proposal.old_pos["z"]
    if (dx * dx + dz * dz) ** 0.5 < 0.5 * max(applied_m, 1e-3):
        pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
        return None
    # Reject if the sim still dropped the object onto the floor (non-gravity path).
    if float(probe_pos["y"]) < float(intended_pos["y"]) - 0.15:
        pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
        return None

    pin_object_pose(controller, proposal.object_id, proposal.old_pos, proposal.old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())
    # Re-apply the *validated floating* pose (force OK — already known reachable).
    pin_object_pose(controller, proposal.object_id, moved_pos, moved_rot)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    # Final pin so rgb1 metadata matches the floating pose.
    pin_obj = pin_object_pose(controller, proposal.object_id, moved_pos, moved_rot)
    if pin_obj.metadata["lastActionSuccess"]:
        event1 = teleport_full(controller, **cam.teleport_kwargs())
        pin_object_pose(controller, proposal.object_id, moved_pos, moved_rot)

    world_delta = np.array(
        [
            float(moved_pos["x"] - proposal.old_pos["x"]),
            float(moved_pos["y"] - proposal.old_pos["y"]),
            float(moved_pos["z"] - proposal.old_pos["z"]),
        ],
        dtype=np.float64,
    )
    move = MoveAction(
        object_id=proposal.object_id,
        object_type=proposal.object_type,
        direction=proposal.direction,
        meters=applied_m,
        delta_world=Vec3(
            x=float(world_delta[0]),
            y=float(world_delta[1]),
            z=float(world_delta[2]),
        ),
        position_before=Vec3.from_dict(proposal.old_pos),
        position_after=Vec3.from_dict(moved_pos),
        rotation_before=dict(proposal.old_rot),
        distance_before=proposal.distance,
        angular_size_proxy=proposal.angular_size_proxy,
        still_visible_in_rgb1=True,
        text=(
            f"move the {proposal.object_type.lower()} "
            f"{format_move_displacement(proposal.direction, applied_m, world_delta)}"
        ),
        edit_meta={
            "settle_gravity": False,
            "binary_slide": True,
            "binary_slide_trace": binary_trace,
            "planned_meters": float(proposal.meters),
            "applied_meters": float(applied_m),
        },
    )
    return move, event0, event1


def find_edit_for_view(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    vlm: Optional[VlmClient] = None,
    angular_size_min: float = ANGULAR_SIZE_MIN,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    max_tries: int = 40,
) -> Optional[tuple[MoveAction, Any, Any, str]]:
    """Pick the first valid edit for this camera (VLM may reorder candidates)."""
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    proposals = enumerate_edit_proposals(
        event,
        basis,
        dist_min=dist_min,
        dist_max=dist_max,
        angular_size_min=angular_size_min,
    )
    if not proposals:
        return None

    order = proposals
    rank_notes = "heuristic_angular_size"
    if vlm is not None:
        # Compact list for the VLM; indices map back into ``proposals``.
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
            rank_notes = "vlm_ranked"

    for proposal in order[:max_tries]:
        result = try_edit(controller, cam, proposal)
        if result is not None:
            move, event0, event1 = result
            return move, event0, event1, rank_notes
    return None
