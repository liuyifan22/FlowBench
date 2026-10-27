"""Agent-eye camera search for CIT eval.

Algorithm (single-room / iTHOR)
-------------------------------
1. ``GetReachablePositions`` → subsample with farthest-point sampling (FPS)
   on the floor (xz) to ``max_positions`` (default 20; fast mode 15) well-spread spots.
2. For each position × yaw (8×45° default, or 4×90° in fast mode): flip a coin
   for horizon (0° or 30°), then ``TeleportFull`` and score the view.
3. Keep poses that already show at least one preferred/moveable object in the
   distance band ``(dist_min, dist_max)``.
4. Return up to ``n_views`` poses that are spatially separated on the floor plan.

Apartment mode (ProcTHOR)
-------------------------
Propose 2–3+ views **per room polygon** (FPS inside the room, 8 yaws, coin-flip
horizon), rejecting near-duplicate camera xyz within a room, then globally
select with hard room coverage + diversity scoring.

Scoring rewards mid-band furniture and open depth at the image center
(``GetCoordinateFromRaycast``), and penalizes staring into a nearby wall.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional, Sequence

from ai2thor.controller import Controller

from cit_pipeline.config import DIST_MAX, DIST_MIN, PREFERRED_OBJECT_TYPES
from cit_pipeline.schemas import CameraPose, Vec3
from cit_pipeline.size_filter import is_small_enough_to_move, scene_table_footprint_limit
from cit_pipeline.thor_env import teleport_full


def camera_basis(yaw_deg: float, horizon_deg: float) -> dict[str, Any]:
    """AI2-THOR agent camera axes in world coordinates (Y-up, right-handed).

    Parameters
    ----------
    yaw_deg:
        Agent body yaw (metadata ``agent.rotation.y``).
    horizon_deg:
        Camera pitch. **Positive looks down** (AI2-THOR convention).

    Returns
    -------
    dict with unit vectors ``forward``, ``right``, ``up``.
    ``forward`` points into the scene (look direction).
    """
    import numpy as np

    yaw = math.radians(yaw_deg)
    hor = math.radians(horizon_deg)
    forward = np.array(
        [math.sin(yaw) * math.cos(hor), -math.sin(hor), math.cos(yaw) * math.cos(hor)],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    # right = up × forward  (Y-up)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        # Looking nearly straight up/down — fall back to yaw-only right.
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return {"forward": forward, "right": right, "up": up}


def objects_in_distance_band(
    event: Any,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    preferred_types: Sequence[str] = PREFERRED_OBJECT_TYPES,
    *,
    som: bool = False,
) -> list[dict[str, Any]]:
    """Visible objects with camera distance in ``(dist_min, dist_max)``.

    Modification 1: footprint strictly smaller than scene tables (tables
    excluded). Returns **preferred + other pickupable/moveable** (no hard gate);
    preferred types are listed first only as a soft cue for callers that still
    want a mild prior — edit search should shuffle survivors equally.

    Modification 12: ``som=True`` relaxes Mod-6/9 tiny-mover bans.
    """
    max_fp = scene_table_footprint_limit(event)
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    mid = 0.5 * (dist_min + dist_max)
    for obj in event.metadata["objects"]:
        if not obj.get("visible"):
            continue
        d = obj.get("distance")
        if d is None or not (dist_min < d < dist_max):
            continue
        if not is_small_enough_to_move(obj, max_fp, event, som=som):
            continue
        if obj["objectType"] in preferred_types:
            preferred.append(obj)
        elif obj.get("pickupable") or obj.get("moveable"):
            fallback.append(obj)
    # Soft preference only: preferred first within mid-band sort; never drop fallback.
    preferred.sort(key=lambda o: abs((o.get("distance") or mid) - mid))
    fallback.sort(key=lambda o: abs((o.get("distance") or mid) - mid))
    return preferred + fallback


def center_ray_depth(controller: Controller, event: Any) -> Optional[float]:
    """Depth (meters) from camera to the surface hit at image center."""
    hit_event = controller.step(action="GetCoordinateFromRaycast", x=0.5, y=0.5)
    if not hit_event.metadata["lastActionSuccess"]:
        return None
    hit = hit_event.metadata.get("actionReturn")
    if not hit:
        return None
    cam = event.metadata.get("cameraPosition") or event.metadata["agent"]["position"]
    return math.dist((cam["x"], cam["y"], cam["z"]), (hit["x"], hit["y"], hit["z"]))


def score_view(
    controller: Controller,
    event: Any,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
) -> float:
    """Higher is better: band-aware preferred objects, not staring at a wall."""
    band = objects_in_distance_band(event, dist_min=dist_min, dist_max=dist_max)
    preferred = [o for o in band if o["objectType"] in PREFERRED_OBJECT_TYPES]
    visible = [
        o
        for o in event.metadata["objects"]
        if o.get("visible") and not o["objectType"].startswith("Floor")
    ]
    score = 4.0 * len(preferred) + 1.5 * len(band) + 0.1 * len(visible)

    depth = center_ray_depth(controller, event)
    if depth is None:
        score -= 5.0
    elif depth < 0.6:
        score -= 8.0  # facing a wall / too close
    else:
        score += min(depth, 4.0)
    return score


def fps_sample_positions(
    positions: Sequence[dict[str, float]],
    k: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    """Farthest-point sample ``k`` standing positions spread across the floor (xz)."""
    n = len(positions)
    if n == 0:
        return []
    if k <= 0 or n <= k:
        return [dict(p) for p in positions]

    xs = [float(p["x"]) for p in positions]
    zs = [float(p["z"]) for p in positions]
    chosen: list[int] = [rng.randrange(n)]
    # min squared distance from each point to the chosen set
    min_d2 = [float("inf")] * n
    for _ in range(1, k):
        cx, cz = xs[chosen[-1]], zs[chosen[-1]]
        for i in range(n):
            dx = xs[i] - cx
            dz = zs[i] - cz
            d2 = dx * dx + dz * dz
            if d2 < min_d2[i]:
                min_d2[i] = d2
        # Farthest from the set so far (deterministic argmax; ties → lowest index).
        best_i = 0
        best_d = -1.0
        for i, d2 in enumerate(min_d2):
            if d2 > best_d:
                best_d = d2
                best_i = i
        chosen.append(best_i)
    return [dict(positions[i]) for i in chosen]


def find_agent_eye_cameras(
    controller: Controller,
    *,
    n_views: int = 2,
    max_positions: int = 20,
    n_yaws: int = 8,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    min_separation: float = 1.0,
    positions: Optional[list[dict[str, float]]] = None,
    position_filter: Optional[Callable[[dict[str, float]], bool]] = None,
    rng: Optional[random.Random] = None,
    require_distance_band: bool = True,
) -> list[CameraPose]:
    """Search agent-eye poses that already see move candidates in the distance band.

    Optional ``positions`` / ``position_filter`` restrict the search (ProcTHOR
    per-room cameras). When omitted, uses full ``GetReachablePositions``.

    Defaults: FPS down to 20 floor positions, yaw every 45° (8 dirs), one random
    horizon (0° or 30°) per position×yaw (~160 teleports). Fast mode: 15 FPS × 4
    yaws (90°) ≈ 60 teleports.

    If ``require_distance_band`` is False, keep any successful teleport (used as a
    last-resort fallback so every room with reachable floor still gets a camera).
    """
    rng = rng or random.Random(0)
    if positions is None:
        positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if not positions:
        raise RuntimeError("No reachable positions in scene")
    if position_filter is not None:
        positions = [p for p in positions if position_filter(p)]
    if not positions:
        raise RuntimeError("No reachable positions after position filter")

    candidates = fps_sample_positions(positions, int(max_positions), rng)
    yaws = _yaw_choices(n_yaws, rng)
    horizon_choices = (0, 30)

    scored: list[CameraPose] = []
    for pos in candidates:
        for yaw in yaws:
            # One horizon per view candidate (coin flip), not both.
            horizon = horizon_choices[rng.randrange(2)]
            event = teleport_full(
                controller,
                position=pos,
                rotation=dict(x=0, y=float(yaw), z=0),
                horizon=horizon,
                standing=True,
            )
            if not event.metadata["lastActionSuccess"]:
                continue
            if require_distance_band and not objects_in_distance_band(
                event, dist_min=dist_min, dist_max=dist_max
            ):
                continue
            s = score_view(controller, event, dist_min=dist_min, dist_max=dist_max)
            cam_pos = event.metadata.get("cameraPosition") or event.metadata["agent"]["position"]
            scored.append(
                CameraPose(
                    position=Vec3.from_dict(pos),
                    rotation_y=float(yaw),
                    horizon=float(horizon),
                    standing=True,
                    score=s,
                    camera_position=Vec3.from_dict(cam_pos),
                )
            )

    if not scored:
        raise RuntimeError("Could not find agent-eye pose with objects in distance band")

    # Greedy diversity: take best scores that are ≥ min_separation apart in xz.
    scored.sort(key=lambda c: c.score, reverse=True)
    selected: list[CameraPose] = []
    for cam in scored:
        if all(_pos_dist(cam, other) >= min_separation for other in selected):
            selected.append(cam)
        if len(selected) >= n_views:
            break
    if not selected:
        selected = scored[:n_views]
    return selected


def _pos_dist(a: CameraPose, b: CameraPose) -> float:
    """Floor-plan (xz) distance between agent standing positions."""
    return math.dist(
        (a.position.x, a.position.z),
        (b.position.x, b.position.z),
    )


def _cam_xyz_dist(a: CameraPose, b: CameraPose) -> float:
    """Euclidean distance between camera xyz (falls back to agent position)."""
    pa = a.camera_position or a.position
    pb = b.camera_position or b.position
    return math.dist((pa.x, pa.y, pa.z), (pb.x, pb.y, pb.z))


def _yaw_choices(n_yaws: int, rng: random.Random) -> list[float]:
    n_yaws = max(1, int(n_yaws))
    if n_yaws >= 8:
        return [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    if n_yaws == 4:
        offset = float(rng.choice((0, 15, 30, 45)))
        return [(offset + 90 * i) % 360 for i in range(4)]
    step = 360.0 / float(n_yaws)
    offset = float(rng.uniform(0.0, step))
    return [(offset + step * i) % 360 for i in range(n_yaws)]


def find_apartment_cameras(
    controller: Controller,
    *,
    n_cameras: int = 10,
    rooms: Sequence[dict[str, Any]],
    positions: Optional[list[dict[str, float]]] = None,
    max_positions: int = 48,
    max_positions_per_room: Optional[int] = None,
    proposals_per_room: int = 3,
    n_yaws: int = 8,
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    min_separation: float = 1.0,
    room_diversity_bonus: float = 1000.0,
    require_distance_band: bool = True,
    rng: Optional[random.Random] = None,
) -> list[tuple[CameraPose, str, str]]:
    """Select ``n_cameras`` diverse agent-eye poses across a whole apartment.

    Proposal (per room)
    -------------------
    Reachable positions are filtered to each room polygon, FPS-subsampled, then
    scored over yaw×horizon (horizon is a coin flip of 0°/30°). Each room keeps
    up to ``proposals_per_room`` views that are ≥ ``min_separation`` apart in
    **camera xyz** (so nearby standing spots cannot flood the pool with near-
    duplicate views). When ``n_rooms * proposals_per_room < n_cameras``, the
    per-room budget is raised so studios can still fill ``y``.

    Selection (global)
    ------------------
    Hard room coverage first (best remaining proposal from each uncovered room
    id, subject to separation), then fill with
    ``view_score + room_diversity_bonus`` / density prior and a small same-room
    yaw clash penalty.
    """
    from cit_pipeline.procthor_rooms import (
        filter_positions_in_room,
        room_label,
    )

    rng = rng or random.Random(0)
    n_cameras = max(1, int(n_cameras))
    if positions is None:
        positions = controller.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ]
    if not positions:
        raise RuntimeError("No reachable positions in apartment")

    room_list = list(rooms)
    yaws = _yaw_choices(n_yaws, rng)
    horizon_choices = (0, 30)
    sep = float(min_separation)
    per_room_fps = int(
        max_positions_per_room
        if max_positions_per_room is not None
        else max(10, min(int(max_positions), 16))
    )

    # Rooms that actually have reachable floor.
    rooms_with_floor: list[tuple[dict[str, Any], list[dict[str, float]]]] = []
    for room in room_list:
        room_pos = filter_positions_in_room(list(positions), room)
        if room_pos:
            rooms_with_floor.append((room, room_pos))
    if not rooms_with_floor:
        raise RuntimeError("No reachable positions inside any room polygon")

    n_floor_rooms = len(rooms_with_floor)
    # Never more than 3 viewpoints per room (even if requested y is larger).
    n_cameras = min(n_cameras, 3 * n_floor_rooms)
    # Propose up to 3 per room; do not inflate beyond that to chase a larger y.
    per_room_budget = min(3, max(2, int(proposals_per_room)))

    proposals: list[tuple[CameraPose, str, str]] = []
    for room, room_pos in rooms_with_floor:
        rid = str(room.get("id") or "unknown")
        rtype = str(room.get("roomType") or "Unknown")
        candidates = fps_sample_positions(room_pos, per_room_fps, rng)
        scored_room: list[CameraPose] = []
        for pos in candidates:
            for yaw in yaws:
                horizon = horizon_choices[rng.randrange(2)]
                event = teleport_full(
                    controller,
                    position=pos,
                    rotation=dict(x=0, y=float(yaw), z=0),
                    horizon=horizon,
                    standing=True,
                )
                if not event.metadata["lastActionSuccess"]:
                    continue
                if require_distance_band and not objects_in_distance_band(
                    event, dist_min=dist_min, dist_max=dist_max
                ):
                    continue
                s = score_view(controller, event, dist_min=dist_min, dist_max=dist_max)
                cam_pos = event.metadata.get("cameraPosition") or event.metadata[
                    "agent"
                ]["position"]
                scored_room.append(
                    CameraPose(
                        position=Vec3.from_dict(pos),
                        rotation_y=float(yaw),
                        horizon=float(horizon),
                        standing=True,
                        score=s,
                        camera_position=Vec3.from_dict(cam_pos),
                    )
                )
        if not scored_room:
            continue
        scored_room.sort(key=lambda c: c.score, reverse=True)
        room_picked: list[CameraPose] = []
        for cam in scored_room:
            # Do not propose near-duplicate camera xyz inside this room.
            if any(_cam_xyz_dist(cam, other) < sep for other in room_picked):
                continue
            room_picked.append(cam)
            proposals.append((cam, rid, rtype))
            if len(room_picked) >= per_room_budget:
                break

    if not proposals:
        raise RuntimeError("Could not find apartment camera poses")

    selected: list[tuple[CameraPose, str, str]] = []
    covered_rooms: set[str] = set()
    remaining = list(proposals)
    bonus = float(room_diversity_bonus)
    work_sep = sep

    def _accept(pick: tuple[CameraPose, str, str]) -> None:
        selected.append(pick)
        covered_rooms.add(pick[1])

    # Pass 1 — hard coverage: one camera per room id when possible.
    uncovered = {rid for _, rid, _ in remaining}
    while uncovered and len(selected) < n_cameras and remaining:
        best_i = -1
        best_score = float("-inf")
        for i, (cam, rid, _rtype) in enumerate(remaining):
            if rid not in uncovered:
                continue
            if any(_cam_xyz_dist(cam, other[0]) < work_sep for other in selected):
                continue
            if float(cam.score) > best_score:
                best_score = float(cam.score)
                best_i = i
        if best_i < 0:
            # Separation blocked coverage; relax once then stop this pass.
            if work_sep > 0.35:
                work_sep = max(0.35, work_sep * 0.5)
                continue
            break
        pick = remaining.pop(best_i)
        uncovered.discard(pick[1])
        _accept(pick)

    # Pass 2 — fill with coverage bonus + spatial / yaw diversity.
    work_sep = sep
    while len(selected) < n_cameras and remaining:
        best_i = -1
        best_key = float("-inf")
        for i, (cam, rid, _rtype) in enumerate(remaining):
            if any(_cam_xyz_dist(cam, other[0]) < work_sep for other in selected):
                continue
            room_term = (
                bonus
                if rid not in covered_rooms
                else (
                    0.05
                    * bonus
                    / (1.0 + sum(1 for s in selected if s[1] == rid))
                )
            )
            yaw_pen = 0.0
            for ocam, orid, _ in selected:
                if orid == rid:
                    dyaw = abs((cam.rotation_y - ocam.rotation_y + 180) % 360 - 180)
                    if dyaw < 35:
                        yaw_pen += 2.0
            key = float(cam.score) + room_term - yaw_pen
            if key > best_key:
                best_key = key
                best_i = i
        if best_i < 0:
            if work_sep > 0.35:
                work_sep = max(0.35, work_sep * 0.5)
                continue
            break
        pick = remaining.pop(best_i)
        _accept(pick)

    if not selected:
        proposals.sort(key=lambda t: t[0].score, reverse=True)
        selected = proposals[:n_cameras]
        covered_rooms = {rid for _, rid, _ in selected}

    labels = [f"{room_label({'id': r, 'roomType': t})}" for _, r, t in selected]
    print(
        f"Apartment cameras: selected={len(selected)}/{n_cameras} "
        f"rooms_covered={len(covered_rooms)}/{n_floor_rooms} "
        f"proposals={len(proposals)} per_room_budget={per_room_budget} "
        f"labels={labels}",
        flush=True,
    )
    return selected
