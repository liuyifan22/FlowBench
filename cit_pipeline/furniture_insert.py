"""Furniture-insertion counterfactual: put a large object between / near anchors.

Example commands
----------------
- ``Put a sofa between the table and the window.``
- ``Put a coffee table near the window.``

Unlike object-centric *relocation* of a small pickupable already in the scene,
this **spawns** a comparatively large procedural asset (``SpawnAsset``) into
free floor space between (or beside) visible landmarks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np
from ai2thor.controller import Controller

from cit_pipeline.camera_search import camera_basis
from cit_pipeline.config import DIST_MAX, DIST_MIN
from cit_pipeline.schemas import CameraPose, MoveAction, Vec3
from cit_pipeline.size_filter import aabb_bottom_y, aabb_footprint, aabb_xz_extent
from cit_pipeline.size_filter import object_mask_pixel_count
from cit_pipeline.thor_env import (
    get_object,
    pin_object_pose,
    teleport_full,
    teleport_object,
)

Relation = Literal["between", "near"]

# Furniture we insert (large / mid floor props — NOT small pickupables).
FURNITURE_INSERT_TYPES = (
    "Sofa",
    "ArmChair",
    "CoffeeTable",
    "SideTable",
    "Chair",
    "Ottoman",
    "Stool",
    "Desk",
    "Dresser",
    "TVStand",
    "ShelvingUnit",
)

# Prefer these for demos (readable size, fit in corridors / open floor).
FURNITURE_PREFERRED = (
    "SideTable",
    "Ottoman",
    "Chair",
    "ArmChair",
    "Sofa",
    "CoffeeTable",
    "Stool",
    "TVStand",
    "Desk",
)

# Landmark anchors for "between A and B" / "near A".
FURNITURE_ANCHOR_TYPES = (
    "Window",
    "DiningTable",
    "Sofa",
    "Television",
    "CounterTop",
    "Bed",
    "Painting",
    "ShelvingUnit",
    "Doorway",
    "ArmChair",
    "Desk",
    "Dresser",
    "FloorLamp",
    "HousePlant",
    "TVStand",
    "Fireplace",
)

_TYPE_PHRASE = {
    "CoffeeTable": "coffee table",
    "SideTable": "side table",
    "ArmChair": "armchair",
    "TVStand": "TV stand",
    "ShelvingUnit": "shelving unit",
    "DiningTable": "dining table",
    "CounterTop": "counter",
    "FloorLamp": "floor lamp",
    "HousePlant": "plant",
    "Doorway": "doorway",
    "Television": "television",
}

ANCHOR_MASK_MIN_PX = 120
FURNITURE_MASK_MIN_PX = 80
ANCHOR_SEP_MIN_M = 1.2
ANCHOR_SEP_MAX_M = 5.0
FLOOR_CLEARANCE_M = 0.18
BETWEEN_T_LO = 0.25
BETWEEN_T_HI = 0.75


@dataclass
class FurniturePlan:
    furniture_type: str
    asset_id: str
    relation: Relation
    anchor_a_id: str
    anchor_a_type: str
    anchor_b_id: Optional[str] = None
    anchor_b_type: Optional[str] = None
    score: float = 0.0
    target_xz: Optional[tuple[float, float]] = None
    yaw: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FurnitureResult:
    move: MoveAction
    event0: Any
    event1: Any
    plan: FurniturePlan
    furniture_id: str
    place_meta: dict[str, Any] = field(default_factory=dict)


def _phrase(typ: str) -> str:
    if typ in _TYPE_PHRASE:
        return _TYPE_PHRASE[typ]
    out, buf = [], ""
    for ch in str(typ or "object"):
        if ch.isupper() and buf:
            out.append(buf.lower())
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf.lower())
    return " ".join(out) if out else str(typ).lower()


def format_furniture_command(plan: FurniturePlan) -> str:
    f = _phrase(plan.furniture_type)
    a = _phrase(plan.anchor_a_type)
    article = "an" if f[0] in "aeiou" else "a"
    if plan.relation == "between" and plan.anchor_b_type:
        b = _phrase(plan.anchor_b_type)
        return f"Put {article} {f} between the {a} and the {b}."
    return f"Put {article} {f} near the {a}."


def harvest_furniture_assets(
    house_indices: Sequence[int],
    *,
    split: str = "train",
) -> dict[str, list[str]]:
    """Collect ``assetId``s for insertable furniture from ProcTHOR house JSON."""
    from cit_pipeline.procthor_rooms import load_procthor_house

    by: dict[str, set[str]] = {t: set() for t in FURNITURE_INSERT_TYPES}
    prefixes = {
        "Sofa": ("Sofa_", "RoboTHOR_sofa"),
        "CoffeeTable": ("Coffee_Table_",),
        "SideTable": ("Side_Table_",),
        "Chair": ("Chair_", "RoboTHOR_chair"),
        "ArmChair": ("Armchair_", "Arm_Chair_"),
        "Ottoman": ("Ottoman_",),
        "TVStand": ("TV_Stand_",),
        "Desk": ("Desk_",),
        "Dresser": ("Dresser_",),
        "ShelvingUnit": ("Shelving_Unit_",),
        "Stool": ("Stool_",),
    }

    def walk(objs: Any) -> None:
        for o in objs or []:
            aid = o.get("assetId")
            oid = str(o.get("id") or "")
            typ = oid.split("|", 1)[0] if "|" in oid else None
            if aid and typ in by:
                by[typ].add(aid)
            elif aid:
                for t, prefs in prefixes.items():
                    if any(str(aid).startswith(p) for p in prefs):
                        by[t].add(str(aid))
            walk(o.get("children") or [])

    for i in house_indices:
        house = load_procthor_house(int(i), split=split)
        walk(house.get("objects") or [])
    return {t: sorted(v) for t, v in by.items() if v}


def _floor_y(event: Any) -> float:
    for o in event.metadata["objects"]:
        if o.get("objectType") == "Floor":
            return float(o["position"]["y"])
    ys = []
    for o in event.metadata["objects"]:
        if o.get("objectType") in ("Chair", "Sofa", "DiningTable", "SideTable"):
            b = aabb_bottom_y(o)
            if b is not None:
                ys.append(b)
    return float(min(ys)) if ys else 0.0


def _xz(obj: dict[str, Any]) -> tuple[float, float]:
    p = obj["position"]
    return float(p["x"]), float(p["z"])


def _dist_xz(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _project_between(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Return (t, dist_to_segment) for p projected onto AB in XZ."""
    ax, az = a
    bx, bz = b
    px, pz = p
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    if L2 < 1e-8:
        return 0.5, _dist_xz(p, a)
    t = ((px - ax) * dx + (pz - az) * dz) / L2
    qx, qz = ax + t * dx, az + t * dz
    return float(t), float(math.hypot(px - qx, pz - qz))


def _anchor_candidates(
    event: Any,
    *,
    min_px: int = ANCHOR_MASK_MIN_PX,
) -> list[dict[str, Any]]:
    out = []
    for o in event.metadata["objects"]:
        typ = str(o.get("objectType") or "")
        if typ not in FURNITURE_ANCHOR_TYPES:
            continue
        if not o.get("visible"):
            continue
        n = object_mask_pixel_count(event, o["objectId"])
        if n is None or int(n) < int(min_px):
            continue
        out.append(o)
    return out


def _cell_clear(
    event: Any,
    xz: tuple[float, float],
    *,
    clearance: float,
    ignore_ids: set[str],
) -> bool:
    for o in event.metadata["objects"]:
        oid = o.get("objectId")
        typ = str(o.get("objectType") or "")
        if not oid or oid in ignore_ids:
            continue
        if typ in ("Floor", "Wall", "Ceiling", "Window", "Doorway", "Doorframe"):
            continue
        fp = aabb_footprint(o)
        if fp is None or fp < 0.25:
            continue
        ext = aabb_xz_extent(o)
        if ext is None:
            ox, oz = _xz(o)
            if math.hypot(xz[0] - ox, xz[1] - oz) < clearance + 0.5 * fp:
                return False
            continue
        xmin, xmax, zmin, zmax = ext
        # Expand box by clearance.
        if (
            xmin - clearance <= xz[0] <= xmax + clearance
            and zmin - clearance <= xz[1] <= zmax + clearance
        ):
            return False
    return True


def _nearest_reachable(
    reachable: Sequence[dict[str, float]],
    xz: tuple[float, float],
) -> Optional[dict[str, float]]:
    if not reachable:
        return None
    best = None
    best_d = 1e9
    for p in reachable:
        d = math.hypot(float(p["x"]) - xz[0], float(p["z"]) - xz[1])
        if d < best_d:
            best_d = d
            best = p
    return best


def enumerate_furniture_plans(
    event: Any,
    *,
    asset_catalog: dict[str, list[str]],
    reachable: Sequence[dict[str, float]],
    rng: Optional[random.Random] = None,
    max_plans: int = 40,
) -> list[FurniturePlan]:
    """Ranked (furniture, between/near anchors, target cell) plans."""
    rng = rng or random.Random(0)
    anchors = _anchor_candidates(event)
    if not anchors:
        return []
    agent = event.metadata["agent"]
    cam = event.metadata.get("cameraPosition") or agent["position"]
    cam_xz = (float(cam["x"]), float(cam["z"]))

    plans: list[FurniturePlan] = []
    # Two-anchor between.
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            if a["objectType"] == b["objectType"]:
                # Same-type anchors are ambiguous in text ("the table and the table").
                continue
            axz, bxz = _xz(a), _xz(b)
            sep = _dist_xz(axz, bxz)
            if not (ANCHOR_SEP_MIN_M <= sep <= ANCHOR_SEP_MAX_M):
                continue
            mid = (0.5 * (axz[0] + bxz[0]), 0.5 * (axz[1] + bxz[1]))
            cell = _nearest_reachable(reachable, mid)
            if cell is None:
                continue
            cxz = (float(cell["x"]), float(cell["z"]))
            t, d_seg = _project_between(cxz, axz, bxz)
            if not (BETWEEN_T_LO <= t <= BETWEEN_T_HI) or d_seg > 0.85:
                continue
            if not _cell_clear(
                event,
                cxz,
                clearance=FLOOR_CLEARANCE_M,
                ignore_ids={a["objectId"], b["objectId"]},
            ):
                continue
            # Yaw along A→B.
            yaw = math.degrees(math.atan2(bxz[0] - axz[0], bxz[1] - axz[1]))
            for ftype in FURNITURE_PREFERRED:
                assets = asset_catalog.get(ftype) or []
                if not assets:
                    continue
                # Avoid inserting same type as an anchor when it would confuse SoM.
                if ftype in (a["objectType"], b["objectType"]):
                    continue
                aid = rng.choice(assets)
                score = 3.0 - abs(sep - 2.5) - 0.3 * d_seg
                score += 0.15 if ftype in ("SideTable", "Ottoman", "Chair", "ArmChair") else 0.0
                # Prefer mid-band to camera.
                d_cam = _dist_xz(cxz, cam_xz)
                if DIST_MIN < d_cam < DIST_MAX + 0.5:
                    score += 0.4
                plans.append(
                    FurniturePlan(
                        furniture_type=ftype,
                        asset_id=aid,
                        relation="between",
                        anchor_a_id=a["objectId"],
                        anchor_a_type=str(a["objectType"]),
                        anchor_b_id=b["objectId"],
                        anchor_b_type=str(b["objectType"]),
                        score=score,
                        target_xz=cxz,
                        yaw=float(yaw),
                        meta={"sep_m": sep, "t": t, "d_seg": d_seg},
                    )
                )

    # Single-anchor near (fallback / diversity).
    for a in anchors:
        axz = _xz(a)
        # Offset toward camera from A.
        vx, vz = cam_xz[0] - axz[0], cam_xz[1] - axz[1]
        n = math.hypot(vx, vz) + 1e-6
        for dist in (1.0, 1.4, 1.8):
            mid = (axz[0] + dist * vx / n, axz[1] + dist * vz / n)
            cell = _nearest_reachable(reachable, mid)
            if cell is None:
                continue
            cxz = (float(cell["x"]), float(cell["z"]))
            if _dist_xz(cxz, axz) < 0.7:
                continue
            if not _cell_clear(
                event, cxz, clearance=FLOOR_CLEARANCE_M, ignore_ids={a["objectId"]}
            ):
                continue
            yaw = math.degrees(math.atan2(vx, vz))
            for ftype in FURNITURE_PREFERRED:
                assets = asset_catalog.get(ftype) or []
                if not assets or ftype == a["objectType"]:
                    continue
                aid = rng.choice(assets)
                score = 1.2 - 0.1 * abs(_dist_xz(cxz, axz) - 1.3)
                plans.append(
                    FurniturePlan(
                        furniture_type=ftype,
                        asset_id=aid,
                        relation="near",
                        anchor_a_id=a["objectId"],
                        anchor_a_type=str(a["objectType"]),
                        score=score,
                        target_xz=cxz,
                        yaw=float(yaw),
                        meta={"near_dist": _dist_xz(cxz, axz)},
                    )
                )

    plans.sort(key=lambda p: -p.score)
    # Soft diversify top band.
    if len(plans) > 4:
        top = plans[0].score
        band = [p for p in plans if p.score >= top - 0.8]
        rest = [p for p in plans if p.score < top - 0.8]
        rng.shuffle(band)
        plans = band + rest
    return plans[:max_plans]


def _snap_to_floor(
    controller: Controller,
    object_id: str,
    floor_y: float,
) -> Any:
    ev = controller.step(dict(action="Pass"))
    obj = get_object(ev, object_id)
    if obj is None:
        return ev
    bottom = aabb_bottom_y(obj)
    pos = dict(obj["position"])
    rot = dict(obj.get("rotation") or {"x": 0.0, "y": 0.0, "z": 0.0})
    if bottom is not None:
        pos["y"] = float(pos["y"]) + (float(floor_y) - float(bottom))
    return teleport_object(
        controller, object_id, pos, rot, force=True, kinematic=True
    )


def _spawn_furniture(
    controller: Controller,
    plan: FurniturePlan,
    *,
    floor_y: float,
    generated_id: str,
) -> Optional[Any]:
    assert plan.target_xz is not None
    x, z = plan.target_xz
    ev = controller.step(
        action="SpawnAsset",
        assetId=plan.asset_id,
        generatedId=generated_id,
        position={"x": float(x), "y": float(floor_y), "z": float(z)},
        rotation={"x": 0.0, "y": float(plan.yaw), "z": 0.0},
    )
    if not ev.metadata.get("lastActionSuccess"):
        return None
    _snap_to_floor(controller, generated_id, floor_y)
    return teleport_full  # placeholder type; caller re-teleports camera


def _placement_ok(
    event0: Any,
    event1: Any,
    plan: FurniturePlan,
    furniture_id: str,
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {}
    furn = get_object(event1, furniture_id)
    if furn is None:
        return False, {"note": "missing_furniture"}
    f_px = object_mask_pixel_count(event1, furniture_id)
    report["furniture_mask_px"] = f_px
    if f_px is None or int(f_px) < int(FURNITURE_MASK_MIN_PX):
        return False, {**report, "note": "furniture_not_visible"}

    a0 = get_object(event0, plan.anchor_a_id)
    a1 = get_object(event1, plan.anchor_a_id)
    if a0 is None or a1 is None:
        return False, {**report, "note": "missing_anchor_a"}
    a_px = object_mask_pixel_count(event1, plan.anchor_a_id)
    report["anchor_a_mask_px"] = a_px
    if a_px is None or int(a_px) < int(ANCHOR_MASK_MIN_PX // 2):
        return False, {**report, "note": "anchor_a_not_visible"}

    if plan.relation == "between" and plan.anchor_b_id:
        b1 = get_object(event1, plan.anchor_b_id)
        if b1 is None:
            return False, {**report, "note": "missing_anchor_b"}
        b_px = object_mask_pixel_count(event1, plan.anchor_b_id)
        report["anchor_b_mask_px"] = b_px
        if b_px is None or int(b_px) < int(ANCHOR_MASK_MIN_PX // 2):
            return False, {**report, "note": "anchor_b_not_visible"}
        # Geometric between check after spawn.
        t, d_seg = _project_between(_xz(furn), _xz(a1), _xz(b1))
        report["t"] = t
        report["d_seg"] = d_seg
        if not (0.15 <= t <= 0.85) or d_seg > 1.1:
            return False, {**report, "note": "not_between_anchors"}
    else:
        d = _dist_xz(_xz(furn), _xz(a1))
        report["near_dist"] = d
        if d < 0.5 or d > 2.8:
            return False, {**report, "note": "not_near_anchor"}

    # Reject mid-air / sunk.
    bottom = aabb_bottom_y(furn)
    report["bottom_y"] = bottom
    if bottom is not None and (bottom < -0.05 or bottom > 0.25):
        return False, {**report, "note": "bad_floor_snap"}
    return True, report


def try_furniture_plan(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    plan: FurniturePlan,
    *,
    house: Any,
    spawn_counter: list[int],
) -> Optional[FurnitureResult]:
    """Spawn furniture for one plan; restore via house reset on failure."""
    if plan.target_xz is None:
        return None
    floor_y = _floor_y(event)
    spawn_counter[0] += 1
    furniture_id = f"{plan.furniture_type}|cit_ins|{spawn_counter[0]}"

    # rgb0 before spawn.
    event0 = teleport_full(controller, **cam.teleport_kwargs())

    ev = controller.step(
        action="SpawnAsset",
        assetId=plan.asset_id,
        generatedId=furniture_id,
        position={
            "x": float(plan.target_xz[0]),
            "y": float(floor_y),
            "z": float(plan.target_xz[1]),
        },
        rotation={"x": 0.0, "y": float(plan.yaw), "z": 0.0},
    )
    if not ev.metadata.get("lastActionSuccess"):
        return None
    _snap_to_floor(controller, furniture_id, floor_y)
    event1 = teleport_full(controller, **cam.teleport_kwargs())
    ok, report = _placement_ok(event0, event1, plan, furniture_id)
    if not ok:
        # Soft cleanup: disable if possible, else caller resets house.
        try:
            controller.step(action="DisableObject", objectId=furniture_id)
        except Exception:
            pass
        return None

    furn = get_object(event1, furniture_id)
    assert furn is not None
    new_pos = dict(furn["position"])
    # Synthetic move: "from nowhere" → placed pose (delta from floor origin under target).
    old_pos = {
        "x": float(plan.target_xz[0]),
        "y": float(floor_y),
        "z": float(plan.target_xz[1]),
    }
    delta = {
        "x": new_pos["x"] - old_pos["x"],
        "y": new_pos["y"] - old_pos["y"],
        "z": new_pos["z"] - old_pos["z"],
    }
    text = format_furniture_command(plan)
    place_meta = {
        "relation": plan.relation,
        "furniture_type": plan.furniture_type,
        "asset_id": plan.asset_id,
        "furniture_id": furniture_id,
        "anchor_a_id": plan.anchor_a_id,
        "anchor_a_type": plan.anchor_a_type,
        "anchor_b_id": plan.anchor_b_id,
        "anchor_b_type": plan.anchor_b_type,
        "validation": report,
        "motion_kind": "furniture_insert",
    }
    move = MoveAction(
        object_id=furniture_id,
        object_type=plan.furniture_type,
        direction="forward",
        meters=float(math.sqrt(delta["x"] ** 2 + delta["y"] ** 2 + delta["z"] ** 2) or 0.01),
        delta_world=Vec3.from_dict(delta),
        position_before=Vec3.from_dict(old_pos),
        position_after=Vec3.from_dict(new_pos),
        rotation_before={"x": 0.0, "y": float(plan.yaw), "z": 0.0},
        distance_before=0.0,
        angular_size_proxy=0.0,
        still_visible_in_rgb1=True,
        text=text,
        edit_meta={
            "furniture_insert": place_meta,
            "landmark_id": plan.anchor_a_id,
            "landmark_b_id": plan.anchor_b_id,
        },
    )
    return FurnitureResult(
        move=move,
        event0=event0,
        event1=event1,
        plan=plan,
        furniture_id=furniture_id,
        place_meta=place_meta,
    )


def find_furniture_insert(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    house: Any,
    asset_catalog: dict[str, list[str]],
    reachable: Sequence[dict[str, float]],
    max_tries: int = 20,
    rng: Optional[random.Random] = None,
    spawn_counter: Optional[list[int]] = None,
    prefer_between: bool = True,
) -> Optional[FurnitureResult]:
    rng = rng or random.Random(0)
    spawn_counter = spawn_counter if spawn_counter is not None else [0]
    plans = enumerate_furniture_plans(
        event, asset_catalog=asset_catalog, reachable=reachable, rng=rng, max_plans=60
    )
    if prefer_between:
        plans = sorted(
            plans, key=lambda p: (0 if p.relation == "between" else 1, -p.score)
        )
    attempts = 0
    for plan in plans:
        if attempts >= max_tries:
            break
        attempts += 1
        hit = try_furniture_plan(
            controller, cam, event, plan, house=house, spawn_counter=spawn_counter
        )
        if hit is not None:
            return hit
        # Disable may leave ghost; reset house + camera for next try.
        from cit_pipeline.thor_env import reset_scene

        reset_scene(controller, house)
        event = teleport_full(controller, **cam.teleport_kwargs())
        reachable = (
            controller.step(action="GetReachablePositions").metadata["actionReturn"]
            or reachable
        )
    return None
