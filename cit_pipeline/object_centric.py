"""Object-centric counterfactual motion: Put A onto / next to B.

This is a preliminary motion type for ProcTHOR / iTHOR: instead of sliding an
object along a camera axis, we pick a movable object A and a receptacle /
landmark B, then place A relative to B and re-render.

Relations
----------
- ``on`` / ``onto``: place A on B's supporting surface (spawn-above + place /
  gravity settle).
- ``next_to``: place A beside B in XZ (height held), e.g. chair next to table.

Does not replace the existing eight QA categories; it is a motion primitive
that can later feed new questions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np
from ai2thor.controller import Controller

from cit_pipeline.camera_search import camera_basis
from cit_pipeline.config import (
    ANGULAR_SIZE_MIN,
    DIST_MAX,
    DIST_MIN,
    FORBIDDEN_MOVE_TYPES,
    SOM_MOVER_MASK_MIN_PIXELS,
)
from cit_pipeline.scene_integrity import scene_collateral_ok
from cit_pipeline.schemas import CameraPose, MoveAction, Vec3
from cit_pipeline.spatial_metrics import aabb_surface_dist, point_aabb_surface_dist
from cit_pipeline.thor_env import (
    get_object,
    pin_object_pose,
    settle_object_with_gravity,
    still_visible,
    teleport_full,
    teleport_object,
)
from cit_pipeline.size_filter import (
    BLOCKED_MOVE_TYPES,
    TABLE_TYPES,
    TOO_TINY_MOVE_TYPES,
    aabb_bottom_y,
    aabb_footprint,
    aabb_xz_extent,
    is_small_enough_to_move,
    is_visible_mask_large_enough,
    object_mask_pixel_count,
    scene_table_footprint_limit,
)

Relation = Literal["on", "next_to"]

# ---------------------------------------------------------------------------
# Type vocabularies (A = mover, B = receptacle / landmark)
# ---------------------------------------------------------------------------

# Small / mid pickupables that make sense as "put … on …" subjects.
# Chairs / stools / ottomans are intentionally excluded — sliding furniture is
# visually ambiguous (occlusion, already-adjacent seats, mislabeled targets).
OBJECT_CENTRIC_A_TYPES = (
    # Food / tabletop
    "Apple",
    "Tomato",
    "Egg",
    "Bread",
    "Potato",
    "Lettuce",
    # Dishware / bottles
    "Plate",
    "Bowl",
    "Cup",
    "Mug",
    "Pot",
    "Pan",
    "WineBottle",
    "Glassbottle",
    "Bottle",
    # Electronics / personal
    "CellPhone",
    "RemoteControl",
    "Laptop",
    "AlarmClock",
    "Book",
    # Decor / misc
    "Vase",
    "Statue",
    "Candle",
    "TeddyBear",
    "Pillow",
    "Box",
    "BasketBall",
    "BaseballBat",
    "HousePlant",
    "SoapBottle",
    "SprayBottle",
    "TissueBox",
    "PepperShaker",
    "SaltShaker",
    "SoapBar",
    "DishSponge",
    "Cloth",
    "HandTowel",
    "ToiletPaper",
    "Newspaper",
    "CD",
)

# Large furniture / surfaces that can receive A.
OBJECT_CENTRIC_B_TYPES = (
    # Primary surfaces
    "DiningTable",
    "CoffeeTable",
    "SideTable",
    "Desk",
    "CounterTop",
    # Soft furniture
    "Sofa",
    "Bed",
    # Seats as receptacles (small A on chair / stool / armchair)
    "Chair",
    "Stool",
    "ArmChair",
    # Secondary stands / storage
    "Dresser",
    "ShelvingUnit",
    "TVStand",
)


# Never use these as A (structural / fixtures / too large / too tiny).
OBJECT_CENTRIC_A_BAN = frozenset(BLOCKED_MOVE_TYPES) | frozenset(FORBIDDEN_MOVE_TYPES) | frozenset(
    TOO_TINY_MOVE_TYPES
) | frozenset(
    {
        "Sink",
        "SinkBasin",
        "Toilet",
        "Bathtub",
        "BathtubBasin",
        "Fridge",
        "Refrigerator",
        "Microwave",
        "Television",
        "FloorLamp",
        "Painting",
        "Mirror",
        "Window",
        "Door",
        "Stairs",
        "LightSwitch",
        "Faucet",
        "StoveKnob",
        "StoveBurner",
        "Cabinet",
        "Drawer",
        "Shelf",
        # Ambiguous furniture movers (see comments_on_v9 follow-up).
        "Chair",
        "Stool",
        "Ottoman",
        "ArmChair",
    }
)

# Stairs / fixtures that look like receptacles but are bad targets.
OBJECT_CENTRIC_B_BAN = frozenset(
    {
        "Floor",
        "Wall",
        "Ceiling",
        "Window",
        "Door",
        "Doorway",
        "Doorframe",
        "Stairs",
        "Sink",
        "SinkBasin",
        "Toilet",
        "Bathtub",
        "BathtubBasin",
        "ShowerGlass",
        "ShowerDoor",
        "ShowerHead",
        "Faucet",
        "LightSwitch",
        "Rug",
    }
)

# Relations preferred per mover class.
# next_to was chair-centric; chairs are no longer movers (A-ban), kept empty.
_NEXT_TO_PREFERRED_A = frozenset()
# Surfaces that look like real "put X on Y" targets (includes seats).
_ON_PREFERRED_B = frozenset(TABLE_TYPES) | frozenset(
    {
        "Sofa",
        "Bed",
        "Dresser",
        "ShelvingUnit",
        "TVStand",
        "Chair",
        "Stool",
        "ArmChair",
    }
)
# next_to landmarks should be large furniture, not another seat.
_NEXT_TO_PREFERRED_B = frozenset(TABLE_TYPES) | frozenset(
    {"Sofa", "Bed", "Dresser", "ShelvingUnit", "TVStand", "CounterTop"}
)

# Human-readable phrases for motion commands.
_TYPE_PHRASE_OVERRIDE = {
    "RemoteControl": "remote",
    "CellPhone": "phone",
    "CounterTop": "counter",
    "DiningTable": "dining table",
    "CoffeeTable": "coffee table",
    "SideTable": "side table",
    "HousePlant": "plant",
    "TVStand": "TV stand",
    "ShelvingUnit": "shelving unit",
    "ArmChair": "armchair",
    "Glassbottle": "glass bottle",
    "WineBottle": "wine bottle",
    "SoapBottle": "soap bottle",
    "SprayBottle": "spray bottle",
    "AlarmClock": "alarm clock",
    "TeddyBear": "teddy bear",
    "BasketBall": "basketball",
    "BaseballBat": "baseball bat",
    "PepperShaker": "pepper shaker",
    "SaltShaker": "salt shaker",
    "DishSponge": "sponge",
    "ToiletPaper": "toilet paper",
    "HandTowel": "hand towel",
    "SoapBar": "soap",
    "TissueBox": "tissue box",
}

NEXT_TO_GAP_M = 0.08
NEXT_TO_MAX_SURFACE_DIST_M = 0.55
ON_MAX_SURFACE_DIST_M = 0.35
MIN_HORIZONTAL_MOVE_M = 0.18
SPAWN_JITTER_TRIES = 8
# Visibility gates (patch): B must be a large on-screen receptacle; A must
# remain clearly visible after the place (not just metadata-visible).
RECEPTACLE_MASK_MIN_PIXELS = 200
MOVER_AFTER_MASK_MIN_PIXELS = 40
# Clearance from other objects already on B (XZ), in addition to PlaceObjectAtPoint.
ON_SURFACE_CLEARANCE_M = 0.12


@dataclass
class PlacePair:
    mover_id: str
    mover_type: str
    receptacle_id: str
    receptacle_type: str
    relation: Relation
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaceResult:
    move: MoveAction
    event0: Any
    event1: Any
    pair: PlacePair
    place_meta: dict[str, Any] = field(default_factory=dict)


def _type_phrase(object_type: str) -> str:
    typ = str(object_type or "object")
    if typ in _TYPE_PHRASE_OVERRIDE:
        return _TYPE_PHRASE_OVERRIDE[typ]
    # CamelCase → spaced words: "SideTable" already handled; fallback split.
    out = []
    buf = ""
    for ch in typ:
        if ch.isupper() and buf:
            out.append(buf.lower())
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf.lower())
    return " ".join(out) if out else typ.lower()


def format_object_centric_command(
    mover_type: str,
    receptacle_type: str,
    relation: Relation,
) -> str:
    a = _type_phrase(mover_type)
    b = _type_phrase(receptacle_type)
    if relation == "next_to":
        return f"Put the {a} next to the {b}."
    return f"Put the {a} on the {b}."


def _is_floor_parent(pid: Any) -> bool:
    if not pid or not isinstance(pid, str):
        return False
    return pid == "Floor" or pid.startswith("Floor")


def _parents(obj: dict[str, Any]) -> list[str]:
    return [p for p in (obj.get("parentReceptacles") or []) if isinstance(p, str)]


def _already_on(mover: dict[str, Any], receptacle_id: str) -> bool:
    return receptacle_id in _parents(mover)


def _objects_on_receptacle(
    event: Any,
    receptacle_id: str,
    *,
    exclude_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Objects currently parented to ``receptacle_id`` (exclude mover)."""
    by_id: dict[str, dict[str, Any]] = {}
    for o in event.metadata["objects"]:
        oid = o.get("objectId")
        if not oid or oid == exclude_id or oid == receptacle_id:
            continue
        parents = _parents(o)
        if receptacle_id in parents or any(
            p.startswith(receptacle_id + "|") or p.startswith(receptacle_id + "___")
            for p in parents
        ):
            by_id[str(oid)] = o
    rec = get_object(event, receptacle_id)
    if rec is not None:
        for cid in rec.get("receptacleObjectIds") or []:
            if not cid or cid == exclude_id:
                continue
            if cid in by_id:
                continue
            obj = get_object(event, cid)
            if obj is not None:
                by_id[str(cid)] = obj
    return list(by_id.values())


def _candidate_clear_of_occupants(
    pos: dict[str, float],
    mover: dict[str, Any],
    occupants: Sequence[dict[str, Any]],
    *,
    clearance_m: float = ON_SURFACE_CLEARANCE_M,
) -> bool:
    """True if spawn XZ is far enough from other objects already on B."""
    a_fp = aabb_footprint(mover) or 0.12
    need = clearance_m + 0.5 * float(a_fp)
    for o in occupants:
        o_fp = aabb_footprint(o) or 0.10
        need_o = need + 0.5 * float(o_fp)
        # Prefer AABB surface distance in XZ when available.
        ext = aabb_xz_extent(o)
        if ext is not None:
            xmin, xmax, zmin, zmax = ext
            dx = max(0.0, max(xmin - float(pos["x"]), float(pos["x"]) - xmax))
            dz = max(0.0, max(zmin - float(pos["z"]), float(pos["z"]) - zmax))
            # Outside AABB: Euclidean to box; inside → colliding.
            if dx == 0.0 and dz == 0.0:
                return False
            if math.hypot(dx, dz) < clearance_m:
                return False
        else:
            op = o.get("position") or {}
            if "x" not in op or "z" not in op:
                continue
            if math.hypot(float(pos["x"]) - float(op["x"]), float(pos["z"]) - float(op["z"])) < need_o:
                return False
    return True


def _no_surface_collision_after(
    event: Any,
    mover_id: str,
    receptacle_id: str,
    *,
    clearance_m: float = ON_SURFACE_CLEARANCE_M,
) -> tuple[bool, dict[str, Any]]:
    """Reject placements that interpenetrate other objects on the same surface."""
    mover = get_object(event, mover_id)
    if mover is None:
        return False, {"note": "missing_mover"}
    occupants = _objects_on_receptacle(event, receptacle_id, exclude_id=mover_id)
    closest = None
    for o in occupants:
        d = aabb_surface_dist(mover, o)
        if d is None:
            continue
        if closest is None or d < closest:
            closest = d
        if d < clearance_m:
            return False, {
                "note": "surface_collision",
                "other_id": o["objectId"],
                "other_type": o.get("objectType"),
                "aabb_surface_dist": d,
            }
    return True, {"occupants": len(occupants), "min_occupant_dist": closest}


def _type_id_consistent(obj: dict[str, Any]) -> bool:
    """Drop ProcTHOR quirks where objectType disagrees with objectId prefix."""
    typ = str(obj.get("objectType") or "")
    oid = str(obj.get("objectId") or "")
    if not typ or not oid:
        return False
    prefix = oid.split("|", 1)[0]
    return prefix == typ


def _aabb_top_y(obj: dict[str, Any]) -> Optional[float]:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    size = aabb.get("size")
    if center and size and "y" in center and "y" in size:
        return float(center["y"]) + 0.5 * float(size["y"])
    return None


def _mover_height(obj: dict[str, Any]) -> float:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    size = aabb.get("size") or {}
    if "y" in size:
        return max(0.02, float(size["y"]))
    return 0.12


def is_object_centric_a(obj: dict[str, Any], event: Any, *, som: bool = True) -> bool:
    """True if ``obj`` is a sensible movable subject A."""
    typ = str(obj.get("objectType") or "")
    if typ in OBJECT_CENTRIC_A_BAN:
        return False
    if not obj.get("visible"):
        return False
    if not (obj.get("pickupable") or obj.get("moveable")):
        return False
    # Prefer explicit allow-list; also accept other pickupables that pass size.
    if typ not in OBJECT_CENTRIC_A_TYPES and not obj.get("pickupable"):
        return False
    max_fp = scene_table_footprint_limit(event)
    if not is_small_enough_to_move(obj, max_fp, event, som=som):
        return False
    if not is_visible_mask_large_enough(event, obj, min_pixels=SOM_MOVER_MASK_MIN_PIXELS):
        return False
    return True


def is_object_centric_b(
    obj: dict[str, Any],
    event: Any,
    *,
    relation: Relation,
    min_mask_pixels: int = RECEPTACLE_MASK_MIN_PIXELS,
) -> bool:
    """True if ``obj`` is a sensible, large on-screen receptacle / landmark B."""
    typ = str(obj.get("objectType") or "")
    if typ in OBJECT_CENTRIC_B_BAN:
        return False
    if not obj.get("visible"):
        return False
    if not _type_id_consistent(obj):
        return False
    # Patch 2: B must cover enough pixels so the target is unambiguous.
    n_px = object_mask_pixel_count(event, obj["objectId"])
    if n_px is None or int(n_px) < int(min_mask_pixels):
        return False
    if relation == "on":
        if typ in _ON_PREFERRED_B:
            return True
        if typ in OBJECT_CENTRIC_B_TYPES:
            return True
        return False
    # next_to: large furniture landmarks only.
    if typ in _NEXT_TO_PREFERRED_B:
        return True
    if bool(obj.get("receptacle")) and typ not in OBJECT_CENTRIC_A_BAN:
        fp = aabb_footprint(obj)
        if fp is not None and fp >= 0.45:
            return True
    return False


def _pair_allowed(
    mover: dict[str, Any],
    receptacle: dict[str, Any],
    relation: Relation,
) -> bool:
    """Hard filters for human-like Put A onto / next to B."""
    a_typ = str(mover["objectType"])
    b_typ = str(receptacle["objectType"])
    if mover["objectId"] == receptacle["objectId"]:
        return False
    if _already_on(mover, receptacle["objectId"]):
        return False
    if _already_on(receptacle, mover["objectId"]):
        return False
    # Never same-type pairs.
    if a_typ == b_typ:
        return False

    d = aabb_surface_dist(mover, receptacle)
    if relation == "next_to":
        # Chair-style next_to movers were removed; keep hook for future use.
        if not _NEXT_TO_PREFERRED_A or a_typ not in _NEXT_TO_PREFERRED_A:
            return False
        if b_typ not in _NEXT_TO_PREFERRED_B:
            return False
        if d is not None and d < 0.08:
            return False
        return True

    # relation == "on"
    if not mover.get("pickupable"):
        return False
    if a_typ in OBJECT_CENTRIC_A_BAN:
        return False
    return b_typ in _ON_PREFERRED_B or b_typ in OBJECT_CENTRIC_B_TYPES


def _pair_score(
    mover: dict[str, Any],
    receptacle: dict[str, Any],
    relation: Relation,
) -> float:
    d = aabb_surface_dist(mover, receptacle)
    if d is None:
        d = math.dist(
            (mover["position"]["x"], mover["position"]["z"]),
            (receptacle["position"]["x"], receptacle["position"]["z"]),
        )
    # Prefer mid-distance pairs (not already stacked, not across the house).
    score = 3.0 - abs(float(d) - 1.0)
    a_typ = str(mover["objectType"])
    b_typ = str(receptacle["objectType"])
    if relation == "on":
        if b_typ in TABLE_TYPES:
            score += 2.5
        elif b_typ in ("Sofa", "Bed"):
            score += 2.0
        elif b_typ in ("Chair", "Stool", "ArmChair"):
            score += 1.2
        elif b_typ in _ON_PREFERRED_B:
            score += 1.5
        if mover.get("pickupable"):
            score += 0.5
        # Classic demos: apple/remote/plate/phone → table/sofa/counter.
        if a_typ in (
            "Apple",
            "Plate",
            "Cup",
            "Mug",
            "Bowl",
            "RemoteControl",
            "CellPhone",
            "Book",
            "Vase",
            "Candle",
            "HousePlant",
        ):
            score += 0.8
    else:
        if a_typ in _NEXT_TO_PREFERRED_A and b_typ in TABLE_TYPES:
            score += 3.0
        elif a_typ in _NEXT_TO_PREFERRED_A and b_typ in _NEXT_TO_PREFERRED_B:
            score += 1.5
    for o in (mover, receptacle):
        dist = o.get("distance")
        if dist is not None and DIST_MIN < float(dist) < DIST_MAX:
            score += 0.25
    if 0.25 < float(d) < 2.5:
        score += 0.4
    return float(score)


def enumerate_place_pairs(
    event: Any,
    *,
    relations: Sequence[Relation] = ("on", "next_to"),
    dist_min: float = DIST_MIN,
    dist_max: float = DIST_MAX,
    som: bool = True,
    max_pairs: int = 40,
    rng: Optional[random.Random] = None,
) -> list[PlacePair]:
    """Enumerate ranked (A, B, relation) candidates visible in ``event``."""
    rng = rng or random.Random(0)
    movers = [
        o
        for o in event.metadata["objects"]
        if is_object_centric_a(o, event, som=som)
        and (o.get("distance") is None or dist_min < float(o["distance"]) < dist_max + 0.75)
    ]
    if not movers:
        return []

    pairs: list[PlacePair] = []
    for relation in relations:
        receptacles = [
            o
            for o in event.metadata["objects"]
            if is_object_centric_b(o, event, relation=relation)
            and (o.get("distance") is None or float(o["distance"]) < dist_max + 1.5)
        ]
        for mover in movers:
            for rec in receptacles:
                if not _pair_allowed(mover, rec, relation):
                    continue
                score = _pair_score(mover, rec, relation)
                # Prefer larger on-screen receptacles.
                b_px = object_mask_pixel_count(event, rec["objectId"]) or 0
                score += min(1.5, 0.0015 * float(b_px))
                pairs.append(
                    PlacePair(
                        mover_id=mover["objectId"],
                        mover_type=str(mover["objectType"]),
                        receptacle_id=rec["objectId"],
                        receptacle_type=str(rec["objectType"]),
                        relation=relation,
                        score=score,
                        meta={"receptacle_mask_px": int(b_px)},
                    )
                )
    pairs.sort(key=lambda p: -p.score)
    # Soft diversify only among near-tied top scores (keep strong pairs first).
    if len(pairs) > 3:
        top_score = pairs[0].score
        band = [p for p in pairs if p.score >= top_score - 0.75]
        rest = [p for p in pairs if p.score < top_score - 0.75]
        rng.shuffle(band)
        pairs = band + rest
    return pairs[:max_pairs]


def _spawn_coords_above(
    controller: Controller,
    receptacle_id: str,
) -> list[dict[str, float]]:
    """Ask AI2-THOR for candidate points above a receptacle surface."""
    coords: list[dict[str, float]] = []
    for anywhere in (True, False):
        ev = controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=anywhere,
        )
        if not ev.metadata.get("lastActionSuccess"):
            continue
        ret = ev.metadata.get("actionReturn") or []
        if isinstance(ret, list):
            for c in ret:
                if isinstance(c, dict) and "x" in c and "y" in c and "z" in c:
                    coords.append(
                        {"x": float(c["x"]), "y": float(c["y"]), "z": float(c["z"])}
                    )
        if coords:
            break
    return coords


def _aabb_surface_candidates(
    receptacle: dict[str, Any],
    mover: dict[str, Any],
    *,
    rng: random.Random,
    n: int = SPAWN_JITTER_TRIES,
) -> list[dict[str, float]]:
    """Fallback: sample points on B's AABB top face, slightly above."""
    aabb = receptacle.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    size = aabb.get("size")
    if not center or not size:
        return []
    cx, cy, cz = float(center["x"]), float(center["y"]), float(center["z"])
    sx, sy, sz = float(size["x"]), float(size["y"]), float(size["z"])
    top_y = cy + 0.5 * sy
    h = _mover_height(mover)
    # Shrink inset so we land on the surface, not the edge lip.
    inset_x = max(0.02, 0.35 * sx)
    inset_z = max(0.02, 0.35 * sz)
    out: list[dict[str, float]] = []
    for _ in range(n):
        dx = rng.uniform(-inset_x, inset_x)
        dz = rng.uniform(-inset_z, inset_z)
        out.append({"x": cx + dx, "y": top_y + 0.5 * h + 0.02, "z": cz + dz})
    # Always include dead center.
    out.insert(0, {"x": cx, "y": top_y + 0.5 * h + 0.02, "z": cz})
    return out


def _next_to_candidates(
    receptacle: dict[str, Any],
    mover: dict[str, Any],
    basis: dict[str, np.ndarray],
    *,
    rng: random.Random,
) -> list[dict[str, float]]:
    """XZ offsets around B's footprint; keep A's original height."""
    ext = aabb_xz_extent(receptacle)
    if ext is None:
        return []
    xmin, xmax, zmin, zmax = ext
    a_fp = aabb_footprint(mover) or 0.25
    gap = NEXT_TO_GAP_M + 0.5 * a_fp
    y = float(mover["position"]["y"])
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    sides = [
        {"x": xmax + gap, "y": y, "z": cz},
        {"x": xmin - gap, "y": y, "z": cz},
        {"x": cx, "y": y, "z": zmax + gap},
        {"x": cx, "y": y, "z": zmin - gap},
    ]
    # Prefer the side toward the camera (more likely still visible).
    right = np.asarray(basis["right"], dtype=np.float64)
    forward = np.asarray(basis["forward"], dtype=np.float64)
    scored: list[tuple[float, dict[str, float]]] = []
    for p in sides:
        delta = np.array([p["x"] - cx, 0.0, p["z"] - cz], dtype=np.float64)
        # Prefer toward camera (-forward) and a bit of lateral variety.
        toward_cam = -float(np.dot(delta, forward))
        scored.append((toward_cam + 0.05 * rng.random(), p))
    scored.sort(key=lambda t: -t[0])
    # Also add a couple of diagonal corners.
    for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sides_extra = {
            "x": (xmax if sx > 0 else xmin) + sx * gap * 0.7,
            "y": y,
            "z": (zmax if sz > 0 else zmin) + sz * gap * 0.7,
        }
        scored.append((rng.random() * 0.1, sides_extra))
    scored.sort(key=lambda t: -t[0])
    # Dedup roughly.
    out: list[dict[str, float]] = []
    for _, p in scored:
        if all(
            math.hypot(p["x"] - q["x"], p["z"] - q["z"]) > 0.05 for q in out
        ):
            out.append(p)
    return out


def _place_at_point(
    controller: Controller,
    object_id: str,
    position: dict[str, float],
    rotation: dict[str, float],
    *,
    force: bool = False,
) -> Any:
    """Prefer PlaceObjectAtPoint; fall back to TeleportObject.

    AI2-THOR 5.0 ``PlaceObjectAtPoint`` takes ``forceKinematic`` (not
    ``forceAction``). When the spawn cell is occupied, fall back to a
    kinematic teleport slightly above the point + optional gravity settle
    in the caller.
    """
    event = controller.step(
        action="PlaceObjectAtPoint",
        objectId=object_id,
        position=position,
        rotation=rotation,
        forceKinematic=True,
    )
    if event.metadata.get("lastActionSuccess"):
        return event
    # Teleport slightly above the spawn point; caller may settle.
    lift = {
        "x": float(position["x"]),
        "y": float(position["y"]) + 0.04,
        "z": float(position["z"]),
    }
    return teleport_object(
        controller,
        object_id,
        lift,
        rotation,
        force=force,
        allow_surface_place=False,
        kinematic=True,
    )


def _horizontal_move_m(a: dict[str, float], b: dict[str, float]) -> float:
    return float(math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"])))


def _placement_ok(
    event0: Any,
    event1: Any,
    pair: PlacePair,
    *,
    old_pos: dict[str, float],
) -> tuple[bool, dict[str, Any]]:
    """Validate A moved, stayed visible, and is on/near B without collateral tip."""
    report: dict[str, Any] = {}
    after = get_object(event1, pair.mover_id)
    before = get_object(event0, pair.mover_id)
    rec1 = get_object(event1, pair.receptacle_id)
    rec0 = get_object(event0, pair.receptacle_id)
    rec = rec1 or rec0
    if after is None or before is None or rec is None:
        return False, {"note": "missing_object"}
    if bool(after.get("isBroken")):
        return False, {"note": "mover_broken"}

    # Patch 1: A must be clearly on-screen after the move (mask, not metadata).
    a_px1 = object_mask_pixel_count(event1, pair.mover_id)
    report["mover_mask_px_after"] = a_px1
    if a_px1 is None or int(a_px1) < int(MOVER_AFTER_MASK_MIN_PIXELS):
        return False, {**report, "note": "mover_not_visible_enough"}
    if not still_visible(event1, pair.mover_id):
        return False, {**report, "note": "mover_not_visible"}

    # Patch 2: B must be large on-screen before *and* after.
    b_px0 = object_mask_pixel_count(event0, pair.receptacle_id)
    b_px1 = object_mask_pixel_count(event1, pair.receptacle_id)
    report["receptacle_mask_px_before"] = b_px0
    report["receptacle_mask_px_after"] = b_px1
    if b_px0 is None or int(b_px0) < int(RECEPTACLE_MASK_MIN_PIXELS):
        return False, {**report, "note": "receptacle_too_small_before"}
    if b_px1 is None or int(b_px1) < int(RECEPTACLE_MASK_MIN_PIXELS):
        return False, {**report, "note": "receptacle_too_small_after"}
    if not still_visible(event1, pair.receptacle_id):
        return False, {**report, "note": "receptacle_not_visible"}

    new_pos = dict(after["position"])
    horiz = _horizontal_move_m(old_pos, new_pos)
    dy = float(new_pos["y"]) - float(old_pos["y"])
    report["horizontal_m"] = horiz
    report["dy"] = dy
    if horiz < MIN_HORIZONTAL_MOVE_M and abs(dy) < 0.08:
        return False, {**report, "note": "too_small_move"}

    parents = _parents(after)
    report["parents"] = parents
    surf = point_aabb_surface_dist(new_pos, rec)
    report["surface_dist"] = surf

    if pair.relation == "on":
        # Require B (or a same-type parent id) in parentReceptacles.
        on_b = pair.receptacle_id in parents
        if not on_b:
            rid = pair.receptacle_id
            on_b = any(
                (not _is_floor_parent(p))
                and (
                    p == rid
                    or p.startswith(rid + "|")
                    or p.startswith(rid + "___")
                    or rid.startswith(p + "|")
                    or rid.startswith(p + "___")
                )
                for p in parents
            )
        if not on_b:
            rec_type = pair.receptacle_type
            on_b = any(
                (not _is_floor_parent(p)) and p.split("|", 1)[0] == rec_type
                for p in parents
            )
        near = surf is not None and surf <= ON_MAX_SURFACE_DIST_M
        bottom = aabb_bottom_y(after)
        top = _aabb_top_y(rec)
        height_ok = (
            bottom is not None
            and top is not None
            and abs(float(bottom) - float(top)) < 0.25
        )
        report["on_b"] = on_b
        report["near"] = near
        report["height_ok"] = height_ok
        if not on_b:
            return False, {**report, "note": "not_on_receptacle"}
        if not (height_ok or near):
            return False, {**report, "note": "not_near_surface"}
        ok_clear, clear_meta = _no_surface_collision_after(
            event1, pair.mover_id, pair.receptacle_id
        )
        report["clearance"] = clear_meta
        if not ok_clear:
            return False, {**report, "note": clear_meta.get("note", "surface_collision")}
    else:
        if surf is None or surf > NEXT_TO_MAX_SURFACE_DIST_M:
            return False, {**report, "note": "not_next_to"}
        if pair.receptacle_id in parents:
            return False, {**report, "note": "became_on_instead"}

    ok_col, col_report = scene_collateral_ok(event0, event1, pair.mover_id)
    report["collateral"] = col_report
    if not ok_col:
        return False, {**report, "note": "collateral"}
    return True, report


def try_place_pair(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    pair: PlacePair,
    *,
    rng: Optional[random.Random] = None,
    settle: bool = True,
) -> Optional[PlaceResult]:
    """Attempt one (A, B, relation) placement; restore on failure.

    Returns rgb0 (pre-move) and rgb1 (post-move) events with A pinned at the
    final pose for rendering.
    """
    rng = rng or random.Random(0)
    mover0 = get_object(event, pair.mover_id)
    rec0 = get_object(event, pair.receptacle_id)
    if mover0 is None or rec0 is None:
        return None
    old_pos = dict(mover0["position"])
    old_rot = dict(mover0.get("rotation") or {"x": 0.0, "y": 0.0, "z": 0.0})
    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])

    if pair.relation == "on":
        occupants = _objects_on_receptacle(
            event, pair.receptacle_id, exclude_id=pair.mover_id
        )
        candidates = _spawn_coords_above(controller, pair.receptacle_id)
        if candidates:
            rng.shuffle(candidates)
            # Prefer points on the camera-facing side of B so A stays visible.
            cam_pos = event.metadata.get("cameraPosition") or agent["position"]
            center = rec0["position"]

            def _toward_camera_score(c: dict[str, float]) -> float:
                d_cam = math.hypot(
                    float(c["x"]) - float(cam_pos["x"]),
                    float(c["z"]) - float(cam_pos["z"]),
                )
                vx = float(c["x"]) - float(center["x"])
                vz = float(c["z"]) - float(center["z"])
                cx = float(cam_pos["x"]) - float(center["x"])
                cz = float(cam_pos["z"]) - float(center["z"])
                n = math.hypot(vx, vz) * math.hypot(cx, cz) + 1e-6
                align = (vx * cx + vz * cz) / n
                return d_cam - 1.5 * align

            # Drop occupied cells first (avoid #17-style interpenetration).
            clear = [
                c
                for c in candidates
                if _candidate_clear_of_occupants(c, mover0, occupants)
            ]
            if clear:
                candidates = clear
            candidates.sort(key=_toward_camera_score)
            if len(candidates) > 28:
                step = max(1, len(candidates) // 28)
                candidates = candidates[::step][:28]
        fallback = [
            c
            for c in _aabb_surface_candidates(rec0, mover0, rng=rng)
            if _candidate_clear_of_occupants(c, mover0, occupants)
        ]
        candidates.extend(fallback)
    else:
        candidates = _next_to_candidates(rec0, mover0, basis, rng=rng)

    if not candidates:
        return None

    # Capture rgb0 first (clean pose), then try placements from a restored state.
    pin_object_pose(controller, pair.mover_id, old_pos, old_rot)
    event0 = teleport_full(controller, **cam.teleport_kwargs())

    # For "on": prefer collision-checked PlaceObjectAtPoint (force=False path
    # via no TeleportObject clip). Only fall back to kinematic teleport without
    # force-through-furniture when place fails.
    used_force_flags = (False,) if pair.relation == "on" else (False, True)
    for pos in candidates[:18]:
        for force in used_force_flags:
            # Restore before each attempt.
            pin_object_pose(controller, pair.mover_id, old_pos, old_rot)
            teleport_full(controller, **cam.teleport_kwargs())

            # Prefer clear PlaceObjectAtPoint; skip teleport fallback when the
            # cell is occupied (PlaceObjectAtPoint already failed).
            place_ev = controller.step(
                action="PlaceObjectAtPoint",
                objectId=pair.mover_id,
                position=pos,
                rotation=old_rot,
                forceKinematic=True,
            )
            used_place_action = True
            if not place_ev.metadata.get("lastActionSuccess"):
                if pair.relation == "on":
                    # Occupied / blocked — try next spawn, do not force-clip.
                    continue
                place_ev = _place_at_point(
                    controller,
                    pair.mover_id,
                    pos,
                    old_rot,
                    force=force,
                )
                used_place_action = (
                    str(place_ev.metadata.get("lastAction") or "") == "PlaceObjectAtPoint"
                )
                if not place_ev.metadata.get("lastActionSuccess"):
                    continue

            placed = get_object(place_ev, pair.mover_id)
            if placed is None:
                continue
            intended = dict(placed["position"])
            intended_rot = dict(placed.get("rotation") or old_rot)

            # Gravity settle when we teleported above the surface (not when
            # PlaceObjectAtPoint already snapped onto B).
            did_settle = False
            if (
                settle
                and pair.relation == "on"
                and not used_place_action
                and mover0.get("pickupable")
            ):
                settle_object_with_gravity(
                    controller,
                    pair.mover_id,
                    intended,
                    intended_rot,
                    max_steps=60,
                    time_step=0.05,
                    min_steps=10,
                    force_place=True,
                )
                did_settle = True
            elif settle and pair.relation == "next_to" and mover0.get("pickupable"):
                # Light settle so small objects don't float beside B.
                settle_object_with_gravity(
                    controller,
                    pair.mover_id,
                    intended,
                    intended_rot,
                    max_steps=40,
                    time_step=0.05,
                    min_steps=8,
                    force_place=True,
                )
                did_settle = True

            event1 = teleport_full(controller, **cam.teleport_kwargs())
            ok, report = _placement_ok(event0, event1, pair, old_pos=old_pos)
            if not ok:
                continue

            after = get_object(event1, pair.mover_id)
            assert after is not None
            new_pos = dict(after["position"])
            new_rot = dict(after.get("rotation") or intended_rot)
            # Pin final pose so rgb1 is stable for SoM / flow.
            pin_object_pose(controller, pair.mover_id, new_pos, new_rot)
            event1 = teleport_full(controller, **cam.teleport_kwargs())

            delta = {
                "x": new_pos["x"] - old_pos["x"],
                "y": new_pos["y"] - old_pos["y"],
                "z": new_pos["z"] - old_pos["z"],
            }
            meters = float(math.sqrt(delta["x"] ** 2 + delta["y"] ** 2 + delta["z"] ** 2))
            right = np.asarray(basis["right"], dtype=np.float64)
            forward = np.asarray(basis["forward"], dtype=np.float64)
            dvec = np.array([delta["x"], 0.0, delta["z"]], dtype=np.float64)
            comps = [
                ("right", float(np.dot(dvec, right))),
                ("left", float(np.dot(dvec, -right))),
                ("forward", float(np.dot(dvec, forward))),
                ("back", float(np.dot(dvec, -forward))),
            ]
            direction = max(comps, key=lambda t: t[1])[0]
            text = format_object_centric_command(
                pair.mover_type, pair.receptacle_type, pair.relation
            )
            place_meta = {
                "relation": pair.relation,
                "receptacle_id": pair.receptacle_id,
                "receptacle_type": pair.receptacle_type,
                "force": force,
                "candidate": pos,
                "validation": report,
                "motion_kind": "object_centric",
                "used_place_action": used_place_action,
                "did_settle": did_settle,
            }
            move = MoveAction(
                object_id=pair.mover_id,
                object_type=pair.mover_type,
                direction=direction,  # type: ignore[arg-type]
                meters=meters,
                delta_world=Vec3.from_dict(delta),
                position_before=Vec3.from_dict(old_pos),
                position_after=Vec3.from_dict(new_pos),
                rotation_before=old_rot,
                distance_before=float(mover0.get("distance") or 0.0),
                angular_size_proxy=float(ANGULAR_SIZE_MIN),
                still_visible_in_rgb1=True,
                text=text,
                edit_meta={
                    "object_centric": place_meta,
                    "landmark_id": pair.receptacle_id,
                    "landmark_type": pair.receptacle_type,
                },
            )
            return PlaceResult(
                move=move,
                event0=event0,
                event1=event1,
                pair=pair,
                place_meta=place_meta,
            )

    # Failure: restore.
    pin_object_pose(controller, pair.mover_id, old_pos, old_rot)
    teleport_full(controller, **cam.teleport_kwargs())
    return None


def find_object_centric_edit(
    controller: Controller,
    cam: CameraPose,
    event: Any,
    *,
    relations: Sequence[Relation] = ("on", "next_to"),
    max_tries: int = 24,
    rng: Optional[random.Random] = None,
    som: bool = True,
    used_movers: Optional[set[str]] = None,
    prefer_relation: Optional[Relation] = None,
) -> Optional[PlaceResult]:
    """Search ranked pairs until one placement validates."""
    rng = rng or random.Random(0)
    rels = list(relations)
    if prefer_relation is not None and prefer_relation in rels:
        rels = [prefer_relation] + [r for r in rels if r != prefer_relation]
    pairs = enumerate_place_pairs(
        event, relations=tuple(rels), som=som, max_pairs=80, rng=rng
    )
    if used_movers:
        pairs = sorted(
            pairs,
            key=lambda p: (p.mover_id in used_movers, -p.score),
        )
    # Soft prefer requested relation without dropping the other.
    if prefer_relation is not None:
        pairs = sorted(
            pairs,
            key=lambda p: (0 if p.relation == prefer_relation else 1, -p.score),
        )
    attempts = 0
    for pair in pairs:
        if attempts >= max_tries:
            break
        attempts += 1
        hit = try_place_pair(controller, cam, event, pair, rng=rng)
        if hit is not None:
            return hit
        event = teleport_full(controller, **cam.teleport_kwargs())
    return None
