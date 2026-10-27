"""Object size helpers and move-target filtering (Modification 1 + 6 + 9).

Rule: only move objects **strictly smaller than a reference table** in the scene
(DiningTable / CoffeeTable / SideTable / Desk / CounterTop). Tables themselves
are never move targets. If no table exists, use ``DEFAULT_MAX_FOOTPRINT_M``.

Modification 6: also reject **too-tiny** movers (credit card / pencil class) —
see ``is_large_enough_mover``.

Modification 9: reject movers whose instance segmentation mask is smaller than
``MOVER_MASK_MIN_PIXELS`` (default 1000 at 800×600).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cit_pipeline.config import (
    MOVER_MASK_MIN_PIXELS,
    SOM_MOVER_MASK_MIN_PIXELS,
    SUPPORT_QA_FLOOR_BOTTOM_Y_M,
    SUPPORT_QA_MIN_BOTTOM_Y_M,
)

# Horizontal footprint (max of AABB x/z) must be below the smallest table's footprint.
TABLE_TYPES = ("DiningTable", "CoffeeTable", "SideTable", "Desk", "CounterTop")

# Never move these regardless of size (too large / structural / landmarks).
BLOCKED_MOVE_TYPES = TABLE_TYPES + (
    "Sofa",
    "Bed",
    "ShelvingUnit",
    "Shelf",
    "Television",
    "Fridge",
    "Refrigerator",
    "Dresser",
    "FloorLamp",
    "Window",
    "Door",
    "Wall",
    "Floor",
    "Ceiling",
    # Fixed fixtures (Modification 8+): not free-standing movers.
    "Cabinet",
    "StoveKnob",
    "StoveBurner",
    "Faucet",
    "Drawer",
    # v7.2: never move bathtubs / basins.
    "Bathtub",
    "BathtubBasin",
)

DEFAULT_MAX_FOOTPRINT_M = 0.85  # ~smaller than a typical dining table short side

# Modification 3: occludee must be smaller than a chair (not sofa/table).
CHAIR_REF_TYPES = ("Chair", "Stool", "Ottoman", "ArmChair")
DEFAULT_CHAIR_FOOTPRINT_M = 0.55
# Huge types never make sense as occludees for "is X occluded by the mover?".
BLOCKED_OCCLUDEE_TYPES = BLOCKED_MOVE_TYPES + (
    "CounterTop",
    "Cabinet",
    "Shelf",
    "Painting",
    "Mirror",
)

# ---------------------------------------------------------------------------
# Modification 6: ban credit-card / pencil-scale movers
# ---------------------------------------------------------------------------
# Explicit type ban (and near-duplicates). Newspaper / KeyChain are also in the
# "tiny class" examples and are not useful as primary movers.
TOO_TINY_MOVE_TYPES = frozenset(
    {
        "CreditCard",
        "Pencil",
        "Pen",
        "KeyChain",
        "Watch",
        "Newspaper",
    }
)
# Size ceiling from these refs: movers must be *strictly larger* than the largest
# of these footprints present in the scene (else defaults).
TINY_MOVE_REF_TYPES = ("CreditCard", "Pencil", "Newspaper", "KeyChain")
DEFAULT_TINY_REF_FOOTPRINT_M = {
    "CreditCard": 0.09,
    "Pencil": 0.18,
    "Newspaper": 0.30,
    "KeyChain": 0.06,
}
# Explicitly allowed even if they sit near the tiny size band.
ALLOW_SMALL_MOVE_TYPES = frozenset({"Egg", "RemoteControl"})


def aabb_footprint(obj: dict[str, Any]) -> Optional[float]:
    """Return max(horizontal AABB extents), or None if missing."""
    aabb = obj.get("axisAlignedBoundingBox") or {}
    size = aabb.get("size")
    if not size:
        return None
    return max(float(size.get("x", 0.0)), float(size.get("z", 0.0)))


def aabb_volume(obj: dict[str, Any]) -> Optional[float]:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    size = aabb.get("size")
    if not size:
        return None
    return float(size.get("x", 0.0)) * float(size.get("y", 0.0)) * float(size.get("z", 0.0))


def aabb_bottom_y(obj: dict[str, Any]) -> Optional[float]:
    """World-y of the AABB bottom face (for support-QA height filter)."""
    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    size = aabb.get("size")
    if center and size and "y" in center and "y" in size:
        return float(center["y"]) - 0.5 * float(size["y"])
    pos = obj.get("position") or {}
    if "y" in pos:
        return float(pos["y"])
    return None


def _is_floor_receptacle_id(parent_id: Any) -> bool:
    if not parent_id or not isinstance(parent_id, str):
        return False
    return parent_id == "Floor" or parent_id.startswith("Floor")


def object_resting_on_floor(obj: dict[str, Any]) -> bool:
    """True when the object is already on / near the floor."""
    parents = obj.get("parentReceptacles") or []
    has_floor = any(_is_floor_receptacle_id(p) for p in parents)
    has_furniture = any(p and not _is_floor_receptacle_id(p) for p in parents)
    if has_floor and not has_furniture:
        return True
    bottom = aabb_bottom_y(obj)
    if bottom is not None and bottom < SUPPORT_QA_FLOOR_BOTTOM_Y_M:
        return True
    return False


def support_qa_eligible(obj: dict[str, Any]) -> bool:
    """Elevated pickupable not already on the floor (support QA mover filter).

    Requires ``pickupable`` (so gravity settle can drop the object), AABB bottom
    ≥ ``SUPPORT_QA_MIN_BOTTOM_Y_M``, not floor-resting, and a non-floor parent
    receptacle (already supported by furniture before the move).
    """
    if not bool(obj.get("pickupable")):
        return False
    if object_resting_on_floor(obj):
        return False
    bottom = aabb_bottom_y(obj)
    if bottom is None or bottom < SUPPORT_QA_MIN_BOTTOM_Y_M:
        return False
    parents = obj.get("parentReceptacles") or []
    if not any(p and not _is_floor_receptacle_id(p) for p in parents):
        return False
    return True


def aabb_xz_extent(obj: dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
    """Return (min_x, max_x, min_z, max_z) from AABB, or None."""
    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    size = aabb.get("size")
    if center and size:
        hx = 0.5 * float(size.get("x", 0.0))
        hz = 0.5 * float(size.get("z", 0.0))
        cx, cz = float(center["x"]), float(center["z"])
        return cx - hx, cx + hx, cz - hz, cz + hz
    pos = obj.get("position") or {}
    if "x" in pos and "z" in pos:
        x, z = float(pos["x"]), float(pos["z"])
        return x, x, z, z
    return None


def scene_table_footprint_limit(event: Any) -> float:
    """Max allowed mover footprint = min table footprint in the scene (exclusive)."""
    footprints: list[float] = []
    for obj in event.metadata["objects"]:
        if obj.get("objectType") in TABLE_TYPES:
            fp = aabb_footprint(obj)
            if fp is not None and fp > 0.05:
                footprints.append(fp)
    if not footprints:
        return DEFAULT_MAX_FOOTPRINT_M
    # Strictly smaller than the smallest table.
    return min(footprints) - 1e-3


def scene_chair_footprint_limit(event: Any) -> float:
    """Max allowed occludee footprint = min chair-like footprint (exclusive)."""
    footprints: list[float] = []
    for obj in event.metadata["objects"]:
        if obj.get("objectType") in CHAIR_REF_TYPES:
            fp = aabb_footprint(obj)
            if fp is not None and fp > 0.05:
                footprints.append(fp)
    if not footprints:
        return DEFAULT_CHAIR_FOOTPRINT_M
    return min(footprints) - 1e-3


def tiny_mover_footprint_ceiling(event: Optional[Any] = None) -> float:
    """Movers must be strictly larger than this (Mod 6).

    Uses the **max** footprint among CreditCard / Pencil / Newspaper / KeyChain
    in the scene when present; otherwise the max of hardcoded defaults.
    """
    fps: list[float] = []
    if event is not None:
        for obj in event.metadata["objects"]:
            if obj.get("objectType") not in TINY_MOVE_REF_TYPES:
                continue
            fp = aabb_footprint(obj)
            if fp is not None and fp > 1e-4:
                fps.append(fp)
    if fps:
        return max(fps)
    return max(DEFAULT_TINY_REF_FOOTPRINT_M.values())


def is_large_enough_mover(
    obj: dict[str, Any],
    event: Optional[Any] = None,
    *,
    som: bool = False,
) -> bool:
    """Modification 6: reject credit-card / pencil-scale movers.

    Egg and RemoteControl are always allowed (size-wise footprint).
    Modification 12: ``som=True`` skips the tiny-type / footprint floor so SoM
    can still mark smaller props.
    """
    if som:
        return True
    otype = obj.get("objectType")
    if otype in ALLOW_SMALL_MOVE_TYPES:
        return True
    if otype in TOO_TINY_MOVE_TYPES:
        return False
    fp = aabb_footprint(obj)
    if fp is None:
        # Unknown size: allow other pickupables that are not on the tiny type list.
        return bool(obj.get("pickupable"))
    return fp > tiny_mover_footprint_ceiling(event)


def object_mask_pixel_count(event: Any, object_id: str) -> Optional[int]:
    """Return instance-mask pixel count, or None if no mask is available."""
    masks = getattr(event, "instance_masks", None)
    if not masks:
        return None
    m = masks.get(object_id)
    if m is None:
        return None
    return int(np.count_nonzero(np.asarray(m)))


def is_visible_mask_large_enough(
    event: Any,
    obj: dict[str, Any],
    *,
    min_pixels: int = MOVER_MASK_MIN_PIXELS,
) -> bool:
    """On-screen mask must cover at least ``min_pixels`` (Modification 9)."""
    n = object_mask_pixel_count(event, obj["objectId"])
    if n is None:
        # No segmentation → cannot verify on-screen size; reject as mover.
        return False
    return n >= int(min_pixels)


def is_small_enough_to_move(
    obj: dict[str, Any],
    max_footprint: float,
    event: Optional[Any] = None,
    *,
    som: bool = False,
) -> bool:
    """Pass if below table-size cap, not too tiny (Mod 6), and mask large enough (Mod 9).

    Modification 12: ``som=True`` relaxes Mod-6 tiny bans and uses
    ``SOM_MOVER_MASK_MIN_PIXELS`` for the on-screen mask floor.
    """
    if obj.get("objectType") in BLOCKED_MOVE_TYPES:
        return False
    if not is_large_enough_mover(obj, event, som=som):
        return False
    if event is not None:
        min_px = SOM_MOVER_MASK_MIN_PIXELS if som else MOVER_MASK_MIN_PIXELS
        if not is_visible_mask_large_enough(event, obj, min_pixels=int(min_px)):
            return False
    fp = aabb_footprint(obj)
    if fp is None:
        # No AABB: allow only clearly small pickupables (already passed Mod 6/9).
        return bool(obj.get("pickupable"))
    return fp < max_footprint


def is_small_enough_occludee(obj: dict[str, Any], max_footprint: float) -> bool:
    """Modification 3: occludee must be smaller than a chair (no sofas/tables)."""
    if obj.get("objectType") in BLOCKED_OCCLUDEE_TYPES:
        return False
    fp = aabb_footprint(obj)
    if fp is None:
        return bool(obj.get("pickupable"))
    return fp < max_footprint
