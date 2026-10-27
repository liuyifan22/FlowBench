"""Post-edit scene integrity checks (Modification 20).

``forceAction=True`` flip-plan teleports can tunnel a mover into furniture and
tip / break collateral objects (e.g. laptop forced into an ArmChair). Metadata
signals we use:

- mover ``isBroken``
- large pose change on non-mover furniture (Sofa / ArmChair / Chair / …)
"""

from __future__ import annotations

from typing import Any, Optional

# Types whose sudden tip/roll indicates a bad force-teleport.
COLLATERAL_TYPES = frozenset(
    {
        "Sofa",
        "ArmChair",
        "Chair",
        "Stool",
        "Ottoman",
        "Bed",
        "Dresser",
        "ShelvingUnit",
        "TVStand",
        "CoffeeTable",
        "DiningTable",
        "SideTable",
        "Desk",
        "FloorLamp",
        "Television",
    }
)


def _rot_delta_deg(a: dict[str, float], b: dict[str, float]) -> float:
    """Max absolute Euler delta (degrees), naive but enough for tip detection."""
    return max(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in ("x", "y", "z"))


def _pos_delta_m(a: dict[str, float], b: dict[str, float]) -> float:
    return float(
        (
            (float(a["x"]) - float(b["x"])) ** 2
            + (float(a["y"]) - float(b["y"])) ** 2
            + (float(a["z"]) - float(b["z"])) ** 2
        )
        ** 0.5
    )


def scene_collateral_ok(
    event0: Any,
    event1: Any,
    mover_id: str,
    *,
    max_furniture_rot_deg: float = 15.0,
    max_furniture_pos_m: float = 0.15,
    max_furniture_y_jump_m: float = 0.10,
) -> tuple[bool, dict[str, Any]]:
    """Return (ok, report). Reject if mover broke or furniture tipped/moved."""
    before = {o["objectId"]: o for o in event0.metadata["objects"]}
    after = {o["objectId"]: o for o in event1.metadata["objects"]}
    report: dict[str, Any] = {"mover_broken": False, "collateral": []}

    mov1 = after.get(mover_id)
    if mov1 is not None and bool(mov1.get("isBroken")):
        report["mover_broken"] = True
        return False, report

    for oid, o0 in before.items():
        if oid == mover_id:
            continue
        typ = o0.get("objectType")
        if typ not in COLLATERAL_TYPES:
            continue
        o1 = after.get(oid)
        if o1 is None:
            continue
        drot = _rot_delta_deg(o0["rotation"], o1["rotation"])
        dpos = _pos_delta_m(o0["position"], o1["position"])
        dy = abs(float(o1["position"]["y"]) - float(o0["position"]["y"]))
        bad = (
            drot >= max_furniture_rot_deg
            or dpos >= max_furniture_pos_m
            or dy >= max_furniture_y_jump_m
            or bool(o1.get("isBroken"))
        )
        if bad:
            report["collateral"].append(
                {
                    "objectId": oid,
                    "objectType": typ,
                    "drot_deg": drot,
                    "dpos_m": dpos,
                    "dy_m": dy,
                    "isBroken": bool(o1.get("isBroken")),
                    "isMoving": bool(o1.get("isMoving")),
                }
            )
    return (len(report["collateral"]) == 0 and not report["mover_broken"]), report


def object_on_screen(
    event: Any,
    obj: dict[str, Any],
    *,
    min_pixels: int = 40,
) -> bool:
    """True if instance mask has ≥ ``min_pixels`` (iTHOR has no room labels).

    Objects that are metadata-``visible`` but missing from instance segmentation
    (common for tiny / out-of-view items flagged visible by THOR) are rejected.
    """
    from cit_pipeline.size_filter import object_mask_pixel_count

    n = object_mask_pixel_count(event, obj["objectId"])
    if n is None:
        return False
    return int(n) >= int(min_pixels)
