"""Ordinal referring expressions via GT camera-order + optional VLM polish.

Modification 1 §6: for duplicate types prefer \"the 3rd chair from the left\".

Modification 4: if more than ``ORDINAL_MAX_SAME_TYPE`` visible instances of a
type exist, ordinals are banned for that type (``referring_allowed`` is False).

Modification 8: leftmost / rightmost instances remain referable even when the
type exceeds the ordinal cap (``instance_referring_allowed``).
"""

from __future__ import annotations

from typing import Any, Optional

from cit_pipeline.camera_search import camera_basis
from cit_pipeline.config import ORDINAL_MAX_SAME_TYPE


def camera_frame_x(event: Any, position: dict[str, float]) -> float:
    """Signed camera-right coordinate (positive = right of view center)."""
    import numpy as np

    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    cam = event.metadata.get("cameraPosition") or agent["position"]
    p = np.array(
        [position["x"] - cam["x"], position["y"] - cam["y"], position["z"] - cam["z"]],
        dtype=np.float64,
    )
    return float(np.dot(p, basis["right"]))


def camera_bearing_deg(event: Any, position: dict[str, float]) -> Optional[float]:
    """Horizontal bearing degrees vs camera forward (atan2(x_cam, z_cam))."""
    import math
    import numpy as np

    agent = event.metadata["agent"]
    basis = camera_basis(agent["rotation"]["y"], agent["cameraHorizon"])
    cam = event.metadata.get("cameraPosition") or agent["position"]
    p = np.array(
        [position["x"] - cam["x"], position["y"] - cam["y"], position["z"] - cam["z"]],
        dtype=np.float64,
    )
    x_cam = float(np.dot(p, basis["right"]))
    z_cam = float(np.dot(p, basis["forward"]))
    if z_cam <= 1e-4:
        return None
    return math.degrees(math.atan2(x_cam, z_cam))


def visible_same_type_count(event: Any, object_type: str) -> int:
    return sum(
        1
        for obj in event.metadata["objects"]
        if obj.get("objectType") == object_type and obj.get("visible")
    )


def referring_allowed(
    event: Any,
    object_type: str,
    *,
    max_same: int = ORDINAL_MAX_SAME_TYPE,
) -> bool:
    """Modification 4: ban types with too many visible duplicates."""
    return visible_same_type_count(event, object_type) <= max_same


def ordinal_from_left(
    event: Any,
    object_id: str,
    object_type: str,
) -> tuple[int, int, str]:
    """Return (1-based index from left, count of visible same type, default phrase)."""
    same: list[tuple[float, str]] = []
    for obj in event.metadata["objects"]:
        if obj.get("objectType") != object_type:
            continue
        if not obj.get("visible"):
            continue
        x = camera_frame_x(event, obj["position"])
        same.append((x, obj["objectId"]))
    same.sort(key=lambda t: t[0])
    ids = [oid for _, oid in same]
    if object_id not in ids:
        return 1, max(1, len(ids)), f"the {object_type.lower()}"
    idx = ids.index(object_id) + 1
    n = len(ids)
    if n <= 1:
        phrase = f"the {object_type.lower()}"
    else:
        phrase = f"the {idx}{_ordinal_suffix(idx)} {object_type.lower()} from the left"
    return idx, n, phrase


def extreme_side(idx: int, n: int) -> Optional[str]:
    """Return 'leftmost' / 'rightmost' / 'only', or None if not an extreme."""
    if n <= 0:
        return None
    if n == 1:
        return "only"
    if idx == 1:
        return "leftmost"
    if idx == n:
        return "rightmost"
    return None


def instance_phrase(event: Any, object_id: str, object_type: str) -> str:
    """Phrase for one instance; uses leftmost/rightmost when ordinals are banned."""
    idx, n, gt_phrase = ordinal_from_left(event, object_id, object_type)
    if n <= ORDINAL_MAX_SAME_TYPE:
        return gt_phrase
    side = extreme_side(idx, n)
    if side == "leftmost":
        return f"the leftmost {object_type.lower()}"
    if side == "rightmost":
        return f"the rightmost {object_type.lower()}"
    if side == "only":
        return f"the {object_type.lower()}"
    return gt_phrase


def extreme_instance_phrase(event: Any, object_id: str, object_type: str) -> str:
    """Always name extremes as leftmost/rightmost (Modification 11 affordance).

    Avoids awkward ordinals like \"the 4th chair from the left\" when we only
    ever select the leftmost or rightmost seat.
    """
    idx, n, _ = ordinal_from_left(event, object_id, object_type)
    side = extreme_side(idx, n)
    if side == "leftmost":
        return f"the leftmost {object_type.lower()}"
    if side == "rightmost":
        return f"the rightmost {object_type.lower()}"
    return f"the {object_type.lower()}"


def instance_referring_allowed(
    event: Any,
    object_id: str,
    object_type: str,
    *,
    max_same: int = ORDINAL_MAX_SAME_TYPE,
) -> bool:
    """True if this instance can be referred to (ordinal or leftmost/rightmost)."""
    idx, n, _ = ordinal_from_left(event, object_id, object_type)
    if n <= max_same:
        return True
    return extreme_side(idx, n) in ("leftmost", "rightmost", "only")


def _ordinal_suffix(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def referring_phrase(
    event: Any,
    object_id: str,
    object_type: str,
    *,
    vlm: Any = None,
    rgb_path: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a referring phrase; optionally VLM-polish while keeping GT ordinal."""
    idx, n, gt_ordinal = ordinal_from_left(event, object_id, object_type)
    side = extreme_side(idx, n)
    if n <= ORDINAL_MAX_SAME_TYPE:
        allowed = True
        gt_phrase = gt_ordinal
    elif side in ("leftmost", "rightmost"):
        allowed = True
        gt_phrase = f"the {side} {object_type.lower()}"
    elif side == "only":
        allowed = True
        gt_phrase = f"the {object_type.lower()}"
    else:
        allowed = False
        gt_phrase = gt_ordinal

    phrase = gt_phrase
    vlm_polished = False
    if vlm is not None and n > 1 and allowed:
        if n <= ORDINAL_MAX_SAME_TYPE:
            prompt = (
                "Write a short noun phrase referring to ONE object in the image.\n"
                f"Ground-truth: it is the {idx}{_ordinal_suffix(idx)} {object_type.lower()} "
                f"from the left among {n} visible {object_type.lower()}s.\n"
                f"Template: \"{gt_phrase}\"\n"
                "You may lightly polish wording but MUST keep the same ordinal and type. "
                "Return ONLY the noun phrase (no full sentence)."
            )
        else:
            prompt = (
                "Write a short noun phrase referring to ONE object in the image.\n"
                f"Ground-truth: it is the {side} {object_type.lower()} "
                f"among {n} visible {object_type.lower()}s.\n"
                f"Template: \"{gt_phrase}\"\n"
                "You may lightly polish wording but MUST keep leftmost/rightmost and type. "
                "Return ONLY the noun phrase (no full sentence)."
            )
        try:
            images = [rgb_path] if rgb_path is not None else None
            raw = vlm.chat(prompt, images=images, max_tokens=32, temperature=0.0)
            line = next((ln.strip().strip("\"'") for ln in raw.splitlines() if ln.strip()), "")
            if line and object_type.lower() in line.lower():
                phrase = line
                vlm_polished = True
        except Exception:
            pass
    return {
        "phrase": phrase,
        "ordinal_from_left": idx,
        "same_type_visible_count": n,
        "gt_phrase": gt_phrase,
        "extreme_side": side,
        "vlm_polished": vlm_polished,
        "referring_allowed": allowed,
    }
