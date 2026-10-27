"""Spatial metrics for CIT GT (v7.2+).

- **closer_to:** prefer AABB surface (edge) distance over center–center.
- **left_right:** prefer on-screen mask-centroid *u* (pixel x) over view bearing.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from cit_pipeline.config import FOV, HEIGHT, WIDTH


def point_point_dist(a: dict[str, float], b: dict[str, float]) -> float:
    return float(
        math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
    )


def _aabb_center_size(
    obj: dict[str, Any],
) -> Optional[tuple[dict[str, float], dict[str, float]]]:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    size = aabb.get("size")
    if not center or not size:
        return None
    try:
        c = {"x": float(center["x"]), "y": float(center["y"]), "z": float(center["z"])}
        s = {"x": float(size["x"]), "y": float(size["y"]), "z": float(size["z"])}
    except (KeyError, TypeError, ValueError):
        return None
    if min(s["x"], s["y"], s["z"]) <= 0.0:
        return None
    return c, s


def point_aabb_surface_dist(
    point: dict[str, float], obj: dict[str, Any]
) -> Optional[float]:
    """Euclidean distance from a point to an object's AABB surface (0 if inside)."""
    cs = _aabb_center_size(obj)
    if cs is None:
        return None
    c, s = cs
    dx = max(0.0, abs(float(point["x"]) - c["x"]) - 0.5 * s["x"])
    dy = max(0.0, abs(float(point["y"]) - c["y"]) - 0.5 * s["y"])
    dz = max(0.0, abs(float(point["z"]) - c["z"]) - 0.5 * s["z"])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def aabb_surface_dist(a: dict[str, Any], b: dict[str, Any]) -> Optional[float]:
    """Min distance between two AABB surfaces (0 if they overlap / touch)."""
    ca = _aabb_center_size(a)
    cb = _aabb_center_size(b)
    if ca is None or cb is None:
        return None
    c1, s1 = ca
    c2, s2 = cb
    dx = max(0.0, abs(c1["x"] - c2["x"]) - 0.5 * (s1["x"] + s2["x"]))
    dy = max(0.0, abs(c1["y"] - c2["y"]) - 0.5 * (s1["y"] + s2["y"]))
    dz = max(0.0, abs(c1["z"] - c2["z"]) - 0.5 * (s1["z"] + s2["z"]))
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def closer_dist_objects(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, str]:
    """Distance used for closer_to GT. Prefer AABB edges; fall back to centers."""
    d = aabb_surface_dist(a, b)
    if d is not None:
        return d, "aabb_surface"
    return point_point_dist(a["position"], b["position"]), "center"


def closer_dist_point_to_object(
    point: dict[str, float], obj: dict[str, Any]
) -> tuple[float, str]:
    d = point_aabb_surface_dist(point, obj)
    if d is not None:
        return d, "point_aabb_surface"
    return point_point_dist(point, obj["position"]), "point_center"


def project_world_to_pixel_u(
    world: dict[str, float],
    event: Any,
) -> Optional[float]:
    """Project a world point to image *u* (pixel x)."""
    from cit_pipeline.occlusion import _project_world_to_pixel

    agent = event.metadata["agent"]
    cam_pos = event.metadata.get("cameraPosition") or agent["position"]
    yaw = float(agent["rotation"]["y"])
    horizon = float(agent["cameraHorizon"])
    frame = getattr(event, "frame", None)
    if frame is not None:
        height, width = int(frame.shape[0]), int(frame.shape[1])
    else:
        width, height = int(WIDTH), int(HEIGHT)
    fov = float(event.metadata.get("fov") or FOV)
    proj = _project_world_to_pixel(
        world, cam_pos, yaw, horizon, width, height, fov
    )
    if proj is None:
        return None
    return float(proj[0])


def object_screen_u(
    event: Any,
    obj: dict[str, Any],
    *,
    prefer_mask: bool = True,
) -> tuple[Optional[float], str]:
    """Horizontal screen coordinate of an object (larger *u* = righter in image).

    Prefers instance-mask / AABB-projection mask centroid (matches SoM badges).
    Falls back to projecting the metadata position / AABB center.
    """
    if prefer_mask:
        try:
            from cit_pipeline.occlusion import get_object_mask

            mask, method = get_object_mask(event, obj["objectId"])
            if mask is not None:
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    return float(xs.mean()), f"mask_centroid:{method}"
        except Exception:  # noqa: BLE001
            pass

    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center")
    if center:
        u = project_world_to_pixel_u(
            {"x": float(center["x"]), "y": float(center["y"]), "z": float(center["z"])},
            event,
        )
        if u is not None:
            return u, "aabb_center_project"

    pos = obj.get("position")
    if pos:
        u = project_world_to_pixel_u(pos, event)
        if u is not None:
            return u, "position_project"
    return None, "none"


def object_screen_u_at_position(
    event: Any,
    position: dict[str, float],
) -> Optional[float]:
    """Project a hypothetical object position (for metadata-only flip planning)."""
    return project_world_to_pixel_u(position, event)
