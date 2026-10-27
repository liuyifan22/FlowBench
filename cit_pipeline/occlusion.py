"""Occlusion helpers for CIT QA.

Mask method
-----------
1. **Preferred:** ``renderInstanceSegmentation`` → ``event.instance_masks[objectId]``.
2. **Fallback:** project each object's AABB corners into the image and fill the
   axis-aligned rectangle (approximate; can overestimate overlap).

Definition
----------
``occlusion_rate = |mask_occludee ∩ mask_mover| / |mask_occludee|`` in **rgb1**.

A QA label is **yes** if ``rate >= threshold`` (default 0.60). Candidate occludees
are restricted to objects whose masks **intersect** the mover (both yes and no
pools), then we balance yes/no across views in ``qa.sample_two_questions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from cit_pipeline.config import OCCLUDE_THRESHOLD, OCCLUDEE_MASK_MIN_PIXELS
from cit_pipeline.referring import referring_allowed
from cit_pipeline.size_filter import (
    is_small_enough_occludee,
    is_visible_mask_large_enough,
    scene_chair_footprint_limit,
)

MaskMethod = Literal["instance_segmentation", "aabb_projection"]


@dataclass
class OcclusionPair:
    occludee_id: str
    occludee_type: str
    occlusion_rate: float
    is_occluded: bool
    mask_method: MaskMethod
    intersects_mover_in_rgb1: bool


def _instance_mask(event: Any, object_id: str) -> Optional[np.ndarray]:
    masks = getattr(event, "instance_masks", None)
    if not masks:
        return None
    m = masks.get(object_id)
    if m is None:
        return None
    arr = np.asarray(m)
    if arr.dtype != bool:
        arr = arr > 0
    return arr


def _project_world_to_pixel(
    world: dict[str, float],
    cam_pos: dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
    width: int,
    height: int,
    fov_deg: float,
) -> Optional[tuple[float, float]]:
    """Project world point to pixel using agent camera (same basis as camera_search)."""
    from cit_pipeline.camera_search import camera_basis

    basis = camera_basis(yaw_deg, horizon_deg)
    p = np.array(
        [world["x"] - cam_pos["x"], world["y"] - cam_pos["y"], world["z"] - cam_pos["z"]],
        dtype=np.float64,
    )
    x_cam = float(np.dot(p, basis["right"]))
    y_cam = float(np.dot(p, basis["up"]))
    z_cam = float(np.dot(p, basis["forward"]))
    if z_cam <= 1e-4:
        return None
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    u = width * 0.5 + f * (x_cam / z_cam)
    # image y increases downward; camera up is positive y_cam
    v = height * 0.5 - f * (y_cam / z_cam)
    return u, v


def _aabb_mask(event: Any, object_id: str) -> Optional[np.ndarray]:
    obj = None
    for o in event.metadata["objects"]:
        if o["objectId"] == object_id:
            obj = o
            break
    if obj is None:
        return None
    aabb = obj.get("axisAlignedBoundingBox") or {}
    corners = aabb.get("cornerPoints")
    if not corners:
        return None

    agent = event.metadata["agent"]
    cam_pos = event.metadata.get("cameraPosition") or agent["position"]
    yaw = float(agent["rotation"]["y"])
    horizon = float(agent["cameraHorizon"])
    width = int(event.frame.shape[1])
    height = int(event.frame.shape[0])
    fov = float(event.metadata.get("fov") or 90)

    pixels: list[tuple[float, float]] = []
    for c in corners:
        if isinstance(c, dict):
            pt = {"x": c["x"], "y": c["y"], "z": c["z"]}
        else:
            pt = {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
        proj = _project_world_to_pixel(pt, cam_pos, yaw, horizon, width, height, fov)
        if proj is not None:
            pixels.append(proj)
    if len(pixels) < 2:
        return None

    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]
    u0 = int(max(0, math.floor(min(us))))
    u1 = int(min(width - 1, math.ceil(max(us))))
    v0 = int(max(0, math.floor(min(vs))))
    v1 = int(min(height - 1, math.ceil(max(vs))))
    if u1 <= u0 or v1 <= v0:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[v0 : v1 + 1, u0 : u1 + 1] = True
    return mask


def get_object_mask(event: Any, object_id: str) -> tuple[Optional[np.ndarray], MaskMethod]:
    m = _instance_mask(event, object_id)
    if m is not None and m.any():
        return m, "instance_segmentation"
    m = _aabb_mask(event, object_id)
    return m, "aabb_projection"


def occlusion_rate(
    event: Any,
    mover_id: str,
    occludee_id: str,
) -> tuple[float, MaskMethod, bool]:
    mover_mask, method_m = get_object_mask(event, mover_id)
    occludee_mask, method_o = get_object_mask(event, occludee_id)
    method: MaskMethod = (
        "instance_segmentation"
        if method_m == "instance_segmentation" and method_o == "instance_segmentation"
        else "aabb_projection"
    )
    if mover_mask is None or occludee_mask is None:
        return 0.0, method, False
    occludee_area = int(occludee_mask.sum())
    if occludee_area == 0:
        return 0.0, method, False
    inter = int(np.logical_and(mover_mask, occludee_mask).sum())
    rate = inter / occludee_area
    intersects = inter > 0
    return rate, method, intersects


def occlusion_cover_rate(
    event0: Any,
    event1: Any,
    mover_id: str,
    occludee_id: str,
) -> tuple[float, float, MaskMethod, bool]:
    """Cover metric robust to full occlusion of B.

    ``cover = |mask_A_rgb1 ∩ mask_B_rgb0| / |mask_B_rgb0|`` — fraction of B's
    *original* image region now occupied by A. Also returns the remaining
    visible fraction of B in rgb1 relative to rgb0.
    """
    b0, method_b0 = get_object_mask(event0, occludee_id)
    a1, method_a1 = get_object_mask(event1, mover_id)
    b1, _ = get_object_mask(event1, occludee_id)
    method: MaskMethod = (
        "instance_segmentation"
        if method_b0 == "instance_segmentation" and method_a1 == "instance_segmentation"
        else "aabb_projection"
    )
    if b0 is None or a1 is None:
        return 0.0, 1.0, method, False
    b0_area = int(b0.sum())
    if b0_area == 0:
        return 0.0, 1.0, method, False
    cover = float(np.logical_and(a1, b0).sum()) / float(b0_area)
    if b1 is None or not b1.any():
        remaining = 0.0
    else:
        remaining = float(np.logical_and(b1, b0).sum()) / float(b0_area)
    return cover, remaining, method, cover > 0.0


def select_occlusion_pairs(
    event1: Any,
    mover_id: str,
    *,
    threshold: float = OCCLUDE_THRESHOLD,
    balance: bool = True,
    max_occludee_footprint: Optional[float] = None,
    event0: Optional[Any] = None,
    som: bool = False,
) -> tuple[list[OcclusionPair], list[OcclusionPair], MaskMethod]:
    """Split intersecting occludees into yes (rate≥threshold) and no pools.

    Only objects with ``intersects_mover_in_rgb1`` enter either pool — random
    non-overlapping landmarks are excluded so "no" still means a meaningful
    near-miss occlusion rather than unrelated objects.

    Modification 3: occludees must be smaller than a chair (footprint filter).
    Modification 9: occludee on-screen mask (prefer rgb0) must be ≥
    ``OCCLUDEE_MASK_MIN_PIXELS`` so we never ask about tiny speck objects.
    Modification 12: ``som=True`` uses ``SOM_OCCLUDEE_MASK_MIN_PIXELS``.
    """
    from cit_pipeline.config import SOM_OCCLUDEE_MASK_MIN_PIXELS

    yes: list[OcclusionPair] = []
    no: list[OcclusionPair] = []
    method: MaskMethod = "aabb_projection"
    max_fp = (
        max_occludee_footprint
        if max_occludee_footprint is not None
        else scene_chair_footprint_limit(event1)
    )
    # Prefer rgb0 for the size check (before A covers B and shrinks B's mask).
    mask_event = event0 if event0 is not None else event1
    min_px = SOM_OCCLUDEE_MASK_MIN_PIXELS if som else OCCLUDEE_MASK_MIN_PIXELS

    for obj in event1.metadata["objects"]:
        oid = obj["objectId"]
        if oid == mover_id:
            continue
        if not obj.get("visible"):
            continue
        if obj["objectType"] in ("Floor", "Wall", "Ceiling", "Window", "Doorway"):
            continue
        if not is_small_enough_occludee(obj, max_fp):
            continue
        if not is_visible_mask_large_enough(
            mask_event, obj, min_pixels=min_px
        ):
            continue
        # Modification 4: skip types that would need awkward ordinals.
        if not referring_allowed(event1, obj["objectType"]):
            continue
        rate, method, intersects = occlusion_rate(event1, mover_id, oid)
        if not intersects:
            continue
        pair = OcclusionPair(
            occludee_id=oid,
            occludee_type=obj["objectType"],
            occlusion_rate=rate,
            is_occluded=rate >= threshold,
            mask_method=method,
            intersects_mover_in_rgb1=True,
        )
        if pair.is_occluded:
            yes.append(pair)
        else:
            no.append(pair)

    yes.sort(key=lambda p: -p.occlusion_rate)
    no.sort(key=lambda p: p.occlusion_rate)
    if balance:
        # Prefer balanced pools; callers sample one of each when possible.
        pass
    return yes, no, method
