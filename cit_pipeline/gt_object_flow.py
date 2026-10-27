"""Simulator ground-truth object flow (Modification 20).

Rigid depth flow (preferred)
----------------------------
For each before-mask pixel with depth: unproject → add known ``delta_world`` →
reproject. Camera is fixed between rgb0/rgb1 (CIT setup). Requires
``renderDepthImage=True``.

Also records:
  - before/after instance masks (may differ in area)
  - centroid→centroid arrow in pixels
  - optional Middlebury color viz + arrow overlay

No learned flow model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cit_pipeline.camera_search import camera_basis


@dataclass
class ObjectFlowGT:
    """GT flow summary for one moved object between rgb0 and rgb1."""

    object_id: str
    object_type: str
    centroid_before_uv: Optional[tuple[float, float]]
    centroid_after_uv: Optional[tuple[float, float]]
    center_flow_uv: Optional[tuple[float, float]]
    mask_pixels_before: int
    mask_pixels_after: int
    delta_world: tuple[float, float, float]
    method: str  # "rigid_depth" | "uniform_mask" | "center_arrow"
    mean_flow_mag_px: float = 0.0
    max_flow_mag_px: float = 0.0
    flow_hw2: Optional[np.ndarray] = field(default=None, repr=False)

    def meta_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("flow_hw2", None)
        return d


def mask_centroid_uv(mask: np.ndarray) -> Optional[tuple[float, float, int]]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), int(len(xs))


def get_instance_mask(event: Any, object_id: str) -> Optional[np.ndarray]:
    masks = getattr(event, "instance_masks", None) or {}
    m = masks.get(object_id)
    if m is None:
        return None
    return np.asarray(m).astype(bool)


def center_flow_arrow(
    mask_before: np.ndarray,
    mask_after: np.ndarray,
) -> tuple[
    Optional[tuple[float, float]],
    Optional[tuple[float, float]],
    Optional[tuple[float, float]],
    int,
    int,
]:
    cb = mask_centroid_uv(mask_before)
    ca = mask_centroid_uv(mask_after)
    n0 = 0 if cb is None else cb[2]
    n1 = 0 if ca is None else ca[2]
    c0 = None if cb is None else (cb[0], cb[1])
    c1 = None if ca is None else (ca[0], ca[1])
    if c0 is None or c1 is None:
        return c0, c1, None, n0, n1
    return c0, c1, (c1[0] - c0[0], c1[1] - c0[1]), n0, n1


def uniform_mask_flow(
    mask_before: np.ndarray,
    flow_uv: tuple[float, float],
) -> np.ndarray:
    h, w = mask_before.shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    m = np.asarray(mask_before).astype(bool)
    flow[m, 0] = float(flow_uv[0])
    flow[m, 1] = float(flow_uv[1])
    return flow


def _camera_K(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K; AI2-THOR ``fieldOfView`` is vertical FOV in degrees."""
    fy = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    fx = fy
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _as_xyz(delta_world: Any) -> np.ndarray:
    if isinstance(delta_world, dict):
        return np.array(
            [
                float(delta_world.get("x", 0.0)),
                float(delta_world.get("y", 0.0)),
                float(delta_world.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
    if hasattr(delta_world, "x"):
        return np.array(
            [float(delta_world.x), float(delta_world.y), float(delta_world.z)],
            dtype=np.float64,
        )
    arr = np.asarray(delta_world, dtype=np.float64).reshape(-1)
    out = np.zeros(3, dtype=np.float64)
    out[: min(3, arr.size)] = arr[:3]
    return out


def rigid_depth_flow(
    mask_before: np.ndarray,
    depth_before: np.ndarray,
    *,
    delta_world: np.ndarray,
    cam_position: dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
    fov_deg: float,
) -> np.ndarray:
    """Dense forward flow: unproject mask pixels → translate → reproject."""
    h, w = mask_before.shape[:2]
    depth = np.asarray(depth_before, dtype=np.float64)
    if depth.shape[:2] != (h, w):
        raise ValueError(f"depth shape {depth.shape} != mask {(h, w)}")
    m = np.asarray(mask_before).astype(bool)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    if not np.any(m):
        return flow

    K = _camera_K(w, h, fov_deg)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    basis = camera_basis(yaw_deg, horizon_deg)
    right = np.asarray(basis["right"], dtype=np.float64)
    up = np.asarray(basis["up"], dtype=np.float64)
    forward = np.asarray(basis["forward"], dtype=np.float64)
    cam = np.array(
        [float(cam_position["x"]), float(cam_position["y"]), float(cam_position["z"])],
        dtype=np.float64,
    )
    delta = np.asarray(delta_world, dtype=np.float64).reshape(3)

    ys, xs = np.where(m)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 1e-4) & (z < 100.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if len(xs) == 0:
        return flow

    x_c = (xs.astype(np.float64) - cx) / fx * z
    y_c = (ys.astype(np.float64) - cy) / fy * z
    pts = (
        cam[None, :]
        + x_c[:, None] * right[None, :]
        + (-y_c)[:, None] * up[None, :]
        + z[:, None] * forward[None, :]
    )
    pts2 = pts + delta[None, :]

    rel = pts2 - cam[None, :]
    z2 = rel @ forward
    x2 = rel @ right
    y2 = -(rel @ up)
    ok = z2 > 1e-4
    u2 = fx * (x2 / z2) + cx
    v2 = fy * (y2 / z2) + cy
    flow[ys[ok], xs[ok], 0] = (u2[ok] - xs[ok]).astype(np.float32)
    flow[ys[ok], xs[ok], 1] = (v2[ok] - ys[ok]).astype(np.float32)
    return flow


def compute_object_flow_gt(
    event0: Any,
    event1: Any,
    *,
    object_id: str,
    object_type: str,
    delta_world: Any,
    cam_position: Optional[dict[str, float]] = None,
    yaw_deg: Optional[float] = None,
    horizon_deg: Optional[float] = None,
    fov_deg: float = 90.0,
    prefer_depth: bool = True,
) -> ObjectFlowGT:
    """Build GT flow for the moved object from simulator events."""
    m0 = get_instance_mask(event0, object_id)
    m1 = get_instance_mask(event1, object_id)
    h, w = event0.frame.shape[:2]
    if m0 is None:
        m0 = np.zeros((h, w), dtype=bool)
    if m1 is None:
        m1 = np.zeros((h, w), dtype=bool)

    c0, c1, arrow, n0, n1 = center_flow_arrow(m0, m1)
    dw = _as_xyz(delta_world)
    dw_t = (float(dw[0]), float(dw[1]), float(dw[2]))

    depth = getattr(event0, "depth_frame", None)
    flow = None
    method = "center_arrow"
    if (
        prefer_depth
        and depth is not None
        and cam_position is not None
        and yaw_deg is not None
        and horizon_deg is not None
        and n0 > 0
    ):
        flow = rigid_depth_flow(
            m0,
            depth,
            delta_world=dw,
            cam_position=cam_position,
            yaw_deg=float(yaw_deg),
            horizon_deg=float(horizon_deg),
            fov_deg=float(fov_deg),
        )
        method = "rigid_depth"
    elif arrow is not None and n0 > 0:
        flow = uniform_mask_flow(m0, arrow)
        method = "uniform_mask"

    mean_mag = max_mag = 0.0
    if flow is not None and n0 > 0:
        m = m0.astype(bool)
        mag = np.sqrt(flow[m, 0] ** 2 + flow[m, 1] ** 2)
        if mag.size:
            mean_mag = float(mag.mean())
            max_mag = float(mag.max())

    return ObjectFlowGT(
        object_id=object_id,
        object_type=object_type,
        centroid_before_uv=c0,
        centroid_after_uv=c1,
        center_flow_uv=arrow,
        mask_pixels_before=n0,
        mask_pixels_after=n1,
        delta_world=dw_t,
        method=method,
        mean_flow_mag_px=mean_mag,
        max_flow_mag_px=max_mag,
        flow_hw2=flow,
    )


def save_flow_artifacts(
    gt: ObjectFlowGT,
    out_dir: Path,
    stem: str,
    *,
    mask_before: Optional[np.ndarray] = None,
    mask_after: Optional[np.ndarray] = None,
    rgb0: Optional[np.ndarray] = None,
) -> dict[str, str]:
    """Write masks / dense flow / Middlebury viz / center arrow overlay."""
    from PIL import Image, ImageDraw

    from cit_pipeline.flow_viz import flow_color_image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if mask_before is not None:
        p = out_dir / f"{stem}_mask0.png"
        Image.fromarray((np.asarray(mask_before).astype(np.uint8) * 255)).save(p)
        paths["mask_before"] = p.name
    if mask_after is not None:
        p = out_dir / f"{stem}_mask1.png"
        Image.fromarray((np.asarray(mask_after).astype(np.uint8) * 255)).save(p)
        paths["mask_after"] = p.name

    if gt.flow_hw2 is not None:
        p_npy = out_dir / f"{stem}_gt_flow.npy"
        np.save(p_npy, gt.flow_hw2.astype(np.float32))
        paths["gt_flow_npy"] = p_npy.name

        # Middlebury on full image (zeros outside mask → dark).
        color = flow_color_image(gt.flow_hw2)
        p_col = out_dir / f"{stem}_gt_flow.png"
        Image.fromarray(color).save(p_col)
        paths["gt_flow_png"] = p_col.name

        # Masked Middlebury: black outside mover.
        if mask_before is not None:
            m = np.asarray(mask_before).astype(bool)
            masked = color.copy()
            masked[~m] = 0
            p_m = out_dir / f"{stem}_gt_flow_masked.png"
            Image.fromarray(masked).save(p_m)
            paths["gt_flow_masked_png"] = p_m.name

    if rgb0 is not None and gt.center_flow_uv is not None and gt.centroid_before_uv is not None:
        img = Image.fromarray(np.asarray(rgb0).copy()).convert("RGB")
        draw = ImageDraw.Draw(img)
        u0, v0 = gt.centroid_before_uv
        du, dv = gt.center_flow_uv
        u1, v1 = u0 + du, v0 + dv
        draw.line([(u0, v0), (u1, v1)], fill=(220, 30, 30), width=3)
        r = 4
        draw.ellipse([u0 - r, v0 - r, u0 + r, v0 + r], fill=(40, 180, 40))
        draw.ellipse([u1 - r, v1 - r, u1 + r, v1 + r], fill=(255, 220, 40))
        # Also scatter a few dense-flow vectors if available.
        if gt.flow_hw2 is not None and mask_before is not None:
            m = np.asarray(mask_before).astype(bool)
            ys, xs = np.where(m)
            if len(xs) > 0:
                n = min(48, len(xs))
                rng = np.random.default_rng(0)
                idx = rng.choice(len(xs), size=n, replace=False)
                for i in idx:
                    x, y = int(xs[i]), int(ys[i])
                    fu, fv = float(gt.flow_hw2[y, x, 0]), float(gt.flow_hw2[y, x, 1])
                    if abs(fu) + abs(fv) < 0.5:
                        continue
                    draw.line([(x, y), (x + fu, y + fv)], fill=(30, 120, 255), width=1)
        p = out_dir / f"{stem}_gt_flow_arrow.png"
        img.save(p)
        paths["gt_flow_arrow"] = p.name

    return paths


def attach_gt_flow_to_view(
    *,
    event0: Any,
    event1: Any,
    move: Any,
    cam: Any,
    sample_dir: Path,
    view_id: str,
    fov_deg: float = 90.0,
) -> dict[str, Any]:
    """Compute rigid-depth GT flow, save artifacts, return meta for ``edit_meta``."""
    cam_pos = None
    if getattr(cam, "camera_position", None) is not None:
        cp = cam.camera_position
        cam_pos = cp.to_dict() if hasattr(cp, "to_dict") else dict(cp)
    else:
        # Fallback: agent eye ~0.675 m above body for standing agent.
        pos = cam.position.to_dict() if hasattr(cam.position, "to_dict") else dict(cam.position)
        cam_pos = {"x": pos["x"], "y": float(pos["y"]) + 0.675, "z": pos["z"]}
        md = getattr(event0, "metadata", None) or {}
        agent = md.get("agent") or {}
        if agent.get("cameraPosition"):
            cam_pos = dict(agent["cameraPosition"])

    gt = compute_object_flow_gt(
        event0,
        event1,
        object_id=move.object_id,
        object_type=move.object_type,
        delta_world=move.delta_world,
        cam_position=cam_pos,
        yaw_deg=float(cam.rotation_y),
        horizon_deg=float(cam.horizon),
        fov_deg=float(fov_deg),
        prefer_depth=True,
    )
    m0 = get_instance_mask(event0, move.object_id)
    m1 = get_instance_mask(event1, move.object_id)
    paths = save_flow_artifacts(
        gt,
        Path(sample_dir),
        view_id,
        mask_before=m0,
        mask_after=m1,
        rgb0=getattr(event0, "frame", None),
    )
    meta = gt.meta_dict()
    meta["artifacts"] = paths
    return meta
