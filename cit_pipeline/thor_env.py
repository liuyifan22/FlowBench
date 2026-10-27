"""AI2-THOR controller helpers for iTHOR and ProcTHOR eval paths.

Cluster notes
-------------
- ``/tmp`` is often full on shared nodes → redirect TMPDIR to ``$CIT_TMP_DIR`` (default ``~/.cache/cit/tmp``).
- Use ``CloudRendering`` (Vulkan) for headless GPUs.
- ``visibilityDistance`` should be ~6 m so mid-room landmarks stay marked visible.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image

from cit_pipeline.config import (
    DEFAULT_ITHOR_SCENE,
    FOV,
    HEIGHT,
    TMP_DIR,
    VISIBILITY_DISTANCE,
    WIDTH,
)


def ensure_tmpdir(tmp_dir: Path = TMP_DIR) -> Path:
    """Point process temp vars at a writable NFS path; set NVIDIA Vulkan ICD."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = str(tmp_dir)
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    # AI2-THOR shells out to ``vulkaninfo``; keep the cit env bin on PATH.
    cit_bin = os.path.dirname(sys.executable)
    path0 = os.environ.get("PATH", "")
    if cit_bin not in path0.split(":"):
        os.environ["PATH"] = f"{cit_bin}:{path0}"
    return tmp_dir


def start_controller(
    scene: Any = DEFAULT_ITHOR_SCENE,
    width: int = WIDTH,
    height: int = HEIGHT,
    fov: int = FOV,
    visibility_distance: float = VISIBILITY_DISTANCE,
    render_instance_segmentation: bool = True,
    render_depth: bool = False,
) -> Controller:
    """Start CloudRendering for an iTHOR FloorPlan string **or** a ProcTHOR house dict."""
    ensure_tmpdir()
    return Controller(
        platform=CloudRendering,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=fov,
        visibilityDistance=visibility_distance,
        renderDepthImage=render_depth,
        # Needed for occlusion QA masks (event.instance_masks).
        renderInstanceSegmentation=render_instance_segmentation,
    )


def reset_scene(controller: Controller, scene: Any) -> Any:
    """Reset to an iTHOR name or reload a ProcTHOR house dict."""
    return controller.reset(scene)


def save_rgb(event: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(event.frame).save(path)
    return path


def _object_type_lookup(event: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for obj in (event.metadata or {}).get("objects") or []:
        oid = obj.get("objectId")
        if oid:
            out[str(oid)] = str(obj.get("objectType") or oid.split("|", 1)[0])
    return out


def _pack_rgb_color(rgb: tuple[int, ...] | list[int]) -> int:
    """Pack an RGB(A) color tuple into the uint32 key used by AI2-THOR."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return int(r) | (int(g) << 8) | (int(b) << 16)


def build_segmentation_id_maps(
    event: Any,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Build a dense uint16 seg-id map and id→{category, instance_id} labels.

    Pixel value ``0`` is background / unmapped. IDs ``1..N`` are consecutive
    integers assigned to unique instance colors present in the frame.
    """
    seg_rgb = getattr(event, "instance_segmentation_frame", None)
    if seg_rgb is None:
        raise RuntimeError("event.instance_segmentation_frame is missing")

    seg_rgb = np.asarray(seg_rgb)
    h, w = int(seg_rgb.shape[0]), int(seg_rgb.shape[1])
    packed = (
        seg_rgb[:, :, 0].astype(np.uint32)
        | (seg_rgb[:, :, 1].astype(np.uint32) << 8)
        | (seg_rgb[:, :, 2].astype(np.uint32) << 16)
    )

    color_to_oid = getattr(event, "color_to_object_id", None) or {}
    packed_to_oid: dict[int, str] = {}
    for color, oid in color_to_oid.items():
        if color is None or oid is None:
            continue
        try:
            packed_to_oid[_pack_rgb_color(tuple(color))] = str(oid)
        except Exception:  # noqa: BLE001
            continue

    type_by_oid = _object_type_lookup(event)
    unique = np.unique(packed)
    id_map = np.zeros((h, w), dtype=np.uint16)
    labels: dict[str, dict[str, Any]] = {
        "0": {
            "category": "background",
            "instance_id": None,
            "object_id": None,
            "color_rgb": [0, 0, 0],
            "color_packed": 0,
        }
    }
    next_id = 1
    for key in unique.tolist():
        key_i = int(key)
        if key_i == 0:
            continue
        oid = packed_to_oid.get(key_i)
        if oid is None:
            # Unmapped colors (sky / unlisted) stay background.
            continue
        if next_id >= 65535:
            break
        sid = next_id
        next_id += 1
        id_map[packed == key_i] = sid
        category = type_by_oid.get(oid) or oid.split("|", 1)[0]
        labels[str(sid)] = {
            "category": category,
            "instance_id": oid,
            "object_id": oid,
            "object_type": category,
            "color_rgb": [key_i & 255, (key_i >> 8) & 255, (key_i >> 16) & 255],
            "color_packed": key_i,
        }
    return id_map, labels


def save_depth_map(event: Any, path: Path) -> Path:
    """Save float32 depth (meters) as ``.npy``."""
    depth = getattr(event, "depth_frame", None)
    if depth is None:
        raise RuntimeError("event.depth_frame is missing (start_controller render_depth=True)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32))
    return path


def save_segmentation_maps(
    event: Any,
    *,
    seg_rgb_path: Path,
    seg_ids_path: Path,
    labels_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Save RGB instance seg, uint16 id map, and optional labels JSON."""
    seg_rgb = getattr(event, "instance_segmentation_frame", None)
    if seg_rgb is None:
        raise RuntimeError("event.instance_segmentation_frame is missing")
    seg_rgb_path = Path(seg_rgb_path)
    seg_ids_path = Path(seg_ids_path)
    seg_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(seg_rgb)).save(seg_rgb_path)
    id_map, labels = build_segmentation_id_maps(event)
    np.save(seg_ids_path, id_map)
    if labels_path is not None:
        Path(labels_path).write_text(json.dumps(labels, indent=2))
    return labels


def save_aux_maps_for_rgb(
    event: Any,
    sample_dir: Path,
    stem: str,
) -> dict[str, Any]:
    """Save seg + depth artifacts next to an RGB frame.

    Writes:
      ``{stem}_seg.png``       — color instance segmentation
      ``{stem}_seg_ids.npy``   — uint16 HxW segmentation ids
      ``{stem}_seg_labels.json`` — id → {category, instance_id, …}
      ``{stem}_depth.npy``     — float32 HxW depth meters
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    seg_rgb_path = sample_dir / f"{stem}_seg.png"
    seg_ids_path = sample_dir / f"{stem}_seg_ids.npy"
    labels_path = sample_dir / f"{stem}_seg_labels.json"
    depth_path = sample_dir / f"{stem}_depth.npy"
    labels = save_segmentation_maps(
        event,
        seg_rgb_path=seg_rgb_path,
        seg_ids_path=seg_ids_path,
        labels_path=labels_path,
    )
    save_depth_map(event, depth_path)
    return {
        "seg_path": str(seg_rgb_path),
        "seg_ids_path": str(seg_ids_path),
        "seg_labels_path": str(labels_path),
        "depth_path": str(depth_path),
        "seg_id_to_label": labels,
    }


def merge_seg_label_dicts(
    labels_rgb0: Optional[dict[str, dict[str, Any]]] = None,
    labels_rgb1: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Camera-level copy of per-frame seg-id → {category, instance_id} maps."""
    return {
        "rgb0": dict(labels_rgb0 or {}),
        "rgb1": dict(labels_rgb1 or {}),
    }


def get_object(event: Any, object_id: str) -> Optional[dict[str, Any]]:
    for obj in event.metadata["objects"]:
        if obj["objectId"] == object_id:
            return obj
    return None


def still_visible(event: Any, object_id: str) -> bool:
    """True if metadata marks the object visible under current visibilityDistance."""
    obj = get_object(event, object_id)
    return bool(obj and obj.get("visible"))


def teleport_full(controller: Controller, **kwargs: Any) -> Any:
    """Snap the agent (and its camera) to an exact pose."""
    return controller.step(action="TeleportFull", **kwargs)


def teleport_object(
    controller: Controller,
    object_id: str,
    position: dict[str, float],
    rotation: dict[str, float],
    *,
    force: bool = False,
    allow_surface_place: bool = False,
    kinematic: bool = True,
) -> Any:
    """Move an object to an absolute pose.

    Modification 2: default ``force=False`` so collisions / blocked teleports
    fail instead of tunneling through counters. Restores may use ``force=True``.

    Modification 3 / 10: default ``forceKinematic=True`` so objects hold pose
    for mid-pipeline renders. Pass ``kinematic=False`` to release for gravity
    settle (support QA).
    """
    event = controller.step(
        action="TeleportObject",
        objectId=object_id,
        position=position,
        rotation=rotation,
        forceAction=force,
        forceKinematic=bool(kinematic),
    )
    if event.metadata["lastActionSuccess"]:
        return event
    if allow_surface_place:
        return controller.step(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position=position,
            rotation=rotation,
            forceKinematic=bool(kinematic),
        )
    return event


def pin_object_pose(
    controller: Controller,
    object_id: str,
    position: dict[str, float],
    rotation: dict[str, float],
) -> Any:
    """Force-kinematic re-pin so gravity cannot settle the object mid-pipeline."""
    return teleport_object(
        controller,
        object_id,
        position,
        rotation,
        force=True,
        allow_surface_place=False,
        kinematic=True,
    )


def settle_object_with_gravity(
    controller: Controller,
    object_id: str,
    position: dict[str, float],
    rotation: dict[str, float],
    *,
    max_steps: int = 80,
    time_step: float = 0.05,
    min_steps: int = 12,
    force_place: bool = True,
) -> Any:
    """Place the object non-kinematic and advance physics until the scene rests.

    Modification 10 (support QA): after a collision-checked slide, release the
    object so it can fall onto furniture or the floor. Uses AI2-THOR's
    PausePhysicsAutoSim / AdvancePhysicsStep / UnpausePhysicsAutoSim loop.

    Always runs at least ``min_steps`` advances: kinematic / heavy props often
    report ``isSceneAtRest`` immediately, which previously skipped gravity.
    A 2 cm lift before release helps free-fall engage when the slide left the
    object clipped into a surface or still marked kinematic.
    """
    controller.step(action="PausePhysicsAutoSim")
    drop_pos = {
        "x": float(position["x"]),
        "y": float(position["y"]) + 0.02,
        "z": float(position["z"]),
    }
    teleport_object(
        controller,
        object_id,
        drop_pos,
        rotation,
        force=force_place,
        allow_surface_place=False,
        kinematic=False,
    )
    # Best-effort: some builds expose an explicit kinematic toggle.
    try:
        controller.step(
            action="SetObjectKinematicState",
            objectId=object_id,
            kinematic=False,
            forceAction=True,
        )
    except Exception:
        pass
    steps = 0
    max_steps_i = int(max_steps)
    min_steps_i = max(0, int(min_steps))
    while steps < max_steps_i:
        at_rest = bool(controller.last_event.metadata.get("isSceneAtRest"))
        if at_rest and steps >= min_steps_i:
            break
        controller.step(action="AdvancePhysicsStep", timeStep=float(time_step))
        steps += 1
    controller.step(action="UnpausePhysicsAutoSim")
    return controller.last_event
