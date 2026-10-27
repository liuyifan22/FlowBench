"""Pose snapshot / restore to avoid full ``controller.reset`` between categories.

After a clean room load, store every interactive object's exact pose (+ agent).
Between categories, rewind changed objects with force kinematic teleports and
verify they landed on the snapshot poses. If verification fails, callers fall
back to a full house reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ai2thor.controller import Controller

from cit_pipeline.thor_env import get_object, pin_object_pose, teleport_full, teleport_object

# Structural / non-teleportable ids — never snapshot or restore these.
_SKIP_TYPES = frozenset(
    {
        "Floor",
        "Wall",
        "Ceiling",
        "Doorway",
        "Doorframe",
        "Window",
        "Rug",
    }
)

# Tight tolerances: "exact" rewind with float noise allowance.
POS_TOL_M = 0.02
ROT_TOL_DEG = 2.0
AGENT_POS_TOL_M = 0.05
AGENT_ROT_TOL_DEG = 1.0


def _pos_delta(a: dict[str, float], b: dict[str, float]) -> float:
    return float(
        (
            (float(a["x"]) - float(b["x"])) ** 2
            + (float(a["y"]) - float(b["y"])) ** 2
            + (float(a["z"]) - float(b["z"])) ** 2
        )
        ** 0.5
    )


def _rot_delta_deg(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in ("x", "y", "z"))


def _copy_pose(d: dict[str, Any]) -> dict[str, float]:
    return {k: float(d[k]) for k in ("x", "y", "z")}


def _should_track(obj: dict[str, Any]) -> bool:
    typ = str(obj.get("objectType") or "")
    if typ in _SKIP_TYPES:
        return False
    if not obj.get("objectId"):
        return False
    pos = obj.get("position")
    rot = obj.get("rotation")
    if not isinstance(pos, dict) or not isinstance(rot, dict):
        return False
    if not all(k in pos for k in ("x", "y", "z")):
        return False
    if not all(k in rot for k in ("x", "y", "z")):
        return False
    return True


@dataclass
class ObjectPoseSnap:
    object_id: str
    object_type: str
    position: dict[str, float]
    rotation: dict[str, float]
    is_broken: bool = False
    is_open: Optional[bool] = None


@dataclass
class SceneSnapshot:
    """Exact poses captured after a clean load (pre-category)."""

    objects: dict[str, ObjectPoseSnap] = field(default_factory=dict)
    agent_position: dict[str, float] = field(default_factory=dict)
    agent_rotation: dict[str, float] = field(default_factory=dict)
    agent_horizon: float = 0.0
    agent_standing: bool = True


def capture_scene_snapshot(event: Any) -> SceneSnapshot:
    """Record object + agent poses from the current event metadata."""
    objects: dict[str, ObjectPoseSnap] = {}
    for obj in event.metadata.get("objects") or []:
        if not _should_track(obj):
            continue
        oid = str(obj["objectId"])
        is_open = obj.get("isOpen")
        objects[oid] = ObjectPoseSnap(
            object_id=oid,
            object_type=str(obj.get("objectType") or ""),
            position=_copy_pose(obj["position"]),
            rotation=_copy_pose(obj["rotation"]),
            is_broken=bool(obj.get("isBroken")),
            is_open=bool(is_open) if is_open is not None else None,
        )
    agent = event.metadata.get("agent") or {}
    return SceneSnapshot(
        objects=objects,
        agent_position=_copy_pose(agent.get("position") or {"x": 0, "y": 0, "z": 0}),
        agent_rotation=_copy_pose(agent.get("rotation") or {"x": 0, "y": 0, "z": 0}),
        agent_horizon=float(agent.get("cameraHorizon") or 0.0),
        agent_standing=bool(agent.get("isStanding", True)),
    )


def objects_changed_vs_snapshot(
    event: Any,
    snapshot: SceneSnapshot,
    *,
    pos_tol_m: float = POS_TOL_M,
    rot_tol_deg: float = ROT_TOL_DEG,
) -> list[str]:
    """Object ids whose pose / broken / open state drifted from the snapshot."""
    changed: list[str] = []
    cur = {
        str(o["objectId"]): o
        for o in (event.metadata.get("objects") or [])
        if o.get("objectId")
    }
    for oid, snap in snapshot.objects.items():
        obj = cur.get(oid)
        if obj is None:
            changed.append(oid)
            continue
        if bool(obj.get("isBroken")) != bool(snap.is_broken):
            changed.append(oid)
            continue
        if snap.is_open is not None and obj.get("isOpen") is not None:
            if bool(obj.get("isOpen")) != bool(snap.is_open):
                changed.append(oid)
                continue
        if _pos_delta(obj["position"], snap.position) > pos_tol_m:
            changed.append(oid)
            continue
        if _rot_delta_deg(obj["rotation"], snap.rotation) > rot_tol_deg:
            changed.append(oid)
            continue
    return changed


def restore_scene_snapshot(
    controller: Controller,
    snapshot: SceneSnapshot,
    *,
    event: Optional[Any] = None,
    restore_agent: bool = True,
    pos_tol_m: float = POS_TOL_M,
    rot_tol_deg: float = ROT_TOL_DEG,
) -> tuple[bool, dict[str, Any]]:
    """Teleport changed objects (and optionally agent) back to snapshot poses.

    Uses ``force=True`` kinematic pins so restores reach the *exact* stored
    coordinates. Returns ``(ok, report)``; ``ok=False`` means caller should
    ``controller.reset`` and recapture.
    """
    report: dict[str, Any] = {
        "n_snapshot": len(snapshot.objects),
        "n_changed": 0,
        "n_restored": 0,
        "restore_failures": [],
        "verify_failures": [],
        "open_state_mismatches": [],
        "missing_objects": [],
        "method": "snapshot_restore",
    }

    # Need a current event to know what changed; cheap agent teleport if missing.
    if event is None:
        event = teleport_full(
            controller,
            position=dict(snapshot.agent_position),
            rotation=dict(snapshot.agent_rotation),
            horizon=float(snapshot.agent_horizon),
            standing=bool(snapshot.agent_standing),
        )

    changed = objects_changed_vs_snapshot(
        event, snapshot, pos_tol_m=pos_tol_m, rot_tol_deg=rot_tol_deg
    )
    report["n_changed"] = len(changed)
    report["changed_ids"] = list(changed)

    for oid in changed:
        snap = snapshot.objects.get(oid)
        if snap is None:
            continue
        # Open/close mismatch cannot be fixed by TeleportObject alone → fail safe.
        cur = get_object(event, oid)
        if (
            snap.is_open is not None
            and cur is not None
            and cur.get("isOpen") is not None
            and bool(cur.get("isOpen")) != bool(snap.is_open)
        ):
            report["open_state_mismatches"].append(oid)
            report["ok"] = False
            report["reason"] = "open_state_mismatch"
            return False, report

        ev = pin_object_pose(controller, oid, dict(snap.position), dict(snap.rotation))
        if not ev.metadata.get("lastActionSuccess"):
            # One retry with explicit teleport_object (same force path).
            ev = teleport_object(
                controller,
                oid,
                dict(snap.position),
                dict(snap.rotation),
                force=True,
                kinematic=True,
            )
        if not ev.metadata.get("lastActionSuccess"):
            report["restore_failures"].append(oid)
            report["ok"] = False
            report["reason"] = "teleport_failed"
            return False, report
        report["n_restored"] += 1
        event = ev

    if restore_agent:
        event = teleport_full(
            controller,
            position=dict(snapshot.agent_position),
            rotation=dict(snapshot.agent_rotation),
            horizon=float(snapshot.agent_horizon),
            standing=bool(snapshot.agent_standing),
        )
        if not event.metadata.get("lastActionSuccess"):
            report["ok"] = False
            report["reason"] = "agent_teleport_failed"
            return False, report

    ok, verify = verify_scene_snapshot(
        event,
        snapshot,
        pos_tol_m=pos_tol_m,
        rot_tol_deg=rot_tol_deg,
        check_agent=restore_agent,
    )
    report["verify"] = verify
    if not ok:
        report["verify_failures"] = list(verify.get("failures") or [])
        report["ok"] = False
        report["reason"] = "verify_failed"
        return False, report

    report["ok"] = True
    report["reason"] = "ok"
    return True, report


def verify_scene_snapshot(
    event: Any,
    snapshot: SceneSnapshot,
    *,
    pos_tol_m: float = POS_TOL_M,
    rot_tol_deg: float = ROT_TOL_DEG,
    check_agent: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Safety check: restored poses match the snapshot within tight tolerances."""
    failures: list[dict[str, Any]] = []
    cur = {
        str(o["objectId"]): o
        for o in (event.metadata.get("objects") or [])
        if o.get("objectId")
    }
    for oid, snap in snapshot.objects.items():
        obj = cur.get(oid)
        if obj is None:
            failures.append({"objectId": oid, "reason": "missing"})
            continue
        if bool(obj.get("isBroken")) != bool(snap.is_broken):
            failures.append(
                {
                    "objectId": oid,
                    "reason": "isBroken",
                    "snapshot": snap.is_broken,
                    "current": bool(obj.get("isBroken")),
                }
            )
            continue
        if snap.is_open is not None and obj.get("isOpen") is not None:
            if bool(obj.get("isOpen")) != bool(snap.is_open):
                failures.append(
                    {
                        "objectId": oid,
                        "reason": "isOpen",
                        "snapshot": snap.is_open,
                        "current": bool(obj.get("isOpen")),
                    }
                )
                continue
        dpos = _pos_delta(obj["position"], snap.position)
        drot = _rot_delta_deg(obj["rotation"], snap.rotation)
        if dpos > pos_tol_m or drot > rot_tol_deg:
            failures.append(
                {
                    "objectId": oid,
                    "reason": "pose_mismatch",
                    "dpos_m": round(dpos, 4),
                    "drot_deg": round(drot, 3),
                    "snapshot_pos": dict(snap.position),
                    "current_pos": _copy_pose(obj["position"]),
                }
            )

    if check_agent:
        agent = event.metadata.get("agent") or {}
        apos = agent.get("position") or {}
        arot = agent.get("rotation") or {}
        if apos and snapshot.agent_position:
            ad = _pos_delta(apos, snapshot.agent_position)
            if ad > AGENT_POS_TOL_M:
                failures.append(
                    {"objectId": "agent", "reason": "agent_pos", "dpos_m": round(ad, 4)}
                )
        if arot and snapshot.agent_rotation:
            ar = abs(float(arot.get("y", 0.0)) - float(snapshot.agent_rotation.get("y", 0.0)))
            ar = min(ar, 360.0 - ar)
            if ar > AGENT_ROT_TOL_DEG:
                failures.append(
                    {"objectId": "agent", "reason": "agent_yaw", "drot_deg": round(ar, 3)}
                )

    return (len(failures) == 0), {"failures": failures, "n_failures": len(failures)}
