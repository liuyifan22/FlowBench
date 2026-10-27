"""Camera-frame move wording helpers (Modification 20).

Fixes misleading single-cardinal labels when the actual world displacement is
diagonal in the camera frame (e.g. wall-axis ±X with yaw=315° → labeled
\"left\" but ~45° forward as well).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cit_pipeline.config import MOVE_TEXT_VERTICAL_MIN_M


def format_direction_phrase(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d == "forward":
        return "along the camera forward axis"
    if d in {"back", "backward"}:
        return "along the camera backward axis"
    if d == "down":
        return "down (world −Y)"
    if d == "up":
        return "up (world +Y)"
    return d


def _as_xyz(delta_world: Any) -> np.ndarray:
    if delta_world is None:
        return np.zeros(3, dtype=np.float64)
    if hasattr(delta_world, "x"):
        return np.array(
            [float(delta_world.x), float(delta_world.y), float(delta_world.z)],
            dtype=np.float64,
        )
    if isinstance(delta_world, dict):
        return np.array(
            [
                float(delta_world.get("x", 0.0)),
                float(delta_world.get("y", 0.0)),
                float(delta_world.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
    arr = np.asarray(delta_world, dtype=np.float64).reshape(-1)
    out = np.zeros(3, dtype=np.float64)
    out[: min(3, arr.size)] = arr[:3]
    return out


def camera_frame_horizontal_components(
    delta_world: Any,
    basis: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    delta = _as_xyz(delta_world)
    delta = np.array([delta[0], 0.0, delta[2]], dtype=np.float64)
    right = np.asarray(basis["right"], dtype=np.float64)
    forward = np.asarray(basis["forward"], dtype=np.float64)
    right_h = np.array([right[0], 0.0, right[2]], dtype=np.float64)
    fwd_h = np.array([forward[0], 0.0, forward[2]], dtype=np.float64)
    if float(np.linalg.norm(right_h)) < 1e-8 or float(np.linalg.norm(fwd_h)) < 1e-8:
        return []
    right_h /= np.linalg.norm(right_h)
    fwd_h /= np.linalg.norm(fwd_h)
    return [
        ("left", float(np.dot(delta, -right_h))),
        ("right", float(np.dot(delta, right_h))),
        ("forward", float(np.dot(delta, fwd_h))),
        ("back", float(np.dot(delta, -fwd_h))),
    ]


def primary_camera_direction(delta_world: Any, basis: dict[str, np.ndarray]) -> str:
    comps = camera_frame_horizontal_components(delta_world, basis)
    if not comps:
        return "forward"
    return max(comps, key=lambda t: t[1])[0]


def alignment_score(delta_world: Any, basis: dict[str, np.ndarray]) -> float:
    comps = camera_frame_horizontal_components(delta_world, basis)
    if not comps:
        return 0.0
    best = max(v for _, v in comps)
    n = float(np.linalg.norm(_as_xyz(delta_world)[[0, 2]]))
    if n < 1e-8:
        return 0.0
    return float(best / n)


def format_move_displacement(
    direction: str,
    meters: float,
    delta_world: Any = None,
    basis: Optional[dict[str, np.ndarray]] = None,
    *,
    dy_min: float = MOVE_TEXT_VERTICAL_MIN_M,
    compound_frac: float = 0.30,
) -> str:
    """Format horizontal (+ optional downward) move with optional compound axes."""
    delta = _as_xyz(delta_world)
    dy = float(delta[1])

    if basis is not None and delta_world is not None:
        comps = camera_frame_horizontal_components(delta, basis)
        pos = [(name, val) for name, val in comps if val > 0.02]
        pos.sort(key=lambda t: -t[1])
        if pos:
            primary_name, primary_m = pos[0]
            parts = [f"{primary_m:.2f}m {format_direction_phrase(primary_name)}"]
            if len(pos) > 1:
                sec_name, sec_m = pos[1]
                if sec_m >= compound_frac * primary_m:
                    parts.append(f"{sec_m:.2f}m {format_direction_phrase(sec_name)}")
            horiz = " and ".join(parts)
        else:
            horiz = f"{float(meters):.2f}m {format_direction_phrase(direction)}"
    else:
        horiz = f"{float(meters):.2f}m {format_direction_phrase(direction)}"

    if dy < -float(dy_min):
        return f"{horiz} and {abs(dy):.2f}m down"
    return horiz
