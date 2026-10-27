"""ProcTHOR house / room helpers (compatible with iTHOR single-room flow).

AI2-THOR cannot conveniently load a single ProcTHOR room as its own scene, so we
load the **full apartment** once and restrict camera search to each room's
floor polygon. Each room then uses the same category-driven edit+QA path as iTHOR.

Dataset loading
---------------
Prefer the shared local checkout at
``data/procthor-10k`` (override with env
``PROCTHOR_DATA_DIR``) so workers never hit the GitHub API. Falls back to
``prior.load_dataset(..., offline=True)`` only if local files are missing.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_PROCTHOR_DATA_DIR = Path("data/procthor-10k")

# Process-level cache: split -> random-access house list / LazyJsonDataset.
_SPLIT_CACHE: dict[str, Any] = {}


def procthor_data_dir() -> Path:
    return Path(os.environ.get("PROCTHOR_DATA_DIR") or DEFAULT_PROCTHOR_DATA_DIR)


def _load_split_local(split: str, data_dir: Path) -> Any:
    """Load one split from local ``{split}.jsonl.gz`` (LazyJsonDataset if available)."""
    path = data_dir / f"{split}.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ProcTHOR split file: {path}")

    sizes = {"train": 10_000, "val": 1_000, "test": 1_000}
    total = sizes.get(split)
    with gzip.open(path, "rt") as f:
        lines = [line for line in f]

    if total is not None and len(lines) != total:
        # Don't hard-fail on count drift; warn via print for build logs.
        print(
            f"[warn] ProcTHOR {split}: expected {total} houses, found {len(lines)} in {path}",
            flush=True,
        )

    try:
        from prior import LazyJsonDataset

        return LazyJsonDataset(data=lines, dataset="procthor-dataset", split=split)
    except Exception:
        # Plain list of dicts (eager) — still fine for 10k houses.
        return [json.loads(line) for line in lines]


def _load_split_prior(split: str, *, revision: Optional[str] = None) -> Any:
    """Fallback: prior package, forced offline to avoid GitHub 429s."""
    import prior

    kwargs: dict[str, Any] = {"offline": True}
    if revision:
        kwargs["revision"] = revision
    ds = prior.load_dataset("procthor-10k", **kwargs)
    try:
        return ds[split]
    except Exception as exc:  # noqa: BLE001
        raise KeyError(f"ProcTHOR split {split!r} missing") from exc


def get_procthor_split(
    split: str = "train",
    *,
    revision: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> Any:
    """Return a cached train/val/test house collection (indexable)."""
    data_dir = Path(data_dir) if data_dir is not None else procthor_data_dir()
    cache_key = f"{data_dir}::{split}::{revision or ''}"
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

    local_gz = data_dir / f"{split}.jsonl.gz"
    if local_gz.is_file():
        print(f"[procthor] loading {split} from {local_gz}", flush=True)
        split_ds = _load_split_local(split, data_dir)
    else:
        print(
            f"[procthor] local {local_gz} missing; falling back to prior offline",
            flush=True,
        )
        split_ds = _load_split_prior(split, revision=revision)

    _SPLIT_CACHE[cache_key] = split_ds
    return split_ds


def load_procthor_house(
    house_index: int,
    *,
    split: str = "train",
    revision: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Load one ProcTHOR-10K house dict from the shared local dataset."""
    split_ds = get_procthor_split(split, revision=revision, data_dir=data_dir)
    house = split_ds[int(house_index)]
    if isinstance(house, (bytes, str)):
        house = json.loads(house)
    if not isinstance(house, dict):
        raise TypeError(
            f"ProcTHOR house {house_index} in split {split!r} is {type(house)}, expected dict"
        )
    return house


def room_label(room: dict[str, Any]) -> str:
    rid = str(room.get("id") or "room")
    rtype = str(room.get("roomType") or "Unknown")
    # Filesystem-friendly.
    safe = rid.replace("|", "_").replace("/", "_")
    return f"{safe}_{rtype}"


def point_in_floor_polygon(x: float, z: float, polygon: list[dict[str, Any]]) -> bool:
    """Ray-cast point-in-polygon on the XZ floor plan."""
    if not polygon or len(polygon) < 3:
        return False
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = float(polygon[i]["x"]), float(polygon[i]["z"])
        xj, zj = float(polygon[j]["x"]), float(polygon[j]["z"])
        intersect = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / ((zj - zi) + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def filter_positions_in_room(
    positions: list[dict[str, float]],
    room: dict[str, Any],
    *,
    margin: float = 0.05,
) -> list[dict[str, float]]:
    """Keep reachable agent positions whose (x,z) lie in the room polygon."""
    poly = room.get("floorPolygon") or []
    if not poly:
        return []
    # Slight inward bias is hard without polygon shrink; just use the polygon.
    out = []
    for p in positions:
        if point_in_floor_polygon(float(p["x"]), float(p["z"]), poly):
            out.append(p)
    return out


def list_rooms(house: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = list(house.get("rooms") or [])
    # Stable order: Kitchen / Living / Bedroom / Bathroom then id.
    order = {"Kitchen": 0, "LivingRoom": 1, "Bedroom": 2, "Bathroom": 3}

    def key(r: dict[str, Any]) -> tuple:
        return (order.get(str(r.get("roomType")), 9), str(r.get("id") or ""))

    return sorted(rooms, key=key)


def room_containing_position(
    rooms: list[dict[str, Any]],
    position: dict[str, float],
) -> Optional[dict[str, Any]]:
    """Return the first room whose floor polygon contains ``position`` (x,z)."""
    x, z = float(position["x"]), float(position["z"])
    for room in rooms:
        poly = room.get("floorPolygon") or []
        if poly and point_in_floor_polygon(x, z, poly):
            return room
    return None


def room_by_id(rooms: list[dict[str, Any]], room_id: str) -> Optional[dict[str, Any]]:
    rid = str(room_id)
    for room in rooms:
        if str(room.get("id") or "") == rid:
            return room
    return None


def room_floor_wall_segments(
    room: dict[str, Any],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """XZ wall segments from ``floorPolygon`` edges (treat each edge as a wall)."""
    poly = room.get("floorPolygon") or []
    if len(poly) < 3:
        return []
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        segs.append(
            (
                (float(a["x"]), float(a["z"])),
                (float(b["x"]), float(b["z"])),
            )
        )
    return segs


def _closest_point_on_segment(
    px: float,
    pz: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> tuple[float, float, float]:
    """Return (qx, qz, dist) from point P to segment AB in XZ."""
    abx, abz = bx - ax, bz - az
    len_sq = abx * abx + abz * abz
    if len_sq < 1e-12:
        dist = ((px - ax) ** 2 + (pz - az) ** 2) ** 0.5
        return ax, az, dist
    t = max(0.0, min(1.0, ((px - ax) * abx + (pz - az) * abz) / len_sq))
    qx = ax + t * abx
    qz = az + t * abz
    dist = ((px - qx) ** 2 + (pz - qz) ** 2) ** 0.5
    return qx, qz, dist


def object_room_wall_clearance(
    obj: dict[str, Any],
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, tuple[float, float]]:
    """Min center→wall distance (m) and unit push direction (xz) toward the wall."""
    from cit_pipeline.size_filter import aabb_xz_extent

    pos = obj.get("position") or {}
    px = float(pos.get("x") or 0.0)
    pz = float(pos.get("z") or 0.0)
    extent = aabb_xz_extent(obj)
    if extent:
        min_x, max_x, min_z, max_z = extent
        px = 0.5 * (min_x + max_x)
        pz = 0.5 * (min_z + max_z)
    best_d = float("inf")
    best_push = (0.0, 0.0)
    for (ax, az), (bx, bz) in segments:
        qx, qz, d = _closest_point_on_segment(px, pz, ax, az, bx, bz)
        if d < best_d:
            best_d = d
            dx, dz = qx - px, qz - pz
            n = (dx * dx + dz * dz) ** 0.5
            if n > 1e-8:
                best_push = (dx / n, dz / n)
            else:
                # On the wall plane: use inward normal from edge (left of A→B).
                ex, ez = bx - ax, bz - az
                el = (ex * ex + ez * ez) ** 0.5 or 1.0
                best_push = (-ez / el, ex / el)
    return best_d, best_push
