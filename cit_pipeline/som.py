"""Set-of-Marks (SoM) helpers for CIT RGB frames.

Reconstructed API used by ``pipeline`` / ``som_hooks`` / ``reannotate_som_marks``.
Draws numbered badges on instance masks and rewrites phrases as ``the bowl (1)``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cit_pipeline.config import (
    SOM_MARK_FILL,
    SOM_MARK_MIN_SEP_PX,
    SOM_MARK_OFFSET_RADII,
    SOM_MARK_OUTLINE,
    SOM_MARK_RADIUS,
    SOM_TEXT_FILL,
)
from cit_pipeline.occlusion import get_object_mask


@dataclass
class SomMark:
    id: str
    object_id: str
    label: str
    uv: tuple[float, float]
    role: str = "mover"
    uv_centroid: Optional[tuple[float, float]] = None


def som_marked_phrase(object_type: Any, mark_id: str = "1") -> str:
    """``Bowl`` / ``the bowl`` → ``the bowl (1)``."""
    raw = str(object_type or "object").strip()
    if not raw:
        raw = "object"
    # Strip existing mark suffix.
    raw = re.sub(r"\s*\(\d+\)\s*$", "", raw).strip()
    low = raw.lower()
    if low.startswith("the "):
        phrase = low
    else:
        # CamelCase → spaced words then lower.
        spaced = re.sub(r"(?<!^)([A-Z])", r" \1", raw).strip()
        phrase = f"the {spaced.lower()}"
    return f"{phrase} ({mark_id})"


def _dedupe_specs(
    specs: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for mid, oid, role in specs:
        key = str(oid)
        if key in seen:
            continue
        seen.add(key)
        out.append((str(mid), str(oid), str(role)))
    return out


def build_mark_specs_for_question(
    q: Any,
    move: Any,
    *,
    frame: str = "both",
) -> list[tuple[str, str, str]]:
    """Return ``(mark_id, object_id, role)`` tuples for one QA category."""
    del frame  # specs are category-level; frame filtering is call-site.
    cat = str(getattr(q, "category", "") or "")
    meta = dict(getattr(q, "gt_meta", None) or {})
    mover_id = str(getattr(move, "object_id", "") or meta.get("mover_id") or "")
    specs: list[tuple[str, str, str]] = []
    if mover_id:
        specs.append(("1", mover_id, "mover"))

    if cat == "occlusion":
        oid = meta.get("occludee_id")
        if oid:
            specs.append(("2", str(oid), "occludee"))
    elif cat == "closer_to":
        # Mark the correct landmark when available; else first choice id.
        oid = meta.get("closer_id") or meta.get("landmark_id")
        if not oid:
            ids = list(meta.get("choice_ids") or meta.get("candidate_ids") or [])
            oid = ids[0] if ids else None
        if oid:
            specs.append(("2", str(oid), "landmark"))
    elif cat == "left_right_landmark":
        oid = meta.get("landmark_id")
        if oid:
            specs.append(("2", str(oid), "landmark"))
    elif cat == "contact_wall":
        pass  # mover only
    elif cat == "feasibility":
        pass
    elif cat == "affordance":
        oid = meta.get("seat_id")
        if oid:
            specs.append(("2", str(oid), "seat"))
    elif cat in ("closest_three", "count_near"):
        ids = list(meta.get("candidate_ids") or [])
        letters = "ABCDEF"
        for i, oid in enumerate(ids[:6]):
            specs.append((letters[i], str(oid), "candidate"))
    elif cat == "near_object":
        oid = meta.get("landmark_id")
        if oid:
            specs.append(("2", str(oid), "landmark"))
    elif cat == "support":
        pass

    return _dedupe_specs(specs)


def object_mark_uv(
    event: Any,
    object_id: str,
    *,
    width: int,
    height: int,
    offset_radii: float = SOM_MARK_OFFSET_RADII,
    radius: int = SOM_MARK_RADIUS,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return ``(badge_uv, centroid_uv)`` for an object mask, or None."""
    mask, _method = get_object_mask(event, object_id)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    # Prefer a point near the top of the mask for the badge.
    top_y = float(ys.min())
    top_xs = xs[ys == ys.min()]
    bx = float(top_xs.mean()) if len(top_xs) else cx
    by = max(radius + 2.0, top_y - float(offset_radii) * float(radius))
    bx = float(np.clip(bx, radius + 1, width - radius - 2))
    by = float(np.clip(by, radius + 1, height - radius - 2))
    return (bx, by), (cx, cy)


def resolve_marks_for_event(
    event: Any,
    specs: Sequence[tuple[str, str, str]],
    *,
    width: int,
    height: int,
) -> list[SomMark]:
    marks: list[SomMark] = []
    placed: list[tuple[float, float]] = []
    for mid, oid, role in specs:
        uv_pair = object_mark_uv(event, oid, width=width, height=height)
        if uv_pair is None:
            continue
        (u, v), (cu, cv) = uv_pair
        # Nudge away from already-placed badges.
        for _ in range(8):
            ok = True
            for pu, pv in placed:
                if (u - pu) ** 2 + (v - pv) ** 2 < float(SOM_MARK_MIN_SEP_PX) ** 2:
                    ok = False
                    break
            if ok:
                break
            u = float(np.clip(u + SOM_MARK_RADIUS, SOM_MARK_RADIUS, width - SOM_MARK_RADIUS))
            v = float(np.clip(v + SOM_MARK_RADIUS, SOM_MARK_RADIUS, height - SOM_MARK_RADIUS))
        placed.append((u, v))
        label = str(mid)
        marks.append(
            SomMark(
                id=str(mid),
                object_id=str(oid),
                label=label,
                uv=(u, v),
                role=str(role),
                uv_centroid=(cu, cv),
            )
        )
    return marks


def marks_to_meta(marks: Sequence[SomMark]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in marks or []:
        out.append(
            {
                "id": m.id,
                "object_id": m.object_id,
                "label": m.label,
                "uv": [float(m.uv[0]), float(m.uv[1])],
                "role": m.role,
                "uv_centroid": (
                    [float(m.uv_centroid[0]), float(m.uv_centroid[1])]
                    if m.uv_centroid is not None
                    else None
                ),
            }
        )
    return out


def marks_from_meta(meta_list: Any) -> list[SomMark]:
    out: list[SomMark] = []
    for d in meta_list or []:
        if not isinstance(d, dict):
            continue
        uv = d.get("uv") or [0.0, 0.0]
        uc = d.get("uv_centroid")
        out.append(
            SomMark(
                id=str(d.get("id") or ""),
                object_id=str(d.get("object_id") or ""),
                label=str(d.get("label") or d.get("id") or ""),
                uv=(float(uv[0]), float(uv[1])),
                role=str(d.get("role") or "mover"),
                uv_centroid=(float(uc[0]), float(uc[1])) if uc else None,
            )
        )
    return out


def reposition_marks(
    marks: Sequence[SomMark],
    *,
    old_offset_radii: float,
    new_offset_radii: float,
    radius: int,
    width: int,
    height: int,
    avoid_overlap: bool = True,
) -> list[SomMark]:
    """Shift badge UVs when mark radius / offset policy changes."""
    del old_offset_radii
    scale = float(new_offset_radii)
    out: list[SomMark] = []
    placed: list[tuple[float, float]] = []
    for m in marks:
        u, v = float(m.uv[0]), float(m.uv[1])
        if m.uv_centroid is not None:
            cu, cv = m.uv_centroid
            # Keep relative to centroid with new offset.
            u, v = float(cu), float(cv) - scale * float(radius)
        u = float(np.clip(u, radius + 1, width - radius - 2))
        v = float(np.clip(v, radius + 1, height - radius - 2))
        if avoid_overlap:
            for _ in range(6):
                if all((u - pu) ** 2 + (v - pv) ** 2 >= SOM_MARK_MIN_SEP_PX**2 for pu, pv in placed):
                    break
                u = float(np.clip(u + radius, radius, width - radius))
                v = float(np.clip(v + radius, radius, height - radius))
            placed.append((u, v))
        out.append(
            SomMark(
                id=m.id,
                object_id=m.object_id,
                label=m.label,
                uv=(u, v),
                role=m.role,
                uv_centroid=m.uv_centroid,
            )
        )
    return out


def draw_som_on_image(
    image: Image.Image,
    marks: Sequence[SomMark],
    *,
    radius: int = SOM_MARK_RADIUS,
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, radius)
        )
    except Exception:
        font = ImageFont.load_default()
    for m in marks or []:
        u, v = float(m.uv[0]), float(m.uv[1])
        bbox = [u - radius, v - radius, u + radius, v + radius]
        draw.ellipse(bbox, fill=tuple(SOM_MARK_FILL), outline=tuple(SOM_MARK_OUTLINE), width=2)
        label = str(m.label or m.id)
        # Center text approximately.
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:
            tw, th = (6 * len(label), 10)
        draw.text(
            (u - tw / 2.0, v - th / 2.0),
            label,
            fill=tuple(SOM_TEXT_FILL),
            font=font,
        )
    return img


# Colour SoM: text-free discs with a thin white rim, referred to as ``the bowl (marked blue)``.
# Basic colour terms (Berlin–Kay) minus white/gray, which clash with the rim and walls.
SOM_COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (230, 25, 35),
    "orange": (255, 140, 0),
    "yellow": (255, 225, 25),
    "green": (30, 170, 50),
    "cyan": (0, 215, 230),
    "blue": (20, 60, 230),
    "purple": (140, 40, 190),
    "pink": (255, 120, 200),
    "brown": (130, 75, 25),
    "black": (0, 0, 0),
}
SOM_COLOR_RADIUS = SOM_MARK_RADIUS  # same footprint as the numbered badges
SOM_COLOR_RIM = 2


def assign_mark_colors(mark_ids: Iterable[str], rng: Any) -> dict[str, str]:
    """Map each unique mark id to a distinct palette colour, uniformly at random."""
    ids = list(dict.fromkeys(str(m) for m in mark_ids))
    names = list(SOM_COLOR_PALETTE)
    if len(ids) > len(names):
        raise ValueError(f"{len(ids)} marks > {len(names)} palette colours")
    picked = rng.sample(names, len(ids))
    return dict(zip(ids, picked))


def som_color_tag(color: str) -> str:
    """``blue`` → ``(marked blue)``; "marked" keeps it from reading as the object's own colour."""
    return f"(marked {color})"


def som_color_phrase(object_type: Any, color: str) -> str:
    """``Bowl`` / ``the bowl`` / ``the bowl (1)`` → ``the bowl (marked blue)``."""
    return re.sub(r"\([^()]*\)$", som_color_tag(color), som_marked_phrase(object_type, "x"))


def draw_color_som_on_image(
    image: Image.Image,
    marks: Sequence[SomMark],
    color_by_id: dict[str, str],
    *,
    radius: int = SOM_COLOR_RADIUS,
    rim: int = SOM_COLOR_RIM,
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for m in marks or []:
        u, v = float(m.uv[0]), float(m.uv[1])
        fill = SOM_COLOR_PALETTE[color_by_id[str(m.id)]]
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=(255, 255, 255))
        r = radius - rim
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill)
    return img


def save_som_rgb(event: Any, marks: Sequence[SomMark], out_path: Path) -> Path:
    """Save event RGB with SoM badges to ``out_path``."""
    frame = getattr(event, "frame", None)
    if frame is None:
        raise ValueError("event has no frame for SoM render")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr[..., :3])
    out = draw_som_on_image(img, marks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def attach_som_to_question(
    q: Any,
    move: Any,
    event0: Any,
    event1: Any,
) -> None:
    """Fill ``text_som`` + mark metadata on a Question (in-place)."""
    specs = build_mark_specs_for_question(q, move, frame="both")
    frame = getattr(event1, "frame", None)
    h, w = (int(frame.shape[0]), int(frame.shape[1])) if frame is not None else (600, 800)
    marks0 = resolve_marks_for_event(event0, specs, width=w, height=h)
    marks1 = resolve_marks_for_event(event1, specs, width=w, height=h)

    meta = dict(getattr(q, "gt_meta", None) or {})
    meta["som_marks"] = marks_to_meta(marks1 or marks0)
    meta["som_marks_rgb0"] = marks_to_meta(marks0)
    meta["som_marks_rgb1"] = marks_to_meta(marks1)
    q.gt_meta = meta

    # Default SoM stem: replace natural mover phrase with marked phrase.
    natural = str(getattr(q, "text", "") or "")
    mover_p = meta.get("mover_phrase") or ""
    marked = som_marked_phrase(getattr(move, "object_type", None) or "object", "1")
    text_som = natural
    if mover_p and mover_p in text_som:
        text_som = text_som.replace(mover_p, marked, 1)
    else:
        # Fallback: prepend nothing; keep natural text.
        text_som = natural
    # Occlusion / others may already rewrite text_som downstream.
    if not getattr(q, "text_som", None):
        q.text_som = text_som
