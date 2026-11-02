"""Row filters and text fixes shared by the train / test set builders and the release builder."""

from __future__ import annotations

import re
from typing import Any


# Movers that are part of the room structure or of fixed furniture are dropped:
# windows, doors (doorway / doorframe), the room shell ("the floor"), cabinet /
# drawer parts, sinks, faucets and countertops.
STRUCTURAL_MOVER_TYPES = {"window", "door", "room", "wall", "sink", "sinkbasin", "faucet", "countertop"}
STRUCTURAL_MOVER_PHRASES = {
    "the window", "the doorway", "the doorframe", "the door", "the cabinet", "the cupboard",
    "the drawer", "the wall", "the sink", "the faucet", "the countertop", "the counter top",
}
_MOVER_PHRASE_RE = re.compile(r"move (.+?) \(marked \w+\)")


def structural_mover(r: dict[str, Any]) -> bool:
    if str(r.get("mover_object_id") or "").split("|")[0].lower() in STRUCTURAL_MOVER_TYPES:
        return True
    m = _MOVER_PHRASE_RE.match(str(r.get("motion_command") or ""))
    return bool(m) and m[1] in STRUCTURAL_MOVER_PHRASES


def usable(r: dict[str, Any]) -> bool:
    return bool(r.get("gt_complete")) and bool(re.fullmatch(r"[A-F]+", str(r.get("answer") or "")))


# Natural near_object / count_near static questions refer to the mover as a bare "it"
# ("Is it within 0.50m of the cellphone?"); name it from the motion command instead.
_MOVER_FROM_MOTION = re.compile(r"^move (.+?) -?\d+(?:\.\d+)?\s?m\b", re.IGNORECASE)
DANGLING_IT_CATS = {"near_object", "count_near"}


def name_dangling_it(r: dict[str, Any]) -> str | None:
    q = r.get("static_question")
    if r.get("category") not in DANGLING_IT_CATS or not isinstance(q, str):
        return q
    m = _MOVER_FROM_MOTION.match(str(r.get("motion_command") or ""))
    if not m or len(re.findall(r"\bit\b", q, flags=re.IGNORECASE)) != 1:
        return q
    phrase = m[1]
    out = re.sub(r"\bit\b", phrase, q, count=1, flags=re.IGNORECASE)
    return out[:1].upper() + out[1:]
