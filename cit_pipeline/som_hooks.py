"""Runtime SoM hooks for Mod-30 categories (som.py ships as .pyc only)."""

from __future__ import annotations

from typing import Any


def apply_som_hooks() -> None:
    """Extend ``build_mark_specs_for_question`` for v30 categories."""
    import cit_pipeline.som as som

    if getattr(som, "_v30_hooks_applied", False):
        return

    _orig = som.build_mark_specs_for_question
    _dedupe = som._dedupe_specs

    def build_mark_specs_for_question(
        q: Any,
        move: Any,
        *,
        frame: str = "both",
    ) -> list[tuple[str, str, str]]:
        cat = str(getattr(q, "category", "") or "")
        # Reuse closest_three lettering for count_near by temporarily aliasing.
        if cat == "count_near":
            saved = q.category
            try:
                q.category = "closest_three"
                specs = list(_orig(q, move, frame=frame))
            finally:
                q.category = saved
            return _dedupe(specs)

        specs = list(_orig(q, move, frame=frame))
        meta = dict(getattr(q, "gt_meta", None) or {})
        if cat == "near_object":
            oid = meta.get("landmark_id")
            if oid:
                specs.append(("2", str(oid), "landmark"))
            return _dedupe(specs)
        return specs

    som.build_mark_specs_for_question = build_mark_specs_for_question
    som._v30_hooks_applied = True
