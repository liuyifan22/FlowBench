"""Dataset selection for Qwen VL finetuning on CIT MCQ jsonl.

Currently: keep only *usable* rows (GT-complete letter answer + images present).
Balancing hooks are stubbed for later (yes/no targets, category caps, etc.).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


LETTER_SET = set("ABCDEF")


@dataclass
class DatasetFilterConfig:
    """Controls which jsonl rows enter training.

    Modes 1A / 2A are chosen at train time; this file only decides *which*
    examples are eligible.
    """

    require_gt_complete: bool = True
    require_letter_answer: bool = True
    require_images_exist: bool = True
    # Optional category allow-list (None = all).
    categories: Optional[tuple[str, ...]] = None
    # Future balancing (unused when None / False).
    balance_yes_no: bool = False
    target_yes_frac: Optional[float] = None  # e.g. 0.5 within yes/no cats
    max_per_category: Optional[int] = None
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.categories is not None:
            d["categories"] = list(self.categories)
        return d


DEFAULT_FILTER = DatasetFilterConfig()


def _letter_answer(rec: dict[str, Any]) -> Optional[str]:
    ans = str(rec.get("answer") or "").strip().upper()
    if not ans or ans in {"NO ANSWER", "NONE", "NULL"}:
        return None
    letters = "".join(c for c in ans if c in LETTER_SET)
    if not letters:
        return None
    # Prefer single letter; multi-letter (e.g. ACF) kept as sorted unique.
    if len(letters) == 1:
        return letters
    return "".join(sorted(set(letters)))


def is_usable(
    rec: dict[str, Any],
    *,
    dataset_root: Path,
    cfg: DatasetFilterConfig = DEFAULT_FILTER,
) -> bool:
    """Return True if ``rec`` may be used for finetuning under ``cfg``."""
    if cfg.require_gt_complete and rec.get("gt_complete") is False:
        return False
    if cfg.categories is not None:
        cat = str(rec.get("category") or "")
        if cat not in cfg.categories:
            return False
    if cfg.require_letter_answer:
        if _letter_answer(rec) is None:
            return False
    if cfg.require_images_exist:
        for key in ("image_before", "image_after"):
            rel = rec.get(key)
            if not rel:
                return False
            path = Path(str(rel))
            if not path.is_file():
                path = dataset_root / path
            if not path.is_file():
                return False
    return True


def iter_usable_records(
    jsonl_path: Path,
    *,
    dataset_root: Path,
    cfg: DatasetFilterConfig = DEFAULT_FILTER,
    limit: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield usable records (optionally apply light post-filters later)."""
    n_out = 0
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not is_usable(rec, dataset_root=dataset_root, cfg=cfg):
                continue
            yield rec
            n_out += 1
            if limit and n_out >= limit:
                return


def apply_balancing(
    records: list[dict[str, Any]],
    *,
    cfg: DatasetFilterConfig,
) -> list[dict[str, Any]]:
    """Placeholder for yes/no / category balancing.

    Currently a no-op (returns ``records`` unchanged) unless flags are set;
    when ``balance_yes_no`` is True, raises until implemented so we do not
    silently ignore a requested balance.
    """
    if cfg.balance_yes_no or cfg.target_yes_frac is not None or cfg.max_per_category:
        raise NotImplementedError(
            "Balancing rules are not implemented yet. "
            "Set balance_yes_no=False and leave target_yes_frac/max_per_category "
            "as None (default: pick usable only)."
        )
    return records


def load_filtered_records(
    jsonl_path: Path,
    *,
    dataset_root: Path,
    cfg: DatasetFilterConfig = DEFAULT_FILTER,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Load usable rows, then apply optional balancing."""
    rows = list(
        iter_usable_records(
            jsonl_path, dataset_root=dataset_root, cfg=cfg, limit=limit
        )
    )
    return apply_balancing(rows, cfg=cfg)


def write_filtered_jsonl(
    src_jsonl: Path,
    dst_jsonl: Path,
    *,
    dataset_root: Path,
    cfg: DatasetFilterConfig = DEFAULT_FILTER,
    limit: int = 0,
) -> dict[str, int]:
    """Materialize a filtered jsonl; returns simple counts."""
    rows = load_filtered_records(
        src_jsonl, dataset_root=dataset_root, cfg=cfg, limit=limit
    )
    dst_jsonl = Path(dst_jsonl)
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with dst_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_in = sum(1 for _ in Path(src_jsonl).open() if _.strip())
    return {"n_in": n_in, "n_out": len(rows), "n_dropped": n_in - len(rows)}


def summarize_filter(
    jsonl_path: Path,
    *,
    dataset_root: Path,
    cfg: DatasetFilterConfig = DEFAULT_FILTER,
) -> dict[str, Any]:
    """Count kept / dropped without writing."""
    from collections import Counter

    n_in = 0
    n_out = 0
    by_cat = Counter()
    with Path(jsonl_path).open() as f:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            rec = json.loads(line)
            if is_usable(rec, dataset_root=dataset_root, cfg=cfg):
                n_out += 1
                by_cat[str(rec.get("category") or "?")] += 1
    return {
        "n_in": n_in,
        "n_out": n_out,
        "keep_frac": (n_out / n_in) if n_in else 0.0,
        "by_category": dict(by_cat),
        "filter": cfg.to_dict(),
    }
