"""MCQ training-set export layered on top of eval sample dirs.

Backward compatibility
----------------------
``build_one_eval_sample`` still writes the usual layout::

    out_dir/ithor_FloorPlan…_<id>/
      sample.json
      view_*_rgb0.png
      view_*_rgb1.png
      view_*_rgb0_som.png   # Modification 12
      view_*_rgb1_som.png

Debug / visualization tools
keep working unchanged. This module only *additionally* flattens complete
questions into ``dataset.jsonl`` + a shared ``images/`` folder.

Modification 12 dual export
---------------------------
One ``sample.json`` (both image + question variants) can be exported to:

* ``ithor_v3`` — original images + non-SoM questions
* ``ithor_v3_som`` — SoM images + SoM questions

Answer-flip quotas (train ~50% identical / bench ≤10%) apply per export root.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from cit_pipeline.config import (
    ANSWER_FLIP_BENCH_IDENTICAL_FRAC,
    ANSWER_FLIP_TRAIN_KEEP_IDENTICAL_FRAC,
)
from cit_pipeline.jsonl_writer import LockedJsonlWriter
from cit_pipeline.mcq_export import AnswerFlipQuota, export_sample_to_mcq_records


def default_mcq_jsonl(out_dir: Path) -> Path:
    return Path(out_dir) / "dataset.jsonl"


def default_mcq_images_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "images"


def open_mcq_writer(
    out_dir: Path,
    *,
    jsonl_path: Optional[Path] = None,
    flush_every: int = 10,
) -> LockedJsonlWriter:
    path = Path(jsonl_path) if jsonl_path else default_mcq_jsonl(out_dir)
    return LockedJsonlWriter(path, flush_every=flush_every)


def flip_quota_for_mode(
    mode: str,
    *,
    identical_frac: Optional[float] = None,
) -> Optional[AnswerFlipQuota]:
    """Build a quota tracker for ``train`` / ``bench`` / ``off``."""
    m = (mode or "off").lower().strip()
    if m in ("off", "none", ""):
        return None
    if identical_frac is not None:
        frac = float(identical_frac)
    elif m == "bench":
        frac = float(ANSWER_FLIP_BENCH_IDENTICAL_FRAC)
    else:
        frac = float(ANSWER_FLIP_TRAIN_KEEP_IDENTICAL_FRAC)
    return AnswerFlipQuota(frac)


def export_sample_dict_to_mcq(
    sample: dict[str, Any],
    *,
    sample_dir: Path,
    images_dir: Path,
    writer: LockedJsonlWriter,
    seed: int = 0,
    copy_images: bool = True,
    som: bool = False,
    flip_quota: Optional[AnswerFlipQuota] = None,
) -> list[dict[str, Any]]:
    """Convert one sample dict → MCQ records, append via ``writer``."""
    rng = random.Random(seed)
    records = export_sample_to_mcq_records(
        sample,
        sample_dir=Path(sample_dir),
        images_dir=Path(images_dir),
        rng=rng,
        copy_images=copy_images,
        som=som,
        flip_quota=flip_quota,
    )
    writer.extend(records)
    return records


def export_sample_dict_dual(
    sample: dict[str, Any],
    *,
    sample_dir: Path,
    out_dir: Path,
    out_dir_som: Optional[Path],
    writer: LockedJsonlWriter,
    writer_som: Optional[LockedJsonlWriter],
    seed: int = 0,
    flip_mode: str = "train",
    identical_frac: Optional[float] = None,
    flip_quota: Optional[AnswerFlipQuota] = None,
    flip_quota_som: Optional[AnswerFlipQuota] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Export non-SoM + SoM MCQ records to separate roots (Modification 12)."""
    if flip_quota is None:
        flip_quota = flip_quota_for_mode(flip_mode, identical_frac=identical_frac)
    recs = export_sample_dict_to_mcq(
        sample,
        sample_dir=sample_dir,
        images_dir=default_mcq_images_dir(out_dir),
        writer=writer,
        seed=seed,
        copy_images=True,
        som=False,
        flip_quota=flip_quota,
    )
    recs_som: list[dict[str, Any]] = []
    if out_dir_som is not None and writer_som is not None:
        if flip_quota_som is None:
            flip_quota_som = flip_quota_for_mode(
                flip_mode, identical_frac=identical_frac
            )
        Path(out_dir_som).mkdir(parents=True, exist_ok=True)
        recs_som = export_sample_dict_to_mcq(
            sample,
            sample_dir=sample_dir,
            images_dir=default_mcq_images_dir(out_dir_som),
            writer=writer_som,
            seed=seed + 17,
            copy_images=True,
            som=True,
            flip_quota=flip_quota_som,
        )
    return recs, recs_som


def export_sample_dir_to_mcq(
    sample_dir: Path,
    *,
    images_dir: Path,
    writer: LockedJsonlWriter,
    seed: int = 0,
    copy_images: bool = True,
    som: bool = False,
    flip_quota: Optional[AnswerFlipQuota] = None,
) -> list[dict[str, Any]]:
    sample_path = Path(sample_dir) / "sample.json"
    sample = json.loads(sample_path.read_text())
    return export_sample_dict_to_mcq(
        sample,
        sample_dir=Path(sample_dir),
        images_dir=images_dir,
        writer=writer,
        seed=seed,
        copy_images=copy_images,
        som=som,
        flip_quota=flip_quota,
    )
