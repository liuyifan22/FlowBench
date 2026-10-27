#!/usr/bin/env python3
"""Generate full iTHOR CIT MCQ dataset (procthor_v9 categories on iTHOR 120 scenes) → ithor_v9.

Scenes: FloorPlan1–30, 201–230, 301–330, 401–430 (120 total).

v7.2 + Mod 30 balance profile (~45–55% Yes on yes/no categories):
  - ``fast_find_edit=True`` + snapshot restore (enabled explicitly for iTHOR)
  - Higher fast tries (contact/occlusion/feasibility) + wider wall select band
  - Manual Yes/No targets at 50% No for occlusion/contact/feasibility/left_right
    (closer_to unchanged)
  - Wall knobs: ``wall_away_prob``, ``fast_wall_select_near_m``
  - VLM question wording text-only (no rgb0/rgb1)
  - Dual MCQ roots under ``ithor_v9/{natural,som}``

Example::

  python build_v9_distributed_ithor.py \\
    --out …/ithor_v9/natural --out-som …/ithor_v9/som \\
    --gpus 1 --worker-id 0 --num-workers 16 \\
    --vlm-base-url http://localhost:19857/v1

Run one such process per AI2-THOR GPU (see README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cit_pipeline.config import (
    DEFAULT_DATASET_PARENT,
    QA_CATEGORIES,
    ITHOR_CONTACT_TARGET_NO_FRAC,
    ITHOR_FAST_CONTACT_MAX_TRIES,
    ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS,
    ITHOR_FAST_OCCLUSION_MAX_TRIES,
    ITHOR_FAST_WALL_SELECT_NEAR_M,
    ITHOR_FEASIBILITY_TARGET_NO_FRAC,
    ITHOR_FULL_SCENES,
    ITHOR_LEFT_RIGHT_TARGET_NO_FRAC,
    ITHOR_OCCLUDE_TARGET_NO_FRAC,
    ITHOR_WALL_AWAY_PROB,
    MODIFICATION_TAG,
    VLM_BASE_URL,
    VLM_MODEL,
    PipelineConfig,
)

DEFAULT_OUT = DEFAULT_DATASET_PARENT / "ithor_v9" / "natural"
DEFAULT_OUT_SOM = DEFAULT_DATASET_PARENT / "ithor_v9" / "som"
DEFAULT_VLM = VLM_BASE_URL
MODIFICATION = f"{MODIFICATION_TAG}_ithor_v9"


def _scene_already_done(out: Path, scene: str) -> bool:
    """True if any sample for ``scene`` already has views on disk."""
    for sp in out.glob(f"ithor_{scene}_*/sample.json"):
        try:
            data = json.loads(sp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if data.get("scene") != scene:
            continue
        if data.get("views"):
            return True
    return False


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _gpu_shard_worker(payload: dict) -> dict:
    """Build a shard of scenes on one GPU; flush MCQ jsonl every N items."""
    t_shard0 = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu"])
    cit_bin = os.path.dirname(sys.executable)
    path0 = os.environ.get("PATH", "")
    if cit_bin not in path0.split(":"):
        os.environ["PATH"] = f"{cit_bin}:{path0}"
    os.environ.setdefault("TMPDIR", os.environ.get("CIT_TMP_DIR", str(Path.home() / ".cache" / "cit" / "tmp")))
    os.environ.setdefault("TMP", os.environ["TMPDIR"])
    os.environ.setdefault("TEMP", os.environ["TMPDIR"])
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    from cit_pipeline.dataset_export import (
        default_mcq_images_dir,
        export_sample_dict_dual,
        flip_quota_for_mode,
        open_mcq_writer,
    )
    from cit_pipeline.pipeline import build_one_eval_sample, write_manifest

    out = Path(payload["out"])
    out_som = Path(payload["out_som"]) if payload.get("out_som") else None
    gpu = payload["gpu"]
    use_vlm = bool(payload["use_vlm"])
    vlm_optional = bool(payload.get("vlm_optional", False))
    vlm_connect_retries = int(payload.get("vlm_connect_retries", 8))
    attach_cot = bool(payload.get("attach_cot", False))
    attach_pretended = bool(payload.get("attach_pretended_cot", False))
    export_mcq = bool(payload.get("export_mcq", True))
    flush_every = int(payload.get("flush_every", 1))
    scenes = list(payload["scenes"])
    seed_base = int(payload["seed_base"])
    vlm_base_url = str(payload.get("vlm_base_url") or VLM_BASE_URL)
    vlm_model = str(payload.get("vlm_model") or VLM_MODEL)
    worker_tag = payload.get("worker_tag") or f"gpu{gpu}"
    flip_mode = str(payload.get("flip_mode") or "train")
    identical_frac = payload.get("identical_frac")
    fast_find_edit = bool(payload.get("fast_find_edit", True))
    snapshot_restore = payload.get("snapshot_restore")
    vlm_word_use_images = bool(payload.get("vlm_word_use_images", False))
    vlm_referring_polish = bool(payload.get("vlm_referring_polish", False))
    # Mod 30 iTHOR balance profile (optional payload overrides).
    balance = payload.get("ithor_balance") or {}
    occlude_target_no_frac = float(
        balance.get("occlude_target_no_frac", ITHOR_OCCLUDE_TARGET_NO_FRAC)
    )
    contact_target_no_frac = float(
        balance.get("contact_target_no_frac", ITHOR_CONTACT_TARGET_NO_FRAC)
    )
    feasibility_target_no_frac = float(
        balance.get("feasibility_target_no_frac", ITHOR_FEASIBILITY_TARGET_NO_FRAC)
    )
    left_right_target_no_frac = float(
        balance.get("left_right_target_no_frac", ITHOR_LEFT_RIGHT_TARGET_NO_FRAC)
    )
    wall_away_prob = float(balance.get("wall_away_prob", ITHOR_WALL_AWAY_PROB))
    fast_wall_select_near_m = float(
        balance.get("fast_wall_select_near_m", ITHOR_FAST_WALL_SELECT_NEAR_M)
    )
    fast_contact_max_tries = int(
        balance.get("fast_contact_max_tries", ITHOR_FAST_CONTACT_MAX_TRIES)
    )
    fast_occlusion_max_tries = int(
        balance.get("fast_occlusion_max_tries", ITHOR_FAST_OCCLUSION_MAX_TRIES)
    )
    fast_feasibility_max_attempts = int(
        balance.get(
            "fast_feasibility_max_attempts", ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS
        )
    )
    categories = payload.get("categories")
    if categories is not None:
        categories = tuple(str(c) for c in categories if str(c).strip())
        if not categories:
            categories = None

    writer = None
    writer_som = None
    images_dir = default_mcq_images_dir(out)
    flip_quota = flip_quota_for_mode(flip_mode, identical_frac=identical_frac)
    flip_quota_som = flip_quota_for_mode(flip_mode, identical_frac=identical_frac)
    if export_mcq:
        writer = open_mcq_writer(
            out,
            jsonl_path=Path(payload["mcq_jsonl"]) if payload.get("mcq_jsonl") else None,
            flush_every=flush_every,
        )
        images_dir.mkdir(parents=True, exist_ok=True)
        if out_som is not None:
            out_som.mkdir(parents=True, exist_ok=True)
            writer_som = open_mcq_writer(
                out_som,
                jsonl_path=(
                    Path(payload["mcq_jsonl_som"])
                    if payload.get("mcq_jsonl_som")
                    else None
                ),
                flush_every=flush_every,
            )

    results: list[dict] = []
    n_mcq = 0
    n_mcq_som = 0
    print(
        f"[{worker_tag}] shard start: {len(scenes)} scene(s) "
        f"mcq={export_mcq} som={out_som} vlm={vlm_base_url} "
        f"use_vlm={use_vlm} vlm_optional={vlm_optional} retries={vlm_connect_retries} "
        f"fast={fast_find_edit} snap={snapshot_restore} mod={MODIFICATION} "
        f"flip_mode={flip_mode} cats={categories or 'ALL'} "
        f"balance={{occ_no={occlude_target_no_frac}, contact_no={contact_target_no_frac}, "
        f"feas_no={feasibility_target_no_frac}, lr_no={left_right_target_no_frac}, "
        f"wall_away={wall_away_prob}, wall_near={fast_wall_select_near_m}, "
        f"tries=c{fast_contact_max_tries}/o{fast_occlusion_max_tries}/"
        f"f{fast_feasibility_max_attempts}}}",
        flush=True,
    )

    try:
        for scene, seed_idx in scenes:
            t0 = time.perf_counter()
            print(
                f"[{worker_tag}] START {scene} seed={seed_base + seed_idx}",
                flush=True,
            )
            try:
                cfg = PipelineConfig(
                    scene=scene,
                    dataset="ithor",
                    out_dir=out,
                    out_dir_som=out_som,
                    seed=seed_base + seed_idx,
                    n_views=1,
                    questions_per_view=1,
                    category_driven=True,
                    use_vlm_ranking=use_vlm,
                    vlm_optional=vlm_optional,
                    vlm_connect_retries=vlm_connect_retries,
                    vlm_base_url=vlm_base_url,
                    vlm_model=vlm_model,
                    attach_cot=attach_cot,
                    attach_pretended_cot=attach_pretended and attach_cot,
                    vlm_word_use_images=vlm_word_use_images,
                    vlm_referring_polish=vlm_referring_polish,
                    export_mcq=export_mcq,
                    enable_som=True,
                    answer_flip_mode=flip_mode,
                    answer_flip_identical_frac=(
                        float(identical_frac) if identical_frac is not None else None
                    ),
                    modification=MODIFICATION,
                    # iTHOR default fast is off; v7 explicitly enables it.
                    fast_find_edit=fast_find_edit,
                    snapshot_restore=(
                        True
                        if snapshot_restore is None
                        else bool(snapshot_restore)
                    ),
                    # Mod 30: manual Yes/No + wall/fast enumeration profile.
                    occlude_target_no_frac=occlude_target_no_frac,
                    contact_target_no_frac=contact_target_no_frac,
                    feasibility_target_no_frac=feasibility_target_no_frac,
                    left_right_target_no_frac=left_right_target_no_frac,
                    wall_away_prob=wall_away_prob,
                    fast_wall_select_near_m=fast_wall_select_near_m,
                    fast_contact_max_tries=fast_contact_max_tries,
                    fast_occlusion_max_tries=fast_occlusion_max_tries,
                    fast_feasibility_max_attempts=fast_feasibility_max_attempts,
                    categories=categories,
                )
                sample = build_one_eval_sample(cfg)
                write_manifest(out, sample)

                sample_dir = out / sample.sample_id
                sample_path = sample_dir / "sample.json"
                data = json.loads(sample_path.read_text())
                if use_vlm and attach_cot:
                    n_complete = 0
                    n_warm = 0
                    n_pret = 0
                    for view in data.get("views") or []:
                        for q in view.get("questions") or []:
                            if not q.get("gt_complete", True):
                                continue
                            n_complete += 1
                            cot = q.get("cot") or {}
                            if cot.get("warm"):
                                n_warm += 1
                            if cot.get("pretended"):
                                n_pret += 1
                    if n_complete and n_warm < n_complete:
                        raise RuntimeError(
                            f"incomplete warm CoT ({n_warm}/{n_complete}); "
                            "refusing MCQ export"
                        )
                    if attach_pretended and n_complete and n_pret < n_complete:
                        raise RuntimeError(
                            f"incomplete pretended CoT ({n_pret}/{n_complete}); "
                            "refusing MCQ export"
                        )

                mcq_n = 0
                mcq_som_n = 0
                if writer is not None:
                    recs, recs_som = export_sample_dict_dual(
                        data,
                        sample_dir=sample_dir,
                        out_dir=out,
                        out_dir_som=out_som,
                        writer=writer,
                        writer_som=writer_som,
                        seed=seed_base + seed_idx,
                        flip_mode=flip_mode,
                        identical_frac=(
                            float(identical_frac)
                            if identical_frac is not None
                            else None
                        ),
                        flip_quota=flip_quota,
                        flip_quota_som=flip_quota_som,
                    )
                    mcq_n = len(recs)
                    mcq_som_n = len(recs_som)
                    n_mcq += mcq_n
                    n_mcq_som += mcq_som_n

                warm_n = 0
                pretended_n = 0
                for view in data.get("views") or []:
                    for q in view.get("questions") or []:
                        cot = q.get("cot") or {}
                        if cot.get("warm"):
                            warm_n += 1
                        if cot.get("pretended"):
                            pretended_n += 1

                elapsed = time.perf_counter() - t0
                n_views = len(data.get("views") or sample.views)
                row = {
                    "ok": True,
                    "scene": scene,
                    "sample_id": sample.sample_id,
                    "n_views": n_views,
                    "n_mcq": mcq_n,
                    "n_mcq_som": mcq_som_n,
                    "warm_cot": warm_n,
                    "pretended_cot": pretended_n,
                    "elapsed_s": elapsed,
                    "gpu": gpu,
                    "worker": worker_tag,
                    "categories": (data.get("meta") or {}).get("categories_done"),
                }
                results.append(row)
                print(
                    f"[{worker_tag}] OK {scene} -> {sample.sample_id} "
                    f"views={n_views} mcq={mcq_n} mcq_som={mcq_som_n} "
                    f"warm={warm_n} pretended={pretended_n} {elapsed:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                elapsed = time.perf_counter() - t0
                for sp in out.glob(f"ithor_{scene}_*"):
                    if sp.is_dir():
                        try:
                            import shutil

                            shutil.rmtree(sp)
                            print(
                                f"[{worker_tag}] removed partial {sp.name}",
                                flush=True,
                            )
                        except Exception as rm_exc:  # noqa: BLE001
                            print(
                                f"[{worker_tag}] partial cleanup failed {sp}: {rm_exc}",
                                flush=True,
                            )
                print(
                    f"[{worker_tag}] FAIL {scene} ({elapsed:.1f}s): {exc}",
                    flush=True,
                )
                results.append(
                    {
                        "ok": False,
                        "scene": scene,
                        "error": str(exc),
                        "elapsed_s": elapsed,
                        "gpu": gpu,
                        "worker": worker_tag,
                    }
                )
    finally:
        flushed = writer.close() if writer is not None else 0
        flushed_som = writer_som.close() if writer_som is not None else 0
        if flushed:
            print(
                f"[{worker_tag}] final jsonl flush: {flushed} record(s)",
                flush=True,
            )
        if flushed_som:
            print(
                f"[{worker_tag}] final SoM jsonl flush: {flushed_som} record(s)",
                flush=True,
            )

    ok = sum(1 for r in results if r.get("ok"))
    bad = len(results) - ok
    return {
        "gpu": gpu,
        "worker": worker_tag,
        "n_ok": ok,
        "n_fail": bad,
        "n_mcq": n_mcq,
        "n_mcq_som": n_mcq_som,
        "elapsed_s": time.perf_counter() - t_shard0,
        "per_scene": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--out-som",
        type=Path,
        default=DEFAULT_OUT_SOM,
        help="SoM MCQ root (dataset.jsonl + images/). Pass empty / none to skip.",
    )
    p.add_argument("--seed-base", type=int, default=9200)
    p.add_argument("--gpus", default="0", help="AI2-THOR GPU id(s), comma-separated")
    p.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="This process's shard index in [0, num-workers)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total distributed workers (round-robin scene sharding)",
    )
    p.add_argument(
        "--vlm-base-url",
        default=DEFAULT_VLM,
        help="OpenAI-compatible VLM base URL (…/v1)",
    )
    p.add_argument("--vlm-model", default=VLM_MODEL)
    p.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM entirely; export GT template question text",
    )
    p.add_argument(
        "--vlm-optional",
        action="store_true",
        help="If VLM connect fails, continue with GT template questions",
    )
    p.add_argument(
        "--vlm-connect-retries",
        type=int,
        default=None,
        help="VLM connect attempts (default 8; --no-vlm forces 0)",
    )
    p.add_argument(
        "--vlm-word-images",
        action="store_true",
        help="Pass rgb0/rgb1 into VLM wording (slow; off by default for v7)",
    )
    p.add_argument(
        "--vlm-referring-polish",
        action="store_true",
        help="VLM polish referring phrases (slow; off by default for v7)",
    )
    p.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable Mod-28 fast_find_edit (on by default for ithor_v7)",
    )
    p.add_argument(
        "--no-snapshot-restore",
        action="store_true",
        help="Disable pose snapshot rewind between categories",
    )
    p.add_argument(
        "--with-cot",
        action="store_true",
        help="Attach warm CoT via VLM (skipped by default for speed)",
    )
    p.add_argument(
        "--with-pretended-cot",
        action="store_true",
        help="Also attach pretended CoT (implies --with-cot; skipped by default)",
    )
    p.add_argument(
        "--no-cot",
        action="store_true",
        help="Deprecated no-op (warm/pretended CoT already off by default)",
    )
    p.add_argument(
        "--no-pretended-cot",
        action="store_true",
        help="Deprecated no-op (pretended CoT already off by default)",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip scenes with samples")
    p.add_argument(
        "--no-mcq",
        action="store_true",
        help="Do not write dataset.jsonl / images/ (legacy sample dirs only)",
    )
    p.add_argument(
        "--mcq-jsonl",
        type=Path,
        default=None,
        help="MCQ jsonl path (default: OUT/dataset.jsonl)",
    )
    p.add_argument(
        "--mcq-jsonl-som",
        type=Path,
        default=None,
        help="SoM MCQ jsonl path (default: OUT_SOM/dataset.jsonl)",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Append to jsonl under flock every N MCQ records (default 1)",
    )
    p.add_argument(
        "--flip-mode",
        choices=("train", "bench", "off"),
        default="train",
        help="Answer-identical quota: train~50pct, bench<=10pct, off=keep all",
    )
    p.add_argument(
        "--identical-frac",
        type=float,
        default=None,
        help="Override identical-answer keep fraction for flip-mode",
    )
    p.add_argument(
        "--scenes",
        default="",
        help="Comma-separated override; default = all 120 ITHOR_FULL_SCENES",
    )
    p.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Optional cap for smoke tests (0 = all)",
    )
    p.add_argument(
        "--categories",
        default="",
        help="Comma-separated QA categories to build (default: all). "
        "Example: occlusion,contact_wall",
    )
    p.add_argument(
        "--worker-tag",
        default="",
        help="Log label (default: host-gpuN)",
    )
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    out_som_arg = args.out_som
    if out_som_arg is not None and str(out_som_arg).strip().lower() in ("", "-", "none"):
        out_som: Path | None = None
    else:
        out_som = Path(out_som_arg) if out_som_arg is not None else DEFAULT_OUT_SOM
        if out_som is not None:
            out_som.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("Need at least one GPU id")

    worker_id = int(args.worker_id)
    num_workers = max(1, int(args.num_workers))
    if worker_id < 0 or worker_id >= num_workers:
        raise SystemExit(f"--worker-id must be in [0, {num_workers})")

    if args.scenes.strip():
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = list(ITHOR_FULL_SCENES)
    if args.max_scenes and args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    assert len(ITHOR_FULL_SCENES) == 120, len(ITHOR_FULL_SCENES)

    skipped: list[str] = []
    todo_all: list[tuple[str, int]] = []
    for scene in scenes:
        try:
            seed_idx = list(ITHOR_FULL_SCENES).index(scene)
        except ValueError:
            seed_idx = len(todo_all)
        if args.skip_existing and _scene_already_done(out, scene):
            skipped.append(scene)
        else:
            todo_all.append((scene, seed_idx))

    todo = [item for i, item in enumerate(todo_all) if i % num_workers == worker_id]

    host = os.uname().nodename.split(".")[0]
    worker_tag = args.worker_tag.strip() or f"{host}-w{worker_id}"

    shards: list[list[tuple[str, int]]] = [[] for _ in gpus]
    for i, item in enumerate(todo):
        shards[i % len(gpus)].append(item)

    use_vlm = not args.no_vlm
    vlm_optional = bool(args.vlm_optional) or bool(args.no_vlm)
    if args.vlm_connect_retries is not None:
        vlm_connect_retries = max(0, int(args.vlm_connect_retries))
    elif args.no_vlm:
        vlm_connect_retries = 0
    else:
        vlm_connect_retries = 8
    attach_cot = bool(args.with_cot) and (not args.no_cot) and use_vlm
    attach_pretended = (
        attach_cot
        and bool(args.with_pretended_cot)
        and (not args.no_pretended_cot)
        and use_vlm
    )
    export_mcq = not args.no_mcq
    mcq_jsonl = args.mcq_jsonl or (out / "dataset.jsonl")
    mcq_jsonl_som = None
    if out_som is not None:
        mcq_jsonl_som = args.mcq_jsonl_som or (out_som / "dataset.jsonl")
    vlm_base_url = args.vlm_base_url
    fast_find_edit = not bool(args.no_fast)
    snapshot_restore = False if bool(args.no_snapshot_restore) else True
    categories = tuple(
        c.strip() for c in str(args.categories or "").split(",") if c.strip()
    )
    if not categories:
        # Match procthor_v9: eight QA categories (5 core + 3 distance).
        categories = tuple(QA_CATEGORIES)

    jobs = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        jobs.append(
            {
                "gpu": gpu,
                "scenes": shard,
                "out": str(out.resolve()),
                "out_som": str(out_som.resolve()) if out_som is not None else None,
                "seed_base": args.seed_base,
                "use_vlm": use_vlm,
                "vlm_optional": vlm_optional,
                "vlm_connect_retries": vlm_connect_retries,
                "attach_cot": attach_cot,
                "attach_pretended_cot": attach_pretended,
                "vlm_word_use_images": bool(args.vlm_word_images),
                "vlm_referring_polish": bool(args.vlm_referring_polish),
                "fast_find_edit": fast_find_edit,
                "snapshot_restore": snapshot_restore,
                "categories": list(categories) if categories else None,
                "ithor_balance": {
                    "occlude_target_no_frac": ITHOR_OCCLUDE_TARGET_NO_FRAC,
                    "contact_target_no_frac": ITHOR_CONTACT_TARGET_NO_FRAC,
                    "feasibility_target_no_frac": ITHOR_FEASIBILITY_TARGET_NO_FRAC,
                    "left_right_target_no_frac": ITHOR_LEFT_RIGHT_TARGET_NO_FRAC,
                    "wall_away_prob": ITHOR_WALL_AWAY_PROB,
                    "fast_wall_select_near_m": ITHOR_FAST_WALL_SELECT_NEAR_M,
                    "fast_contact_max_tries": ITHOR_FAST_CONTACT_MAX_TRIES,
                    "fast_occlusion_max_tries": ITHOR_FAST_OCCLUSION_MAX_TRIES,
                    "fast_feasibility_max_attempts": ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS,
                },
                "export_mcq": export_mcq,
                "mcq_jsonl": str(mcq_jsonl.resolve()),
                "mcq_jsonl_som": (
                    str(mcq_jsonl_som.resolve()) if mcq_jsonl_som is not None else None
                ),
                "flush_every": max(1, int(args.flush_every)),
                "vlm_base_url": vlm_base_url,
                "vlm_model": args.vlm_model,
                "worker_tag": f"{worker_tag}-gpu{gpu}",
                "flip_mode": args.flip_mode,
                "identical_frac": args.identical_frac,
            }
        )

    log_path = out / f"build_w{worker_id}.log"
    timing_path = out / f"build_timing_w{worker_id}.json"
    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    header = (
        f"=== iTHOR v7 distributed MCQ build (fast + SoM + Mod29 balance) ===\n"
        f"modification={MODIFICATION}\n"
        f"fast_find_edit={fast_find_edit} snapshot_restore={snapshot_restore}\n"
        f"categories={categories or 'ALL'}\n"
        f"balance: occ_no={ITHOR_OCCLUDE_TARGET_NO_FRAC} "
        f"contact_no={ITHOR_CONTACT_TARGET_NO_FRAC} "
        f"feas_no={ITHOR_FEASIBILITY_TARGET_NO_FRAC} "
        f"lr_no={ITHOR_LEFT_RIGHT_TARGET_NO_FRAC}\n"
        f"wall: away_prob={ITHOR_WALL_AWAY_PROB} "
        f"fast_select_near_m={ITHOR_FAST_WALL_SELECT_NEAR_M}\n"
        f"fast_tries: contact={ITHOR_FAST_CONTACT_MAX_TRIES} "
        f"occlusion={ITHOR_FAST_OCCLUSION_MAX_TRIES} "
        f"feasibility={ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS}\n"
        f"vlm_optional={vlm_optional} retries={vlm_connect_retries}\n"
        f"start_utc={start_iso}\n"
        f"host={host} worker_id={worker_id}/{num_workers} tag={worker_tag}\n"
        f"out={out}\n"
        f"out_som={out_som}\n"
        f"gpus={gpus}\n"
        f"scenes_total={len(scenes)} shard_todo={len(todo)} "
        f"global_skipped={len(skipped)}\n"
        f"vlm={not args.no_vlm} url={vlm_base_url} cot={attach_cot} "
        f"pretended={attach_pretended}\n"
        f"mcq={export_mcq} jsonl={mcq_jsonl} som_jsonl={mcq_jsonl_som} "
        f"flush_every={args.flush_every} flip_mode={args.flip_mode}\n"
    )
    print(header, flush=True)
    with log_path.open("a") as f:
        f.write(header + "\n")

    shard_results: list[dict] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {ex.submit(_gpu_shard_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                shard_results.append(r)
                with log_path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "gpu": r.get("gpu"),
                                "worker": r.get("worker"),
                                "n_ok": r.get("n_ok"),
                                "n_fail": r.get("n_fail"),
                                "n_mcq": r.get("n_mcq"),
                                "n_mcq_som": r.get("n_mcq_som"),
                                "elapsed_s": r.get("elapsed_s"),
                            }
                        )
                        + "\n"
                    )

    total_s = time.perf_counter() - t0
    end_iso = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    for sr in shard_results:
        results.extend(sr.get("per_scene") or [])

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    n_mcq = sum(int(r.get("n_mcq") or 0) for r in ok)
    n_mcq_som = sum(int(r.get("n_mcq_som") or 0) for r in ok)

    timing = {
        "modification": MODIFICATION,
        "start_utc": start_iso,
        "end_utc": end_iso,
        "total_wall_seconds": total_s,
        "total_wall_hms": _fmt_hms(total_s),
        "host": host,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "gpus": gpus,
        "vlm_base_url": vlm_base_url,
        "scenes_requested": len(scenes),
        "scenes_todo_shard": len(todo),
        "scenes_skipped_existing_global": skipped,
        "n_ok": len(ok),
        "n_fail": len(bad),
        "n_mcq_records": n_mcq,
        "n_mcq_som_records": n_mcq_som,
        "mcq_jsonl": str(mcq_jsonl) if export_mcq else None,
        "mcq_jsonl_som": str(mcq_jsonl_som) if export_mcq and mcq_jsonl_som else None,
        "flush_every": args.flush_every,
        "flip_mode": args.flip_mode,
        "per_scene": sorted(results, key=lambda r: r.get("scene") or ""),
        "sum_scene_seconds": sum(float(r.get("elapsed_s") or 0) for r in results),
        "out": str(out),
        "out_som": str(out_som) if out_som else None,
    }
    timing_path.write_text(json.dumps(timing, indent=2))

    summary = (
        f"{'=' * 60}\n"
        f"Worker {worker_id}/{num_workers} done: {len(ok)} ok, {len(bad)} failed, "
        f"{n_mcq} MCQ + {n_mcq_som} SoM MCQ rows\n"
        f"Total wall time: {timing['total_wall_hms']} ({total_s:.1f}s)\n"
        f"→ {out}\n"
        f"→ SoM {out_som}\n"
        f"Timing: {timing_path}\n"
    )
    print(summary, flush=True)
    with log_path.open("a") as f:
        f.write(summary)
        for r in sorted(ok, key=lambda x: x["scene"]):
            f.write(
                f"  OK {r['scene']} {r.get('sample_id')} "
                f"views={r.get('n_views')} mcq={r.get('n_mcq')} "
                f"mcq_som={r.get('n_mcq_som')} {r.get('elapsed_s', 0):.1f}s\n"
            )
        for r in bad:
            f.write(f"  FAIL {r.get('scene')}: {r.get('error')}\n")

    if bad and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
