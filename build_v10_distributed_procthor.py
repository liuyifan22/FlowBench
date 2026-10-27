#!/usr/bin/env python3
"""Generate ProcTHOR CIT MCQ dataset → procthor_v10 (distributed).

v10: feasibility + contact_wall only (shortcut-fixed generators).
  - Feasibility: continuous shared-range meters, meter-first polarity pick
  - Contact_wall: always toward; ~45% half-motion Nos
  - ``fast_find_edit=True`` + snapshot restore
  - Dual MCQ roots under ``procthor_v10/{natural,som}``

Example::

  python build_v10_distributed_procthor.py \\
    --out …/procthor_v10/natural \\
    --out-som …/procthor_v10/som \\
    --n-houses 10000 --house-pool 10000 --worker-id 0 --num-workers 48 \\
    --categories contact_wall,feasibility \\
    --vlm-base-url http://localhost:19857/v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cit_pipeline.config import (
    CONTACT_ALWAYS_TOWARD,
    CONTACT_FAST_MAX_CANDIDATES,
    CONTACT_FAST_SINGLE_CAMERA,
    CONTACT_FEAS_FIRST,
    CONTACT_HALF_MOTION_NO_FRAC,
    CONTACT_ROOM_WALL_MAX_M,
    CONTACT_ROOM_WALL_PREFILTER,
    DEFAULT_DATASET_PARENT,
    ITHOR_CONTACT_TARGET_NO_FRAC,
    ITHOR_FAST_CONTACT_MAX_TRIES,
    ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS,
    ITHOR_FAST_WALL_SELECT_NEAR_M,
    ITHOR_FEASIBILITY_TARGET_NO_FRAC,
    ITHOR_WALL_AWAY_PROB,
    MODIFICATION_TAG,
    SOM_MARK_OFFSET_RADII,
    VLM_BASE_URL,
    VLM_MODEL,
    PipelineConfig,
)

DEFAULT_OUT = DEFAULT_DATASET_PARENT / "procthor_v10" / "natural"
DEFAULT_OUT_SOM = DEFAULT_DATASET_PARENT / "procthor_v10" / "som"
MODIFICATION = f"{MODIFICATION_TAG}_procthor_v10_feas_contact"
DEFAULT_CATEGORIES = ("contact_wall", "feasibility")


def _house_already_done(out: Path, house_index: int, split: str) -> bool:
    prefix = f"procthor_{split}{house_index}_"
    for sp in out.glob(f"{prefix}*/sample.json"):
        try:
            data = json.loads(sp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if data.get("views"):
            return True
    return False


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-som", type=Path, default=DEFAULT_OUT_SOM)
    p.add_argument("--seed-base", type=int, default=11000)
    p.add_argument(
        "--categories",
        default="contact_wall,feasibility",
        help="Comma-separated QA categories (default: contact_wall,feasibility)",
    )
    p.add_argument("--gpus", default="0", help="AI2-THOR GPU id (single)")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--vlm-base-url", default=VLM_BASE_URL)
    p.add_argument("--vlm-model", default=VLM_MODEL)
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument(
        "--vlm-optional",
        action="store_true",
        help="If VLM connect fails, continue with GT template questions (non-fatal)",
    )
    p.add_argument(
        "--vlm-connect-retries",
        type=int,
        default=None,
        help="VLM connect attempts (default 8; --no-vlm forces 0)",
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
    p.add_argument("--no-cot", action="store_true", help="Deprecated no-op")
    p.add_argument("--no-pretended-cot", action="store_true", help="Deprecated no-op")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-mcq", action="store_true")
    p.add_argument("--flush-every", type=int, default=5)
    p.add_argument("--flip-mode", choices=("train", "bench", "off"), default="train")
    p.add_argument("--identical-frac", type=float, default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--n-houses", type=int, default=48)
    p.add_argument(
        "--house-indices",
        default="",
        help="Comma-separated override; else random sample of n-houses",
    )
    p.add_argument("--house-pool", type=int, default=10000, help="ProcTHOR train pool size")
    p.add_argument("--max-rooms", type=int, default=0, help="Legacy per-room mode only")
    p.add_argument(
        "--camera-mode",
        choices=("apartment", "per_room"),
        default="apartment",
        help="apartment=y cameras house-wide (v8); per_room=legacy v7",
    )
    p.add_argument(
        "--cameras-per-apartment",
        type=int,
        default=10,
        help="y: apartment-wide cameras / samples (default 10 → ideally 80 QAs)",
    )
    p.add_argument(
        "--room-diversity-bonus",
        type=float,
        default=1000.0,
        help="Score bonus for cameras in not-yet-covered rooms",
    )
    p.add_argument(
        "--camera-min-separation",
        type=float,
        default=1.0,
        help="Min camera-xyz distance (m) for in-room proposal + global select",
    )
    p.add_argument(
        "--proposals-per-room",
        type=int,
        default=3,
        help="Apartment mode: max views proposed per room before global select",
    )
    p.add_argument(
        "--max-positions-per-room",
        type=int,
        default=10,
        help="Apartment mode: FPS subsample size inside each room (default 10)",
    )
    p.add_argument(
        "--cameras-per-room",
        type=int,
        default=0,
        help="Legacy per_room mode: camera pool (0=auto)",
    )
    p.add_argument(
        "--samples-per-room",
        type=int,
        default=1,
        help="Legacy per_room mode: samples per room",
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Process the shared house list from the end (N-1, N-2, …)",
    )
    p.add_argument("--worker-tag", default="")
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    out_som_arg = args.out_som
    if out_som_arg is not None and str(out_som_arg).strip().lower() in ("", "-", "none"):
        out_som: Path | None = None
    else:
        out_som = Path(out_som_arg)
        out_som.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpus) != 1:
        raise SystemExit("Pass exactly one --gpus id per worker process")
    gpu = gpus[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cit_bin = os.path.dirname(sys.executable)
    path0 = os.environ.get("PATH", "")
    if cit_bin not in path0.split(":"):
        os.environ["PATH"] = f"{cit_bin}:{path0}"
    os.environ.setdefault("TMPDIR", os.environ.get("CIT_TMP_DIR", str(Path.home() / ".cache" / "cit" / "tmp")))
    os.environ.setdefault("TMP", os.environ["TMPDIR"])
    os.environ.setdefault("TEMP", os.environ["TMPDIR"])
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    worker_id = int(args.worker_id)
    num_workers = max(1, int(args.num_workers))
    if worker_id < 0 or worker_id >= num_workers:
        raise SystemExit(f"--worker-id must be in [0, {num_workers})")

    rng = random.Random(int(args.seed_base))
    if args.house_indices.strip():
        houses = [int(x) for x in args.house_indices.split(",") if x.strip()]
    else:
        houses = sorted(rng.sample(range(0, int(args.house_pool)), k=int(args.n_houses)))

    # Persist shared house list once so all workers see the same assignment.
    houses_path = out / "houses.json"
    if worker_id == 0 and not houses_path.exists():
        houses_path.write_text(
            json.dumps(
                {
                    "seed_base": args.seed_base,
                    "n_houses": len(houses),
                    "house_pool": int(args.house_pool),
                    "houses": houses,
                    "split": args.split,
                    "max_rooms": int(args.max_rooms or 0),
                    "camera_mode": str(args.camera_mode),
                    "cameras_per_apartment": int(args.cameras_per_apartment),
                    "room_diversity_bonus": float(args.room_diversity_bonus),
                    "camera_min_separation": float(args.camera_min_separation),
                    "proposals_per_room": int(args.proposals_per_room),
                    "max_positions_per_room": int(args.max_positions_per_room),
                    "cameras_per_room": int(args.cameras_per_room),
                    "samples_per_room": int(args.samples_per_room),
                    "som_offset_radii": float(SOM_MARK_OFFSET_RADII),
                    "categories": list(
                        c.strip()
                        for c in str(args.categories or "").split(",")
                        if c.strip()
                    )
                    or list(DEFAULT_CATEGORIES),
                    "dataset": "procthor_v10",
                    "contact_always_toward": bool(CONTACT_ALWAYS_TOWARD),
                    "contact_half_motion_no_frac": float(CONTACT_HALF_MOTION_NO_FRAC),
                },
                indent=2,
            )
            + "\n"
        )
    # Prefer the shared list whenever present (incl. worker 0 / reverse fleet).
    if not houses_path.exists() and worker_id != 0:
        for _ in range(60):
            if houses_path.exists():
                break
            time.sleep(1)
    if houses_path.exists():
        houses = list(json.loads(houses_path.read_text())["houses"])

    # Seeds stay keyed to the canonical ascending list; --reverse only flips
    # iteration order so a second fleet can start from the high end.
    house_pos = {h: i for i, h in enumerate(houses)}
    order = list(reversed(houses)) if args.reverse else list(houses)
    my_houses = [h for i, h in enumerate(order) if i % num_workers == worker_id]
    worker_tag = args.worker_tag or ("wr" if args.reverse else "w") + str(worker_id)
    use_vlm = not args.no_vlm
    vlm_optional = bool(args.vlm_optional) or bool(args.no_vlm)
    if args.vlm_connect_retries is not None:
        vlm_connect_retries = max(0, int(args.vlm_connect_retries))
    elif args.no_vlm:
        vlm_connect_retries = 0
    else:
        vlm_connect_retries = 8
    attach_cot = bool(args.with_cot) and (not args.no_cot)
    attach_pretended = (
        attach_cot
        and bool(args.with_pretended_cot)
        and (not args.no_pretended_cot)
        and (not args.no_vlm)
    )
    export_mcq = not args.no_mcq
    flip_mode = args.flip_mode
    identical_frac = args.identical_frac

    from cit_pipeline.dataset_export import (
        default_mcq_images_dir,
        export_sample_dict_dual,
        flip_quota_for_mode,
        open_mcq_writer,
    )
    from cit_pipeline.pipeline import build_procthor_house_samples, write_manifest

    categories = tuple(
        c.strip() for c in str(args.categories or "").split(",") if c.strip()
    ) or DEFAULT_CATEGORIES

    writer = None
    writer_som = None
    images_dir = default_mcq_images_dir(out)
    flip_quota = flip_quota_for_mode(flip_mode, identical_frac=identical_frac)
    flip_quota_som = flip_quota_for_mode(flip_mode, identical_frac=identical_frac)
    if export_mcq:
        writer = open_mcq_writer(out, flush_every=int(args.flush_every))
        images_dir.mkdir(parents=True, exist_ok=True)
        if out_som is not None:
            writer_som = open_mcq_writer(out_som, flush_every=int(args.flush_every))

    print(
        f"[{worker_tag}] start: {len(my_houses)}/{len(houses)} houses "
        f"gpu={gpu} vlm={args.vlm_base_url} use_vlm={use_vlm} "
        f"vlm_optional={vlm_optional} retries={vlm_connect_retries} "
        f"mod={MODIFICATION} "
        f"mode={args.camera_mode} y={args.cameras_per_apartment} "
        f"room_bonus={args.room_diversity_bonus} "
        f"min_sep={args.camera_min_separation} "
        f"proposals_per_room={args.proposals_per_room} "
        f"som_offset={SOM_MARK_OFFSET_RADII} "
        f"cats={categories} half={CONTACT_HALF_MOTION_NO_FRAC} "
        f"toward={CONTACT_ALWAYS_TOWARD} "
        f"room_prefilter={CONTACT_ROOM_WALL_PREFILTER} "
        f"feas_first={CONTACT_FEAS_FIRST} single_cam={CONTACT_FAST_SINGLE_CAMERA} "
        f"wall_max={CONTACT_ROOM_WALL_MAX_M} max_cand={CONTACT_FAST_MAX_CANDIDATES}",
        flush=True,
    )
    results: list[dict] = []
    n_mcq = 0
    n_mcq_som = 0
    t_shard0 = time.perf_counter()

    try:
        for seed_idx, house_index in enumerate(my_houses):
            # Global seed index aligned to canonical (pre-reverse) house list.
            global_idx = house_pos[house_index]
            if args.skip_existing and _house_already_done(out, house_index, args.split):
                print(f"[{worker_tag}] SKIP house {house_index} (exists)", flush=True)
                results.append(
                    {
                        "ok": True,
                        "house_index": house_index,
                        "skipped": True,
                        "gpu": gpu,
                        "worker": worker_tag,
                    }
                )
                continue

            t0 = time.perf_counter()
            print(
                f"[{worker_tag}] START house {house_index} "
                f"seed={args.seed_base + global_idx}",
                flush=True,
            )
            try:
                cfg = PipelineConfig(
                    dataset="procthor",
                    house_index=int(house_index),
                    house_split=str(args.split),
                    out_dir=out,
                    out_dir_som=out_som,
                    seed=int(args.seed_base) + global_idx,
                    n_views=1,
                    questions_per_view=1,
                    category_driven=True,
                    categories=categories,
                    use_vlm_ranking=use_vlm,
                    vlm_optional=vlm_optional,
                    vlm_connect_retries=vlm_connect_retries,
                    vlm_base_url=str(args.vlm_base_url),
                    vlm_model=str(args.vlm_model),
                    attach_cot=attach_cot,
                    attach_pretended_cot=attach_pretended and attach_cot,
                    export_mcq=export_mcq,
                    enable_som=True,
                    answer_flip_mode=flip_mode,
                    answer_flip_identical_frac=(
                        float(identical_frac) if identical_frac is not None else None
                    ),
                    modification=MODIFICATION,
                    max_rooms=int(args.max_rooms or 0),
                    procthor_camera_mode=str(args.camera_mode),
                    procthor_cameras_per_apartment=max(
                        1, int(args.cameras_per_apartment)
                    ),
                    procthor_room_diversity_bonus=float(args.room_diversity_bonus),
                    procthor_camera_min_separation=float(args.camera_min_separation),
                    procthor_proposals_per_room=max(2, int(args.proposals_per_room)),
                    procthor_max_positions_per_room=max(
                        8, int(args.max_positions_per_room)
                    ),
                    procthor_cameras_per_room=int(args.cameras_per_room),
                    procthor_samples_per_room=max(1, int(args.samples_per_room)),
                    fast_find_edit=True,
                    snapshot_restore=True,
                    contact_target_no_frac=float(ITHOR_CONTACT_TARGET_NO_FRAC),
                    feasibility_target_no_frac=float(ITHOR_FEASIBILITY_TARGET_NO_FRAC),
                    wall_away_prob=float(ITHOR_WALL_AWAY_PROB),
                    fast_wall_select_near_m=float(ITHOR_FAST_WALL_SELECT_NEAR_M),
                    fast_contact_max_tries=int(ITHOR_FAST_CONTACT_MAX_TRIES),
                    fast_feasibility_max_attempts=int(
                        ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS
                    ),
                    contact_always_toward=bool(CONTACT_ALWAYS_TOWARD),
                    contact_half_motion_no_frac=float(CONTACT_HALF_MOTION_NO_FRAC),
                    contact_room_wall_prefilter=bool(CONTACT_ROOM_WALL_PREFILTER),
                    contact_room_wall_max_m=float(CONTACT_ROOM_WALL_MAX_M),
                    contact_fast_max_candidates=int(CONTACT_FAST_MAX_CANDIDATES),
                    contact_fast_single_camera=bool(CONTACT_FAST_SINGLE_CAMERA),
                    contact_feas_first=bool(CONTACT_FEAS_FIRST),
                )
                samples = build_procthor_house_samples(cfg)
                house_mcq = 0
                house_mcq_som = 0
                sample_ids = []
                for sample in samples:
                    write_manifest(out, sample)
                    sample_dir = out / sample.sample_id
                    sample_path = sample_dir / "sample.json"
                    data = json.loads(sample_path.read_text())
                    sample_ids.append(sample.sample_id)

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
                                f"incomplete warm CoT ({n_warm}/{n_complete})"
                            )
                        if attach_pretended and n_complete and n_pret < n_complete:
                            raise RuntimeError(
                                f"incomplete pretended CoT ({n_pret}/{n_complete})"
                            )

                    if writer is not None:
                        recs, recs_som = export_sample_dict_dual(
                            data,
                            sample_dir=sample_dir,
                            out_dir=out,
                            out_dir_som=out_som,
                            writer=writer,
                            writer_som=writer_som,
                            seed=int(args.seed_base) + global_idx,
                            flip_mode=flip_mode,
                            identical_frac=(
                                float(identical_frac)
                                if identical_frac is not None
                                else None
                            ),
                            flip_quota=flip_quota,
                            flip_quota_som=flip_quota_som,
                        )
                        house_mcq += len(recs)
                        house_mcq_som += len(recs_som)

                n_mcq += house_mcq
                n_mcq_som += house_mcq_som
                elapsed = time.perf_counter() - t0
                row = {
                    "ok": True,
                    "house_index": house_index,
                    "n_samples": len(samples),
                    "sample_ids": sample_ids,
                    "n_mcq": house_mcq,
                    "n_mcq_som": house_mcq_som,
                    "elapsed_s": elapsed,
                    "gpu": gpu,
                    "worker": worker_tag,
                }
                results.append(row)
                print(
                    f"[{worker_tag}] OK house {house_index}: "
                    f"rooms={len(samples)} mcq={house_mcq} mcq_som={house_mcq_som} "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                elapsed = time.perf_counter() - t0
                print(
                    f"[{worker_tag}] FAIL house {house_index} ({elapsed:.1f}s): {exc}",
                    flush=True,
                )
                results.append(
                    {
                        "ok": False,
                        "house_index": house_index,
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
            print(f"[{worker_tag}] final jsonl flush: {flushed}", flush=True)
        if flushed_som:
            print(f"[{worker_tag}] final SoM jsonl flush: {flushed_som}", flush=True)

    ok = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    bad = sum(1 for r in results if not r.get("ok"))
    summary = {
        "worker": worker_tag,
        "worker_id": worker_id,
        "gpu": gpu,
        "n_ok": ok,
        "n_fail": bad,
        "n_skipped": skipped,
        "n_mcq": n_mcq,
        "n_mcq_som": n_mcq_som,
        "elapsed_s": time.perf_counter() - t_shard0,
        "houses": my_houses,
        "per_house": results,
        "end_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "logs").mkdir(parents=True, exist_ok=True)
    summary_name = (
        f"rev_worker{worker_id}_summary.json"
        if args.reverse
        else f"worker{worker_id}_summary.json"
    )
    (out / "logs" / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[{worker_tag}] DONE ok={ok} fail={bad} skip={skipped} "
        f"mcq={n_mcq} wall={_fmt_hms(summary['elapsed_s'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
