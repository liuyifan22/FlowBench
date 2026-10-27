#!/usr/bin/env python3
"""Backfill **warm** CoT for procthor_v9 (no pretended CoT).

Mirrors ``backfill_procthor_v7_cot.py`` but:

- Only processes questions whose ``id`` is in the train/val/test splits
  (``procthor_v9_som_balanced_splits``), i.e. not incomplete / discarded rows.
- Generates cold template + **warm** VLM polish only (``polish_pretended=False``).
- Resumable: skips questions that already have a non-empty ``*_warm.txt``.

Default VLM: Qwen3-VL-32B-Instruct at ``localhost:19857``.

Example:
  python backfill_procthor_v9_cot.py --workers 48
  python backfill_procthor_v9_cot.py --limit 64 --workers 8

  # Split remaining work across two VLM servers (shard by sample_id):
  python backfill_procthor_v9_cot.py --vlm-url http://localhost:19857/v1 \\
    --shard 0 --num-shards 2 --workers 48 --progress .../cot/_progress_s0.jsonl
  python backfill_procthor_v9_cot.py --vlm-url http://localhost:19857/v1 \\
    --shard 1 --num-shards 2 --workers 48 --progress .../cot/_progress_s1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_NATURAL = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v9/natural"
)
DEFAULT_SPLITS = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/"
    "qwen_training/procthor_v9_som_balanced_splits"
)
DEFAULT_VLM_URL = "http://localhost:19857/v1"


def _txt_paths(cot_dir: Path, rid: str) -> dict[str, Path]:
    return {
        "cold": cot_dir / f"{rid}_cold.txt",
        "warm": cot_dir / f"{rid}_warm.txt",
    }


def _write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _warm_done(paths: dict[str, Path]) -> bool:
    p = paths["warm"]
    return p.is_file() and p.stat().st_size > 0


def _load_keep_ids(splits_dir: Path) -> dict[str, str]:
    """Map record id → split name (train/val/test)."""
    keep: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing split jsonl: {path}")
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("id") or "")
                if rid:
                    keep[rid] = name
    return keep


def _sample_shard(sample_id: str, num_shards: int) -> int:
    """Stable shard id in [0, num_shards) from sample_id (keeps one sample together)."""
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_shards


def _iter_jobs(
    natural: Path,
    cot_dir: Path,
    keep_ids: Optional[dict[str, str]],
    *,
    force: bool,
    limit: Optional[int],
    shard: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    jsonl = natural / "dataset.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"Missing {jsonl}")
    jobs: list[dict[str, Any]] = []
    n_skip_done = 0
    n_skip_not_in_splits = 0
    n_skip_incomplete = 0
    n_skip_shard = 0
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or "")
            if keep_ids is not None and rid not in keep_ids:
                n_skip_not_in_splits += 1
                continue
            if rec.get("gt_complete") is False:
                n_skip_incomplete += 1
                continue
            sample_id = str(rec.get("sample_id") or "")
            qid = str(rec.get("question_id") or "")
            if not rid or not sample_id or not qid:
                continue
            if num_shards > 1 and _sample_shard(sample_id, num_shards) != shard:
                n_skip_shard += 1
                continue
            paths = _txt_paths(cot_dir, rid)
            if not force and _warm_done(paths):
                n_skip_done += 1
                continue
            jobs.append(
                {
                    "sample_dir": natural / sample_id,
                    "sample_id": sample_id,
                    "question_id": qid,
                    "rec_id": rid,
                    "category": rec.get("category"),
                    "split": keep_ids[rid] if keep_ids is not None else "all",
                }
            )
            if limit is not None and len(jobs) >= limit:
                break
    print(
        f"jsonl scan: todo={len(jobs)} skip_done={n_skip_done} "
        f"skip_shard={n_skip_shard} "
        f"skip_not_in_splits={n_skip_not_in_splits} "
        f"skip_incomplete_in_splits={n_skip_incomplete} "
        f"(shard={shard}/{num_shards})",
        flush=True,
    )
    return jobs


def _load_job_context(job: dict[str, Any]) -> dict[str, Any]:
    sample_dir: Path = job["sample_dir"]
    sp = sample_dir / "sample.json"
    if not sp.is_file():
        raise FileNotFoundError(f"missing sample.json: {sp}")
    sample = json.loads(sp.read_text())
    qid = job["question_id"]
    for view in sample.get("views") or []:
        for q in view.get("questions") or []:
            if str(q.get("question_id") or "") != qid:
                continue
            view_id = str(view.get("view_id") or "view")
            move = view.get("move") or {}
            rgb0 = sample_dir / f"{view_id}_rgb0.png"
            rgb1 = sample_dir / f"{view_id}_rgb1.png"
            if not rgb0.is_file() and view.get("rgb0_path"):
                rgb0 = Path(view["rgb0_path"])
            if not rgb1.is_file() and view.get("rgb1_path"):
                rgb1 = Path(view["rgb1_path"])
            out = dict(job)
            out.update(
                {
                    "view_id": view_id,
                    "q": q,
                    "move": move,
                    "rgb0": rgb0,
                    "rgb1": rgb1,
                }
            )
            return out
    raise KeyError(f"question_id {qid} not in {sp}")


def _process_one(
    job: dict[str, Any],
    vlm: Any,
    cot_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    from cit_pipeline.cot import attach_cot_to_question

    t0 = time.time()
    paths = _txt_paths(cot_dir, job["rec_id"])
    if not force and _warm_done(paths):
        return {
            "rec_id": job["rec_id"],
            "sample_id": job["sample_id"],
            "sample_dir": str(job["sample_dir"]),
            "question_id": job["question_id"],
            "view_id": None,
            "split": job.get("split"),
            "ok_warm": True,
            "warm_error": None,
            "cot": {},
            "elapsed_s": round(time.time() - t0, 2),
            "skipped_done": True,
        }
    job = _load_job_context(job)
    q = job["q"]
    updated = attach_cot_to_question(
        q,
        job["move"],
        rgb0=job["rgb0"],
        rgb1=job["rgb1"],
        vlm=vlm,
        polish=True,
        polish_pretended=False,
    )
    cot = updated.get("cot") or {}
    paths = _txt_paths(cot_dir, job["rec_id"])
    cold = cot.get("cold") or ""
    warm = cot.get("warm") or ""
    if cold:
        _write_txt(paths["cold"], str(cold))
    ok_warm = bool(warm) and not cot.get("warm_error")
    if ok_warm:
        _write_txt(paths["warm"], str(warm))
    return {
        "rec_id": job["rec_id"],
        "sample_id": job["sample_id"],
        "sample_dir": str(job["sample_dir"]),
        "question_id": job["question_id"],
        "view_id": job["view_id"],
        "split": job.get("split"),
        "ok_warm": ok_warm,
        "warm_error": cot.get("warm_error"),
        "cot": cot,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _patch_sample_json(
    sample_dir: Path,
    question_id: str,
    cot: dict[str, Any],
    lock: threading.Lock,
) -> None:
    with lock:
        sp = sample_dir / "sample.json"
        sample = json.loads(sp.read_text())
        found = False
        for view in sample.get("views") or []:
            for i, q in enumerate(view.get("questions") or []):
                if str(q.get("question_id") or "") != question_id:
                    continue
                q = dict(q)
                existing = dict(q.get("cot") or {})
                # Keep any existing pretended; overwrite cold/warm.
                existing.update(cot)
                q["cot"] = existing
                view["questions"][i] = q
                found = True
                break
            if found:
                break
        if not found:
            return
        meta = dict(sample.get("meta") or {})
        tag = "mod7_cot_cold_warm"
        prev = str(meta.get("modification") or "").strip()
        if prev and tag not in prev:
            meta["modification"] = f"{prev}+{tag}"
        elif not prev:
            meta["modification"] = tag
        cot_meta = dict(meta.get("cot") or {})
        cot_meta["cold"] = True
        cot_meta["warm"] = True
        meta["cot"] = cot_meta
        sample["meta"] = meta
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        os.replace(tmp, sp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--natural", type=Path, default=DEFAULT_NATURAL)
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument(
        "--no-splits",
        action="store_true",
        help="Process every gt_complete row (skip train/val/test keep-list).",
    )
    p.add_argument(
        "--cot-dir",
        type=Path,
        default=None,
        help="Output txt dir (default: <natural>/cot)",
    )
    p.add_argument("--vlm-url", default=DEFAULT_VLM_URL)
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-patch-sample", action="store_true")
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard index in [0, num-shards). Jobs partitioned by sample_id.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shards for parallel VLM servers (default: 1 = no split).",
    )
    args = p.parse_args()

    natural: Path = args.natural
    cot_dir: Path = args.cot_dir or (natural / "cot")
    cot_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress or (cot_dir / "_progress.jsonl")
    shard = int(args.shard)
    num_shards = int(args.num_shards)
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if shard < 0 or shard >= num_shards:
        raise SystemExit(f"--shard must be in [0, {num_shards})")

    from cit_pipeline.config import VLM_MODEL
    from cit_pipeline.vlm_client import VlmClient

    model = args.model or VLM_MODEL
    print(
        f"natural={natural}\nsplits={'NONE (--no-splits)' if args.no_splits else args.splits_dir}\n"
        f"cot_dir={cot_dir}\n"
        f"vlm={model} @ {args.vlm_url}\nworkers={args.workers} (warm only)\n"
        f"shard={shard}/{num_shards} progress={progress_path}",
        flush=True,
    )

    if args.no_splits:
        keep_ids = None
        print("split ids: all gt_complete rows (no keep-list)", flush=True)
    else:
        keep_ids = _load_keep_ids(args.splits_dir)
        print(
            f"split ids: {len(keep_ids)} "
            f"(train+val+test unique)",
            flush=True,
        )
    jobs = _iter_jobs(
        natural,
        cot_dir,
        keep_ids,
        force=args.force,
        limit=args.limit,
        shard=shard,
        num_shards=num_shards,
    )
    print(f"questions to process: {len(jobs)}", flush=True)
    if not jobs:
        print("Nothing to do.", flush=True)
        return

    thread_local = threading.local()
    sample_locks: dict[str, threading.Lock] = {}
    sample_locks_guard = threading.Lock()

    def get_vlm() -> Any:
        v = getattr(thread_local, "vlm", None)
        if v is None:
            v = VlmClient(
                base_url=args.vlm_url,
                model=model,
                enable_thinking=False,
            )
            thread_local.vlm = v
        return v

    def get_sample_lock(sample_dir: str) -> threading.Lock:
        with sample_locks_guard:
            lock = sample_locks.get(sample_dir)
            if lock is None:
                lock = threading.Lock()
                sample_locks[sample_dir] = lock
            return lock

    n_ok = 0
    n_fail = 0
    n_done = 0
    t_start = time.time()
    progress_lock = threading.Lock()
    elapsed_ok: list[float] = []

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        return _process_one(job, get_vlm(), cot_dir, force=args.force)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {ex.submit(run_job, job): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            n_done += 1
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                result = {
                    "rec_id": job["rec_id"],
                    "ok_warm": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[fail] {job['rec_id']}: {exc}", flush=True)
            else:
                if result["ok_warm"]:
                    n_ok += 1
                    if result.get("elapsed_s") is not None:
                        elapsed_ok.append(float(result["elapsed_s"]))
                    if (
                        not args.no_patch_sample
                        and not result.get("skipped_done")
                        and result.get("cot")
                    ):
                        _patch_sample_json(
                            Path(result["sample_dir"]),
                            result["question_id"],
                            result["cot"],
                            get_sample_lock(result["sample_dir"]),
                        )
                else:
                    n_fail += 1
                    print(
                        f"[partial] {result['rec_id']} "
                        f"warm={result['ok_warm']} werr={result.get('warm_error')}",
                        flush=True,
                    )

            row = {
                k: result.get(k)
                for k in (
                    "rec_id",
                    "ok_warm",
                    "warm_error",
                    "elapsed_s",
                    "split",
                    "error",
                )
                if k in result or result.get(k) is not None
            }
            row["t"] = time.time()
            with progress_lock:
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            if n_done % 20 == 0 or n_done == len(jobs):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - n_done) / rate if rate > 0 else float("inf")
                mean_lat = (
                    sum(elapsed_ok[-200:]) / len(elapsed_ok[-200:])
                    if elapsed_ok
                    else 0.0
                )
                print(
                    f"[{n_done}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.2f} q/s mean_lat={mean_lat:.1f}s "
                    f"eta={eta/3600:.1f}h",
                    flush=True,
                )

    wall = time.time() - t_start
    rate = n_done / wall if wall > 0 else 0.0
    print("=" * 60, flush=True)
    print(
        f"Done: ok={n_ok} fail={n_fail} total={len(jobs)} "
        f"wall={wall:.0f}s rate={rate:.2f} q/s → {cot_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
