#!/usr/bin/env python3
"""Rewrite SoM ``static_question`` so it carries mark ids from ``question``.

procthor_v10/som often has SoM marks in ``question`` (e.g. ``chair (1)``) but
``static_question`` drops them or replaces them with ordinals
(``rightmost chair``, ``2nd book from the left``). This script calls a
text-only Qwen3-VL endpoint to inject the marks back into ``static_question``.

Writes a full jsonl (all rows) with updated ``static_question``. When a rewrite
happens, the previous text is kept as ``static_question_orig``.

Example (one shard against one server):
  python fix_static_question_som.py \\
    --input $CIT_DATA_ROOT/procthor_v10/som/dataset.jsonl \\
    --output $CIT_DATA_ROOT/procthor_v10/som/dataset_static_som.jsonl \\
    --vlm-url http://localhost:19857/v1 \\
    --num-shards 64 --shard-id 0 --workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

MARK_RE = re.compile(r"\((\d+)\)")

SYSTEM = """You rewrite short indoor-scene questions so object mentions use Set-of-Marks (SoM) ids.

Rules:
1. Read QUESTION (already has SoM marks like "chair (1)", "dresser (2)").
2. Read STATIC_QUESTION (often missing marks, or uses ordinals like "rightmost chair", "2nd book from the left", "the peppershaker").
3. Output ONE rewritten static question that keeps STATIC_QUESTION's concise style, but replaces object references with the matching SoM forms from QUESTION (e.g. "the chair (1)", "the book (1)").
4. Every mark id that appears on an object in QUESTION must appear on that same object in the output.
5. Do NOT invent new marks. Do NOT add long preambles like "Starting from the after image".
6. Preserve distances, directions, yes/no intent, and punctuation style of STATIC_QUESTION when possible.
7. Output ONLY the rewritten static question text — no quotes, no explanation."""


def _needs_fix(question: str, static_question: str) -> bool:
    q_marks = set(MARK_RE.findall(question or ""))
    if not q_marks:
        return False
    sq_marks = set(MARK_RE.findall(static_question or ""))
    return not q_marks.issubset(sq_marks)


def _clean_output(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    # Drop wrapping quotes.
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    # First non-empty line if the model babbling.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer a line that looks like a question.
    for ln in lines:
        if "?" in ln or ln.lower().startswith(("would", "is ", "are ", "which", "how", "after")):
            return ln
    return lines[0]


def _rewrite(
    client: OpenAI,
    model: str,
    question: str,
    static_question: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> str:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"STATIC_QUESTION:\n{static_question}\n\n"
        "Rewritten STATIC_QUESTION:"
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}
        }
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning and not text:
        text = str(reasoning).strip()
    return _clean_output(text)


def _valid_rewrite(question: str, original_sq: str, new_sq: str) -> bool:
    if not new_sq or len(new_sq) < 8:
        return False
    q_marks = set(MARK_RE.findall(question))
    new_marks = set(MARK_RE.findall(new_sq))
    if not q_marks.issubset(new_marks):
        return False
    # Reject obvious dumps of the full question preamble.
    if new_sq.lower().startswith("starting from the after image"):
        return False
    return True


def _process_row(
    rec: dict[str, Any],
    client: OpenAI,
    model: str,
    *,
    force: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    """Return (updated_rec, status) where status in {skip, ok, fail, dry}."""
    q = str(rec.get("question") or "")
    sq = str(rec.get("static_question") or "")
    if not force and not _needs_fix(q, sq):
        return rec, "skip"
    if dry_run:
        out = dict(rec)
        out["static_question_would_fix"] = True
        return out, "dry"
    try:
        new_sq = _rewrite(client, model, q, sq)
        if not _valid_rewrite(q, sq, new_sq):
            # one retry with a stricter reminder
            new_sq = _rewrite(
                client,
                model,
                q,
                sq + "\n\n(Reminder: you MUST include every SoM mark from QUESTION.)",
            )
        if not _valid_rewrite(q, sq, new_sq):
            return rec, "fail"
        out = dict(rec)
        out["static_question_orig"] = sq
        out["static_question"] = new_sq
        return out, "ok"
    except Exception:
        return rec, "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vlm-url", type=str, required=True, help="OpenAI-compatible base URL (.../v1)")
    ap.add_argument("--model", type=str, default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="Thread workers inside this process")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to rewrite in this shard (0=all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--examples-file", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"bad shard-id {args.shard_id} for num-shards {args.num_shards}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.vlm_url, api_key="EMPTY", timeout=120.0)

    # Stream input; keep only this shard's line indices for rewrite jobs,
    # but write ALL rows belonging to this shard to the shard output.
    rows: list[tuple[int, dict[str, Any]]] = []
    with args.input.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if (i % args.num_shards) != args.shard_id:
                continue
            rows.append((i, json.loads(line)))

    todo_idx = [
        j
        for j, (_i, rec) in enumerate(rows)
        if args.force
        or _needs_fix(str(rec.get("question") or ""), str(rec.get("static_question") or ""))
    ]
    if args.limit and args.limit > 0:
        todo_idx = todo_idx[: args.limit]

    print(
        f"[shard {args.shard_id}/{args.num_shards}] rows={len(rows)} "
        f"todo={len(todo_idx)} url={args.vlm_url} workers={args.workers}",
        flush=True,
    )

    results: dict[int, tuple[dict[str, Any], str]] = {}
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0, "skip": 0, "dry": 0}
    examples: list[dict[str, str]] = []
    t0 = time.time()

    def _job(j: int) -> None:
        _i, rec = rows[j]
        out, status = _process_row(
            rec, client, args.model, force=args.force, dry_run=args.dry_run
        )
        with lock:
            results[j] = (out, status)
            counts[status] = counts.get(status, 0) + 1
            if status == "ok" and len(examples) < args.max_examples:
                examples.append(
                    {
                        "id": str(out.get("id") or ""),
                        "question": str(out.get("question") or ""),
                        "static_question_orig": str(out.get("static_question_orig") or ""),
                        "static_question": str(out.get("static_question") or ""),
                    }
                )
            done = counts["ok"] + counts["fail"] + counts["dry"]
            if done and done % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[shard {args.shard_id}] done={done}/{len(todo_idx)} "
                    f"ok={counts['ok']} fail={counts['fail']} "
                    f"{rate:.2f} r/s elapsed={elapsed:.0f}s",
                    flush=True,
                )

    # Rows not in todo are skips written as-is.
    for j, (_i, rec) in enumerate(rows):
        if j not in set(todo_idx):
            results[j] = (rec, "skip")
            counts["skip"] += 1

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_job, j) for j in todo_idx]
        for fut in as_completed(futs):
            exc = fut.exception()
            if exc is not None:
                print(f"[shard {args.shard_id}] worker error: {exc}", flush=True)
                traceback.print_exc()

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w") as out_f:
        for j in range(len(rows)):
            rec, _status = results[j]
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)

    if args.examples_file is not None:
        args.examples_file.parent.mkdir(parents=True, exist_ok=True)
        args.examples_file.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    elapsed = time.time() - t0
    rewritten = counts["ok"] + counts["dry"]
    rate = rewritten / max(elapsed, 1e-6)
    print(
        f"[shard {args.shard_id}] FINISH wrote={args.output} "
        f"ok={counts['ok']} fail={counts['fail']} skip={counts['skip']} "
        f"dry={counts['dry']} elapsed={elapsed:.1f}s rate={rate:.2f} rewrite/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
