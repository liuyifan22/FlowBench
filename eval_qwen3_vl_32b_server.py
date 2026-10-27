#!/usr/bin/env python3
"""Evaluate Qwen3-VL-32B on CIT iTHOR MCQ via the shared vLLM OpenAI server.

Same 2×2 protocol as ``eval_qwen3_vl_8b.py``, but inference hits a remote
server (default: ``localhost:19857``) instead of loading weights locally. Several
processes can call the same server in parallel (one process per eval cell).

Axes
----
``--image-setting {before,after|1,2}``
  1 / before : Question + options + motion cue + **rgb0**
  2 / after  : ``static_question`` + options + **rgb1**

``--answer-mode {direct,cot|A,B}``
  A / direct : short letter (max 16 tokens)
  B / cot    : reason, then ``<answer>A</answer>`` (max 512 tokens;
               Thinking: larger budget, adapted prompt, sampling)

``--thinking``
  For Qwen3-*-Thinking servers: do not ask "step by step"; strip ``<think>``;
  default temp=0.6 / top_p=0.95. Auto-enabled if ``--model`` contains
  ``Thinking``.

Examples
--------
Sanity cell 1A (6 items)::

  python eval_qwen3_vl_32b_server.py \\
    --image-setting before --answer-mode direct --limit 6 \\
    --base-url http://localhost:19857/v1

Thinking 1B/2B (server must be Qwen3-VL-32B-Thinking)::

  bash run_eval_thinking_1b2b_qwen32b.sh

All four cells in parallel (shared server)::

  bash run_eval_sanity_4way_qwen32b.sh
"""

from __future__ import annotations

import os
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cit_pipeline.benchmark_eval import (
    build_force_answer_continuation,
    build_user_prompt,
    extract_pred_answer,
    force_answer_at_tokens,
    format_category_stats_table,
    has_complete_answer_tag,
    max_new_tokens_for_mode,
    normalize_gt_answer,
    resolve_image_path,
    setting_tag,
)
from cit_pipeline.benchmark_eval import _normalize_answer_mode as _normalize_answer_mode_local
from cit_pipeline.vlm_client import VlmClient

DEFAULT_DATASET = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11/som"
)
DEFAULT_BASE_URL = "http://localhost:19857/v1"
DEFAULT_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_OUT_ROOT = _ROOT / "outputs" / "evals"


def _load_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _wait_for_server(base_url: str, *, timeout_s: float = 600.0) -> None:
    """Poll ``/health`` (or ``/models``) until the OpenAI-compatible server is up."""
    import urllib.error
    import urllib.request

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        health = root[: -len("/v1")] + "/health"
        models = root + "/models"
    else:
        health = root + "/health"
        models = root + "/models"

    t0 = time.time()
    while True:
        for url in (health, models):
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= int(resp.status) < 300:
                        print(f"[server] ready: {url}", flush=True)
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"VLM server not ready after {timeout_s}s ({base_url}): {last}"
            )
        print(
            f"[server] waiting for {base_url} ({time.time() - t0:.0f}s)…",
            flush=True,
        )
        time.sleep(5)


class ServerEngine:
    """Thin wrapper: multimodal chat.completions against the shared vLLM."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
        image_first: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        self.image_first = bool(image_first)
        self.client = VlmClient(base_url=base_url, model=model, timeout=timeout,
                                enable_thinking=enable_thinking)
        self.model = model
        self.base_url = base_url
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_s = float(retry_delay_s)

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        force_answer_at: Optional[int] = None,
    ) -> str:
        """Generate; optionally force ``<answer>`` near a thinking budget.

        When ``force_answer_at`` is set and phase-1 output lacks a complete
        ``<answer>…</answer>``, continue from
        ``partial + FORCE_ANSWER_NUDGE`` with the remaining token budget.
        """
        phase1_budget = int(max_new_tokens)
        if force_answer_at is not None:
            phase1_budget = max(1, min(int(force_answer_at), int(max_new_tokens)))

        partial, _meta = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=phase1_budget,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=None,
        )
        if force_answer_at is None or has_complete_answer_tag(partial):
            return partial

        continuation = build_force_answer_continuation(partial)
        remaining = max(32, int(max_new_tokens) - phase1_budget)
        suffix, _meta2 = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=remaining,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=continuation,
        )
        # Continuation API returns only the newly generated suffix.
        if suffix.startswith(continuation):
            return suffix.strip()
        return (continuation + suffix).strip()

    def _generate_once(
        self,
        *,
        prompt: str,
        image_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        assistant_prefix: Optional[str],
        followup: Optional[tuple[str, str]] = None,
    ) -> tuple[str, dict[str, Any]]:
        from cit_pipeline.vlm_client import _encode_image

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image(image_path)},
                    },
                ]
                if self.image_first:  # same order as training / eval_qwen3_vl_8b.py
                    content.reverse()
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": content},
                ]
                continue_final = False
                if followup is not None:  # (assistant reply, next user message)
                    messages += [{"role": "assistant", "content": followup[0]},
                                 {"role": "user", "content": followup[1]}]
                if assistant_prefix:
                    messages.append(
                        {"role": "assistant", "content": assistant_prefix}
                    )
                    continue_final = True
                text, meta = self.client.chat_messages(
                    messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    continue_final_message=continue_final,
                    return_meta=True,
                )
                return text, meta
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(
                    f"[server] generate attempt {attempt}/{self.max_retries} "
                    f"failed ({exc})",
                    flush=True,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * attempt)
        raise RuntimeError(f"VLM generate failed after retries: {last_exc}")


_STRICT_LETTERS_RE = re.compile(r"^\s*[\(\[]?([A-F](?:\s*[,/&]?\s*(?:and\s+)?[A-F]){0,2})[\)\]]?(?:[.:)]|\s|$)")
# Follow-up user turn when a direct reply is not a clean letter (continuing a partial assistant
# message via continue_final_message is unreliable across chat templates).
DIRECT_RESCUE_FOLLOWUP = "Please answer with only the option letter(s), e.g. B or ACF."


# Answer-box markers some models wrap their final answer in (GLM-4.xV: <|begin_of_box|>A<|end_of_box|>).
_ANSWER_BOX_RE = re.compile(r"<\|(?:begin|end)_of_box\|>")


def strict_letters(text: str) -> Optional[str]:
    """Upper-case option letters at the very start of ``text`` ("B", "(B)", "BDF", "B, D, F")."""
    m = _STRICT_LETTERS_RE.match(_ANSWER_BOX_RE.sub("", text or ""))
    return normalize_gt_answer("".join(re.findall(r"[A-F]", m.group(1)))) if m else None


def _resolve_som_flag(args: argparse.Namespace, rec: dict[str, Any]) -> bool:
    """CLI --som / --no-som override; else follow record['som'] or dataset path."""
    mode = str(getattr(args, "som_mode", "auto") or "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    if "som" in rec:
        return bool(rec.get("som"))
    ds = str(getattr(args, "dataset", "") or "")
    return "/som" in ds or ds.rstrip("/").endswith("som")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    thinking = bool(args.thinking)
    temperature = float(args.temperature)
    top_p = float(args.top_p)
    if thinking and args.temperature == 0.0 and not args.force_greedy:
        temperature = 0.6
        if abs(top_p - 1.0) < 1e-9:
            top_p = 0.95

    dataset_root = Path(args.dataset)
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / "dataset.jsonl"
    tag = setting_tag(args.image_setting, args.answer_mode)
    default_max_tokens = max_new_tokens_for_mode(
        args.answer_mode, thinking=thinking
    )
    out_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUT_ROOT
        / f"qwen3vl32b_server_{'thinking_' if thinking else ''}{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_server_wait:
        _wait_for_server(args.base_url, timeout_s=float(args.server_wait_timeout))

    if not jsonl_path.is_file():
        raise SystemExit(f"Missing dataset jsonl: {jsonl_path}")

    records = _load_jsonl(jsonl_path, limit=int(args.limit or 0))
    from cit_pipeline.eval_formats import build_formal_prompt, permute_for_seed
    records = permute_for_seed(records, int(args.choice_seed))
    if not records:
        raise SystemExit(f"No records in {jsonl_path}")

    # Thinking completions are long; bump HTTP timeout unless user set higher.
    timeout = float(args.timeout)
    if thinking and timeout <= 300.0:
        timeout = 600.0

    print(
        f"[eval] cell={tag} image_setting={args.image_setting} "
        f"answer_mode={args.answer_mode} thinking={thinking} "
        f"som_mode={getattr(args, 'som_mode', 'auto')} n={len(records)} "
        f"max_tokens_default={default_max_tokens} "
        f"temp={temperature} top_p={top_p} model={args.model} "
        f"base_url={args.base_url}",
        flush=True,
    )

    engine = ServerEngine(
        base_url=args.base_url,
        model=args.model,
        timeout=timeout,
        max_retries=int(args.max_retries),
        retry_delay_s=float(args.retry_delay),
        image_first=bool(args.image_first),
        enable_thinking={"on": True, "off": False}.get(args.enable_thinking),
    )
    # Cheap ping (text-only) so we fail fast if the model name is wrong.
    try:
        ping = engine.client.chat_text("Reply with OK.", max_tokens=4)
        print(f"[server] ping OK: {ping!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"VLM ping failed ({args.base_url}): {exc}") from exc

    pred_path = out_dir / "predictions.jsonl"
    results: list[dict[str, Any]] = []
    n_correct = 0
    n_parse_fail = 0
    # per category: [correct, total, parse_fail]
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    t0 = time.perf_counter()
    with pred_path.open("w") as fout:
        for i, rec in enumerate(records):
            cat = str(rec.get("category") or "unknown")
            max_tokens = max_new_tokens_for_mode(
                args.answer_mode, category=cat, thinking=thinking
            )
            force_at = force_answer_at_tokens(thinking=thinking, category=cat)
            use_som = _resolve_som_flag(args, rec)
            if args.prompt_format == "formal":
                prompt = build_formal_prompt(rec, image_setting=args.image_setting,
                                             direct_hint=bool(args.formal_direct_hint))
            else:
                prompt = build_user_prompt(
                    rec,
                    image_setting=args.image_setting,
                    answer_mode=args.answer_mode,
                    thinking=thinking,
                    som=use_som,
                )
            img = resolve_image_path(
                rec, dataset_root=dataset_root, image_setting=args.image_setting
            )
            gt = normalize_gt_answer(rec.get("answer"))
            t_item = time.perf_counter()
            try:
                raw = engine.generate(
                    prompt=prompt,
                    image_path=img,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    force_answer_at=force_at,
                )
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = str(exc)
                print(f"[eval] FAIL {rec.get('id')}: {exc}", flush=True)

            rescued = False
            if args.direct_rescue and _normalize_answer_mode_local(args.answer_mode) == "direct" and not err:
                # Accept only a clean letter answer; otherwise nudge once for the letters.
                pred = strict_letters(raw)
                if pred is None:
                    rescued = True
                    try:
                        suffix, _ = engine._generate_once(
                            prompt=prompt, image_path=img, max_new_tokens=8,
                            temperature=temperature, top_p=top_p, assistant_prefix=None,
                            followup=(raw, DIRECT_RESCUE_FOLLOWUP),
                        )
                    except Exception as exc:  # noqa: BLE001
                        suffix, err = "", str(exc)
                    pred = strict_letters(suffix)
                    raw = f"{raw}\n[user] {DIRECT_RESCUE_FOLLOWUP}\n[assistant] {suffix}"
            else:
                pred = extract_pred_answer(
                    raw, answer_mode=args.answer_mode, thinking=thinking
                )
            parse_ok = pred is not None
            if not parse_ok:
                n_parse_fail += 1
                by_cat[cat][2] += 1
            correct = bool(parse_ok and pred == gt)
            if correct:
                n_correct += 1
                by_cat[cat][0] += 1
            by_cat[cat][1] += 1

            row = {
                "id": rec.get("id"),
                "category": cat,
                "mcq_type": rec.get("mcq_type"),
                "gt": gt,
                "pred": pred,
                "correct": correct,
                "parse_ok": parse_ok,
                "raw": raw,
                "error": err,
                "image": str(img),
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "force_answer_at": force_at,
                "direct_rescued": rescued,
                "elapsed_s": time.perf_counter() - t_item,
                "cell": tag,
                "thinking": thinking,
                "som": use_som,
            }
            results.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[{i + 1}/{len(records)}] {cat} gt={gt} pred={pred} "
                f"{'OK' if correct else 'WRONG'} tok={max_tokens} "
                f"{row['elapsed_s']:.1f}s",
                flush=True,
            )

    total_s = time.perf_counter() - t0
    n = len(results)
    per_category = {
        c: {
            "n": tot,
            "n_correct": cor,
            "n_parse_fail": pf,
            "accuracy": (cor / tot) if tot else 0.0,
        }
        for c, (cor, tot, pf) in sorted(by_cat.items())
    }
    metrics = {
        "cell": tag,
        "prompt_format": args.prompt_format,
        "choice_seed": int(args.choice_seed),
        "image_setting": args.image_setting,
        "answer_mode": args.answer_mode,
        "thinking": thinking,
        "som_mode": getattr(args, "som_mode", "auto"),
        "temperature": temperature,
        "top_p": top_p,
        "model": args.model,
        "backend": "openai_server",
        "base_url": args.base_url,
        "dataset_jsonl": str(jsonl_path),
        "n": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "max_new_tokens_default": default_max_tokens,
        "max_new_tokens_closest_three_cot": max_new_tokens_for_mode(
            args.answer_mode, category="closest_three", thinking=thinking
        ),
        "limit": args.limit,
        "elapsed_s": total_s,
        "per_category": per_category,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(
        "==== category-wise ====\n"
        + format_category_stats_table(
            per_category, n_total=n, n_correct=n_correct
        ),
        flush=True,
    )
    print(f"[eval] wrote {pred_path}", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image-setting",
        default="before",
        help="before|after (or 1|2). 1=rgb0+motion, 2=rgb1 after-image",
    )
    p.add_argument(
        "--answer-mode",
        default="direct",
        help="direct|cot (or A|B). A=letter only, B=CoT + <answer> tags",
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--jsonl", type=Path, default=None, help="Default: DATASET/dataset.jsonl")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--prompt-format", choices=("cit", "formal"), default="cit",
                   help="cit = build_user_prompt (training / earlier evals); formal = eval_formats.build_formal_prompt")
    p.add_argument("--formal-direct-hint", action="store_true",
                   help="formal format: append 'Please directly give the index of the answer.'")
    p.add_argument("--choice-seed", type=int, default=0,
                   help="0 = dataset order; other seeds re-permute answer choices (eval_formats.permute_for_seed)")
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Qwen3-*-Thinking: adapted B prompt, strip <think>, larger budget, "
        "default temp=0.6/top_p=0.95",
    )
    p.add_argument(
        "--direct-rescue",
        action="store_true",
        help="Direct mode: accept only a clean leading letter answer; if the reply is anything else "
        "(e.g. a truncated explanation), ask once in a follow-up turn for only the option letter(s).",
    )
    p.add_argument(
        "--image-first",
        action="store_true",
        help="Send [image, text] instead of [text, image] (matches training and eval_qwen3_vl_8b.py).",
    )
    p.add_argument(
        "--enable-thinking",
        choices=("on", "off"),
        default=None,
        help="chat_template_kwargs.enable_thinking for hybrid-thinking models (Qwen3.5: off = Instruct mode)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--force-greedy",
        action="store_true",
        help="Keep temperature=0 even with --thinking (not recommended)",
    )
    p.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout seconds")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--skip-server-wait", action="store_true")
    p.add_argument("--server-wait-timeout", type=float, default=600.0)
    p.add_argument(
        "--som",
        dest="som_mode",
        action="store_const",
        const="on",
        default="auto",
        help="Force SoM grounding note in the prompt (ithor_v3/som)",
    )
    p.add_argument(
        "--no-som",
        dest="som_mode",
        action="store_const",
        const="off",
        help="Disable SoM grounding note even if records have som=true",
    )
    args = p.parse_args()
    name = str(args.model).lower()
    if "thinking" in name and not args.thinking:
        print("[eval] model name contains 'Thinking' → enabling --thinking", flush=True)
        args.thinking = True
    evaluate(args)


if __name__ == "__main__":
    main()
