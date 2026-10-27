#!/usr/bin/env python3
"""Evaluate Qwen3-VL-8B on the CIT iTHOR MCQ benchmark (2×2 protocol).

Axes (CLI flags)
----------------
``--image-setting {before,after,fake2a|1,2}``
  1 / before : Question + options + motion cue + **rgb0**
  2 / after  : static question + options + **rgb1**
  fake2a     : same prompt as after, but image is **rgb0** (ablation)

``--answer-mode {direct,cot|A,B}``
  A / direct : "Please directly give the index…"  (max 16 tokens)
  B / cot    : reason, then ``<answer>A</answer>`` (max 512 tokens)

Examples
--------
Sanity (6 items), setting 1A on GPU 0::

  CUDA_VISIBLE_DEVICES=0 python eval_qwen3_vl_8b.py \\
    --image-setting before --answer-mode direct --limit 6 --gpus 0

Full cell 2B on GPUs 5,6,7::

  CUDA_VISIBLE_DEVICES=5,6,7 python eval_qwen3_vl_8b.py \\
    --image-setting after --answer-mode cot --gpus 5,6,7

fake2a ablation (2A prompt, rgb0 image) on GPUs 4,5::

  python eval_qwen3_vl_8b.py \\
    --image-setting fake2a --answer-mode direct --gpus 4,5

Or launch all four sanity cells::

  bash run_eval_sanity_4way.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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

DEFAULT_DATASET = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11/som"
)
DEFAULT_MODEL = Path("models/Qwen3-VL-8B-Instruct")
DEFAULT_OUT_ROOT = _ROOT / "outputs" / "evals"


# Appended when a generous-budget CoT run (--cot-max-new-tokens) reaches the budget
# without an <answer> tag, so the instruct model still commits to a letter.
COT_BUDGET_NUDGE = "\n\nI must give my final answer now. <answer>"
# Averages are reported over all categories and without these two (6-cat ithor evals).
SIX_CAT_EXCLUDED = ("contact_wall", "near_object")


def category_averages(per_category: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, cats in (("all", sorted(per_category)),
                      ("subset", sorted(c for c in per_category if c not in SIX_CAT_EXCLUDED))):
        rows = [per_category[c] for c in cats]
        n = sum(r["n"] for r in rows)
        cor = sum(r["n_correct"] for r in rows)
        out[key] = {
            "categories": cats,
            "n": n,
            "average_accuracy": cor / n if n else 0.0,
            "category_average_accuracy": sum(r["accuracy"] for r in rows) / len(rows) if rows else 0.0,
        }
    return out


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


def _wait_for_model(model_dir: Path, timeout_s: float = 7200.0) -> None:
    """Block until safetensor shards referenced by the index exist."""
    index = model_dir / "model.safetensors.index.json"
    t0 = time.time()
    while True:
        if index.is_file():
            meta = json.loads(index.read_text())
            weight_map = meta.get("weight_map") or {}
            shards = sorted(set(weight_map.values()))
            missing = [s for s in shards if not (model_dir / s).is_file()]
            if not missing:
                print(f"[model] ready: {model_dir} ({len(shards)} shards)", flush=True)
                return
            print(
                f"[model] waiting for {len(missing)} shard(s); "
                f"e.g. {missing[0]} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            print(f"[model] waiting for index at {index}", flush=True)
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Model not ready after {timeout_s}s: {model_dir}")
        time.sleep(30)


class TransformersEngine:
    """Single-/multi-GPU HF generate path (good for small sanity runs)."""

    def __init__(self, model_path: Path, *, gpus: list[str]) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if len(gpus) > 1 else {"": 0}
        print(
            f"[hf] loading {model_path} device_map={device_map} dtype={dtype}",
            flush=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.torch = torch

    def generate(
        self,
        *,
        prompt: str,
        image_path: Optional[Path],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        force_answer_at: Optional[int] = None,
        nudge: Optional[str] = None,
    ) -> str:
        phase1_budget = int(max_new_tokens)
        if force_answer_at is not None:
            phase1_budget = max(1, min(int(force_answer_at), int(max_new_tokens)))
        partial = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=phase1_budget,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=None,
        )
        if force_answer_at is None or has_complete_answer_tag(partial):
            return partial
        continuation = (
            build_force_answer_continuation(partial)
            if nudge is None
            else partial.rstrip() + nudge
        )
        remaining = max(32, int(max_new_tokens) - phase1_budget)
        suffix = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=remaining,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=continuation,
        )
        if suffix.startswith(continuation):
            return suffix.strip()
        return (continuation + suffix).strip()

    def _generate_once(
        self,
        *,
        prompt: str,
        image_path: Optional[Path],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        assistant_prefix: Optional[str],
    ) -> str:
        from PIL import Image

        content: list[dict[str, Any]] = []
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": content,
            }
        ]
        continue_final = False
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
            continue_final = True
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
            return_dict=True,
            return_tensors="pt",
        )
        # Move tensors to the first parameter device.
        device = next(self.model.parameters()).device
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with self.torch.inference_mode():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        # Trim prompt tokens.
        prompt_len = inputs["input_ids"].shape[-1]
        gen = out_ids[:, prompt_len:]
        text = self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return (text or "").strip()


def _compatible_tp(model_path: Path, n_gpus: int) -> int:
    """Largest TP ≤ n_gpus that divides the model's attention-head count."""
    import json

    n_gpus = max(1, int(n_gpus))
    cfg_path = Path(model_path) / "config.json"
    n_heads = 32
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        text = cfg.get("text_config") or cfg
        n_heads = int(text.get("num_attention_heads") or cfg.get("num_attention_heads") or 32)
    tp = n_gpus
    while tp > 1 and (n_heads % tp != 0):
        tp -= 1
    return tp


class VllmEngine:
    """vLLM offline engine (supports tensor parallel across --gpus)."""

    def __init__(
        self,
        model_path: Path,
        *,
        gpus: list[str],
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        # Caller should already set CUDA_VISIBLE_DEVICES; TP must divide attn heads.
        n_visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if n_visible <= 0 or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
            n_visible = max(1, len(gpus) if gpus else 1)
        tp = _compatible_tp(model_path, n_visible)
        print(
            f"[vllm] loading {model_path} tp={tp} "
            f"gpu_memory_utilization={gpu_memory_utilization} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        self.llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            # Fixed KV-cache size (GiB per GPU): skips vLLM's memory profiling, whose "free memory must not
            # grow" assertion fails on GPUs shared with jobs whose usage fluctuates.
            **({"kv_cache_memory_bytes": int(float(os.environ["CIT_VLLM_KV_CACHE_GIB"]) * (1 << 30))}
               if os.environ.get("CIT_VLLM_KV_CACHE_GIB") else {}),
        )

    def generate(
        self,
        *,
        prompt: str,
        image_path: Optional[Path],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        force_answer_at: Optional[int] = None,
        nudge: Optional[str] = None,
    ) -> str:
        phase1_budget = int(max_new_tokens)
        if force_answer_at is not None:
            phase1_budget = max(1, min(int(force_answer_at), int(max_new_tokens)))
        partial = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=phase1_budget,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=None,
        )
        if force_answer_at is None or has_complete_answer_tag(partial):
            return partial
        continuation = (
            build_force_answer_continuation(partial)
            if nudge is None
            else partial.rstrip() + nudge
        )
        remaining = max(32, int(max_new_tokens) - phase1_budget)
        suffix = self._generate_once(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=remaining,
            temperature=temperature,
            top_p=top_p,
            assistant_prefix=continuation,
        )
        if suffix.startswith(continuation):
            return suffix.strip()
        return (continuation + suffix).strip()

    def generate_many(self, items: list[dict[str, Any]], *, temperature: float = 0.0,
                      top_p: float = 1.0) -> list[str]:
        """Batched ``generate``: same two phases (free generation up to ``force_answer_at``, then the
        forced-answer continuation for items without a complete <answer> tag), but each phase is one
        vLLM call over all items. Items: prompt, image_path, max_new_tokens, force_answer_at, nudge."""
        budgets = [int(it["max_new_tokens"]) if it.get("force_answer_at") is None
                   else max(1, min(int(it["force_answer_at"]), int(it["max_new_tokens"]))) for it in items]
        reqs = [self._build_request(it["prompt"], it["image_path"], None) for it in items]
        outs = self.llm.generate(reqs, sampling_params=[self._sampling(b, temperature, top_p) for b in budgets])
        partials = [(o.outputs[0].text or "").strip() for o in outs]
        results = list(partials)
        todo = [i for i, it in enumerate(items)
                if it.get("force_answer_at") is not None and not has_complete_answer_tag(partials[i])]
        if todo:
            conts = {i: (build_force_answer_continuation(partials[i]) if items[i].get("nudge") is None
                         else partials[i].rstrip() + items[i]["nudge"]) for i in todo}
            reqs2 = [self._build_request(items[i]["prompt"], items[i]["image_path"], conts[i]) for i in todo]
            sps2 = [self._sampling(max(32, int(items[i]["max_new_tokens"]) - budgets[i]), temperature, top_p)
                    for i in todo]
            for i, o in zip(todo, self.llm.generate(reqs2, sampling_params=sps2)):
                suffix = (o.outputs[0].text or "").strip()
                results[i] = suffix.strip() if suffix.startswith(conts[i]) else (conts[i] + suffix).strip()
        return results

    def _sampling(self, max_new_tokens: int, temperature: float, top_p: float):
        if temperature and temperature > 0:
            return self.SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return self.SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)

    def _build_request(self, prompt: str, image_path: Optional[Path], assistant_prefix: Optional[str]) -> dict:
        from PIL import Image

        content: list[dict[str, Any]] = []
        image = None
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        continue_final = bool(assistant_prefix)
        if continue_final:
            messages.append({"role": "assistant", "content": assistant_prefix})
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not continue_final,
            continue_final_message=continue_final)
        req: dict[str, Any] = {"prompt": text_prompt}
        if image is not None:
            req["multi_modal_data"] = {"image": image}
        return req

    def _generate_once(
        self,
        *,
        prompt: str,
        image_path: Optional[Path],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        assistant_prefix: Optional[str],
    ) -> str:
        from PIL import Image

        content: list[dict[str, Any]] = []
        image = None
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": content,
            }
        ]
        continue_final = False
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
            continue_final = True
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
        )
        # vLLM rejects temperature=0 with some samplers; use greedy when ~0.
        if temperature and temperature > 0:
            sp = self.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            sp = self.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
        req: dict[str, Any] = {"prompt": text_prompt}
        if image is not None:
            req["multi_modal_data"] = {"image": image}
        outs = self.llm.generate([req], sampling_params=sp)
        return (outs[0].outputs[0].text or "").strip()


def _build_engine(backend: str, model_path: Path, gpus: list[str], args: argparse.Namespace):
    if backend == "vllm":
        return VllmEngine(
            model_path,
            gpus=gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    if backend == "hf":
        return TransformersEngine(model_path, gpus=gpus)
    raise ValueError(backend)


def _resolve_som_flag(args: argparse.Namespace, rec: dict[str, Any]) -> bool:
    """CLI --som / --no-som override; else follow record['som'] or dataset path."""
    mode = str(getattr(args, "som_mode", "auto") or "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    if "som" in rec:
        return bool(rec.get("som"))
    ds = str(getattr(args, "dataset", "") or "").lower()
    return "/som" in ds or ds.rstrip("/").endswith("som")


def dry_run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Validate dataset paths + print SoM-adapted prompts; no model load."""
    dataset_root = Path(args.dataset)
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / "dataset.jsonl"
    records = _load_jsonl(jsonl_path, limit=int(args.limit or 0) or 3)
    if not records:
        raise SystemExit(f"No records in {jsonl_path}")

    tag = setting_tag(args.image_setting, args.answer_mode)
    out_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUT_ROOT / f"dryrun_qwen3vl8b_{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    n_som = 0
    previews: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        som = _resolve_som_flag(args, rec)
        if som:
            n_som += 1
        prompt = build_user_prompt(
            rec,
            image_setting=args.image_setting,
            answer_mode=args.answer_mode,
            thinking=bool(args.thinking),
            som=som,
        )
        try:
            img = resolve_image_path(
                rec, dataset_root=dataset_root, image_setting=args.image_setting
            )
            img_ok = True if img is None else img.is_file()
        except Exception as exc:  # noqa: BLE001
            img = Path("?")
            img_ok = False
            print(f"[dry-run] image FAIL {rec.get('id')}: {exc}", flush=True)
        if not img_ok:
            n_missing += 1
        row = {
            "id": rec.get("id"),
            "category": rec.get("category"),
            "som": som,
            "gt": normalize_gt_answer(rec.get("answer")),
            "image": None if img is None else str(img),
            "image_ok": img_ok,
            "prompt": prompt,
        }
        previews.append(row)
        print("=" * 72, flush=True)
        print(
            f"[dry-run {i + 1}/{len(records)}] {rec.get('id')} "
            f"cat={rec.get('category')} som={som} img_ok={img_ok}",
            flush=True,
        )
        print(f"image: {img}", flush=True)
        print(prompt, flush=True)

    summary = {
        "dry_run": True,
        "cell": tag,
        "image_setting": args.image_setting,
        "answer_mode": args.answer_mode,
        "som_mode": getattr(args, "som_mode", "auto"),
        "dataset_jsonl": str(jsonl_path),
        "n_preview": len(records),
        "n_som_flag": n_som,
        "n_missing_images": n_missing,
        "previews": previews,
    }
    (out_dir / "dry_run.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[dry-run] wrote {out_dir / 'dry_run.json'} "
        f"missing_images={n_missing} som_rows={n_som}/{len(records)}",
        flush=True,
    )
    if n_missing:
        raise SystemExit(f"dry-run found {n_missing} missing image(s)")
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "dry_run", False)):
        return dry_run_eval(args)

    dataset_root = Path(args.dataset)
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / "dataset.jsonl"
    model_path = Path(args.model)
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]

    # Pin devices before any CUDA/vLLM init. For vLLM, shrink to a TP that
    # divides the model's attention-head count (8B → 32 heads ⇒ TP∈{1,2,4,…}).
    if gpus:
        if args.backend == "vllm":
            if not args.skip_model_wait:
                _wait_for_model(model_path, timeout_s=args.model_wait_timeout)
            tp = _compatible_tp(model_path, len(gpus))
            if tp < len(gpus):
                print(
                    f"[eval] TP {len(gpus)} unsupported for this model; "
                    f"using first {tp} GPU(s): {gpus[:tp]}",
                    flush=True,
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:tp])
        elif "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    tag = setting_tag(args.image_setting, args.answer_mode)
    thinking = bool(args.thinking)
    # Thinking models: Qwen recommends sampling (not greedy). Default 0.6 / 0.95
    # unless the user explicitly overrides temperature.
    temperature = float(args.temperature)
    top_p = float(args.top_p)
    if thinking and args.temperature == 0.0 and not args.force_greedy:
        temperature = 0.6
        if abs(top_p - 1.0) < 1e-9:
            top_p = 0.95
    default_max_tokens = max_new_tokens_for_mode(
        args.answer_mode, thinking=thinking
    )
    cot_budget = (
        int(args.cot_max_new_tokens)
        if args.cot_max_new_tokens and str(args.answer_mode).lower() in {"cot", "b"}
        else 0
    )
    out_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUT_ROOT / f"qwen3vl8b_{'thinking_' if thinking else ''}{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_model_wait:
        _wait_for_model(model_path, timeout_s=args.model_wait_timeout)

    records = _load_jsonl(jsonl_path, limit=int(args.limit or 0))
    if not records:
        raise SystemExit(f"No records in {jsonl_path}")
    if int(args.choice_seed):  # answer-order shuffle, same permutations as the other eval scripts
        from cit_pipeline.eval_formats import permute_for_seed

        records = permute_for_seed(records, int(args.choice_seed))

    print(
        f"[eval] cell={tag} image_setting={args.image_setting} "
        f"answer_mode={args.answer_mode} thinking={thinking} "
        f"som_mode={getattr(args, 'som_mode', 'auto')} n={len(records)} "
        f"max_tokens_default={default_max_tokens} "
        f"temp={temperature} top_p={top_p} backend={args.backend} "
        f"gpus={gpus} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    engine = _build_engine(args.backend, model_path, gpus or ["0"], args)

    pred_path = out_dir / "predictions.jsonl"
    results: list[dict[str, Any]] = []
    n_correct = 0
    n_parse_fail = 0
    # per category: [correct, total, parse_fail]
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    t0 = time.perf_counter()
    batched: dict[int, str] = {}
    bs = int(getattr(args, "batch_size", 1) or 1)
    if bs > 1 and hasattr(engine, "generate_many"):
        print(f"[eval] batched generation, batch size {bs}", flush=True)
    with pred_path.open("w") as fout:
        for i, rec in enumerate(records):
            if bs > 1 and hasattr(engine, "generate_many") and i % bs == 0:
                # Generate the next chunk in one batched call (same prompts / budgets / nudge as below).
                chunk = []
                for r in records[i:i + bs]:
                    c = str(r.get("category") or "unknown")
                    mt = max_new_tokens_for_mode(args.answer_mode, category=c, thinking=thinking)
                    fa = force_answer_at_tokens(thinking=thinking, category=c)
                    nd = None
                    if cot_budget and not thinking:
                        fa, mt, nd = cot_budget, cot_budget + 32, COT_BUDGET_NUDGE
                    chunk.append({"prompt": build_user_prompt(r, image_setting=args.image_setting,
                                                              answer_mode=args.answer_mode, thinking=thinking,
                                                              som=_resolve_som_flag(args, r)),
                                  "image_path": resolve_image_path(r, dataset_root=dataset_root,
                                                                   image_setting=args.image_setting),
                                  "max_new_tokens": mt, "force_answer_at": fa, "nudge": nd})
                tb = time.perf_counter()
                for k, txt in enumerate(engine.generate_many(chunk, temperature=temperature, top_p=top_p)):
                    batched[i + k] = txt
                print(f"[eval] batch {i // bs + 1}: {len(chunk)} items in {time.perf_counter() - tb:.0f}s", flush=True)
            cat = str(rec.get("category") or "unknown")
            max_tokens = max_new_tokens_for_mode(
                args.answer_mode, category=cat, thinking=thinking
            )
            force_at = force_answer_at_tokens(thinking=thinking, category=cat)
            nudge = None
            if cot_budget and not thinking:
                # 1B with a generous budget: reason freely for cot_budget tokens; if no
                # <answer> tag by then, append COT_BUDGET_NUDGE and allow a few tokens.
                force_at, max_tokens, nudge = cot_budget, cot_budget + 32, COT_BUDGET_NUDGE
            prompt = build_user_prompt(
                rec,
                image_setting=args.image_setting,
                answer_mode=args.answer_mode,
                thinking=thinking,
                som=_resolve_som_flag(args, rec),
            )
            img = resolve_image_path(
                rec, dataset_root=dataset_root, image_setting=args.image_setting
            )
            gt = normalize_gt_answer(rec.get("answer"))
            t_item = time.perf_counter()
            try:
                if i in batched:
                    raw = batched.pop(i)
                else:
                    raw = engine.generate(
                        prompt=prompt,
                        image_path=img,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        force_answer_at=force_at,
                        nudge=nudge,
                    )
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = str(exc)
                print(f"[eval] FAIL {rec.get('id')}: {exc}", flush=True)

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
                "image": None if img is None else str(img),
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "force_answer_at": force_at,
                "forced_answer": bool(nudge and nudge.strip() in raw),
                "elapsed_s": time.perf_counter() - t_item,
                "cell": tag,
                "thinking": thinking,
                "som": _resolve_som_flag(args, rec),
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
        "image_setting": args.image_setting,
        "answer_mode": args.answer_mode,
        "thinking": thinking,
        "som_mode": getattr(args, "som_mode", "auto"),
        "temperature": temperature,
        "top_p": top_p,
        "model": str(model_path),
        "backend": args.backend,
        "gpus": gpus,
        "dataset_jsonl": str(jsonl_path),
        "choice_seed": int(args.choice_seed),
        "n": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "max_new_tokens_default": default_max_tokens,
        "max_new_tokens_closest_three_cot": max_new_tokens_for_mode(
            args.answer_mode, category="closest_three", thinking=thinking
        ),
        "cot_max_new_tokens": cot_budget or None,
        "n_forced_answer": sum(int(r.get("forced_answer", False)) for r in results),
        "averages": category_averages(per_category),
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
        help="before|after|fake2a|text (or 1|2). 1=rgb0+motion, 2=rgb1+static, "
        "fake2a=static prompt + rgb0, text=static prompt with no image",
    )
    p.add_argument(
        "--answer-mode",
        default="direct",
        help="direct|cot (or A|B). A=letter only, B=CoT + <answer> tags",
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--jsonl", type=Path, default=None, help="Default: DATASET/dataset.jsonl")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--gpus", default="0", help="Physical GPU ids (also sets CUDA_VISIBLE_DEVICES if unset)")
    p.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="Inference backend (vllm recommended for multi-GPU TP)",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Qwen3-*-Thinking: adapted B prompt, strip <think>, larger budget, "
        "default temp=0.6/top_p=0.95",
    )
    p.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=0,
        help="Generous CoT budget for instruct models in answer-mode cot (e.g. 4096); "
        "0 keeps the default 512 / 1024. If no <answer> tag by then, a short nudge forces one.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--force-greedy",
        action="store_true",
        help="Keep temperature=0 even with --thinking (not recommended)",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--skip-model-wait", action="store_true")
    p.add_argument("--model-wait-timeout", type=float, default=7200.0)
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
    p.add_argument("--batch-size", type=int, default=1,
                   help="vLLM backend: generate this many items per batched call (1 = one at a time, as before)")
    p.add_argument("--choice-seed", type=int, default=0,
                   help="0 = dataset order; k > 0 = balanced answer-order shuffle k (eval_formats.permute_for_seed)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts + check image paths; do not load the model",
    )
    args = p.parse_args()
    # Auto-enable thinking when the model path name suggests it.
    name = str(args.model).lower()
    if "thinking" in name and not args.thinking:
        print("[eval] model path contains 'Thinking' → enabling --thinking", flush=True)
        args.thinking = True
    evaluate(args)


if __name__ == "__main__":
    main()
