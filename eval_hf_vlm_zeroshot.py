#!/usr/bin/env python3
"""Zero-shot CIT MCQ eval with plain transformers, for VLMs vLLM cannot serve
(e.g. LLaVA-OneVision-1.5 / -2, which ship their own modelling code).

Same protocol as ``eval_qwen3_vl_32b_server.py --image-first --direct-rescue``:
prompt from ``cit_pipeline.benchmark_eval.build_user_prompt``, one user turn
[image, text], greedy decoding, 16 tokens; if the reply does not start with a clean
option letter, ask once "Please answer with only the option letter(s)..." for 8
tokens (follow-up user turn). Writes predictions.jsonl + metrics.json in the same format.

  CUDA_VISIBLE_DEVICES=0 python eval_hf_vlm_zeroshot.py --model /path/LLaVA-OneVision-2-8B-Instruct \
      --name LLaVA-OneVision-2-8B --image-setting after --out .../2A
  # several answer-order seeds in one process (model loaded once):
  ... --choice-seed 0,1,2 --out .../1A_seed{seed}
"""

from __future__ import annotations

import os
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from cit_pipeline.benchmark_eval import (
    build_user_prompt,
    format_category_stats_table,
    normalize_gt_answer,
    resolve_image_path,
    setting_tag,
)
from eval_qwen3_vl_32b_server import DIRECT_RESCUE_FOLLOWUP, strict_letters

DEFAULT_DATASET = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11/som")


def load(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    # GPU guard (no-op without GPU_GUARD_DIR): take the GPU over from the holder and release it
    # right before loading; transformers reserves the model's full size on the GPU at load start.
    from qwen_finetune.gpu_guard import guard_client

    guard_client.take_over(0, 0.0, headroom_gib=2.0)
    guard_client.drop()
    last = None
    for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            for dkw in ("dtype", "torch_dtype"):  # transformers >= 5 uses dtype, 4.x torch_dtype
                try:
                    model = cls.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda",
                                                **{dkw: torch.bfloat16}).eval()
                    return processor, model
                except TypeError as exc:
                    if dkw not in str(exc):
                        raise
                    last = exc
        except (ValueError, KeyError) as exc:  # wrong auto class for this checkpoint
            last = exc
    raise RuntimeError(f"could not load {model_dir}: {last}")


@torch.inference_mode()
def generate(processor, model, image: Image.Image, prompt: str, max_new_tokens: int,
             followup: tuple[str, str] | None = None) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    if followup is not None:  # (assistant reply, next user message)
        messages += [{"role": "assistant", "content": [{"type": "text", "text": followup[0]}]},
                     {"role": "user", "content": [{"type": "text", "text": followup[1]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--image-setting", choices=("before", "after"), required=True)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", required=True,
                    help="output dir; may contain {seed} to run several --choice-seed values in one process")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-format", choices=("cit", "formal"), default="cit")
    ap.add_argument("--choice-seed", default="0",
                    help="seed or comma list; several seeds share one loaded model (the GPU is never released)")
    ap.add_argument("--formal-direct-hint", action="store_true")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir (finetuned run) applied on --model")
    ap.add_argument("--no-rescue", action="store_true",
                    help="no follow-up turn for non-letter replies (finetuned models, as eval_train_aligned.py)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.choice_seed.split(",")]
    if len(seeds) > 1 and "{seed}" not in args.out:
        ap.error("--out must contain {seed} when several seeds are given")
    processor, model = load(args.model)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"loaded adapter {args.adapter}", flush=True)
    for seed in seeds:
        out = Path(args.out.format(seed=seed))
        if (out / "metrics.json").is_file():
            print(f"seed {seed}: {out} already done, skipping", flush=True)
            continue
        run_seed(args, processor, model, seed, out)


def run_seed(args, processor, model, choice_seed: int, out: Path) -> None:
    from cit_pipeline.eval_formats import build_formal_prompt, permute_for_seed
    out.mkdir(parents=True, exist_ok=True)
    records = permute_for_seed([json.loads(l) for l in open(args.dataset / "dataset.jsonl")], choice_seed)
    records = records[: args.limit] if args.limit else records
    tag = setting_tag(args.image_setting, "direct")
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    n_correct = n_pf = 0
    t0 = time.perf_counter()
    with open(out / "predictions.jsonl", "w") as fout:
        for i, rec in enumerate(records):
            cat = rec["category"]
            prompt = (build_formal_prompt(rec, image_setting=args.image_setting, direct_hint=args.formal_direct_hint)
                      if args.prompt_format == "formal"
                      else build_user_prompt(rec, image_setting=args.image_setting, answer_mode="direct"))
            img_path = resolve_image_path(rec, dataset_root=args.dataset, image_setting=args.image_setting)
            image = Image.open(img_path).convert("RGB")
            gt = normalize_gt_answer(rec.get("answer"))
            t_item, err = time.perf_counter(), None
            try:
                raw = generate(processor, model, image, prompt, 16)
                pred, rescued = strict_letters(raw), False
                if pred is None and not args.no_rescue:
                    rescued = True
                    suffix = generate(processor, model, image, prompt, 8, followup=(raw, DIRECT_RESCUE_FOLLOWUP))
                    pred, raw = strict_letters(suffix), f"{raw}\n[user] {DIRECT_RESCUE_FOLLOWUP}\n[assistant] {suffix}"
            except Exception as exc:  # noqa: BLE001
                raw, pred, rescued, err = "", None, False, str(exc)[:300]
                print(f"[FAIL] {rec['id']}: {exc}", flush=True)
            ok = pred is not None
            correct = bool(ok and pred == gt)
            n_correct += correct
            n_pf += not ok
            by_cat[cat][0] += correct
            by_cat[cat][1] += 1
            by_cat[cat][2] += not ok
            row = {"id": rec["id"], "category": cat, "mcq_type": rec.get("mcq_type"), "gt": gt, "pred": pred,
                   "correct": correct, "parse_ok": ok, "raw": raw, "error": err, "image": str(img_path),
                   "prompt": prompt, "max_new_tokens": 16, "direct_rescued": rescued,
                   "elapsed_s": time.perf_counter() - t_item, "cell": tag, "som": bool(rec.get("som"))}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i + 1}/{len(records)}] {cat} gt={gt} pred={pred} {'OK' if correct else 'WRONG'} "
                  f"{row['elapsed_s']:.1f}s", flush=True)
    n = len(records)
    per_category = {c: {"n": t, "n_correct": cor, "n_parse_fail": pf, "accuracy": cor / t if t else 0.0}
                    for c, (cor, t, pf) in sorted(by_cat.items())}
    metrics = {"cell": tag, "prompt_format": args.prompt_format, "choice_seed": choice_seed,
               "image_setting": args.image_setting, "answer_mode": "direct", "model": args.name,
               "model_path": args.model, "adapter": args.adapter or None, "backend": "transformers", "dataset_jsonl": str(args.dataset / "dataset.jsonl"),
               "n": n, "accuracy": n_correct / n if n else 0.0, "n_correct": n_correct, "n_parse_fail": n_pf,
               "elapsed_s": time.perf_counter() - t0, "per_category": per_category,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(format_category_stats_table(per_category, n_total=n, n_correct=n_correct), flush=True)


if __name__ == "__main__":
    main()
