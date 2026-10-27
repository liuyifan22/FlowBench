#!/usr/bin/env python3
"""Train-aligned ithor_v7.1 eval for LoRA SFT checkpoints (modes 1A / 2A).

Uses the *same* example builder as training (``qwen_finetune.data.build_sft_example``)
and the same chat-template shape as ``QwenVLDataCollator``:

  user: [image, prompt_text]  →  generate assistant letter

This is intentionally separate from ``eval_qwen3_vl_8b.py`` so format drift cannot
silently creep in. Matched protocol: evaluate the 1A-trained adapter on cell 1A
and the 2A-trained adapter on cell 2A.

Examples::

  CUDA_VISIBLE_DEVICES=1 python qwen_finetune/eval_train_aligned.py \\
      --mode 1A --model .../procthor_v7_som_1A_5ep_.../checkpoint-2000 \\
      --out outputs/evals/.../1A

  # Or launch both matched cells:
  bash qwen_finetune/run_eval_train_aligned_matched.sh
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
_CIT = _ROOT.parent
for p in (_CIT, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cit_pipeline.benchmark_eval import (  # noqa: E402
    extract_pred_answer,
    format_category_stats_table,
    setting_tag,
)
from qwen_finetune.data import build_sft_example  # noqa: E402
from qwen_finetune.dataset_config import (  # noqa: E402
    DatasetFilterConfig,
    load_filtered_records,
)

DEFAULT_DATASET = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/ithor_v11/som"
)
DEFAULT_BASE = Path("models/Qwen3-VL-8B-Instruct")


def _load_lora_engine(model_path: Path, *, base_model: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = Path(model_path)
    adapter_cfg = model_path / "adapter_config.json"
    if not adapter_cfg.is_file():
        # Allow a run root that contains adapters at top level.
        if (model_path / "adapter_model.safetensors").is_file():
            pass
        else:
            raise SystemExit(f"No adapter_config.json under {model_path}")

    meta = json.loads(adapter_cfg.read_text()) if adapter_cfg.is_file() else {}
    base = Path(base_model) if base_model else Path(
        meta.get("base_model_name_or_path") or DEFAULT_BASE
    )
    def _has_processor(p: Path) -> bool:
        return (
            (p / "preprocessor_config.json").is_file()
            or (p / "processor_config.json").is_file()
        )

    # Prefer processor artifacts saved with the run (matches training tokenize).
    # checkpoint-* dirs usually only have adapters; parent run dir has processor_*.
    if _has_processor(model_path):
        proc_src = model_path
    elif _has_processor(model_path.parent):
        proc_src = model_path.parent
    else:
        proc_src = base

    print(f"[train-aligned] base={base}", flush=True)
    print(f"[train-aligned] adapter={model_path}", flush=True)
    print(f"[train-aligned] processor={proc_src}", flush=True)

    processor = AutoProcessor.from_pretrained(str(proc_src), trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base),
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(model_path))
    model.eval()
    return processor, model, torch


def _generate_letter(
    *,
    processor: Any,
    model: Any,
    torch: Any,
    prompt: str,
    image,
    max_new_tokens: int = 16,
) -> str:
    """Mirror training collator: user=[image,text], then generate assistant."""
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
    inputs = processor.apply_chat_template(
        [user_msg],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    gen = out_ids[:, prompt_len:]
    text = processor.batch_decode(
        gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return (text or "").strip()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    mode = args.mode.upper()
    if mode not in {"1A", "2A"}:
        raise SystemExit(f"mode must be 1A or 2A, got {mode}")

    dataset_root = Path(args.dataset)
    jsonl_path = Path(args.jsonl) if args.jsonl else dataset_root / "dataset.jsonl"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetFilterConfig()
    records = load_filtered_records(
        jsonl_path, dataset_root=dataset_root, cfg=cfg, limit=int(args.limit or 0)
    )
    if not records:
        raise SystemExit(f"No usable records in {jsonl_path}")
    from cit_pipeline.eval_formats import permute_for_seed
    records = permute_for_seed(records, int(args.choice_seed))

    # Build SFT examples up front (same path as training).
    examples = []
    for rec in records:
        try:
            examples.append(
                build_sft_example(rec, mode=mode, dataset_root=dataset_root)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {rec.get('id')}: {exc}", flush=True)
    if not examples:
        raise SystemExit("No examples built")

    # Parity dump: first example prompt + paths.
    ex0 = examples[0]
    parity = {
        "mode": mode,
        "id": ex0.id,
        "image_path": str(ex0.image_path),
        "answer": ex0.answer,
        "prompt": ex0.prompt,
        "builder": "qwen_finetune.data.build_sft_example",
        "chat_format": "user:[image,text] + generate assistant (train collator parity)",
        "model": str(args.model),
        "base_model": str(args.base_model),
    }
    (out_dir / "format_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(
        f"[train-aligned] cell={mode} n={len(examples)} "
        f"builder=build_sft_example model={args.model}",
        flush=True,
    )
    print(f"[train-aligned] sample prompt:\n{ex0.prompt[:500]}\n---", flush=True)

    if args.dry_run:
        print("[dry-run] skipping model load", flush=True)
        return parity

    processor, model, torch = _load_lora_engine(
        Path(args.model), base_model=Path(args.base_model)
    )

    pred_path = out_dir / "predictions.jsonl"
    n_correct = 0
    n_parse_fail = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    t0 = time.perf_counter()

    with pred_path.open("w") as fout:
        for i, ex in enumerate(examples):
            t_item = time.perf_counter()
            image = Image.open(ex.image_path).convert("RGB")
            try:
                raw = _generate_letter(
                    processor=processor,
                    model=model,
                    torch=torch,
                    prompt=ex.prompt,
                    image=image,
                    max_new_tokens=int(args.max_new_tokens),
                )
                err = None
            except Exception as exc:  # noqa: BLE001
                raw, err = "", str(exc)[:300]
                print(f"[FAIL] {ex.id}: {exc}", flush=True)

            pred = extract_pred_answer(raw, answer_mode="direct", thinking=False)
            parse_ok = pred is not None
            if not parse_ok:
                n_parse_fail += 1
                by_cat[ex.category][2] += 1
            correct = bool(parse_ok and pred == ex.answer)
            if correct:
                n_correct += 1
                by_cat[ex.category][0] += 1
            by_cat[ex.category][1] += 1

            row = {
                "id": ex.id,
                "category": ex.category,
                "gt": ex.answer,
                "pred": pred,
                "correct": correct,
                "parse_ok": parse_ok,
                "raw": raw,
                "error": err,
                "image": str(ex.image_path),
                "prompt": ex.prompt,
                "elapsed_s": time.perf_counter() - t_item,
                "cell": setting_tag(
                    "before" if mode == "1A" else "after", "direct"
                ),
                "format": "train_aligned_sft",
                "model": str(args.model),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(examples):
                print(
                    f"[{i + 1}/{len(examples)}] {ex.category} "
                    f"gt={ex.answer} pred={pred} "
                    f"{'OK' if correct else 'WRONG'} "
                    f"{row['elapsed_s']:.2f}s",
                    flush=True,
                )

    n = len(examples)
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
        "cell": mode,
        "choice_seed": int(args.choice_seed),
        "image_setting": "before" if mode == "1A" else "after",
        "answer_mode": "direct",
        "format": "train_aligned_sft",
        "builder": "qwen_finetune.data.build_sft_example",
        "model": str(args.model),
        "base_model": str(args.base_model),
        "dataset_jsonl": str(jsonl_path),
        "n": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "per_category": per_category,
        "elapsed_s": time.perf_counter() - t0,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(
        "==== category-wise ====\n"
        + format_category_stats_table(
            per_category, n_total=n, n_correct=n_correct
        ),
        flush=True,
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("1A", "2A"))
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--model", type=Path, required=True, help="LoRA adapter dir or best ckpt")
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--choice-seed", type=int, default=0,
                   help="0 = dataset order; other seeds re-permute answer choices (eval_formats.permute_for_seed)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
