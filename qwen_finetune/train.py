"""Multi-GPU SFT for Qwen3-VL-8B on CIT MCQ data (modes 1A / 2A / 1B). Also Qwen3.5 and
LLaVA-OneVision-2 (see ``model_family``) with the identical recipe.

Launch with torchrun, e.g.::

  torchrun --nproc_per_node=8 -m qwen_finetune.train --mode 2A
  # or
  # full recipe: see README.md (Qwen fine-tuning)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
_CIT = _ROOT.parent
for p in (_CIT, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_config import DatasetFilterConfig, summarize_filter  # noqa: E402
from data import QwenVLDataCollator, make_dataset  # noqa: E402

DEFAULT_MODEL = Path("models/Qwen3-VL-8B-Instruct")
DEFAULT_DATA = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11/qwen_training/"
    "som_balanced_splits"
)
DEFAULT_OUT = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit") + "/qwen_training/runs"
)
# Cap each process to this fraction of its GPU's total memory.
DEFAULT_GPU_MEM_FRACTION = 0.80


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=("1A", "2A", "1B"),
        help="Training cell: 1A (before+motion), 2A (after+static), 1B (before+motion, reasoning target)",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Train jsonl. Default: DATASET/train.jsonl if present else DATASET/dataset.jsonl",
    )
    p.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
        help="Val jsonl. Default: DATASET/val.jsonl if present.",
    )
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Held-out eval jsonl (not used during training). Default: DATASET/eval.jsonl",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = all usable train rows")
    p.add_argument("--val-limit", type=int, default=0, help="0 = all val rows")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument(
        "--eval-every-epochs",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.25), override eval_steps/save_steps to this "
            "fraction of one epoch after the train set size is known"
        ),
    )
    p.add_argument(
        "--eval-strategy",
        default="steps",
        choices=("no", "steps", "epoch"),
        help="HF Trainer eval schedule (uses all DDP ranks)",
    )
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=DEFAULT_GPU_MEM_FRACTION,
        help="Per-process CUDA memory fraction cap (default 0.80)",
    )
    p.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LoRA (default on). Use --no-lora for full finetune.",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module name substrings for LoRA",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--merge-lora",
        action="store_true",
        help="After training, merge LoRA into base and save under output_dir/merged",
    )
    p.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated category allow-list",
    )
    p.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep best checkpoint by eval_loss (requires val set + eval_strategy)",
    )
    p.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip filesystem image existence filter (use when paths are pre-validated / absolute)",
    )
    return p.parse_args()


def _is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _resolve_jsonls(args: argparse.Namespace) -> tuple[Path, Optional[Path], Optional[Path]]:
    dataset_root = Path(args.dataset)
    if args.jsonl is not None:
        train_jsonl = Path(args.jsonl)
    elif (dataset_root / "train.jsonl").is_file():
        train_jsonl = dataset_root / "train.jsonl"
    else:
        train_jsonl = dataset_root / "dataset.jsonl"

    if args.val_jsonl is not None:
        val_jsonl: Optional[Path] = Path(args.val_jsonl)
    elif (dataset_root / "val.jsonl").is_file():
        val_jsonl = dataset_root / "val.jsonl"
    else:
        val_jsonl = None

    if args.eval_jsonl is not None:
        eval_jsonl: Optional[Path] = Path(args.eval_jsonl)
    elif (dataset_root / "eval.jsonl").is_file():
        eval_jsonl = dataset_root / "eval.jsonl"
    else:
        eval_jsonl = None

    return train_jsonl, val_jsonl, eval_jsonl


def _cap_gpu_memory(fraction: float) -> None:
    """Limit each local rank to ``fraction`` of its GPU's total VRAM."""
    if fraction <= 0 or fraction >= 1:
        return
    import torch

    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=local_rank)
    if _is_main():
        props = torch.cuda.get_device_properties(local_rank)
        capped = int(props.total_memory * float(fraction))
        print(
            f"[train] GPU mem fraction={fraction:.2f} "
            f"(~{capped / (1024**3):.1f} GiB / "
            f"{props.total_memory / (1024**3):.1f} GiB on cuda:{local_rank})",
            flush=True,
        )


def model_family(model_path: Path) -> dict[str, Any]:
    """Per-architecture handling. Qwen3-VL keeps the original code path; the others load through
    AutoModelForImageTextToText. llava_onevision2's apply_chat_template drops PIL images, so its
    collator renders text first; qwen3_5 is trained in non-thinking (Instruct) mode."""
    from transformers import AutoConfig

    mt = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True).model_type
    return {"model_type": mt,
            "render_then_process": mt == "llava_onevision2",
            "chat_kwargs": {"enable_thinking": False} if mt == "qwen3_5" else {}}


def save_processor(processor: Any, out_dir: Path, base: Path) -> None:
    """processor.save_pretrained, or (remote-code processors without it, e.g. LLaVA-OneVision-2)
    copy the base model's tokenizer / preprocessor / chat-template / code files."""
    import shutil

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(out_dir))
        return
    for f in base.iterdir():
        if f.suffix in (".py", ".jinja", ".txt") or (f.suffix == ".json" and "safetensors" not in f.name
                                                      and f.name != "config.json"):
            shutil.copy(f, out_dir / f.name)


def _build_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(args.model), trust_remote_code=True
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    fam = model_family(Path(args.model))
    if _is_main():
        print(f"[train] loading base model {args.model} dtype={dtype} family={fam}", flush=True)
    if fam["model_type"] == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model),
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model), dtype=dtype, trust_remote_code=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise SystemExit(
                "peft is required for --lora (default). "
                "Install: pip install peft accelerate datasets  "
                "or pass --no-lora for full finetune."
            ) from exc
        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        if _is_main():
            model.print_trainable_parameters()
    return model, processor


def main() -> None:
    args = _parse_args()
    mode = args.mode.upper()
    dataset_root = Path(args.dataset)
    train_jsonl, val_jsonl, eval_jsonl = _resolve_jsonls(args)
    if not train_jsonl.is_file():
        raise SystemExit(f"Missing train jsonl: {train_jsonl}")

    _cap_gpu_memory(float(args.gpu_mem_fraction))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUT / f"procthor_v7_som_{mode}_{stamp}"
    )
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    cats: Optional[tuple[str, ...]] = None
    if args.categories:
        cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    # Splits are already GT-complete filtered; keep filters as a safety net.
    filter_cfg = DatasetFilterConfig(
        require_gt_complete=True,
        require_letter_answer=True,
        require_images_exist=not bool(args.skip_image_check),
        categories=cats,
        balance_yes_no=False,
        seed=int(args.seed),
    )

    if _is_main():
        stats = summarize_filter(
            train_jsonl, dataset_root=dataset_root, cfg=filter_cfg
        )
        print(f"[filter/train] {json.dumps(stats, indent=2)}", flush=True)
        (out_dir / "filter_stats_train.json").write_text(json.dumps(stats, indent=2))
        if val_jsonl is not None and val_jsonl.is_file():
            vstats = summarize_filter(
                val_jsonl, dataset_root=dataset_root, cfg=filter_cfg
            )
            print(f"[filter/val] {json.dumps(vstats, indent=2)}", flush=True)
            (out_dir / "filter_stats_val.json").write_text(json.dumps(vstats, indent=2))
        meta = {
            "mode": mode,
            "model": str(args.model),
            "dataset": str(dataset_root),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl) if val_jsonl else None,
            "eval_jsonl": str(eval_jsonl) if eval_jsonl else None,
            "filter": filter_cfg.to_dict(),
            "lora": bool(args.lora),
            "gpu_mem_fraction": float(args.gpu_mem_fraction),
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "model_family": model_family(Path(args.model)),
        }
        (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("transformers required") from exc

    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "accelerate is required for multi-GPU Trainer. "
            "pip install accelerate peft datasets"
        ) from exc

    train_ds = make_dataset(
        train_jsonl,
        dataset_root=dataset_root,
        mode=mode,
        filter_cfg=filter_cfg,
        limit=int(args.limit or 0),
    )
    if len(train_ds) == 0:
        raise SystemExit("No usable training examples after filter")

    if args.eval_every_epochs is not None:
        frac = float(args.eval_every_epochs)
        if not (0.0 < frac <= 1.0):
            raise SystemExit(
                f"--eval-every-epochs must be in (0, 1], got {frac}"
            )
        world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        eff_batch = (
            int(args.per_device_train_batch_size)
            * int(args.gradient_accumulation_steps)
            * world
        )
        steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_batch))
        every = max(1, int(round(steps_per_epoch * frac)))
        args.eval_steps = every
        args.save_steps = every
        if _is_main():
            print(
                f"[train] eval_every_epochs={frac} → steps_per_epoch={steps_per_epoch} "
                f"eval_steps=save_steps={every} "
                f"(eff_batch={eff_batch}, world={world})",
                flush=True,
            )

    eval_ds = None
    if val_jsonl is not None and val_jsonl.is_file() and args.eval_strategy != "no":
        eval_ds = make_dataset(
            val_jsonl,
            dataset_root=dataset_root,
            mode=mode,
            filter_cfg=filter_cfg,
            limit=int(args.val_limit or 0),
        )
        if len(eval_ds) == 0:
            if _is_main():
                print("[train] warning: empty val set; disabling eval", flush=True)
            eval_ds = None

    if _is_main():
        print(
            f"[train] mode={mode} n_train={len(train_ds)} "
            f"n_val={len(eval_ds) if eval_ds is not None else 0} out={out_dir}",
            flush=True,
        )
        if eval_jsonl is not None and eval_jsonl.is_file():
            print(
                f"[train] held-out eval jsonl present (not used in-loop): {eval_jsonl}",
                flush=True,
            )

    # GPU guard (gpu_guard/): take this rank's GPU over from the holder and keep it occupied while
    # the model loads on CPU; freed right before the Trainer moves the model onto the GPU.
    from gpu_guard import guard_client

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    guard_client.take_over(local_rank, 0.0, headroom_gib=2.0)  # the model loads on CPU first
    model, processor = _build_model_and_processor(args)
    fam = model_family(Path(args.model))
    collator = QwenVLDataCollator(processor, max_length=int(args.max_length),
                                  render_then_process=fam["render_then_process"],
                                  chat_kwargs=fam["chat_kwargs"])

    do_eval = eval_ds is not None
    eval_strategy = args.eval_strategy if do_eval else "no"
    load_best = bool(args.load_best_model_at_end) and do_eval and eval_strategy != "no"
    save_strategy = "epoch" if eval_strategy == "epoch" else "steps"
    if load_best and save_strategy == "steps":
        # HF requires save_steps to be a multiple of eval_steps.
        save_steps = int(args.save_steps)
        eval_steps = int(args.eval_steps)
        if save_steps % eval_steps != 0:
            raise SystemExit(
                f"save_steps ({save_steps}) must be a multiple of "
                f"eval_steps ({eval_steps}) when load_best_model_at_end"
            )

    ta_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        save_strategy=save_strategy,
        save_steps=int(args.save_steps),
        eval_strategy=eval_strategy,
        save_total_limit=int(args.save_total_limit),
        load_best_model_at_end=load_best,
        bf16=bool(args.bf16),
        remove_unused_columns=False,
        dataloader_num_workers=int(args.dataloader_num_workers),
        report_to="none",
        seed=int(args.seed),
        ddp_find_unused_parameters=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    if eval_strategy == "steps":
        ta_kwargs["eval_steps"] = int(args.eval_steps)
    if load_best:
        ta_kwargs["metric_for_best_model"] = "eval_loss"
        ta_kwargs["greater_is_better"] = False

    training_args = TrainingArguments(**ta_kwargs)

    guard_client.drop()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    if _is_main():
        save_processor(processor, out_dir, Path(args.model))
        print(f"[train] saved checkpoint → {out_dir}", flush=True)

        if args.lora and args.merge_lora:
            merge_dir = out_dir / "merged"
            merge_dir.mkdir(parents=True, exist_ok=True)
            print(f"[train] merging LoRA → {merge_dir}", flush=True)
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merge_dir))
            save_processor(processor, merge_dir, Path(args.model))
            print(f"[train] merged model saved → {merge_dir}", flush=True)

        done = {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out_dir),
            "n_train": len(train_ds),
            "n_val": len(eval_ds) if eval_ds is not None else 0,
            "eval_jsonl": str(eval_jsonl) if eval_jsonl else None,
            "mode": mode,
        }
        (out_dir / "train_done.json").write_text(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
