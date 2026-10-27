#!/usr/bin/env python3
"""Merge a Qwen3-VL LoRA run into its base model so vLLM can serve it.

Loads the base (from adapter_config.json, else Qwen3-VL-8B-Instruct) in bf16 on CPU, applies the
adapter, merge_and_unload(), and saves weights + the run's processor / chat template (the ones
used for training tokenization) to <run>/merged. Writes <run>/merged/MERGED_OK when complete.

  CUDA_VISIBLE_DEVICES= python qwen_finetune/merge_lora.py <run_dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_BASE = "models/Qwen3-VL-8B-Instruct"


def main() -> None:
    run = Path(sys.argv[1])
    out = run / "merged"
    if (out / "MERGED_OK").is_file():
        print(f"{out} already merged")
        return
    base = json.loads((run / "adapter_config.json").read_text()).get("base_model_name_or_path") or DEFAULT_BASE
    print(f"base={base} adapter={run}", flush=True)
    cls = Qwen3VLForConditionalGeneration
    if AutoConfig.from_pretrained(base, trust_remote_code=True).model_type != "qwen3_vl":
        cls = AutoModelForImageTextToText  # Qwen3.5, LLaVA-OneVision-2, ...
    model = cls.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(run)).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(str(run), trust_remote_code=True).save_pretrained(out)
    if (run / "chat_template.jinja").is_file():
        shutil.copy(run / "chat_template.jinja", out / "chat_template.jinja")
    for f in Path(base).glob("*.py"):  # remote-code models need their modelling files next to the weights
        shutil.copy(f, out / f.name)
    (out / "MERGED_OK").write_text(json.dumps({"base": base, "adapter": str(run)}) + "\n")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
