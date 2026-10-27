"""CIT MCQ → Qwen3-VL supervised finetune examples (modes 1A / 2A only)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from torch.utils.data import Dataset

_CIT = Path(__file__).resolve().parents[1]
if str(_CIT) not in sys.path:
    sys.path.insert(0, str(_CIT))

from cit_pipeline.benchmark_eval import (  # noqa: E402
    build_user_prompt,
    normalize_gt_answer,
    resolve_image_path,
)

try:
    from qwen_finetune.dataset_config import (  # noqa: E402
        DatasetFilterConfig,
        load_filtered_records,
    )
except ImportError:  # running as scripts inside qwen_finetune/
    from dataset_config import DatasetFilterConfig, load_filtered_records  # noqa: E402

MODE_TO_SETTING = {
    "1A": "before",
    "2A": "after",
    "1B": "before",  # before image + action, reason then <answer>X</answer>
}


@dataclass
class SFTExample:
    id: str
    mode: str
    category: str
    prompt: str
    answer: str
    image_path: Path


def build_sft_example(rec: dict[str, Any], *, mode: str, dataset_root: Path) -> SFTExample:
    """Build one supervised example for mode ``1A``, ``2A`` or ``1B``.

    1B uses the CoT prompt and supervises ``rec["cot_target"]`` (reasoning followed by
    ``<answer>X</answer>``; see prepare_procthor_v11_1B_splits.py).
    """
    mode = mode.upper()
    if mode not in MODE_TO_SETTING:
        raise ValueError(f"Unsupported mode={mode!r}; use 1A, 2A or 1B")
    setting = MODE_TO_SETTING[mode]
    prompt = build_user_prompt(
        rec,
        image_setting=setting,
        answer_mode="cot" if mode == "1B" else "direct",
        som=bool(rec.get("som")),
    )
    answer = normalize_gt_answer(rec.get("answer"))
    if not answer:
        raise ValueError(f"missing letter answer for {rec.get('id')}")
    if mode == "1B":
        answer = str(rec.get("cot_target") or "")
        if "<answer>" not in answer:
            raise ValueError(f"mode 1B needs cot_target for {rec.get('id')}")
    img = resolve_image_path(rec, dataset_root=dataset_root, image_setting=setting)
    if img is None:
        raise ValueError(f"mode {mode} requires an image for {rec.get('id')}")
    return SFTExample(
        id=str(rec.get("id")),
        mode=mode,
        category=str(rec.get("category") or ""),
        prompt=prompt,
        answer=answer,
        image_path=img,
    )


class CitVLSftDataset(Dataset):
    """Lazy image load; tokenization happens in the collator."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        mode: str,
        dataset_root: Path,
    ) -> None:
        self.mode = mode.upper()
        self.dataset_root = Path(dataset_root)
        self.examples = [
            build_sft_example(r, mode=self.mode, dataset_root=self.dataset_root)
            for r in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return {
            "id": ex.id,
            "mode": ex.mode,
            "category": ex.category,
            "prompt": ex.prompt,
            "answer": ex.answer,
            "image": image,
            "image_path": str(ex.image_path),
        }


def make_dataset(
    jsonl_path: Path,
    *,
    dataset_root: Path,
    mode: str,
    filter_cfg: Optional[DatasetFilterConfig] = None,
    limit: int = 0,
) -> CitVLSftDataset:
    cfg = filter_cfg or DatasetFilterConfig()
    rows = load_filtered_records(
        jsonl_path, dataset_root=dataset_root, cfg=cfg, limit=limit
    )
    return CitVLSftDataset(rows, mode=mode, dataset_root=dataset_root)


class QwenVLDataCollator:
    """Pad a batch; supervise only assistant answer tokens."""

    def __init__(self, processor: Any, *, max_length: int = 2048, render_then_process: bool = False,
                 chat_kwargs: Optional[dict[str, Any]] = None) -> None:
        """``render_then_process``: render the chat template to text, then run the processor on
        (text, image). Needed for processors whose ``apply_chat_template(tokenize=True)`` drops PIL
        images (LLaVA-OneVision-2). ``chat_kwargs`` go to the chat template (Qwen3.5:
        ``enable_thinking=False``)."""
        self.processor = processor
        self.max_length = int(max_length)
        self.render_then_process = bool(render_then_process)
        self.chat_kwargs = dict(chat_kwargs or {})

    def _tokenize(self, msgs: list[dict[str, Any]], image: Any, *, generation_prompt: bool) -> dict[str, Any]:
        if self.render_then_process:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=generation_prompt, **self.chat_kwargs)
            return self.processor(text=[text], images=[image], return_tensors="pt")
        return self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=generation_prompt, return_dict=True,
            return_tensors="pt", **self.chat_kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_inputs: list[dict[str, Any]] = []
        prompt_lens: list[int] = []

        for feat in features:
            image_part = {"type": "image"} if self.render_then_process else {"type": "image", "image": feat["image"]}
            user_msg = {
                "role": "user",
                "content": [
                    image_part,
                    {"type": "text", "text": feat["prompt"]},
                ],
            }
            full_msgs = [
                user_msg,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": feat["answer"]}],
                },
            ]
            prompt_tok = self._tokenize([user_msg], feat["image"], generation_prompt=True)
            full_tok = self._tokenize(full_msgs, feat["image"], generation_prompt=False)
            prompt_lens.append(int(prompt_tok["input_ids"].shape[-1]))
            # Squeeze batch dim from apply_chat_template.
            item = {k: v.squeeze(0) for k, v in full_tok.items()}
            batch_inputs.append(item)

        # Pad text fields.
        input_ids = [b["input_ids"] for b in batch_inputs]
        attn = [b["attention_mask"] for b in batch_inputs]
        pad_id = int(getattr(self.processor.tokenizer, "pad_token_id", 0) or 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn, batch_first=True, padding_value=0
        )
        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            cut = min(plen, labels.shape[-1])
            labels[i, :cut] = -100
        labels[attention_mask == 0] = -100

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Vision tensors: concat patches / stack per-image grid metadata.
        if "pixel_values" in batch_inputs[0]:
            out["pixel_values"] = torch.cat(
                [b["pixel_values"] for b in batch_inputs], dim=0
            )
        if "image_grid_thw" in batch_inputs[0]:
            grids = []
            for b in batch_inputs:
                g = b["image_grid_thw"]
                # apply_chat_template may yield (3,) or (n_img, 3); model wants (N, 3).
                if g.ndim == 1:
                    g = g.unsqueeze(0)
                grids.append(g)
            out["image_grid_thw"] = torch.cat(grids, dim=0)
        for key in ("patch_positions",):  # extra per-patch vision tensors (LLaVA-OneVision-2)
            if key in batch_inputs[0]:
                out[key] = torch.cat([b[key] for b in batch_inputs], dim=0)
        if "mm_token_type_ids" in batch_inputs[0]:
            mm = [b["mm_token_type_ids"] for b in batch_inputs]
            out["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm, batch_first=True, padding_value=0
            )
            if out["mm_token_type_ids"].shape[-1] > self.max_length:
                out["mm_token_type_ids"] = out["mm_token_type_ids"][
                    :, : self.max_length
                ]
        return out
