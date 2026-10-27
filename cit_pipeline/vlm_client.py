"""OpenAI-compatible client for the local Qwen3-VL-32B vLLM server.

Roles in CIT (locked)
---------------------
(a) Language: rephrase question text given category + GT facts
    (see ``qa.word_questions_with_vlm``).
(b) Ranking: reorder edit proposals for a clearer counterfactual
    (see ``rank_proposals`` / ``edit_search.find_edit_for_view``).

Not used (yet)
--------------
(c) Reject filter — deferred until manual failure analysis.

Default endpoint: ``http://127.0.0.1:19857/v1`` (a vLLM server, see README.md).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from openai import OpenAI

from cit_pipeline.config import VLM_API_KEY, VLM_BASE_URL, VLM_MODEL


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix or 'jpeg'}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VlmClient:
    """Thin wrapper around ``openai.OpenAI`` chat.completions."""

    def __init__(
        self,
        base_url: str = VLM_BASE_URL,
        model: str = VLM_MODEL,
        api_key: str = VLM_API_KEY,
        timeout: float = 600.0,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        text: str,
        images: Optional[Sequence[Union[Path, str]]] = None,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_meta: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> Any:
        """Send a multimodal (or text-only) chat request; return assistant text.

        If ``return_meta=True``, returns ``(text, meta)`` where meta includes
        ``finish_reason``.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images or []:
            p = Path(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(p)},
                }
            )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat_messages(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_meta=return_meta,
            enable_thinking=enable_thinking,
        )

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        continue_final_message: bool = False,
        return_meta: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> Any:
        """Low-level chat.completions; optionally continue the final assistant turn."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if temperature and temperature > 0:
            kwargs["top_p"] = top_p
        extra_body: dict[str, Any] = {}
        if continue_final_message:
            # vLLM / HF chat template: keep generating from the last assistant msg.
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
        think = self.enable_thinking if enable_thinking is None else enable_thinking
        if think is not None:
            ctk = dict(extra_body.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = bool(think)
            extra_body["chat_template_kwargs"] = ctk
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        # Some Thinking servers put the chain in reasoning_content and leave
        # content empty until after </think>.
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning and not text:
            text = str(reasoning).strip()
        elif reasoning and text and str(reasoning).strip() not in text:
            text = f"<think>\n{str(reasoning).strip()}\n</think>\n\n{text}"
        meta = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "reasoning_content": reasoning,
        }
        if return_meta:
            return text, meta
        return text

    def chat_text(self, text: str, **kwargs: Any) -> str:
        return self.chat(text, images=None, **kwargs)

    def rank_proposals(
        self,
        proposals: Sequence[dict[str, Any]],
        *,
        rgb0_path: Optional[Path] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Ask the VLM to rank edit proposals; return a reordered copy.

        On parse/API failure, falls back to the original order (best-effort).
        Does **not** drop proposals — only reorders (no reject filter).
        """
        if not proposals:
            return []
        if len(proposals) == 1:
            return list(proposals)

        lines = []
        for i, p in enumerate(proposals):
            lines.append(f"{i}: {json.dumps(p, default=str)}")
        prompt = (
            "You are helping select counterfactual edits for a vision eval dataset.\n"
            "Rank the following proposals from best to worst for a clear, obvious "
            "object move that changes spatial relations in the image.\n"
            "Prefer mid-distance furniture moves that stay in frame.\n"
            f"Return ONLY a JSON list of indices, length <= {top_k}, e.g. [2,0,1].\n\n"
            + "\n".join(lines)
        )
        images = [rgb0_path] if rgb0_path and rgb0_path.exists() else None
        try:
            raw = self.chat(prompt, images=images, max_tokens=128, temperature=0.0)
        except Exception:
            return list(proposals)[:top_k]

        order = _parse_index_list(raw, n=len(proposals))
        if not order:
            return list(proposals)[:top_k]
        ranked = [proposals[i] for i in order if 0 <= i < len(proposals)]
        seen = set(order)
        for i, p in enumerate(proposals):
            if i not in seen:
                ranked.append(p)
        return ranked[: max(top_k, 1)]


def _parse_index_list(text: str, n: int) -> list[int]:
    """Extract a JSON int list from model output; ignore out-of-range / dupes."""
    match = re.search(r"\[[^\]]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out: list[int] = []
    for x in data:
        if isinstance(x, int) and 0 <= x < n and x not in out:
            out.append(x)
    return out
