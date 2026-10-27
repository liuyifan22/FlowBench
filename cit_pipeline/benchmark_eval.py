"""Prompting + answer extraction for CIT MCQ benchmark evals.

Two orthogonal axes (2×2), plus a diagnostic ablation:

* ``image_setting``
  - ``before``: rgb0 + motion command (imagine the outcome)
  - ``after``: rgb1 only (image already shows the outcome)
  - ``fake2a`` / ``fake_after``: same *prompt* as ``after`` (static question,
    no motion text), but feed **rgb0** instead of rgb1 — checks whether 2A
    gains come from the after-image or from question/answer leakage.
* ``answer_mode``
  - ``direct``: short letter answer (max 16 tokens)
  - ``cot``: free reasoning then ``<answer>…</answer>`` (max 512 tokens)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

LETTER_RE = re.compile(r"\b([A-F]{1,3})\b", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(
    r"<answer>\s*([A-Fa-f]{1,6})\s*</answer>", re.IGNORECASE | re.DOTALL
)
# Fallback if the model drops tags but still ends with a letter key.
TRAILING_LETTER_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*|final\s*answer\s*(?:is|:)?\s*)([A-F]{1,3})\s*\.?\s*$",
    re.IGNORECASE,
)
# Qwen3 Thinking models emit reasoning inside <think>...</think>.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# Injected near the token budget so the model closes reasoning and emits <answer>.
# Close an open <think> first (Qwen thinking-budget style), then prime <answer>.
FORCE_ANSWER_NUDGE = (
    "\n</think>\n\nWait, I need to output an answer right now. <answer>"
)
OPEN_ANSWER_RE = re.compile(r"<answer>\s*$", re.IGNORECASE)
HAS_OPEN_ANSWER_RE = re.compile(r"<answer>(?![\s\S]*</answer>)", re.IGNORECASE)
OPEN_THINK_RE = re.compile(
    r"<think>(?:(?!</think>).)*$", re.IGNORECASE | re.DOTALL
)


def choices_from_record(rec: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ordered ``(letter, text)`` pairs present on the record."""
    out: list[tuple[str, str]] = []
    for letter in "ABCDEF":
        key = f"choice_{letter}"
        if key in rec and rec[key] is not None:
            out.append((letter, str(rec[key])))
    return out


def contact_wall_choice_polarity(text: str) -> Optional[str]:
    """Map a contact_wall MCQ option to ``\"yes\"`` / ``\"no\"``, else None."""
    t = str(text or "").strip().lower()
    if not t:
        return None
    if re.search(r":\s*yes\.?\s*$", t) or t.startswith("yes"):
        return "yes"
    if re.search(r":\s*no\.?\s*$", t) or t.startswith("no"):
        return "no"
    yes_hints = (
        "touching a wall",
        "against / touching",
        "against/touching",
        "ends up in contact",
        "in contact with (or blocked",
        "contacting a wall",
    )
    no_hints = (
        "clearance",
        "stays clear",
        "clear of the wall",
        "does not end up",
        "not contacting",
        "not in contact",
    )
    if any(h in t for h in yes_hints):
        return "yes"
    if any(h in t for h in no_hints):
        return "no"
    return None


def staticify_contact_wall_choices(
    rec: dict[str, Any],
) -> list[tuple[str, str]]:
    """Rewrite contact_wall options to plain Yes/No for static / 2A prompts.

    Preserves letter polarity (which letter is Yes vs No). No-ops for other
    categories or when polarity cannot be recovered for every choice.
    """
    if str(rec.get("category") or "") != "contact_wall":
        return choices_from_record(rec)
    choices = choices_from_record(rec)
    if len(choices) < 2:
        return choices
    polarities = [contact_wall_choice_polarity(t) for _, t in choices]
    if any(p is None for p in polarities):
        return choices
    if set(polarities) != {"yes", "no"}:
        return choices
    rewritten: list[tuple[str, str]] = []
    for (letter, _old), pol in zip(choices, polarities):
        new_text = "Yes" if pol == "yes" else "No"
        rec[f"choice_{letter}"] = new_text
        rewritten.append((letter, new_text))
    return rewritten


def format_options_block(choices: list[tuple[str, str]]) -> str:
    return "\n".join(f"({L}) {text}" for L, text in choices)


def som_note_block(rec: dict[str, Any], *, som: Optional[bool] = None) -> str:
    """Short instruction telling the model how to read Set-of-Marks overlays."""
    use = bool(rec.get("som")) if som is None else bool(som)
    if not use:
        return ""
    if rec.get("som_style") == "color":
        # v11 colour SoM: text-free discs, referred to as "(marked <colour>)".
        return (
            "Set-of-Marks (SoM): solid coloured circular marks with a thin white rim are "
            "overlaid on the image near the mentioned objects. Phrases in the question such "
            "as (marked blue) or (marked red) refer to the object carrying the mark of that "
            "colour.\n\n"
        )
    return (
        "Set-of-Marks (SoM): red circular badges with white labels are overlaid "
        "on the image with a bottom-left offset from the mentioned object. "
        "Parenthesized tokens in the question such as (1), (2), (A), (B), … "
        "refer to those badges.\n\n"
    )


def build_user_prompt(
    rec: dict[str, Any],
    *,
    image_setting: str,
    answer_mode: str,
    thinking: bool = False,
    som: Optional[bool] = None,
) -> str:
    """Polish the MCQ stem + instruction for the given 2×2 cell.

    * Setting 1 (before): before-image framing + quoted ``motion_command``,
      then simulate-then-evaluate using ``question`` + options + rgb0.
    * Setting 2 (after): ``static_question`` + options + rgb1 only
      (no motion command / after-motion framing).
    * ``fake2a``: same prompt as setting 2, but paired with rgb0 (see
      ``resolve_image_path``).
    * ``text``: same stem as setting 2, but **no image** (bias / chance probe).

    When ``thinking=True`` (Qwen3-*-Thinking), do **not** ask the model to
    narrate CoT in the user prompt — it already reasons in ``<think>`` blocks.
    Just require a final ``<answer>…</answer>`` letter.

    ``som``: if True, prepend a Set-of-Marks grounding note. If None, follow
    the record's ``som`` flag (ithor_v3/som rows set this). Ignored for
    ``text`` (no image to ground).
    """
    image_setting = _normalize_image_setting(image_setting)
    answer_mode = _normalize_answer_mode(answer_mode)
    # Static / after-image prompts should not mention "after the move" in Yes/No options.
    if image_setting != "before" and str(rec.get("category") or "") == "contact_wall":
        rec = dict(rec)
        staticify_contact_wall_choices(rec)
    choices = choices_from_record(rec)
    n = len(choices)
    multi = n > 2 and str(rec.get("mcq_type") or "") == "mcq6_select3"
    # Text-only: never attach SoM badge instructions.
    som_prefix = "" if image_setting == "text" else som_note_block(rec, som=som)

    options = format_options_block(choices)
    motion = str(rec.get("motion_command") or "").strip()

    if multi:
        answer_shape = (
            "Select exactly three options. Your answer should be the three "
            "letters concatenated in alphabetical order (e.g. ACF)."
        )
    elif n == 2:
        answer_shape = "Your answer should be a single letter: A or B."
    else:
        answer_shape = f"Your answer should be a single letter among {', '.join(L for L, _ in choices)}."

    if answer_mode == "direct":
        instruction = (
            f"{answer_shape}\n"
            "Please directly give the index of the answer. "
        )
    elif thinking:
        # Thinking models already allocate tokens to <think>…</think>.
        instruction = (
            f"{answer_shape}\n"
            "After you finish thinking, put only the final answer inside tags "
            "like <answer>A</answer> (or <answer>ACF</answer> for a three-letter "
            "answer)."
        )
    else:
        about = (
            "the question"
            if image_setting == "text"
            else "the given image and the question"
        )
        instruction = (
            f"{answer_shape}\n"
            f"Please reason about {about} step by step. "
            "After your reasoning, put the final answer inside tags like "
            "<answer>A</answer> (or <answer>ACF</answer> for a three-letter answer)."
        )

    if image_setting == "before":
        # 1A: full ``question`` after simulate-then-evaluate framing.
        # Image is a separate multimodal part before this text
        # (see qwen_finetune.data); keep wording as "The image …".
        question = str(rec.get("question") or "").strip()
        motion_disp = motion[:1].upper() + motion[1:] if motion else motion
        return (
            f"{som_prefix}"
            "The image shows the scene before executing the following action:\n"
            f'"{motion_disp}"\n\n'
            "First simulate this action. Then, in the resulting scene, evaluate:\n"
            f"{question}\n\n"
            f"Options:\n{options}\n\n"
            f"{instruction}"
        )

    # Setting 2 / fake2a / text: static VQA stem (text omits the image).
    static = str(rec.get("static_question") or "").strip()
    if not static:
        from cit_pipeline.mcq_export import derive_static_question

        static = derive_static_question(str(rec.get("question") or ""))
    if image_setting == "text":
        return (
            f"Question: {static}\n\n"
            f"Options:\n{options}\n\n"
            f"{instruction}"
        )
    return (
        f"{som_prefix}"
        f"Question: {static}\n\n"
        f"Options:\n{options}\n\n"
        f"{instruction}"
    )


def resolve_image_path(
    rec: dict[str, Any],
    *,
    dataset_root: Path,
    image_setting: str,
) -> Optional[Path]:
    """Resolve rgb0/rgb1 path for the cell, or ``None`` for text-only."""
    image_setting = _normalize_image_setting(image_setting)
    if image_setting == "text":
        return None
    # before + fake2a → rgb0; after → rgb1
    use_before = image_setting in {"before", "fake2a"}
    rel = rec.get("image_before") if use_before else rec.get("image_after")
    if not rel:
        raise ValueError(f"Missing image path for setting={image_setting}: {rec.get('id')}")
    p = Path(str(rel))
    if not p.is_file():
        p = dataset_root / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def normalize_gt_answer(answer: Any) -> str:
    s = str(answer or "").strip().upper()
    s = re.sub(r"[^A-F]", "", s)
    if len(s) > 1:
        s = "".join(sorted(set(s)))
    return s


def extract_pred_answer(
    text: str,
    *,
    answer_mode: str,
    thinking: bool = False,
) -> Optional[str]:
    """Parse model output into a normalized letter key (or None).

    For B-series (``cot`` / ``thinking``): require a complete
    ``<answer>…</answer>`` tag. No letter-heuristic fallbacks — missing tags
    are parse failures.
    """
    if not text:
        return None
    answer_mode = _normalize_answer_mode(answer_mode)
    raw = text.strip()

    if answer_mode == "cot" or thinking:
        m = ANSWER_TAG_RE.search(raw)
        if m:
            return normalize_gt_answer(m.group(1))
        return None

    # Prefer content after the thinking block for direct-mode leftovers.
    if "<think>" in raw.lower() or "</think>" in raw.lower():
        stripped = THINK_BLOCK_RE.sub("", raw).strip()
        if stripped:
            raw = stripped

    # direct: prefer a bare letter line, else first letter token
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Fa-f]{1,3}", line):
            return normalize_gt_answer(line)
    m = LETTER_RE.search(raw)
    if m:
        return normalize_gt_answer(m.group(1))
    return None


def max_new_tokens_for_mode(
    answer_mode: str,
    *,
    category: Optional[str] = None,
    thinking: bool = False,
) -> int:
    """Token budget for generation.

    Direct answers stay short (16). Instruct CoT defaults to 512, but
    ``closest_three`` often needs a longer chain → 1024.

    Thinking models spend most tokens inside ``<think>``; give a larger budget
    (2048 default, 4096 for closest_three).
    """
    if _normalize_answer_mode(answer_mode) == "direct" and not thinking:
        return 16
    cat = str(category or "").strip().lower()
    if thinking:
        if cat in {"closest_three", "closest-three", "closest3"}:
            return 4096
        return 2048
    if cat in {"closest_three", "closest-three", "closest3"}:
        return 1024
    return 512


def force_answer_at_tokens(
    *,
    thinking: bool,
    category: Optional[str] = None,
) -> Optional[int]:
    """Token count at which Thinking runs inject the force-answer nudge.

    Qwen's cloud ``thinking_budget`` can end ``<think>``, but it does not know
    about our ``<answer>…</answer>`` protocol. For Thinking evals we therefore
    truncate the free-reasoning phase early (500 / 1010) and continue with
    :data:`FORCE_ANSWER_NUDGE` so a letter is emitted before the hard cap.
    """
    if not thinking:
        return None
    cat = str(category or "").strip().lower()
    if cat in {"closest_three", "closest-three", "closest3"}:
        return 1010
    return 500


def has_complete_answer_tag(text: str) -> bool:
    return bool(ANSWER_TAG_RE.search(text or ""))


def build_force_answer_continuation(partial: str) -> str:
    """Close an open ``<think>`` (if any) and append the force-answer nudge.

    Mirrors Qwen's official thinking-budget continuation pattern, but primes
    our ``<answer>`` protocol instead of a free-form final reply.
    """
    p = (partial or "").rstrip()
    if has_complete_answer_tag(p):
        return p
    if OPEN_ANSWER_RE.search(p) or HAS_OPEN_ANSWER_RE.search(p):
        return p
    # If still inside <think>…, close it before asking for <answer>.
    if OPEN_THINK_RE.search(p) or (
        "<think>" in p.lower() and "</think>" not in p.lower()
    ):
        if not p.lower().rstrip().endswith("</think>"):
            p = p + "\n</think>"
        return p + "\n\nWait, I need to output an answer right now. <answer>"
    # No think markers (or already closed): still inject the nudge.
    return p + FORCE_ANSWER_NUDGE if p else FORCE_ANSWER_NUDGE.lstrip()


def format_category_stats_table(
    per_category: dict[str, dict[str, Any]],
    *,
    n_total: int,
    n_correct: int,
) -> str:
    """Human-readable category-wise accuracy table for logs / summaries."""
    lines = [
        f"{'category':<24} {'n':>5} {'correct':>8} {'acc':>7} {'parse_fail':>10}",
        "-" * 58,
    ]
    for cat in sorted(per_category.keys()):
        row = per_category[cat]
        n = int(row.get("n") or 0)
        cor = int(row.get("n_correct") or 0)
        pf = int(row.get("n_parse_fail") or 0)
        acc = (cor / n) if n else 0.0
        lines.append(f"{cat:<24} {n:>5} {cor:>8} {acc:>7.3f} {pf:>10}")
    lines.append("-" * 58)
    tot_acc = (n_correct / n_total) if n_total else 0.0
    lines.append(
        f"{'TOTAL':<24} {n_total:>5} {n_correct:>8} {tot_acc:>7.3f}"
    )
    return "\n".join(lines)


def _normalize_image_setting(value: str) -> str:
    v = str(value).strip().lower().replace("-", "").replace("_", "")
    if v in {"1", "before", "rgb0", "pre", "motion"}:
        return "before"
    if v in {"2", "after", "rgb1", "post", "outcome"}:
        return "after"
    # Ablation: 2A-style static prompt + before image (rgb0).
    if v in {"fake2a", "fakeafter", "fake2", "fakea"}:
        return "fake2a"
    # Ablation: 2A-style static prompt, no image (letter-bias / chance probe).
    if v in {"text", "textonly", "noimage", "0", "none"}:
        return "text"
    raise ValueError(
        f"Unknown image_setting={value!r} "
        f"(use before/after/fake2a/text or 1/2/fake2a/text)"
    )


def _normalize_answer_mode(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"a", "direct", "short", "no_cot", "nocot"}:
        return "direct"
    if v in {"b", "cot", "reason", "reasoning"}:
        return "cot"
    raise ValueError(f"Unknown answer_mode={value!r} (use direct/cot or A/B)")


def setting_tag(image_setting: str, answer_mode: str) -> str:
    """Compact label like ``1A`` / ``2B`` / ``fake2A`` / ``textA``."""
    setting = _normalize_image_setting(image_setting)
    ans = "A" if _normalize_answer_mode(answer_mode) == "direct" else "B"
    if setting == "fake2a":
        return f"fake2{ans}"
    if setting == "text":
        return f"text{ans}"
    img = "1" if setting == "before" else "2"
    return f"{img}{ans}"
