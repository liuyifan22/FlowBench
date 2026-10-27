"""Formal zero-shot prompt format (2026-10-03) and per-seed answer-choice permutations.

``build_formal_prompt`` is the prompt for all non-finetuned models (finetuned Qwen keeps the
training prompt, ``benchmark_eval.build_user_prompt``). Blocks are separated by blank lines:

  <SoM note>                        (only for rows with colour marks)
  <body>                            1A sim:  The image shows the scene before executing the following action:
                                             "<Motion>"  + blank line + <question>
                                    1A real: <command> <question>   (FlowBench's own move text, scale sentence first)
                                    2A sim:  <static question>
                                    2A real: <scale sentence> <static question>
  Options:
  (A) ...
  <answer line>                     "Your answer should be a single letter among A, B, ..." (letters of this
                                    question), or the pick-three instruction for closest_three.

Real (FlowBench) rows carry ``formal_body_before`` / ``formal_body_after``; sim rows are built from
``motion_command`` / ``question`` / ``static_question``.

``permute_for_seed(records, seed)``: seed 0 returns the records unchanged (already balanced);
other seeds re-permute every question's choices with a balanced letter assignment per category
(and per yes/no polarity), using reshuffle_v11_answer_letters.plan / apply_perm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from cit_pipeline.benchmark_eval import (
    choices_from_record,
    som_note_block,
    staticify_contact_wall_choices,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _answer_line(rec: dict[str, Any], letters: list[str]) -> str:
    if len(letters) > 2 and str(rec.get("mcq_type") or "") == "mcq6_select3":
        return ("Select exactly three options. Your answer should be the three letters "
                "concatenated in alphabetical order (e.g. ACF).")
    return f"Your answer should be a single letter among {', '.join(letters)}."


DIRECT_HINT = "Please directly give the index of the answer."


def build_formal_prompt(rec: dict[str, Any], *, image_setting: str, direct_hint: bool = False) -> str:
    before = image_setting == "before"
    if not before and str(rec.get("category") or "") == "contact_wall":
        rec = dict(rec)
        staticify_contact_wall_choices(rec)  # 2A: plain Yes / No, as in the CIT prompt
    choices = choices_from_record(rec)
    note = som_note_block(rec).strip()
    body = rec.get("formal_body_before" if before else "formal_body_after")
    if not body:
        if before:
            motion = str(rec.get("motion_command") or "").strip()
            motion = motion[:1].upper() + motion[1:]
            body = ("The image shows the scene before executing the following action:\n"
                    f'"{motion}"\n\n{str(rec.get("question") or "").strip()}')
        else:
            body = str(rec.get("static_question") or "").strip()
    options = "Options:\n" + "\n".join(f"({L}) {t}" for L, t in choices)
    answer = _answer_line(rec, [L for L, _ in choices])
    if direct_hint:  # optional: re-add the CIT prompt's last line if models do not answer directly
        answer = f"{answer}\n{DIRECT_HINT}"
    blocks = ([note] if note else []) + [body, options, answer]
    return "\n\n".join(blocks)


def permute_for_seed(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if int(seed) == 0:
        return records
    from reshuffle_v11_answer_letters import apply_perm, plan

    perms = plan(records, None, int(seed))
    return [apply_perm(r, perms[r["id"]]) if r["id"] in perms else r for r in records]
