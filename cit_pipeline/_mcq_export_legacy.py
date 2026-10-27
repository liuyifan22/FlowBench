"""Convert CIT eval questions into MCQ training records.

Keeps the existing ``sample.json`` / PNG layout for debug + visualization;
this module only builds flat MCQ dicts for ``dataset.jsonl``.
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Any, Optional

LETTERS4 = ("A", "B", "C", "D")
LETTERS6 = ("A", "B", "C", "D", "E", "F")


class AnswerFlipQuota:
    """Track train/bench quota for keeping non-flipped (identical) answers.

    ``keep_identical_frac`` is the target fraction of yes/no rows whose
    before/after label is unchanged. Call ``allow_identical(rng)`` before
    emitting such a row; flipped rows are always allowed.
    """

    def __init__(self, keep_identical_frac: float = 0.5) -> None:
        self.keep_identical_frac = float(keep_identical_frac)
        self.n_identical = 0
        self.n_flipped = 0

    def allow_identical(self, rng: random.Random) -> bool:
        total = self.n_identical + self.n_flipped
        if total <= 0:
            return bool(rng.random() < self.keep_identical_frac)
        cur = self.n_identical / float(total)
        # Soft: if below target, prefer identical; if above, reject more often.
        if cur < self.keep_identical_frac:
            return True
        # Probability of still accepting scales down as we exceed the target.
        over = cur - self.keep_identical_frac
        return bool(rng.random() < max(0.05, 0.5 - over))

    def observe(self, *, identical: bool) -> None:
        if identical:
            self.n_identical += 1
        else:
            self.n_flipped += 1


def _ensure_question(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "?"
    if t[0].islower():
        t = t[0].upper() + t[1:]
    if not t.endswith("?"):
        t = t.rstrip(".") + "?"
    return t


def build_static_question(
    *,
    category: str,
    question: str,
    gt_meta: Optional[dict[str, Any]] = None,
    som: bool = False,
    **_kwargs: Any,
) -> str:
    """Faithful static VQA stem for setting-2 (rgb1 only) — Modification 11.

    Does **not** naively chop the clause after a comma (that drops the mover
    name). Rebuilds a complete question from ``gt_meta`` referring phrases.
    """
    del som  # stem rebuild is phrase-based; SoM marks handled by caller when needed.
    meta = dict(gt_meta or {})
    mover = (
        meta.get("mover_phrase")
        or (meta.get("mover_referring") or {}).get("phrase")
        or meta.get("seat_phrase")
    )
    cat = str(category or "")

    if cat == "closer_to":
        if mover:
            return _ensure_question(f"Which of the following is {mover} closer to")
    elif cat == "closest_three":
        if mover:
            return _ensure_question(
                f"Which three objects are closest to {mover}"
            )
    elif cat == "contact_wall":
        if mover:
            return _ensure_question(f"Is {mover} in contact with a wall")
    elif cat == "support":
        if mover:
            return _ensure_question(
                f"Will {mover} drop onto the floor instead of staying "
                f"well-supported by furniture"
            )
    elif cat == "occlusion":
        occludee = meta.get("occludee_phrase")
        if occludee and mover:
            thr = meta.get("threshold")
            pct = int(float(thr) * 100) if thr is not None else 30
            return _ensure_question(
                f"Is {occludee} substantially occluded by {mover} "
                f"(at least {pct}% of its visible pixels)"
            )
        if occludee:
            return _ensure_question(f"Is {occludee} substantially occluded")
    elif cat == "left_right_landmark":
        land = meta.get("landmark_phrase")
        asked = str(meta.get("asked") or "left")
        if mover and land:
            return _ensure_question(
                f"Is {mover} to the {asked} of {land} from the camera's viewpoint"
            )
    elif cat == "feasibility":
        direction = meta.get("probe_direction")
        meters = meta.get("probe_meters")
        if mover and direction is not None and meters is not None:
            return _ensure_question(
                f"Would moving {mover} {float(meters):.2f}m {direction} "
                f"be feasible without colliding"
            )
    elif cat == "affordance":
        seat = meta.get("seat_phrase") or mover
        tgt = meta.get("target_phrase")
        if seat and tgt:
            return _ensure_question(
                f"When sitting on {seat}, is {tgt} within your forward view "
                f"(within ±45° of straight ahead)"
            )
        if seat:
            return _ensure_question(
                f"When sitting on {seat}, is the named landmark within your "
                f"forward view (within ±45° of straight ahead)"
            )

    # Fallback: strip temporal framing but keep the rest of the sentence intact.
    return derive_static_question_fallback(question)


def derive_static_question_fallback(question: str) -> str:
    """Legacy stripper — last resort when gt_meta lacks referring phrases."""
    t = (question or "").strip()
    if not t:
        return "?"
    for pat in (
        r"^In the after image,\s*",
        r"^Starting from the after image,\s*",
        r"^After the move,\s*",
        r"^After moving .+?,\s*",
        r"^After the .+? is moved .+?,\s*",
        r"^After .+? is moved .+?,\s*",
    ):
        newt = re.sub(pat, "", t, count=1, flags=re.IGNORECASE)
        if newt != t:
            t = newt
            break
    t = re.sub(r"\s+after it moved[^?]*\?", "?", t, flags=re.IGNORECASE)
    return _ensure_question(t)


# Back-compat alias used by patch_static_question.py / older imports.
def derive_static_question(question: str, **kwargs: Any) -> str:
    if kwargs.get("gt_meta") is not None or kwargs.get("category"):
        return build_static_question(
            category=str(kwargs.get("category") or ""),
            question=question,
            gt_meta=kwargs.get("gt_meta"),
        )
    return derive_static_question_fallback(question)

# (yes_text, no_text) — yes matches a True GT for that category's question.
# Support GT True = drops to floor; False = stays furniture-supported.
_YESNO_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "support": [
        (
            "Yes, it will drop to the floor.",
            "No, it is well-supported by some furniture.",
        ),
        (
            "Yes — it drops onto the floor.",
            "No — it remains well-supported by furniture.",
        ),
        (
            "It will drop to the floor (not furniture-supported).",
            "It is well-supported by some furniture.",
        ),
        (
            "Yes, after the move it falls to the floor.",
            "No, it stays well-supported by furniture.",
        ),
        (
            "No longer supported; it ends up on the floor.",
            "Still well-supported by some furniture after the move.",
        ),
    ],
    "occlusion": [
        (
            "Yes, it is substantially occluded after the move.",
            "No, it is not substantially occluded after the move.",
        ),
        (
            "Yes — the moved object covers it enough to count as occluded.",
            "No — it remains sufficiently visible (not meaningfully occluded).",
        ),
        (
            "Yes, occlusion reaches the stated threshold.",
            "No, occlusion stays below the stated threshold.",
        ),
        (
            "Occluded: yes.",
            "Occluded: no.",
        ),
    ],
    "contact_wall": [
        (
            "Yes, it ends up in contact with (or blocked by) a wall.",
            "No, it does not end up contacting a wall.",
        ),
        (
            "Yes — the object is against / touching a wall after the move.",
            "No — it stays clear of the walls.",
        ),
        (
            "Yes, it is touching a wall after the move.",
            "No, there is still clearance from the walls.",
        ),
        (
            "Wall contact after the move: yes.",
            "Wall contact after the move: no.",
        ),
    ],
    "affordance": [
        (
            "Yes, it is within the forward sitting view (±45°).",
            "No, it is outside the forward sitting view (±45°).",
        ),
        (
            "Yes — from that seat it lies straight ahead within ±45°.",
            "No — from that seat it is not within ±45° ahead.",
        ),
        (
            "Yes, sitting there keeps it in the forward FOV.",
            "No, sitting there does not keep it in the forward FOV.",
        ),
        (
            "Within sitting FOV: yes.",
            "Within sitting FOV: no.",
        ),
    ],
    "left_right_landmark": [
        (
            "Yes, that left/right relation holds in the after image.",
            "No, that left/right relation does not hold.",
        ),
        (
            "Yes — from the camera, the stated side relation is true.",
            "No — from the camera, the stated side relation is false.",
        ),
        (
            "True for the after image.",
            "False for the after image.",
        ),
        ("Yes.", "No."),
    ],
    "feasibility": [
        (
            "Yes, that further move looks physically feasible.",
            "No, that further move looks blocked / infeasible.",
        ),
        (
            "Yes — the probed displacement can succeed without colliding.",
            "No — the probed displacement would collide or fail.",
        ),
        (
            "Yes, the object can make that additional translation.",
            "No, something blocks that additional translation.",
        ),
        (
            "Feasible: yes.",
            "Feasible: no.",
        ),
    ],
    "_default": [
        ("Yes.", "No."),
        ("Yes, that is correct.", "No, that is not correct."),
    ],
}


def _strip_embedded_choices(text: str) -> str:
    """Remove trailing '(A) …' / 'Choose three from:' dumps from stems."""
    t = text.strip()
    t = re.sub(r"\s*\(A\)\s.*$", "", t, flags=re.DOTALL)
    t = re.sub(r"\s*Choose three from:.*$", "", t, flags=re.IGNORECASE | re.DOTALL)
    t = t.strip().rstrip("?")
    return t + "?" if t else "?"


def _choice_fields(choices: dict[str, str]) -> dict[str, str]:
    return {f"choice_{letter}": text for letter, text in choices.items()}


def _append_final_answer(cot: Optional[str], answer: str) -> Optional[str]:
    """Rewrite trailing Answer lines so CoT matches the MCQ letter key."""
    if cot is None:
        return None
    body = cot.rstrip()
    # Drop legacy yes/no / phrase Answer lines; MCQ key is authoritative.
    body = re.sub(
        r"\nAnswer:\s*.*$",
        "",
        body,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    body = re.sub(
        r"\nFinal answer:\s*.*$",
        "",
        body,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # closer_to cold CoT may say "(letter A)" from the pre-shuffle sample.
    body = re.sub(
        r"\(letter\s+[A-F]\)",
        f"(letter {answer[0] if answer else answer})",
        body,
        flags=re.IGNORECASE,
    )
    return body.rstrip() + f"\nAnswer: {answer}"


def _yesno_mcq(
    category: str,
    question_text: str,
    answer_yes: bool,
    rng: random.Random,
) -> tuple[str, dict[str, str], str]:
    variants = _YESNO_VARIANTS.get(category) or _YESNO_VARIANTS["_default"]
    yes_text, no_text = rng.choice(variants)
    if rng.random() < 0.5:
        choices = {"A": yes_text, "B": no_text}
        letter = "A" if answer_yes else "B"
    else:
        choices = {"A": no_text, "B": yes_text}
        letter = "B" if answer_yes else "A"
    return _strip_embedded_choices(question_text), choices, letter


def _mcq4_from_phrases(
    question_text: str,
    choices: list[str],
    correct_phrase: str,
    rng: random.Random,
) -> tuple[str, dict[str, str], str]:
    opts = list(choices)
    if correct_phrase not in opts:
        opts = [correct_phrase] + [c for c in opts if c != correct_phrase]
    opts = opts[:4]
    while len(opts) < 4:
        opts.append(f"the other object {len(opts)}")
    for _ in range(4):
        rng.shuffle(opts)
        if opts[0] != correct_phrase or rng.random() < 0.35:
            break
    if opts[0] == correct_phrase and rng.random() < 0.65:
        j = rng.randint(1, 3)
        opts[0], opts[j] = opts[j], opts[0]
    letter_map = {L: t for L, t in zip(LETTERS4, opts)}
    answer = next(L for L, t in letter_map.items() if t == correct_phrase)
    return _strip_embedded_choices(question_text), letter_map, answer


def _mcq6_select3(
    question_text: str,
    pool: list[str],
    answer_phrases: list[str],
    rng: random.Random,
) -> tuple[str, dict[str, str], str]:
    opts = list(pool)
    for a in answer_phrases:
        if a not in opts:
            opts.append(a)
    opts = opts[:6]
    while len(opts) < 6:
        opts.append(f"the other object {len(opts)}")
    rng.shuffle(opts)
    letter_map = {L: t for L, t in zip(LETTERS6, opts)}
    phrase_to_letter = {t: L for L, t in letter_map.items()}
    letters = [phrase_to_letter[p] for p in answer_phrases if p in phrase_to_letter]
    while len(letters) < 3:
        for L in LETTERS6:
            if L not in letters:
                letters.append(L)
            if len(letters) >= 3:
                break
    # Stable three-letter key, e.g. "ACD" (sorted so GT is not order-dependent).
    answer = "".join(sorted(letters[:3]))
    stem = _strip_embedded_choices(question_text)
    if "which three" not in stem.lower():
        stem = stem.rstrip("?") + " Which three options (letters) are correct?"
    return stem, letter_map, answer


def question_to_mcq(
    q: dict[str, Any],
    *,
    rng: random.Random,
    som: bool = False,
    **_kwargs: Any,
) -> Optional[dict[str, Any]]:
    """Convert one question dict into MCQ fields, or None if incomplete."""
    if not q.get("gt_complete", False):
        return None
    category = str(q.get("category") or "")
    text = str(
        (q.get("text_som") if som and q.get("text_som") else None)
        or q.get("text")
        or ""
    )
    answer = q.get("answer")
    meta = dict(q.get("gt_meta") or {})
    atype = str(q.get("answer_type") or "")

    if category == "closer_to" or atype == "multi_choice":
        choices = list(meta.get("choices") or [])
        correct = meta.get("closer_phrase") or (
            answer if isinstance(answer, str) else None
        )
        if not choices or not correct:
            return None
        stem, choice_map, letter = _mcq4_from_phrases(text, choices, str(correct), rng)
        return {
            "question": stem,
            "answer": letter,
            "mcq_type": "mcq4",
            "n_choices": 4,
            **_choice_fields(choice_map),
            "correct_phrase": str(correct),
        }

    if category == "closest_three" or atype == "set":
        pool = list(meta.get("candidate_pool") or [])
        ans_phrases = list(meta.get("answer_phrases") or [])
        if isinstance(answer, list) and not ans_phrases:
            ans_phrases = [str(x) for x in answer]
        if len(pool) < 6 or len(ans_phrases) < 3:
            return None
        stem, choice_map, letters = _mcq6_select3(text, pool, ans_phrases, rng)
        return {
            "question": stem,
            "answer": letters,
            "mcq_type": "mcq6_select3",
            "n_choices": 6,
            **_choice_fields(choice_map),
            "correct_phrases": ans_phrases[:3],
        }

    if isinstance(answer, bool):
        stem, choice_map, letter = _yesno_mcq(category, text, answer, rng)
        return {
            "question": stem,
            "answer": letter,
            "mcq_type": "mcq2",
            "n_choices": 2,
            **_choice_fields(choice_map),
            "yes_no_gt": bool(answer),
        }

    return None


def export_sample_to_mcq_records(
    sample: dict[str, Any],
    *,
    sample_dir: Path,
    images_dir: Path,
    rng: random.Random,
    copy_images: bool = True,
    som: bool = False,
    flip_quota: Optional[AnswerFlipQuota] = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Build MCQ jsonl records from one eval sample; copy PNGs into images_dir."""
    sample_id = str(sample.get("sample_id") or sample_dir.name)
    scene = str(sample.get("scene") or "")
    records: list[dict[str, Any]] = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for view in sample.get("views") or []:
        view_id = str(view.get("view_id") or "view")
        move = view.get("move") or {}
        motion = str(
            (move.get("text_som") if som and move.get("text_som") else None)
            or move.get("text")
            or ""
        )
        if som:
            rgb0_key, rgb1_key = "rgb0_som_path", "rgb1_som_path"
            img0_name = f"{sample_id}_{view_id}_rgb0_som.png"
            img1_name = f"{sample_id}_{view_id}_rgb1_som.png"
            fallback0 = sample_dir / f"{view_id}_rgb0_som.png"
            fallback1 = sample_dir / f"{view_id}_rgb1_som.png"
        else:
            rgb0_key, rgb1_key = "rgb0_path", "rgb1_path"
            img0_name = f"{sample_id}_{view_id}_rgb0.png"
            img1_name = f"{sample_id}_{view_id}_rgb1.png"
            fallback0 = sample_dir / f"{view_id}_rgb0.png"
            fallback1 = sample_dir / f"{view_id}_rgb1.png"
        rgb0_src = Path(str(view.get(rgb0_key) or ""))
        rgb1_src = Path(str(view.get(rgb1_key) or ""))
        if not rgb0_src.is_file():
            rgb0_src = sample_dir / rgb0_src.name if rgb0_src.name else fallback0
        if not rgb1_src.is_file():
            rgb1_src = sample_dir / rgb1_src.name if rgb1_src.name else fallback1
        if not rgb0_src.is_file():
            rgb0_src = fallback0
        if not rgb1_src.is_file():
            rgb1_src = fallback1
        # Natural fallback if SoM missing.
        if som and not rgb0_src.is_file():
            rgb0_src = sample_dir / f"{view_id}_rgb0.png"
        if som and not rgb1_src.is_file():
            rgb1_src = sample_dir / f"{view_id}_rgb1.png"

        img0_dst = images_dir / img0_name
        img1_dst = images_dir / img1_name
        if copy_images:
            if rgb0_src.is_file():
                shutil.copy2(rgb0_src, img0_dst)
            if rgb1_src.is_file():
                shutil.copy2(rgb1_src, img1_dst)

        for q in view.get("questions") or []:
            gt_meta = dict(q.get("gt_meta") or {})
            identical = gt_meta.get("answer_flipped") is False
            if flip_quota is not None and identical:
                if not flip_quota.allow_identical(rng):
                    continue
            mcq = question_to_mcq(q, rng=rng, som=som)
            if mcq is None:
                continue
            if flip_quota is not None and q.get("gt_complete", False):
                flip_quota.observe(identical=bool(identical))
            cot = dict(q.get("cot") or {})
            ans = str(mcq["answer"])
            q_text = str(mcq["question"])
            gt_meta = dict(q.get("gt_meta") or {})
            # Ensure mover phrase is available for static rebuild.
            if "mover_phrase" not in gt_meta:
                ref = gt_meta.get("mover_referring") or {}
                if isinstance(ref, dict) and ref.get("phrase"):
                    gt_meta["mover_phrase"] = ref["phrase"]
            rec: dict[str, Any] = {
                "id": f"{sample_id}__{q.get('question_id') or view_id}",
                "sample_id": sample_id,
                "scene": scene,
                "view_id": view_id,
                "category": q.get("category"),
                "question_id": q.get("question_id"),
                "image_before": f"images/{img0_name}",
                "image_after": f"images/{img1_name}",
                "motion_command": motion,
                "question": q_text,
                # Static stem for setting-2 (rgb1) eval: faithful rewrite w/ names.
                "static_question": build_static_question(
                    category=str(q.get("category") or ""),
                    question=q_text,
                    gt_meta=gt_meta,
                ),
                "answer": ans,
                "mcq_type": mcq["mcq_type"],
                "n_choices": mcq["n_choices"],
                "gt_complete": True,
                "som": bool(som),
                "answer_flipped": gt_meta.get("answer_flipped"),
                "answer_before": gt_meta.get("answer_before"),
                "answer_after": gt_meta.get("answer_after", q.get("answer")),
                "cot_cold": _append_final_answer(cot.get("cold"), ans),
                "cot_warm": _append_final_answer(cot.get("warm"), ans),
                "cot_pretended": _append_final_answer(cot.get("pretended"), ans),
            }
            for k, v in mcq.items():
                if k.startswith("choice_"):
                    rec[k] = v
            for extra in ("yes_no_gt", "correct_phrase", "correct_phrases"):
                if extra in mcq:
                    rec[extra] = mcq[extra]
            records.append(rec)
    return records
