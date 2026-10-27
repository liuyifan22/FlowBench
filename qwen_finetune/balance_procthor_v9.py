#!/usr/bin/env python3
"""Balance procthor_v9 SoM rows into a *new* jsonl (source untouched).

Pipeline (per category, after GT-complete + letter-answer filter):

1. **Letter / combo balance** (no row drop when remapping works)
   - ``closer_to``, ``count_near``: permute A–D so GT letters are uniform.
   - ``closest_three``: permute A–F so 3-letter answer combos (ABC, ABD, …)
     are as uniform as possible over observed combos.
   - binary ``mcq2``: within each ``yes_no_gt`` bucket (or all rows if no
     polarity), swap A/B so letters are 50/50.

2. **Yes/No balance** (``contact_wall``, ``feasibility`` only)
   Keep 100% of the minority polarity; randomly sample the same count from
   the majority.

3. **Flip-ratio trim** (``left_right_landmark``, ``near_object``, ``count_near``)
   Keep all ``answer_flipped=False``; subsample ``True`` so
   flipped / (flipped+not) ≈ target (default **75%** for left_right,
   **70%** for near_object / count_near). Other categories untouched.

Default is **dry-run** (print expected counts only). Pass ``--out`` to write.

Examples::

  # Preview counts on current SoM (safe while generation still runs)
  python -m qwen_finetune.balance_procthor_v9 --dry-run

  # After procthor_v9 finishes:
  python -m qwen_finetune.balance_procthor_v9 \\
    --dataset …/procthor_v9/som \\
    --out …/qwen_training/procthor_v9_som_balanced/dataset.jsonl
"""

from __future__ import annotations

import os
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
_CIT = _ROOT.parent
for p in (_CIT, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_config import DatasetFilterConfig, is_usable  # noqa: E402

LETTER = "ABCDEF"
YN_BALANCE_CATS = frozenset({"contact_wall", "feasibility"})
FLIP_TRIM_CATS = frozenset(
    {"left_right_landmark", "near_object", "count_near"}
)
# Per-category flip targets (override --target-flip-ratio).
# left_right is very flip-heavy (~90%); 75% keeps more data than 70%.
DEFAULT_FLIP_RATIO_BY_CAT: dict[str, float] = {
    "left_right_landmark": 0.75,
    "near_object": 0.70,
    "count_near": 0.70,
}
PERMUTE_4_CATS = frozenset({"closer_to", "count_near"})
PERMUTE_6_SELECT3 = "closest_three"
BINARY_CATS = frozenset(
    {
        "contact_wall",
        "feasibility",
        "left_right_landmark",
        "near_object",
        "occlusion",
    }
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.environ.get("CIT_DATA_ROOT", "data/cit") + "/"
            "procthor_v9/som"
        ),
        help="Source dataset root (dataset.jsonl + images/)",
    )
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write balanced jsonl here (omit with --dry-run)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats / expected counts (default if --out omitted)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-flip-ratio",
        type=float,
        default=None,
        help=(
            "Global flip target for all FLIP_TRIM_CATS (overrides per-cat defaults). "
            "Default: left_right=0.75, near_object=0.70, count_near=0.70"
        ),
    )
    p.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Faster dry-run: do not require image files on disk",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: alongside --out or stdout only)",
    )
    return p.parse_args()


def _answer_key(rec: dict[str, Any]) -> Optional[str]:
    ans = str(rec.get("answer") or "").strip().upper()
    if not ans or ans in {"NO ANSWER", "NONE", "NULL"}:
        return None
    letters = "".join(c for c in ans if c in LETTER)
    if not letters:
        return None
    if len(letters) == 1:
        return letters
    return "".join(sorted(set(letters)))


def _single_letter(rec: dict[str, Any]) -> str:
    key = _answer_key(rec)
    if not key or len(key) != 1:
        raise ValueError(f"expected single letter, got {key!r} id={rec.get('id')}")
    return key


def _combo_key(rec: dict[str, Any]) -> str:
    key = _answer_key(rec)
    if not key or len(key) != 3:
        raise ValueError(f"expected 3-letter combo, got {key!r} id={rec.get('id')}")
    return key


def flip_ab(rec: dict[str, Any]) -> dict[str, Any]:
    """Swap choice_A/B and remap answer A↔B."""
    out = dict(rec)
    out["choice_A"], out["choice_B"] = out.get("choice_B"), out.get("choice_A")
    L = _single_letter(out)
    out["answer"] = "B" if L == "A" else "A"
    fl = out.get("answer_flipped")
    if fl is True:
        out["answer_flipped"] = False
    elif fl is False:
        out["answer_flipped"] = True
    else:
        out["answer_flipped"] = True
    out["label_balance_op"] = "flip_ab"
    return out


def permute_letter_choices(
    rec: dict[str, Any],
    *,
    letters: list[str],
    target: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Permute choice_* so current GT lands on ``target`` letter/combo."""
    out = dict(rec)
    cur = _answer_key(out)
    if cur is None:
        raise ValueError(f"no answer key for {out.get('id')}")
    if len(target) == 1:
        if len(cur) != 1:
            raise ValueError(f"single target but combo answer {cur}")
        texts = [out.get(f"choice_{L}") for L in letters]
        if any(t is None for t in texts):
            raise ValueError(f"missing choices for {out.get('id')}")
        gt_text = texts[letters.index(cur)]
        others = [texts[i] for i, L in enumerate(letters) if L != cur]
        rng.shuffle(others)
        new_texts: dict[str, Any] = {target: gt_text}
        oi = 0
        for L in letters:
            if L == target:
                continue
            new_texts[L] = others[oi]
            oi += 1
        for L in letters:
            out[f"choice_{L}"] = new_texts[L]
        out["answer"] = target
        out["label_balance_op"] = f"permute_to_{target}"
        out["answer_flipped"] = cur != target
        return out

    # Multi-select: map sorted combo cur → target via a permutation of labels.
    if len(cur) != len(target):
        raise ValueError(f"combo len mismatch {cur} → {target}")
    cur_set = list(cur)
    tgt_set = list(target)
    remaining_src = [L for L in letters if L not in cur_set]
    remaining_tgt = [L for L in letters if L not in tgt_set]
    rng.shuffle(remaining_src)
    # permutation: src_letter → tgt_letter
    perm = {s: t for s, t in zip(cur_set, tgt_set)}
    for s, t in zip(remaining_src, remaining_tgt):
        perm[s] = t
    # Apply: new choice_T = old choice_S where perm[S]=T
    inv = {t: s for s, t in perm.items()}
    old_texts = {L: out.get(f"choice_{L}") for L in letters}
    for T in letters:
        S = inv[T]
        out[f"choice_{T}"] = old_texts[S]
    out["answer"] = target
    out["label_balance_op"] = f"permute_combo_to_{target}"
    out["answer_flipped"] = cur != target
    return out


def _uniform_targets(keys: list[str], universe: list[str], rng: random.Random) -> list[str]:
    """Assign each row a target from ``universe`` for nearly equal counts."""
    n = len(keys)
    if n == 0:
        return []
    base, rem = divmod(n, len(universe))
    bag: list[str] = []
    for i, u in enumerate(universe):
        bag.extend([u] * (base + (1 if i < rem else 0)))
    rng.shuffle(bag)
    return bag


def balance_letters_permute(
    rows: list[dict[str, Any]],
    *,
    category: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"n": 0}

    if category in PERMUTE_4_CATS:
        letters = list("ABCD")
        before = Counter(_answer_key(r) for r in rows)
        targets = _uniform_targets(
            [str(_answer_key(r)) for r in rows], letters, rng
        )
        out = [
            permute_letter_choices(r, letters=letters, target=t, rng=rng)
            for r, t in zip(rows, targets)
        ]
        after = Counter(_answer_key(r) for r in out)
        return out, {
            "mode": "permute_4",
            "n": len(out),
            "before": dict(before),
            "after": dict(after),
        }

    if category == PERMUTE_6_SELECT3:
        letters = list("ABCDEF")
        before = Counter(_combo_key(r) for r in rows)
        universe = sorted(before.keys())
        targets = _uniform_targets(
            [str(_answer_key(r)) for r in rows], universe, rng
        )
        out = [
            permute_letter_choices(r, letters=letters, target=t, rng=rng)
            for r, t in zip(rows, targets)
        ]
        after = Counter(_answer_key(r) for r in out)
        return out, {
            "mode": "permute_combo6",
            "n": len(out),
            "n_combos": len(universe),
            "before": dict(before),
            "after": dict(after),
        }

    if category in BINARY_CATS:
        # Balance A/B within each yes_no_gt group (or one group if missing).
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[r.get("yes_no_gt")].append(r)
        out: list[dict[str, Any]] = []
        detail = {}
        for gkey, grp in groups.items():
            a = [r for r in grp if _single_letter(r) == "A"]
            b = [r for r in grp if _single_letter(r) == "B"]
            # Make |A|≈|B| by flipping excess of the larger side.
            if len(a) > len(b):
                need = (len(a) - len(b)) // 2
                rng.shuffle(a)
                keep_a, to_flip = a[need:], a[:need]
                flipped = [flip_ab(r) for r in to_flip]
                new_grp = keep_a + b + flipped
            elif len(b) > len(a):
                need = (len(b) - len(a)) // 2
                rng.shuffle(b)
                keep_b, to_flip = b[need:], b[:need]
                flipped = [flip_ab(r) for r in to_flip]
                new_grp = a + keep_b + flipped
            else:
                new_grp = a + b
            rng.shuffle(new_grp)
            out.extend(new_grp)
            detail[str(gkey)] = {
                "before_A": len(a),
                "before_B": len(b),
                "after_A": sum(1 for r in new_grp if _single_letter(r) == "A"),
                "after_B": sum(1 for r in new_grp if _single_letter(r) == "B"),
            }
        rng.shuffle(out)
        return out, {"mode": "flip_ab_within_polarity", "n": len(out), "groups": detail}

    # Other categories: leave letters as-is.
    return list(rows), {"mode": "noop", "n": len(rows)}


def balance_yes_no(
    rows: list[dict[str, Any]], *, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    yes = [r for r in rows if r.get("yes_no_gt") is True]
    no = [r for r in rows if r.get("yes_no_gt") is False]
    other = [r for r in rows if r.get("yes_no_gt") not in (True, False)]
    if not yes or not no:
        return list(rows), {
            "skipped": True,
            "yes": len(yes),
            "no": len(no),
            "other": len(other),
        }
    if len(yes) <= len(no):
        weak, strong, wname = yes, no, "yes"
    else:
        weak, strong, wname = no, yes, "no"
    kept_strong = rng.sample(strong, len(weak))
    out = weak + kept_strong + other
    rng.shuffle(out)
    return out, {
        "weak": wname,
        "n_weak": len(weak),
        "n_strong_before": len(strong),
        "n_strong_kept": len(kept_strong),
        "n_other": len(other),
        "after": len(out),
        "yes_before": len(yes),
        "no_before": len(no),
        "yes_after": sum(1 for r in out if r.get("yes_no_gt") is True),
        "no_after": sum(1 for r in out if r.get("yes_no_gt") is False),
    }


def trim_flip_ratio(
    rows: list[dict[str, Any]],
    *,
    target: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (0.0 < target < 1.0):
        raise ValueError(f"target flip ratio must be in (0,1), got {target}")
    fl = [r for r in rows if r.get("answer_flipped") is True]
    nf = [r for r in rows if r.get("answer_flipped") is False]
    other = [
        r
        for r in rows
        if r.get("answer_flipped") not in (True, False)
    ]
    if not nf:
        return list(rows), {
            "skipped": True,
            "reason": "no not_flipped rows",
            "flipped": len(fl),
            "not_flipped": 0,
            "other": len(other),
        }
    # F/(F+N)=target  =>  F = target/(1-target)*N
    max_f = int(round(target / (1.0 - target) * len(nf)))
    max_f = max(0, min(len(fl), max_f))
    kept_f = fl if len(fl) <= max_f else rng.sample(fl, max_f)
    out = kept_f + nf + other
    rng.shuffle(out)
    known = len(kept_f) + len(nf)
    return out, {
        "flipped_before": len(fl),
        "not_flipped_before": len(nf),
        "flipped_kept": len(kept_f),
        "other": len(other),
        "after": len(out),
        "flip_ratio_after": (len(kept_f) / known) if known else None,
        "target": target,
    }


def load_usable(
    jsonl: Path,
    *,
    dataset_root: Path,
    check_images: bool,
) -> list[dict[str, Any]]:
    cfg = DatasetFilterConfig(
        require_gt_complete=True,
        require_letter_answer=True,
        require_images_exist=check_images,
    )
    rows: list[dict[str, Any]] = []
    n_in = 0
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            if not is_usable(rec, dataset_root=dataset_root, cfg=cfg):
                continue
            rows.append(rec)
    return rows


def _flip_target_for(
    category: str, *, global_target: Optional[float]
) -> float:
    if global_target is not None:
        return float(global_target)
    if category in DEFAULT_FLIP_RATIO_BY_CAT:
        return float(DEFAULT_FLIP_RATIO_BY_CAT[category])
    return 0.70


def balance_category(
    rows: list[dict[str, Any]],
    *,
    category: str,
    rng: random.Random,
    target_flip: Optional[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply yn → flip → letter (letter last so labels stay balanced)."""
    report: dict[str, Any] = {"category": category, "n0": len(rows)}
    cur = list(rows)

    if category in YN_BALANCE_CATS:
        cur, st = balance_yes_no(cur, rng=rng)
        report["yes_no"] = st
    report["n_after_yes_no"] = len(cur)

    if category in FLIP_TRIM_CATS:
        tgt = _flip_target_for(category, global_target=target_flip)
        cur, st = trim_flip_ratio(cur, target=tgt, rng=rng)
        report["flip"] = st
    report["n_after_flip"] = len(cur)

    cur, st = balance_letters_permute(cur, category=category, rng=rng)
    report["letter"] = st
    report["n_final"] = len(cur)
    return cur, report


def main() -> None:
    args = _parse_args()
    dry = bool(args.dry_run) or args.out is None
    if not dry and args.out is None:
        raise SystemExit("Pass --out PATH or --dry-run")

    src_root = Path(args.dataset).resolve()
    src_jsonl = Path(args.jsonl) if args.jsonl else src_root / "dataset.jsonl"
    if not src_jsonl.is_file():
        raise SystemExit(f"missing {src_jsonl}")

    check_images = not args.skip_image_check
    print(
        f"[balance] loading usable rows from {src_jsonl} "
        f"(images_check={check_images})",
        flush=True,
    )
    rows = load_usable(
        src_jsonl, dataset_root=src_root, check_images=check_images
    )
    print(f"[balance] usable={len(rows)}", flush=True)

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category") or "?")].append(r)

    rng = random.Random(int(args.seed))
    balanced: list[dict[str, Any]] = []
    cat_reports: dict[str, Any] = {}
    for cat in sorted(by_cat):
        # Independent RNG streams per category for stability if cats change order.
        cat_rng = random.Random(rng.randint(0, 2**31 - 1))
        out_rows, rep = balance_category(
            by_cat[cat],
            category=cat,
            rng=cat_rng,
            target_flip=args.target_flip_ratio,
        )
        # Strip helper keys if any; tag provenance lightly.
        for r in out_rows:
            r2 = dict(r)
            r2["balance_source"] = "procthor_v9_som"
            balanced.append(r2)
        cat_reports[cat] = rep
        print(
            f"  {cat}: {rep['n0']} → yn {rep['n_after_yes_no']} "
            f"→ flip {rep['n_after_flip']} → letter {rep['n_final']}",
            flush=True,
        )

    # Stable shuffle of final pool.
    random.Random(int(args.seed) + 17).shuffle(balanced)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(src_root),
        "source_jsonl": str(src_jsonl),
        "seed": int(args.seed),
        "target_flip_ratio_global": args.target_flip_ratio,
        "target_flip_ratio_by_cat": {
            c: _flip_target_for(c, global_target=args.target_flip_ratio)
            for c in sorted(FLIP_TRIM_CATS)
        },
        "yn_balance_cats": sorted(YN_BALANCE_CATS),
        "flip_trim_cats": sorted(FLIP_TRIM_CATS),
        "n_usable_in": len(rows),
        "n_balanced_out": len(balanced),
        "by_category": cat_reports,
        "dry_run": dry,
    }

    print(json.dumps({
        "n_usable_in": report["n_usable_in"],
        "n_balanced_out": report["n_balanced_out"],
        "per_category_final": {
            c: cat_reports[c]["n_final"] for c in cat_reports
        },
    }, indent=2), flush=True)

    if dry:
        print("[balance] dry-run only — no files written", flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[balance] report → {args.report}", flush=True)
        return

    out_path = Path(args.out).resolve()
    if src_jsonl.resolve() == out_path:
        raise SystemExit("Refusing to overwrite source jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report_path = (
        Path(args.report)
        if args.report
        else out_path.with_suffix(".balance_report.json")
    )
    report["out_jsonl"] = str(out_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    # Symlink images next to out if writing into a fresh dir.
    images_src = src_root / "images"
    images_dst = out_path.parent / "images"
    if images_src.is_dir() and not images_dst.exists():
        images_dst.symlink_to(images_src, target_is_directory=True)
        print(f"[balance] symlinked {images_dst} → {images_src}", flush=True)

    print(f"[balance] wrote {len(balanced)} rows → {out_path}", flush=True)
    print(f"[balance] report → {report_path}", flush=True)


if __name__ == "__main__":
    main()
