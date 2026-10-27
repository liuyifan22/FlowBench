#!/usr/bin/env python3
"""Write $FLOWBENCH_RELEASE_ROOT/README.md (file structure + Hugging Face dataset card).

Counts and sizes are read from the built release, so re-run after any rebuild:

  python flowbench_release/write_readme.py
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release"))
SIM = ROOT / "sim"
STYLES = ("natural", "colour_som", "red_som")
CATS = {
    "closer_to": "Which of four objects is the moved object closer to after the move?",
    "closest_three": "Which three of six objects are closest to the moved object after the move? (answer: 3 letters)",
    "count_near": "How many of the listed objects are within a distance of the moved object after the move?",
    "left_right_landmark": "After the move, is the moved object to the left / right of a landmark (camera view)?",
    "near_object": "Is the moved object within a distance of another object after the move?",
    "occlusion": "After the move, is a landmark object partly / fully hidden by the moved object, or totally visible?",
    "feasibility": "Starting from the after state, can the object make a further stated move without colliding?",
    "contact_wall": "Is the moved object touching a wall after the move?",
}


def rows(split: str, style: str) -> list[dict]:
    p = SIM / split / style / "dataset.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9 if path.exists() else 0.0


def main() -> None:
    counts = {s: collections.Counter(r["category"] for r in rows(s, "colour_som")) for s in ("train", "test")}
    n = {s: sum(c.values()) for s, c in counts.items()}
    internal = collections.Counter(r.get("internal_split") for r in rows("train", "colour_som"))
    match = {s: sum(bool(r.get("choices_match_som")) for r in rows(s, "natural")) for s in ("train", "test")}
    relabel = {s: sum(bool(r.get("som_relabelled")) for r in rows(s, "red_som")) for s in ("train", "test")}
    pq_gb = {st: gb(SIM / "parquet" / st) for st in STYLES}
    cot_n = {st: sum(r.get("cot_0") is not None for r in rows("train", st)) for st in STYLES}
    gt_n = {(sp, st): sum(r.get("cot_gt") is not None for r in rows(sp, st)) for sp in ("train", "test") for st in STYLES}
    cot_cat = {st: collections.Counter(r["category"] for r in rows("train", st) if r.get("cot_0") is not None)
               for st in STYLES}
    cot_table = "\n".join(f"| `{c}` | {cot_cat['natural'][c]:,} | {cot_cat['colour_som'][c]:,} |" for c in CATS)
    cat_table = "\n".join(f"| `{c}` | {counts['train'][c]:,} | {counts['test'][c]:,} | {d} |" for c, d in CATS.items())
    yaml_cfg = "\n".join(
        f"- config_name: sim_{st}\n  data_files:\n  - split: train\n    path: sim/parquet/{st}/train-*.parquet\n"
        f"  - split: test\n    path: sim/parquet/{st}/test-*.parquet" for st in STYLES)

    text = f"""---
pretty_name: FlowBench (sim)
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 100K<n<1M
configs:
{yaml_cfg}
---

# FlowBench — sim

Counterfactual spatial questions about indoor scenes: an object is moved by a stated amount, and the
model must answer about the scene **after** the move. Every question comes with a rendered before / after
image pair and dense ground truth (optical flow, depth, segmentation).

| split | source | questions | scenes (samples) |
|---|---|---|---|
| `train` | ProcTHOR v11 (AI2-THOR ProcTHOR-10k houses) | {n['train']:,} | {len({r['sample_id'] for r in rows('train', 'colour_som')}):,} |
| `test` | iTHOR v11 (AI2-THOR iTHOR floor plans) | {n['test']:,} | {len({r['sample_id'] for r in rows('test', 'colour_som')}):,} |

Each question is released in **three image styles**, with the same question ids, choices order and
answer letter (see `choices_match_som` for the natural exceptions):

| style | images | how text refers to objects |
|---|---|---|
| `natural` | clean renders | names only: "the 2nd kettle from the left" |
| `colour_som` | Set-of-Marks: solid colour discs (white rim, no text), 10 colours sampled per question | "the kettle (marked blue)" |
| `red_som` | Set-of-Marks: red badges with a number (moved object `1`, landmark / occludee `2`) or a letter (choice objects `A`–`F`) | "the kettle (1)"; choices "(A) the plate" |

and with **two question types** per row:

| field | asked on | example |
|---|---|---|
| `question` (with `motion_command`) | the **before** image: predict the outcome of the move | `move the kettle (marked orange) 0.60m right` + "After the move, which three objects are closest to the kettle (marked orange)?" |
| `static_question` | the **after** image: read the outcome off the image | "Which three objects are closest to the kettle (marked orange)?" |

## Categories

| category | train | test | question |
|---|---|---|---|
{cat_table}

Answers are option letters `A`–`F` (`closest_three` answers three letters in alphabetical order, e.g. `ABD`).
Letters are balanced within each category (yes/no questions within each polarity, so a letter never
means "yes"). `contact_wall` and `near_object` are usually left out of headline averages.

## File structure

```
flowbench_release/
  README.md                          this file (also the Hugging Face dataset card)
  sim/
    train/                           ProcTHOR v11, {n['train']:,} questions
    test/                            iTHOR v11, {n['test']:,} questions
      natural/
        dataset.jsonl                one row per question (fields below); image paths relative to this folder
        images/<sample_id>_<view_id>_rgb0.png        before image (800x600 PNG)
        images/<sample_id>_<view_id>_rgb1.png        after image
      colour_som/
        dataset.jsonl
        images/<sample_id>_<view_id>_rgb{{0,1}}_som.png
      red_som/
        dataset.jsonl
        images/<sample_id>_<view_id>_rgb{{0,1}}_som.png
      samples/<sample_id>/           ground truth, one folder per scene sample (shared by all styles)
        sample.json                  scene, camera, move, all questions with gt_meta (object ids, distances, mark positions, ...)
        <view_id>_rgb{{0,1}}.png       clean before / after renders
        <view_id>_rgb{{0,1}}_som.png   red SoM renders as originally generated (original letters)
        <view_id>_gt_flow.npy        (H, W, 2) float32 optical flow before -> after, pixels
        <view_id>_gt_flow{{,_arrow,_masked}}.png   flow visualisations
        <view_id>_rgb{{0,1}}_depth.npy   (H, W) float32 depth, metres
        <view_id>_rgb{{0,1}}_seg.png, _seg_ids.npy, _seg_labels.json   instance segmentation
        <view_id>_mask{{0,1}}.png      moved-object mask
      _build/                        build logs, stats and the file plan (not part of the dataset; do not upload)
    parquet/                         Hugging Face parquet shards (images embedded)
      natural/{{train,test}}-NNNNN-of-MMMMM.parquet
      colour_som/...
      red_som/...
```

`<view_id>` is `view_<category>`: one view (camera + move) per category per sample, and one question per
view, so `<sample_id>_<view_id>` identifies a question's images. The question id is `<sample_id>__<question_id>`.

`samples/*/sample.json` stores the generation-time absolute paths (`$CIT_DATA_ROOT`) in its `*_path` fields;
the files themselves are next to it under the names above.

## Row fields (`dataset.jsonl` and parquet)

| field | |
|---|---|
| `id` | `<sample_id>__<question_id>`, identical across the three styles |
| `sample_id`, `scene`, `view_id`, `question_id` | scene sample, AI2-THOR scene / camera, view, question |
| `category`, `mcq_type`, `n_choices` | `mcq2` (yes/no), `mcq3` / `mcq4`, `mcq6_select3`; `count_near` choices are counts |
| `image_before`, `image_after` | jsonl: path relative to the style folder; parquet: embedded image |
| `motion_command` | the move, e.g. "move the bowl (marked blue) 0.50m left" |
| `question`, `static_question` | see above |
| `choice_A` … `choice_F`, `answer` | options (unused letters are null) and the correct letter(s) |
| `source_version` | generator version of the scene sample (`procthor_v9` / `procthor_v10`, `ithor_v9` / `ithor_v10`) |
| `mover_object_id` | AI2-THOR id of the moved object |
| `internal_split` | train only: the split used in our experiments (`train0`/`train1`/`train2`/`val`/`test`); counts {dict(sorted(internal.items()))} |
| `occludee_px`, `occlusion_cover`, `occlusion_relabelled` | occlusion only: occluded object's pixels in the before image, fraction covered by the mover, relabelled from "visible" |
| `som_colors` | colour_som: mark id → colour (`1` mover, `2` landmark / occludee, `A`–`F` choice / listed objects) |
| `som_labels`, `som_relabelled` | red_som: mark id → letter drawn on the badge; true where badges were redrawn with the release letters |
| `choices_match_som` | natural: true if the options are the same objects in the same order as the SoM rows (then the answer letter is shared too) |
| `cot_gt`, `cot_1`, `cot_0` | reasoning annotations c_gt / c_1 / c_0 (see below); `cot_gt` on train and test, `cot_1` / `cot_0` train only; null where not provided |

In parquet, `som_colors` / `som_labels` are JSON strings.

## Reasoning annotations (`cot_gt`, `cot_1`, `cot_0`)

Three chains of thought, each ending with the question's answer:

| field | paper | written by | from |
|---|---|---|---|
| `cot_gt` | c_gt | deterministic category rules | simulator metadata m_0, m_1 (distances, screen positions, cover rates, collision checks, wall clearances) |
| `cot_1` | c_1 | Qwen3-VL-32B-Instruct (v9 categories) / Qwen3.5-9B (feasibility, contact_wall) | t_q, the before **and** after images, c_gt |
| `cot_0` | c_0 | Qwen3.5-9B (thinking off) | t_q, the **before** image I_0 and t_m only, with c_1 as a private hint |

**`cot_gt`** is written per style, in that row's own wording and option set: names on natural rows,
"(marked <colour>)" on colour_som, badge labels "(1)", "(A)" on red_som. Each text states the rule the
category's ground truth was computed with, the recorded numbers, and the conclusion. The conclusion is
checked against the row's answer.
- Rules: closer_to uses bounding-box surface distances to every option; closest_three ranks the six options
  by centre distance; count_near uses the threshold and the counts before / after; near_object lists the
  objects within the threshold before / after; left_right uses the x of the mask centres in the after image;
  occlusion uses the cover fraction with the v11 rule that any cover counts; feasibility uses the
  simulator's collision-checked probe; contact_wall uses the wall gap and the blocked next step.
- Coverage: {gt_n[("train", "natural")]:,} / {n['train']:,} train and {gt_n[("test", "natural")]:,} / {n['test']:,} test rows. Null on 1,811 train
  and 3 test `closest_three` rows (2.7%) whose stored answer contradicts their own after image. Their
  stored distances agree with the image (checked against depth + segmentation centroids) and rank a
  different top three, so those answers are likely wrong. Also null on 53 natural rows (47 `closest_three`,
  6 `closer_to`) that list two options with the same text.
- count_near records counts, not per-object distances, so its `cot_gt` gives the counts.

**`cot_1` / `cot_0`** were written for the natural question. They are present on natural rows whenever the
question has an accepted c_0 ({cot_n['natural']:,} train rows). On colour_som / red_som rows they are present when
the SoM question asks about the same objects ({cot_n['colour_som']:,} / {cot_n['red_som']:,} rows); most `closer_to` SoM rows offer a
different fourth object and get null. Questions whose answer changed after the CoTs were written
(relabelled occlusion rows) have none.
- Each c_0 was checked automatically: the final answer equals c_1's; no mention of an after / second image
  or of hints; a stated direction never contradicts the camera-axis motion ("backward along the camera
  axis" = toward the camera); word overlap with c_1 at most 0.8; 25-260 words. Failures were regenerated.
- The VLM-written texts describe the scene as the VLM saw it and can be wrong about details that do not
  change the answer.

| category | natural rows with c_0 | SoM rows with c_0 |
|---|---|---|
{cot_table}

## How the styles relate

- The **colour SoM** rows are the reference v11 rows (filters: complete ground truth, no structural movers such
  as windows / doors / cabinets / drawers / sinks / faucets / countertops, every referenced mark drawn,
  occlusion: occludee ≥ 150 px and any mover cover counts as occluded; static questions never mention the move;
  balanced answer letters).
- The **red SoM** rows are the original numbered-SoM rows with the same text fixes and the same choice
  permutation; mapping each red mark id to its colour reproduces the colour row exactly (checked for every row).
  Where the permutation moves lettered choice objects ({relabel['train']:,} train / {relabel['test']:,} test rows), the red badges are
  redrawn with the new letters in the same style (`som_relabelled`).
- The **natural** rows come from the natural export of the same scene samples, with the same fixes. When it asks
  about the same objects as the SoM rows, its options are put in the SoM order ({match['train']:,} / {n['train']:,} train,
  {match['test']:,} / {n['test']:,} test). The rest (most `closer_to` and all `count_near`) use different distractors / object
  lists, so they are separate questions with their own balanced letters.

## Maintenance notes (local copy only)

- Natural images, red-SoM images and everything under `samples/` are **hardlinks** into the generation folders
  (`$CIT_DATA_ROOT/{{procthor,ithor}}_v{{9,10}}`): deleting either copy
  leaves the other intact, and editing a file in place changes both.
- The ProcTHOR v11 colour-SoM images were moved here from $CIT_DATA_ROOT/procthor_v11/som/images.
  `$CIT_DATA_ROOT/procthor_v11/som/images` is now a symlink to
  `sim/train/_build/procthor_v11_som_images/` (hardlinks of `sim/train/colour_som/images/` plus the 56 images of
  questions not in the release), so the v11 training data keeps working. **Do not delete that folder.**
- Build code: `flowbench_release/` (`build_sim_release.py`, `build_sim_parquet.py`,
  `swap_colour_source.py`, `write_readme.py`).

## Loading

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files={{"test": "sim/parquet/colour_som/test-*.parquet"}}, split="test")
row = ds[0]; row["image_before"]  # PIL image
```

or read `sim/<split>/<style>/dataset.jsonl` and open `sim/<split>/<style>/<image_before>`.

## Image encoding and sizes

The folder release keeps every image as the original 800x600 PNG. In parquet, test images are the original
PNG bytes; train images are JPEG (quality 95) to keep the train parquet small. Parquet sizes:
{", ".join(f"{st} {pq_gb[st]:.1f} GB" for st in STYLES)}.
"""
    (ROOT / "README.md").write_text(text)
    print(f"wrote {ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
