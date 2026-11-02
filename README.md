# FlowBench: Counterfactual Imagination Training for Spatial Intelligence

Yifan Liu\*, Baihan Zhang\*, Khai Loong Aw\*, Klemen Kotar\*, Daniel L. K. Yamins

\* Equal contribution

A new benchmark emphasizing counterfactual imagination of object motions inside indoor scenes.

- Automatic Annotations
- Scaled Dataset
- Challenging Benchmark

**Dataset:** [huggingface.co/datasets/StanfordNeuroAILab/FlowBench](https://huggingface.co/datasets/StanfordNeuroAILab/FlowBench). For questions, please contact Yifan: [lyf@stanford.edu](mailto:lyf@stanford.edu).

```python
from datasets import load_dataset
ds = load_dataset("StanfordNeuroAILab/FlowBench", "sim_colour_som", split="test")
```

This repository holds the code that generates **FlowBench-Sim** in AI2-THOR, builds the release, and evaluates and fine-tunes VLMs on it.

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data generation](#data-generation)
- [Building the release](#building-the-release)
- [Evaluating VLMs](#evaluating-vlms)
- [Fine-tuning Qwen3-VL](#fine-tuning-qwen3-vl)
- [Citation](#citation)

## Overview

Each FlowBench question shows a scene, states a move ("move the kettle 0.60m right"), and asks about the scene **after** the move. AI2-THOR renders the before / after pair, and every answer is computed from simulator state. Each sample also stores optical flow, depth, segmentation and object masks.

| split | source | questions |
|---|---|---|
| train | ProcTHOR-10K houses | 419,612 |
| test | iTHOR floor plans (120 scenes) | 714 |

There are eight question categories: `closer_to`, `closest_three`, `count_near`, `left_right_landmark`, `near_object`, `occlusion`, `feasibility`, `contact_wall`. Every question comes in three image styles (`natural`, `colour_som`, `red_som`) and two forms. `question` is asked on the before image together with `motion_command`. `static_question` is asked on the after image.

The pipeline runs in five stages, one folder under `scripts/` each:

```
 1. generation   AI2-THOR + VLM wording     spatial categories (6)          physical categories (2)
                                            generate_procthor_spatial.py     generate_procthor_physical.py
                                            generate_ithor_spatial.py        generate_ithor_physical.py
                                            merge_ithor.py, fix_static_questions.py
        │
        ▼
 2. curation     merge, colour SoM, clean   build_train_set.py, build_test_set.py,
                                            balance_answer_letters.py, fix_occlusion.py
        │
        ▼
 3. release      natural / colour_som / red_som folders and parquet shards
        │
        ▼
 4. reasoning    c_gt (rules), c_1 (VLM, sees the after image), c_0 (VLM, before image only)
        │
        ▼
 5. evaluation   zero-shot VLM evals;  training/ fine-tunes Qwen3-VL
```

The six spatial categories are `closer_to`, `closest_three`, `count_near`, `left_right_landmark`, `near_object` and `occlusion`. The two physical categories are `feasibility` and `contact_wall`.

## Repository layout

```
flowbench/                         core library: AI2-THOR control, edit search, question generators, Set-of-Marks,
                                   GT flow / depth, CoT templates, MCQ export, curation rules, eval prompts
scripts/
  generation/
    generate_procthor_spatial.py   ProcTHOR houses, six spatial categories
    generate_procthor_physical.py  ProcTHOR houses, feasibility + contact_wall
    generate_ithor_spatial.py      iTHOR test scenes, six spatial categories
    generate_ithor_physical.py     iTHOR test scenes, feasibility + contact_wall
    merge_ithor.py                 merges the two iTHOR rounds
    fix_static_questions.py        puts SoM mark ids into static_question (text-only VLM rewrite)
  curation/
    build_train_set.py             ProcTHOR train set: merge, colour SoM, filters, train/val/test splits
    build_test_set.py              iTHOR test benchmark
    balance_answer_letters.py      balances correct-answer letters per category
    fix_occlusion.py               occlusion filter / relabel / rewording
    visualize_test_set.py          HTML viewer for the test benchmark
  release/
    build_release.py               release folders for the three image styles
    build_parquet.py               Hugging Face parquet shards
    add_cot.py                     adds the cot_gt / cot_1 / cot_0 columns
    dedup_colour_images.py         stores the colour-SoM train images only once (optional)
    write_dataset_card.py          writes the dataset card from the built release
  reasoning/
    generate_cot_warm.py           c_1 ("warm" CoT) for every question
    filter_cot.py                  which train rows can reuse their c_1
    generate_cot_pretended.py      c_0 ("pretended" CoT) with Qwen3.5-9B
    integrate_cot.py               collects c_gt / c_1 / c_0 and writes the filters
    build_cot_gt.py, apply_cot_gt.py   rebuilds c_gt from simulator metadata for every row and style
  evaluation/
    eval_server.py                 eval against any OpenAI-compatible server (vLLM)
    eval_hf.py                     transformers eval for VLMs vLLM cannot serve
    eval_qwen_local.py             local Qwen3-VL eval (vLLM or HF backend)
    summarize.py, check_answer_format.py, rescore.py   result tables and answer-format checks
training/                          LoRA SFT of Qwen3-VL-8B (also Qwen3.5-9B, LLaVA-OneVision-2)
  train.py, data.py, dataset_config.py, balance_dataset.py, split_dataset.py,
  build_reasoning_splits.py, evaluate.py, merge_lora.py, visualize_inputs.py, filter_dataset.py
```

## Installation

### Environment

You need Linux with an NVIDIA GPU. AI2-THOR renders headless through Vulkan (`CloudRendering`).

```bash
conda create -n flowbench python=3.10 -y
conda activate flowbench
```

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install vllm==0.25.1
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

`pip install -e .` makes the `flowbench` package importable. The scripts also find it on their own when run from a checkout.

Use vLLM only for the VLM servers and the evals. Data generation itself needs only `ai2thor`, `prior`, `numpy`, `pillow`, `opencv` and `openai`. The c_0 annotations were generated with Qwen3.5-9B on vLLM 0.27.1. If your vLLM cannot load Qwen3.5, serve that model from a second environment with a newer vLLM.

### AI2-THOR

- The NVIDIA Vulkan driver must be installed. The code sets `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` unless you set it yourself, and `vulkaninfo` must be on `PATH`.
- On first start AI2-THOR downloads its Unity build (~1 GB) into `~/.ai2thor`.
- Check the installation:

```bash
python -c "from ai2thor.controller import Controller; from ai2thor.platform import CloudRendering; c = Controller(platform=CloudRendering); print(c.step('MoveAhead').metadata['lastActionSuccess']); c.stop()"
```

### ProcTHOR-10K houses

The ProcTHOR generators read the houses from a local copy, so workers never call the GitHub API. They fall back to `prior.load_dataset("procthor-10k")` when the local files are missing.

```bash
git clone https://github.com/allenai/procthor-10k data/procthor-10k
```

The local copy is expected at `data/procthor-10k/{train,val,test}.jsonl.gz`. Override the location with `PROCTHOR_DATA_DIR`.

### Models

```bash
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir models/Qwen3-VL-32B-Instruct
```

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

```bash
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
```

Qwen3-VL-32B-Instruct writes the question wording and c_1 for the spatial categories. Qwen3.5-9B writes c_1 for the physical categories and all of c_0. Qwen3-VL-8B is the fine-tuning base.

### Paths and environment variables

Run every command from the repository root. Scripts find their data through these variables. The defaults are relative to the repository root.

| variable | default | used for |
|---|---|---|
| `FLOWBENCH_DATA_ROOT` | `data/build` | generation and curation outputs (stage folders below) |
| `FLOWBENCH_RELEASE_ROOT` | `data/flowbench_release` | the release (`sim/train`, `sim/test`, `sim/parquet`) |
| `PROCTHOR_DATA_DIR` | `data/procthor-10k` | ProcTHOR-10K house files |
| `FLOWBENCH_TMP_DIR` | `~/.cache/flowbench/tmp` | TMPDIR for AI2-THOR workers (use a disk with space; `/tmp` often fills up on shared nodes) |

```bash
export FLOWBENCH_DATA_ROOT=/path/to/flowbench_build
export FLOWBENCH_RELEASE_ROOT=/path/to/flowbench_release
```

The stage folders under `FLOWBENCH_DATA_ROOT` keep the generation-round names we used, because those names are also stored in the released rows (`source_version`):

| folder | written by | contents |
|---|---|---|
| `procthor_v9/` | `generate_procthor_spatial.py` | ProcTHOR, spatial categories |
| `procthor_v10/` | `generate_procthor_physical.py` | ProcTHOR, physical categories |
| `ithor_v9/`, `ithor_v10/` | `generate_ithor_{spatial,physical}.py`, `merge_ithor.py` | iTHOR rounds (`ithor_v10/` holds the merge) |
| `procthor_v11/` | `build_train_set.py` | curated train set, CoT files and filters, Qwen training splits |
| `ithor_v11/` | `build_test_set.py` | curated test benchmark |

The full dataset needs a few TB. The two ProcTHOR generation folders hold most of it, because they include the flow / depth / segmentation arrays.

## Data generation

### 0. Start the wording VLM

The generators ask a VLM to phrase questions and options naturally. They fall back to the template text on VLM errors when `--vlm-optional` is set, and `--no-vlm` skips the VLM entirely.

```bash
vllm serve models/Qwen3-VL-32B-Instruct --served-model-name Qwen3-VL-32B-Instruct --host 0.0.0.0 --port 19857 --tensor-parallel-size 4 --gpu-memory-utilization 0.7 --max-model-len 8192
```

```bash
curl -s http://localhost:19857/v1/models
```

Several generators can share one server. Point them at it with `--vlm-base-url http://<host>:19857/v1`.

### 1. Raw generation in AI2-THOR

Each generator process drives one AI2-THOR instance on one GPU and handles every `num-workers`-th house or scene. Workers are resumable (`--skip-existing`), append to a shared `dataset.jsonl` under a file lock, and write two parallel roots: `natural/` (clean images and text) and `som/` (red Set-of-Marks images and mark-aware text). Run one process per GPU. Spread them over several machines by giving each a distinct `--worker-id` and the same `--num-workers`.

The following runs N workers on the GPUs of one machine. The flags are the ones used for the release.

**ProcTHOR, spatial categories.** 10,000 houses, 10 apartment-wide cameras per house:

```bash
N=8; OUT=$FLOWBENCH_DATA_ROOT/procthor_v9; mkdir -p $OUT/natural/logs
for W in $(seq 0 $((N-1))); do
  nohup python -u scripts/generation/generate_procthor_spatial.py \
    --out $OUT/natural --out-som $OUT/som --gpus $W --worker-id $W --num-workers $N \
    --n-houses 10000 --house-pool 10000 --max-rooms 0 \
    --camera-mode apartment --cameras-per-apartment 10 \
    --vlm-base-url http://localhost:19857/v1 --vlm-model Qwen3-VL-32B-Instruct \
    --vlm-optional --vlm-connect-retries 1 \
    --seed-base 9000 --flip-mode train --flush-every 1 --skip-existing --worker-tag w$W \
    > $OUT/natural/logs/worker$W.log 2>&1 &
done
```

**ProcTHOR, physical categories.** Same command with these changes: `scripts/generation/generate_procthor_physical.py`, `OUT=$FLOWBENCH_DATA_ROOT/procthor_v10`, `--seed-base 11000`, and `--categories contact_wall,feasibility`.

**iTHOR test scenes** (FloorPlan 1–30, 201–230, 301–330, 401–430):

```bash
N=8; OUT=$FLOWBENCH_DATA_ROOT/ithor_v9; mkdir -p $OUT/natural/logs
for W in $(seq 0 $((N-1))); do
  nohup python -u scripts/generation/generate_ithor_spatial.py \
    --out $OUT/natural --out-som $OUT/som --gpus $W --worker-id $W --num-workers $N \
    --vlm-base-url http://localhost:19857/v1 --vlm-model Qwen3-VL-32B-Instruct \
    --vlm-optional --vlm-connect-retries 1 \
    --seed-base 9200 --flip-mode train --flush-every 1 --skip-existing --worker-tag w$W \
    > $OUT/natural/logs/worker$W.log 2>&1 &
done
```

Then run `scripts/generation/generate_ithor_physical.py` with `OUT=$FLOWBENCH_DATA_ROOT/ithor_v10`, `--seed-base 10200` and `--categories contact_wall,feasibility`, and merge the two:

```bash
python scripts/generation/merge_ithor.py
```

Use `tail -f $OUT/natural/logs/worker*.log` to follow progress. To stop a run, kill the generator processes; their AI2-THOR children exit with them. Each sample folder `natural/<sample_id>/` holds `sample.json` (scene, camera, move and every question with its `gt_meta`), the before / after renders, `*_gt_flow.npy`, `*_depth.npy`, segmentation and masks. The flat MCQ export is `dataset.jsonl` + `images/`.

### 2. SoM marks in `static_question`

The VLM wording sometimes drops the mark ids from the after-image question. A text-only pass restores them and writes `som/dataset_static_som.jsonl`. Run it on `procthor_v9`, `procthor_v10` and `ithor_v10`. Use `--num-shards` / `--shard-id` to split the work across servers.

```bash
for D in procthor_v9 procthor_v10 ithor_v10; do
  python scripts/generation/fix_static_questions.py --input $FLOWBENCH_DATA_ROOT/$D/som/dataset.jsonl \
    --output $FLOWBENCH_DATA_ROOT/$D/som/dataset_static_som.jsonl \
    --vlm-url http://localhost:19857/v1 --model Qwen3-VL-32B-Instruct --workers 16
done
```

### 3. Curation

None of these steps run AI2-THOR or a VLM. They merge the spatial and physical rounds, redraw the marks as text-free colour discs, drop rows with unresolved marks or structural movers (doors, windows, sinks, countertops, ...), and balance the answer letters.

```bash
python scripts/curation/build_train_set.py --workers 48
```

```bash
python scripts/curation/build_test_set.py
```

```bash
python scripts/curation/balance_answer_letters.py --target procthor
```

```bash
python scripts/curation/balance_answer_letters.py --target ithor
```

The occlusion fix drops occludees under 150 px, labels any cover as occluded, and rewords the choice to "Partly or fully hidden". The `occlusion` letters are then rebalanced:

```bash
for T in procthor ithor; do
  python scripts/curation/fix_occlusion.py --target $T
  python scripts/curation/balance_answer_letters.py --target $T --categories occlusion
done
```

This produces `procthor_v11/{som/dataset.jsonl, train0,train1,train2,val,test.jsonl}` (train) and `ithor_v11/{natural,som}/dataset.jsonl` (the 714-question test benchmark). Write an HTML viewer of the benchmark with:

```bash
python scripts/curation/visualize_test_set.py
```

## Building the release

### Folders and parquet

```bash
python scripts/release/build_release.py --split test --stage text
```

```bash
python scripts/release/build_release.py --split test --stage files
```

```bash
python scripts/release/build_release.py --split train --stage text
```

```bash
python scripts/release/build_release.py --split train --stage files --workers 48
```

```bash
python scripts/release/build_parquet.py --split test
```

```bash
python scripts/release/build_parquet.py --split train --workers 48
```

The release has `sim/<split>/{natural,colour_som,red_som}/{dataset.jsonl,images/}`, `sim/<split>/samples/<sample_id>/` (the GT, hardlinked from the generation folders), and `sim/parquet/<style>/*.parquet` (Hugging Face shards with embedded images). `scripts/release/dedup_colour_images.py` is optional. It replaces the train set's colour-SoM image folder with a symlink into the release, so the images are stored only once.

### Reasoning annotations

| | written by | sees |
|---|---|---|
| `c_gt` | category rules | simulator metadata (distances, screen positions, cover rates, collision checks) |
| `c_1` (warm) | VLM | question, before **and** after image, c_gt |
| `c_0` (pretended) | VLM | question, before image and motion only, with c_1 as a private hint |

**c_1.** Run on the spatial categories with Qwen3-VL-32B, and on the physical categories with Qwen3.5-9B. Shard across servers with `--num-shards K --shard k`. Each run is resumable.

```bash
python scripts/reasoning/generate_cot_warm.py --natural $FLOWBENCH_DATA_ROOT/procthor_v9/natural --no-splits --vlm-url http://localhost:19857/v1 --model Qwen3-VL-32B-Instruct --workers 48
```

```bash
vllm serve models/Qwen3.5-9B --served-model-name Qwen3.5-9B --port 19859 --max-model-len 8192 --limit-mm-per-prompt.image 2 --enable-prefix-caching
```

```bash
python scripts/reasoning/generate_cot_warm.py --natural $FLOWBENCH_DATA_ROOT/procthor_v10/natural --no-splits --vlm-url http://localhost:19859/v1 --model Qwen3.5-9B --workers 32
```

**Filter, then c_0.** `filter_cot.py` keeps the rows whose c_1 exists and agrees with the curated answer. `generate_cot_pretended.py` sends requests to every server listed in `servers.txt` (one `http://<host>:<port>/v1` per line). The file is re-read while the job runs, so you can add servers as they come up. Replies that fail the checks (answer mismatch, mentions of the after image, motion-direction contradictions, too much copying from c_1) are retried.

```bash
python scripts/reasoning/filter_cot.py
```

```bash
python scripts/reasoning/generate_cot_pretended.py --servers-file servers.txt --concurrency 48
```

**Integrate everything.** These steps write `procthor_v11/cot/`, `procthor_v11/filters/`, the reasoning training splits, and the `cot_gt` / `cot_1` / `cot_0` fields of the release:

```bash
python scripts/reasoning/integrate_cot.py
```

```bash
python scripts/release/add_cot.py --workers 24
```

```bash
python scripts/reasoning/build_cot_gt.py
```

```bash
python scripts/reasoning/apply_cot_gt.py --workers 16
```

```bash
python scripts/release/write_dataset_card.py
```

`write_dataset_card.py` writes the dataset card (`$FLOWBENCH_RELEASE_ROOT/README.md`) from the built release.

## Evaluating VLMs

The test benchmark is `ithor_v11/som` (colour SoM). The three settings are:

| setting | image | text | answer |
|---|---|---|---|
| 1A | before | motion + question | option letter(s) |
| 2A | after | `static_question` | option letter(s) |
| 1B | before | motion + question | reasoning, then `<answer>X</answer>` |

**Any model vLLM can serve**, through its OpenAI-compatible server. Below is the protocol used for the zero-shot tables. `--choice-seed` re-permutes the options; we report the mean over seeds 0, 1 and 2.

```bash
vllm serve models/Qwen3-VL-8B-Instruct --served-model-name Qwen3-VL-8B-Instruct --port 19860 --max-model-len 8192 --trust-remote-code --limit-mm-per-prompt '{"image": 1}'
```

```bash
python scripts/evaluation/eval_server.py --dataset $FLOWBENCH_DATA_ROOT/ithor_v11/som --base-url http://localhost:19860/v1 --model Qwen3-VL-8B-Instruct --image-setting before --answer-mode direct --prompt-format formal --direct-rescue --image-first --choice-seed 0 --out outputs/evals/qwen3vl8b_1A_seed0
```

For 2A, use `--image-setting after`. For 1B, use `--answer-mode cot --prompt-format cit`.

**Models vLLM cannot serve** (e.g. LLaVA-OneVision), with plain transformers and the same prompt:

```bash
python scripts/evaluation/eval_hf.py --model <model dir> --name <name> --image-setting before --dataset $FLOWBENCH_DATA_ROOT/ithor_v11/som --prompt-format formal --choice-seed 0,1,2 --out 'outputs/evals/<name>_1A_seed{seed}'
```

**Local Qwen3-VL** (no server):

```bash
python scripts/evaluation/eval_qwen_local.py --model models/Qwen3-VL-8B-Instruct --dataset $FLOWBENCH_DATA_ROOT/ithor_v11/som --image-setting before --answer-mode direct --gpus 0
```

Each run writes `predictions.jsonl` and `metrics.json` with per-category accuracy. The headline numbers average the six categories without `contact_wall` / `near_object`. `scripts/evaluation/summarize.py <root>` collects the seed means into tables. `scripts/evaluation/check_answer_format.py` reports how often the models answer with a bare letter, and `scripts/evaluation/rescore.py` re-scores saved replies after a parser change.

## Fine-tuning Qwen3-VL

Training uses the curated ProcTHOR rows from `train0`–`train2`. The `val` / `test` splits of the train set are never used.

**1. Build the balanced splits.**

```bash
Q=$FLOWBENCH_DATA_ROOT/procthor_v11/qwen_training; mkdir -p $Q/train_pool
cat $FLOWBENCH_DATA_ROOT/procthor_v11/train{0,1,2}.jsonl > $Q/train_pool/dataset.jsonl
ln -sfn $FLOWBENCH_DATA_ROOT/procthor_v11/som/images $Q/train_pool/images
```

```bash
python training/balance_dataset.py --dataset $Q/train_pool --out $Q/som_balanced/dataset.jsonl
```

```bash
ln -sfn $FLOWBENCH_DATA_ROOT/procthor_v11/som/images $Q/som_balanced/images
```

```bash
python training/split_dataset.py --src $Q/som_balanced/dataset.jsonl --out-dir $Q/som_balanced_splits
```

The 1B (reasoning) splits `som_balanced_splits_1B_c0/` are written by `scripts/reasoning/integrate_cot.py`. Their target is c_0 followed by `<answer>X</answer>`.

**2. Train.** This is LoRA r16 with lr 1e-4, 2 epochs, and an effective batch of 32 (here 4 GPUs × 1 × grad-accum 8). For 1B, use `--mode 1B` and the `som_balanced_splits_1B_c0` files. For Qwen3.5-9B or LLaVA-OneVision-2, use the same command with a different `--model`.

```bash
D=$FLOWBENCH_DATA_ROOT/procthor_v11/qwen_training/som_balanced_splits
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 training/train.py \
  --mode 1A --model models/Qwen3-VL-8B-Instruct --dataset $D \
  --jsonl $D/train.jsonl --val-jsonl $D/val.jsonl --eval-jsonl $D/eval.jsonl \
  --output-dir outputs/runs/procthor_v11_som_1A --num-train-epochs 2 --eval-every-epochs 0.25 \
  --gradient-accumulation-steps 8 --save-total-limit 10 --gpu-mem-fraction 0.80 --skip-image-check
```

**3. Evaluate.** `training/evaluate.py` uses exactly the training prompt. To serve the model with vLLM instead, merge the adapter first and use `scripts/evaluation/eval_server.py --prompt-format cit`.

```bash
python training/evaluate.py --mode 1A --model outputs/runs/procthor_v11_som_1A --dataset $FLOWBENCH_DATA_ROOT/ithor_v11/som --choice-seed 0 --out outputs/evals/ft_1A_seed0
```

```bash
python training/merge_lora.py outputs/runs/procthor_v11_som_1A
```

`training/visualize_inputs.py` renders training examples (image, prompt, target) as HTML, so you can check the inputs.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{liu2026flowbench,
  title        = {FlowBench: Counterfactual Imagination Training for Spatial Intelligence},
  author       = {Liu, Yifan and Zhang, Baihan and Aw, Khai Loong and Kotar, Klemen and Yamins, Daniel L. K.},
  year         = {2026},
  howpublished = {\url{https://github.com/liuyifan22/FlowBench}},
  note         = {Yifan Liu, Baihan Zhang, Khai Loong Aw, and Klemen Kotar contributed equally}
}
```
