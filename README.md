# FlowBench: Counterfactual Imagination Training for Spatial Intelligence

Yifan Liu\*, Baihan Zhang\*, Khai Loong Aw\*, Klemen Kotar\*, Daniel L. K. Yamins

\* Equal contribution

A new benchmark emphasizing counterfactual imagination of object motions inside indoor scenes.

- Automatic Annotations
- Scaled Dataset
- Challenging Benchmark

The dataset will be released in about 1.5 months (**before November 2026**). For a download link, please contact Yifan: [lyf@stanford.edu](mailto:lyf@stanford.edu).

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

The release is built in stages. File names keep the internal version numbers, so you can match each script to its stage:

```
 AI2-THOR + VLM wording            v9: 6 spatial categories     v10: feasibility, contact_wall
 ─────────────────────────         ProcTHOR  build_v9_distributed_procthor.py   build_v10_distributed_procthor.py
                                   iTHOR     build_v9_distributed_ithor.py      build_v10_distributed_ithor.py
                                                                  └── merge_ithor_v10_from_v9.py
        │
        ▼  fix_static_question_som.py        (SoM marks in static_question)
        │
 v11 = merged, colour SoM, cleaned           build_v11_colorsom_{procthor,ithor}.py
        │                                    reshuffle_v11_answer_letters.py, apply_v11_occlusion_fix.py
        ▼
 release (natural / colour_som / red_som)    flowbench_release/
        │
        ▼  reasoning annotations             c_gt (rules), c_1 (VLM, sees after image), c_0 (VLM, before image only)
        ▼
 VLM eval / Qwen fine-tuning                 eval_*.py, qwen_finetune/
```

## Repository layout

```
cit_pipeline/                      core library: AI2-THOR control, edit search, QA generators, SoM marks,
                                   GT flow / depth, CoT templates, MCQ export, eval prompts
build_v9_distributed_{procthor,ithor}.py    raw generation, six spatial categories
build_v10_distributed_{procthor,ithor}.py   raw generation, feasibility + contact_wall
merge_ithor_v10_from_v9.py         iTHOR: v10 feasibility / contact_wall + v9 for the other categories
fix_static_question_som.py         puts SoM mark ids into static_question (text-only VLM rewrite)
build_v11_colorsom_procthor.py     ProcTHOR v11: merge v9 + v10, colour SoM, filters, train/val/test splits
build_v11_colorsom_ithor.py        iTHOR v11 (the test benchmark)
reshuffle_v11_answer_letters.py    balances correct-answer letters per category
apply_v11_occlusion_fix.py         occlusion filter / relabel / rewording
backfill_procthor_v9_cot.py        c_1 ("warm" CoT) for every question
make_v11_cot_filter.py             which v11 rows can reuse their c_1
gen_c0_procthor_v11.py             c_0 ("pretended" CoT) with Qwen3.5-9B
integrate_c0_v11.py                collects c_gt / c_1 / c_0 and writes the filters
recover_cgt.py, apply_cgt.py       rebuilds c_gt from simulator metadata for every row and style
flowbench_release/                 release builder: folders, parquet shards, CoT columns, dataset card
eval_qwen3_vl_8b.py                local Qwen3-VL eval (vLLM or HF backend)
eval_qwen3_vl_32b_server.py        eval against any OpenAI-compatible server (vLLM)
eval_hf_vlm_zeroshot.py            transformers eval for VLMs vLLM cannot serve
viz_ithor_v11_benchmark.py         HTML viewer for the test benchmark
qwen_finetune/                     LoRA SFT of Qwen3-VL-8B (also Qwen3.5-9B, LLaVA-OneVision-2) + eval helpers
```

## Installation

### Environment

You need Linux with an NVIDIA GPU. AI2-THOR renders headless through Vulkan (`CloudRendering`).

```bash
conda create -n cit python=3.10 -y
conda activate cit
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

Use vLLM only for the VLM servers and the evals. Data generation itself needs only `ai2thor`, `prior`, `numpy`, `pillow`, `opencv` and `openai`. The c_0 annotations were generated with Qwen3.5-9B on vLLM 0.27.1. If your vLLM cannot load Qwen3.5, serve that model from a second environment with a newer vLLM.

### AI2-THOR

- The NVIDIA Vulkan driver must be installed. The code sets `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` unless you set it yourself, and `vulkaninfo` must be on `PATH`.
- On first start AI2-THOR downloads its Unity build (~1 GB) into `~/.ai2thor`.
- Check the installation:

```bash
python -c "from ai2thor.controller import Controller; from ai2thor.platform import CloudRendering; c = Controller(platform=CloudRendering); print(c.step('MoveAhead').metadata['lastActionSuccess']); c.stop()"
```

### ProcTHOR-10K houses

The ProcTHOR builders read the houses from a local copy, so workers never call the GitHub API. They fall back to `prior.load_dataset("procthor-10k")` when the local files are missing.

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

Qwen3-VL-32B-Instruct writes the question wording and c_1 for the six v9 categories. Qwen3.5-9B writes c_1 for feasibility / contact_wall and all of c_0. Qwen3-VL-8B is the fine-tuning base.

### Paths and environment variables

Run every command from the repository root. Scripts find their data through these variables. The defaults are relative to the repository root.

| variable | default | used for |
|---|---|---|
| `CIT_DATA_ROOT` | `data/cit` | generated datasets: `procthor_v9/`, `procthor_v10/`, `ithor_v9/`, `ithor_v10/`, `procthor_v11/`, `ithor_v11/` |
| `FLOWBENCH_RELEASE_ROOT` | `data/flowbench_release` | the release (`sim/train`, `sim/test`, `sim/parquet`) |
| `PROCTHOR_DATA_DIR` | `data/procthor-10k` | ProcTHOR-10K house files |
| `CIT_TMP_DIR` | `~/.cache/cit/tmp` | TMPDIR for AI2-THOR workers (use a disk with space; `/tmp` often fills up on shared nodes) |

```bash
export CIT_DATA_ROOT=/path/to/cit_data
export FLOWBENCH_RELEASE_ROOT=/path/to/flowbench_release
```

The full dataset needs a few TB. The ProcTHOR v9 + v10 sample folders hold most of it, because they include the flow / depth / segmentation arrays.

## Data generation

### 0. Start the wording VLM

The builders ask a VLM to phrase questions and options naturally. They fall back to the template text on VLM errors when `--vlm-optional` is set, and `--no-vlm` skips the VLM entirely.

```bash
vllm serve models/Qwen3-VL-32B-Instruct --served-model-name Qwen3-VL-32B-Instruct --host 0.0.0.0 --port 19857 --tensor-parallel-size 4 --gpu-memory-utilization 0.7 --max-model-len 8192
```

```bash
curl -s http://localhost:19857/v1/models
```

Several builders can share one server. Point them at it with `--vlm-base-url http://<host>:19857/v1`.

### 1. Raw generation in AI2-THOR

Each builder process drives one AI2-THOR instance on one GPU and handles every `num-workers`-th house or scene. Workers are resumable (`--skip-existing`), append to a shared `dataset.jsonl` under a file lock, and write two parallel roots: `natural/` (clean images and text) and `som/` (red Set-of-Marks images and mark-aware text). Run one process per GPU. Spread them over several machines by giving each a distinct `--worker-id` and the same `--num-workers`.

The following runs N workers on the GPUs of one machine. The flags are the ones used for the release.

**ProcTHOR, six spatial categories (v9).** 10,000 houses, 10 apartment-wide cameras per house:

```bash
N=8; OUT=$CIT_DATA_ROOT/procthor_v9; mkdir -p $OUT/natural/logs
for W in $(seq 0 $((N-1))); do
  nohup python -u build_v9_distributed_procthor.py \
    --out $OUT/natural --out-som $OUT/som --gpus $W --worker-id $W --num-workers $N \
    --n-houses 10000 --house-pool 10000 --max-rooms 0 \
    --camera-mode apartment --cameras-per-apartment 10 \
    --vlm-base-url http://localhost:19857/v1 --vlm-model Qwen3-VL-32B-Instruct \
    --vlm-optional --vlm-connect-retries 1 \
    --seed-base 9000 --flip-mode train --flush-every 1 --skip-existing --worker-tag w$W \
    > $OUT/natural/logs/worker$W.log 2>&1 &
done
```

**ProcTHOR, feasibility + contact_wall (v10).** Same command with these changes: `build_v10_distributed_procthor.py`, `OUT=$CIT_DATA_ROOT/procthor_v10`, `--seed-base 11000`, and `--categories contact_wall,feasibility`.

**iTHOR test scenes** (FloorPlan 1–30, 201–230, 301–330, 401–430):

```bash
N=8; OUT=$CIT_DATA_ROOT/ithor_v9; mkdir -p $OUT/natural/logs
for W in $(seq 0 $((N-1))); do
  nohup python -u build_v9_distributed_ithor.py \
    --out $OUT/natural --out-som $OUT/som --gpus $W --worker-id $W --num-workers $N \
    --vlm-base-url http://localhost:19857/v1 --vlm-model Qwen3-VL-32B-Instruct \
    --vlm-optional --vlm-connect-retries 1 \
    --seed-base 9200 --flip-mode train --flush-every 1 --skip-existing --worker-tag w$W \
    > $OUT/natural/logs/worker$W.log 2>&1 &
done
```

Then run `build_v10_distributed_ithor.py` with `OUT=$CIT_DATA_ROOT/ithor_v10`, `--seed-base 10200` and `--categories contact_wall,feasibility`, and merge the two:

```bash
python merge_ithor_v10_from_v9.py
```

Use `tail -f $OUT/natural/logs/worker*.log` to follow progress. To stop a run, kill the builder processes; their AI2-THOR children exit with them. Each sample folder `natural/<sample_id>/` holds `sample.json` (scene, camera, move and every question with its `gt_meta`), the before / after renders, `*_gt_flow.npy`, `*_depth.npy`, segmentation and masks. The flat MCQ export is `dataset.jsonl` + `images/`.

### 2. SoM marks in `static_question`

The VLM wording sometimes drops the mark ids from the after-image question. A text-only pass restores them and writes `som/dataset_static_som.jsonl`. Run it on `procthor_v9`, `procthor_v10` and `ithor_v10`. Use `--num-shards` / `--shard-id` to split the work across servers.

```bash
for D in procthor_v9 procthor_v10 ithor_v10; do
  python fix_static_question_som.py --input $CIT_DATA_ROOT/$D/som/dataset.jsonl \
    --output $CIT_DATA_ROOT/$D/som/dataset_static_som.jsonl \
    --vlm-url http://localhost:19857/v1 --model Qwen3-VL-32B-Instruct --workers 16
done
```

### 3. Merge, colour SoM and clean (v11)

None of these steps run AI2-THOR or a VLM. They redraw the marks as text-free colour discs, drop rows with unresolved marks or structural movers (doors, windows, sinks, countertops, ...), and balance the answer letters.

```bash
python build_v11_colorsom_procthor.py --workers 48
```

```bash
python build_v11_colorsom_ithor.py
```

```bash
python reshuffle_v11_answer_letters.py --target procthor
```

```bash
python reshuffle_v11_answer_letters.py --target ithor
```

The occlusion fix drops occludees under 150 px, labels any cover as occluded, and rewords the choice to "Partly or fully hidden". The `occlusion` letters are then rebalanced:

```bash
for T in procthor ithor; do
  python apply_v11_occlusion_fix.py --target $T
  python reshuffle_v11_answer_letters.py --target $T --categories occlusion
done
```

This produces `procthor_v11/{som/dataset.jsonl, train0,train1,train2,val,test.jsonl}` (train) and `ithor_v11/{natural,som}/dataset.jsonl` (the 714-question test benchmark). Write an HTML viewer of the benchmark with:

```bash
python viz_ithor_v11_benchmark.py
```

## Building the release

### Folders and parquet

```bash
python flowbench_release/build_sim_release.py --split test --stage text
```

```bash
python flowbench_release/build_sim_release.py --split test --stage files
```

```bash
python flowbench_release/build_sim_release.py --split train --stage text
```

```bash
python flowbench_release/build_sim_release.py --split train --stage files --workers 48
```

```bash
python flowbench_release/build_sim_parquet.py --split test
```

```bash
python flowbench_release/build_sim_parquet.py --split train --workers 48
```

The release has `sim/<split>/{natural,colour_som,red_som}/{dataset.jsonl,images/}`, `sim/<split>/samples/<sample_id>/` (the GT, hardlinked from the generation roots), and `sim/parquet/<style>/*.parquet` (Hugging Face shards with embedded images). `flowbench_release/swap_colour_source.py` is optional. It replaces the ProcTHOR v11 colour-SoM image folder with a symlink into the release, so the images are stored only once.

### Reasoning annotations

| | written by | sees |
|---|---|---|
| `c_gt` | category rules | simulator metadata (distances, screen positions, cover rates, collision checks) |
| `c_1` (warm) | VLM | question, before **and** after image, c_gt |
| `c_0` (pretended) | VLM | question, before image and motion only, with c_1 as a private hint |

**c_1.** Run on the six v9 categories with Qwen3-VL-32B, and on feasibility / contact_wall with Qwen3.5-9B. Shard across servers with `--num-shards K --shard k`. Each run is resumable.

```bash
python backfill_procthor_v9_cot.py --natural $CIT_DATA_ROOT/procthor_v9/natural --no-splits --vlm-url http://localhost:19857/v1 --model Qwen3-VL-32B-Instruct --workers 48
```

```bash
vllm serve models/Qwen3.5-9B --served-model-name Qwen3.5-9B --port 19859 --max-model-len 8192 --limit-mm-per-prompt.image 2 --enable-prefix-caching
```

```bash
python backfill_procthor_v9_cot.py --natural $CIT_DATA_ROOT/procthor_v10/natural --no-splits --vlm-url http://localhost:19859/v1 --model Qwen3.5-9B --workers 32
```

**Filter, then c_0.** `make_v11_cot_filter.py` keeps the rows whose c_1 exists and agrees with the v11 answer. `gen_c0_procthor_v11.py` sends requests to every server listed in `servers.txt` (one `http://<host>:<port>/v1` per line). The file is re-read while the job runs, so you can add servers as they come up. Replies that fail the checks (answer mismatch, mentions of the after image, motion-direction contradictions, too much copying from c_1) are retried.

```bash
python make_v11_cot_filter.py
```

```bash
python gen_c0_procthor_v11.py --servers-file servers.txt --concurrency 48
```

**Integrate everything.** These steps write `procthor_v11/cot/`, `procthor_v11/filters/`, the 1B training splits, and the `cot_gt` / `cot_1` / `cot_0` fields of the release:

```bash
python integrate_c0_v11.py
```

```bash
python flowbench_release/add_cot_to_release.py --workers 24
```

```bash
python recover_cgt.py
```

```bash
python apply_cgt.py --workers 16
```

```bash
python flowbench_release/write_readme.py
```

`write_readme.py` writes the dataset card (`$FLOWBENCH_RELEASE_ROOT/README.md`) from the built release.

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
python eval_qwen3_vl_32b_server.py --dataset $CIT_DATA_ROOT/ithor_v11/som --base-url http://localhost:19860/v1 --model Qwen3-VL-8B-Instruct --image-setting before --answer-mode direct --prompt-format formal --direct-rescue --image-first --choice-seed 0 --out outputs/evals/qwen3vl8b_1A_seed0
```

For 2A, use `--image-setting after`. For 1B, use `--answer-mode cot --prompt-format cit`.

**Models vLLM cannot serve** (e.g. LLaVA-OneVision), with plain transformers and the same prompt:

```bash
python eval_hf_vlm_zeroshot.py --model <model dir> --name <name> --image-setting before --dataset $CIT_DATA_ROOT/ithor_v11/som --prompt-format formal --choice-seed 0,1,2 --out 'outputs/evals/<name>_1A_seed{seed}'
```

**Local Qwen3-VL** (no server):

```bash
python eval_qwen3_vl_8b.py --model models/Qwen3-VL-8B-Instruct --dataset $CIT_DATA_ROOT/ithor_v11/som --image-setting before --answer-mode direct --gpus 0
```

Each run writes `predictions.jsonl` and `metrics.json` with per-category accuracy. The headline numbers average the six categories without `contact_wall` / `near_object`. `qwen_finetune/summarize_formal.py <root>` collects the seed means into tables. `qwen_finetune/formal_compliance.py` reports how often the models answer with a bare letter, and `qwen_finetune/rescore_formal.py` re-scores saved replies after a parser change.

## Fine-tuning Qwen3-VL

Training uses the ProcTHOR v11 colour-SoM rows from `train0`–`train2`. The v11 `val` / `test` splits are never used.

**1. Build the balanced splits.**

```bash
Q=$CIT_DATA_ROOT/procthor_v11/qwen_training; mkdir -p $Q/train_pool
cat $CIT_DATA_ROOT/procthor_v11/train{0,1,2}.jsonl > $Q/train_pool/dataset.jsonl
ln -sfn $CIT_DATA_ROOT/procthor_v11/som/images $Q/train_pool/images
```

```bash
python qwen_finetune/balance_procthor_v9.py --dataset $Q/train_pool --out $Q/som_balanced/dataset.jsonl
```

```bash
ln -sfn $CIT_DATA_ROOT/procthor_v11/som/images $Q/som_balanced/images
```

```bash
python qwen_finetune/split_procthor_balanced.py --src $Q/som_balanced/dataset.jsonl --out-dir $Q/som_balanced_splits
```

The 1B (reasoning) splits `som_balanced_splits_1B_c0/` are written by `integrate_c0_v11.py`. Their target is c_0 followed by `<answer>X</answer>`.

**2. Train.** This is LoRA r16 with lr 1e-4, 2 epochs, and an effective batch of 32 (here 4 GPUs × 1 × grad-accum 8). For 1B, use `--mode 1B` and the `som_balanced_splits_1B_c0` files. For Qwen3.5-9B or LLaVA-OneVision-2, use the same command with a different `--model`.

```bash
D=$CIT_DATA_ROOT/procthor_v11/qwen_training/som_balanced_splits
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 qwen_finetune/train.py \
  --mode 1A --model models/Qwen3-VL-8B-Instruct --dataset $D \
  --jsonl $D/train.jsonl --val-jsonl $D/val.jsonl --eval-jsonl $D/eval.jsonl \
  --output-dir outputs/runs/procthor_v11_som_1A --num-train-epochs 2 --eval-every-epochs 0.25 \
  --gradient-accumulation-steps 8 --save-total-limit 10 --gpu-mem-fraction 0.80 --skip-image-check
```

**3. Evaluate.** `eval_train_aligned.py` uses exactly the training prompt. To serve the model with vLLM instead, merge the adapter first and use `eval_qwen3_vl_32b_server.py --prompt-format cit`.

```bash
python qwen_finetune/eval_train_aligned.py --mode 1A --model outputs/runs/procthor_v11_som_1A --dataset $CIT_DATA_ROOT/ithor_v11/som --choice-seed 0 --out outputs/evals/ft_1A_seed0
```

```bash
python qwen_finetune/merge_lora.py outputs/runs/procthor_v11_som_1A
```

`qwen_finetune/visualize_train_inputs.py` renders training examples (image, prompt, target) as HTML, so you can check the inputs.

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
