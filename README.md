# Nature Identification in Social Media Images

Master's thesis (TFM) for the **BIG-5** project: benchmarking Vision-Language
Models on their ability to detect representations of *nature* in social-media
imagery, and to place what they find on a three-axis taxonomy.

Every image (and every entity within it) is classified on three binary axes:

| axis | question | values |
|---|---|---|
| **nature** | is this nature at all? | nature / no-nature |
| **life category** | living or not? | biotic / abiotic |
| **tangibility** | the real thing, or a depiction of it? | material / immaterial |

The taxonomy definitions are the authoritative reference for all three and live
in [`data/big5_taxonomy/`](data/big5_taxonomy/). They are read at runtime as
system prompts — the model is judged against the same text a human coder used.

Datasets: **BIG-5** (Twitter + Weibo, the target domain, human-annotated) plus
**ImageNet**, **COCO** and **Places365**, whose class vocabularies are mapped
onto the taxonomy so they can serve as additional labelled benchmarks.

---

## Two pipelines

They answer different questions and are never conflated.

**1. VLM pipeline** — *language-based.* What does the model say is in the image?

```
image → caption → entity extraction → WordNet mapping → taxonomy labelling
```

Labelling is a **hybrid**: a WordNet mapping decides an axis when it can, the
VLM decides when it cannot. The mapping is trusted in one direction only — a
node that maps to *nature* is authoritative, a node that maps to *not nature*
is not, because "is nature" is concept-determined while "is not nature" depends
on the instance (a wooden table with visible grain is nature; a painted one is
not, and the class node cannot tell you which this image shows). Tangibility is
**always** the VLM's call, never the mapping's, for the same reason.

**2. Grounding pipeline** — *pixel-based.* Where in the image is it, and how
much of the frame does it occupy?

```
nature entities (from pipeline 1) → SAM3 segmentation → masks → nature relevance score
```

It **enriches the same artifact** produced by pipeline 1 rather than writing a
parallel file, so one record always holds everything predicted for one image.

## Two output files

| file | what it is |
|---|---|
| `vlm_responses_<model>.jsonl` | the **raw prediction record** — caption, entities, per-entity labels and reasoning, hybrid finals, masks, relevance scores. Complete and unflattened, so it can feed metrics not yet invented. Contains no computed metric. |
| `<run>_<dataset>_<model>_predictions.csv` | the **qualitative-review file** — one row per image, everything from the `.jsonl` *plus* every per-image metric computed at scoring time. This file alone should be enough to spot-check a run. |

---

## Layout

```
src/                        importable library — no CLI, no side effects
  vlm_pipeline.py             caption → extract → map → label (+ hybrid resolution)
  grounding_pipeline.py       SAM3 segmentation + nature relevance score
  models/prompts.py           every prompt and response schema, in one place
  models/vlm_models.py        vLLM-backed VLM backends
  loaders/dataset_loader.py   the four datasets + their taxonomy mappings
  loaders/excel_loader.py     the annotated taxonomy graph (WordNet + Excel)
  evaluation/                 clip_metrics · taxonomy_metrics · detection_metrics
                              · grounding_gt_metrics

scripts/                    entry points (run these)
  run_vlm_pipeline.py         THE main entry point — --stage all|infer|score
  run_grounding_pipeline.py   grounding over an existing artifact
  run_pipeline.py             VLM inference → grounding, end to end
  score_grounding_gt.py       score masks against hand-drawn BIG-5 annotations
  job_*.sh                    Slurm launchers — see "Running" below

fine_tuning/                LoRA fine-tuning by rejection sampling (own README)
baseline/                   closed-set CV baselines (the pre-VLM comparison)
labeling_app/               local web app that produced the hand-drawn GT
visualization_app/          local web app for browsing prediction CSVs
data/big5_taxonomy/         taxonomy definitions + the annotated WordNet tree
```

## Running

Everything goes through `scripts/run_vlm_pipeline.py`. `--stage all` runs
inference and scoring as **separate OS subprocesses**, so the VLM's VRAM is
fully released before CLIP loads for scoring.

```bash
python scripts/run_vlm_pipeline.py \
  --dataset big5_twitter \
  --model_family gemma --model_name google/gemma-4-12B-it \
  --big_5_twitter_images_dir  /path/to/big_5/twitter \
  --twitter_en_gt_csv /path/to/twitter-en-6_majority.csv \
  --run_name my_run/big5_twitter/ --output_file results.json
```

On the cluster, the `scripts/job_*.sh` launchers wrap this:

| launcher | what it runs |
|---|---|
| `job_vlm_pipeline.sh` | the main VLM benchmark — model × dataset Slurm array |
| `job_coco_infer_ground.sh` | COCO: VLM inference → SAM3 grounding |
| `job_evaluate_grounding.sh` | grounding scored against COCO **and** BIG-5 GT |
| `job_evaluate_grounding_infer.sh` / `_lora.sh` | the same for a fine-tuned adapter |
| `fine_tuning/job_finetune*.sh`, `job_evaluate.sh` | LoRA training and evaluation |

SAM3 (`facebook/sam3`) is a **gated** HuggingFace repo. Export a token that has
accepted its licence before submitting — never hardcode one in a job script:

```bash
export HF_TOKEN=...
```

## Metrics

Reported per dataset, never merged into a single headline number:

- **Per-axis accuracy / precision / recall / F1** on all three axes.
- **CLIPScore**, **F-CLIPScore** (Oh & Hwang, cited exactly) and
  **Object-CLIPScore** (our F-CLIPScore-inspired variant — deliberately *not*
  called F-CLIPScore).
- **ClipMatch** + **hierarchical precision/recall** (hP/hR/hF1, Wu-Palmer) on
  ImageNet and Places365, which have a closed candidate vocabulary. Hierarchical
  scoring gives partial credit for the right branch of the tree, so predicting
  "bull" for a cow is not scored the same as predicting "airplane".
- **Mask-IoU detection** on COCO — class-agnostic Hungarian matching, an IoU
  sweep across COCO's ladder (@0.50, @0.75, @[.50:.95]) reading out mask
  tightness, and a small/medium/large size split.
- **Nature relevance score** — how much of the frame nature occupies, both as a
  plain coverage ratio and centre-weighted.

Two conventions worth knowing when reading any result: ground-truth-unmapped
instances are *excluded*, prediction-unmapped instances are *penalised as
wrong*, and mapped/unmapped subsets are always reported separately.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Needs a CUDA GPU for the VLM (served via vLLM) and for SAM3.

---

`CLAUDE.md` carries the detailed engineering conventions — exact metric
definitions, routing rules, and the reasoning behind decisions that look
arbitrary from the outside. `data/llm_reference/vlm_pipeline_recap.txt` is the
running design history: what was tried, what was measured, and what was
rejected.
