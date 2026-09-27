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

The caption step is optional. With `--no_caption`, entities are extracted
straight from the image; this is the configuration used for the reported
benchmark runs. The captioned two-pass variant was kept for the caption
ablation (`scripts/significance_test_caption_ablation.py`).

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

## Models evaluated

Fifteen open VLMs from five families, all served through vLLM
(`scripts/job_vlm_pipeline.sh`):

| tier | models |
|---|---|
| lightweight | Qwen3.5-0.8B · InternVL3.5-2B · Ministral-3-3B · Gemma-4-E4B |
| base | Ministral-3-8B · Qwen3.5-9B · Gemma-4-12B · LLaVA-OneVision-2-8B · InternVL3.5-8B |
| mixture-of-experts | Gemma-4-26B-A4B · Qwen3.6-35B-A3B · InternVL3.5-30B-A3B |
| heavyweight | Qwen3.6-27B · Gemma-4-31B · InternVL3.5-38B |

**Closed-set baselines** (`baseline/`): ImageNet classifiers (ConvNeXt-B,
ViT-B/16, Swin-V2-B), a Places365 ResNet-50, a COCO Query2Label multi-label
model, and a DenseNet-121 trained directly on the three taxonomy axes. Their
class predictions go through the same taxonomy mapping, so they can be compared
with the VLMs.

**Fine-tuning** (`fine_tuning/`): LoRA on Gemma-4-12B's language decoder, using
rejection sampling on a 70/10/20 BIG-5 split grouped by post. There are two
variants: self-distillation, and distillation from Gemma-4-26B-A4B. The
fine-tuned model is evaluated through the same pipeline via
`--lora_adapter_path`. See [`fine_tuning/README.md`](fine_tuning/README.md).

---

## Layout

```
src/                          importable library — no CLI, no side effects
  vlm_pipeline.py               caption → extract → map → label (+ hybrid resolution)
  grounding_pipeline.py         SAM3 segmentation + nature relevance score
  models/prompts.py             every prompt and response schema, in one place
  models/vlm_models.py          vLLM-backed VLM backends
  loaders/dataset_loader.py     the four datasets + their taxonomy mappings
  loaders/excel_loader.py       the annotated taxonomy graph (WordNet + Excel)
  evaluation/                   clip_metrics · taxonomy_metrics · detection_metrics
                                · grounding_gt_metrics

scripts/                      entry points
  run_vlm_pipeline.py           THE main entry point — --stage all|infer|score
  run_grounding_pipeline.py     SAM3 grounding over an existing artifact
  run_pipeline.py               VLM inference → grounding, end to end
  convert_grounding_annotations.py  hand-drawn BIG-5 polygons → RLE ground truth
  subset_artifact_for_gt.py     cut an artifact down to the annotated images
  make_grounding_split_file.py  list the annotated images as a --split_file
  score_grounding_gt.py         score masks against the hand-drawn BIG-5 GT
  combine_grounding_metrics.py  merge per-model grounding results into one JSON
  significance_test_caption_ablation.py  paired bootstrap for the caption ablation
  visualize_grounding.py        render one image's nature masks (thesis figures)
  job_*.sh                      Slurm launchers — see "Running" below

fine_tuning/                  LoRA fine-tuning by rejection sampling (own README)
baseline/                     closed-set CV baselines (the pre-VLM comparison)
data/big5_taxonomy/           taxonomy definitions + the annotated WordNet tree
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
  --no_caption \
  --results_dir results --run_name my_run/big5_twitter/
```

Everything lands under `results/my_run/big5_twitter/`: the shared metrics JSON
(`--output_file`, one entry per dataset and model), `responses/` holding the
`.jsonl` artifacts, and `predictions/` holding the per-image CSVs. Use
`--stage infer` or `--stage score` to run one half on its own; scoring never
reruns the VLM.

To add SAM3 grounding to an existing artifact:

```bash
python scripts/run_grounding_pipeline.py \
  --responses_file results/my_run/big5_twitter/responses/vlm_responses_<model>.jsonl \
  --in_place
```

On the cluster, the `scripts/job_*.sh` launchers wrap this:

| launcher | what it runs |
|---|---|
| `scripts/job_vlm_pipeline.sh` | the main VLM benchmark — model × dataset Slurm array |
| `scripts/job_grounding_coco.sh` | COCO: VLM inference → SAM3 grounding → mask-IoU scoring |
| `scripts/job_grounding_big5.sh` | BIG-5: grounding scored against the hand-drawn masks |
| `scripts/job_score_testsplit.sh` | re-score a base model on the fine-tuning test split |
| `fine_tuning/job_finetune*.sh`, `job_evaluate.sh` | LoRA training and evaluation |
| `baseline/run_all_experiments.sh` | every closed-set baseline |

Both grounding launchers also take a LoRA adapter directory as their first
argument, to evaluate a fine-tuned model.

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
  called F-CLIPScore). The first two need a caption, so they read n/a under
  `--no_caption`.
- **ClipMatch** + **hierarchical precision/recall** (hP/hR/hF1, Wu-Palmer) on
  ImageNet and Places365, which have a closed candidate vocabulary. Hierarchical
  scoring gives partial credit for the right branch of the tree, so predicting
  "bull" for a cow is not scored the same as predicting "airplane".
- **Mask-IoU detection** on COCO — class-agnostic Hungarian matching, an IoU
  sweep across COCO's ladder (@0.50, @0.75, @[.50:.95]) reading out mask
  tightness, and a small/medium/large size split.
- **Nature relevance score** — how much of the frame nature occupies, both as a
  plain coverage ratio and centre-weighted.

Three conventions worth knowing when reading any result: ground-truth-unmapped
instances are *excluded*, prediction-unmapped instances are *penalised as
wrong*, and mapped/unmapped subsets are always reported separately.

## Data

No datasets are included. BIG-5 images and annotations belong to the BIG-5
project and are not public. ImageNet, COCO (val2017) and Places365 have to be
downloaded separately, and their paths are passed as CLI flags. The launchers
use the thesis cluster's paths, so edit them for your own setup. The
hand-drawn BIG-5 grounding masks used by `score_grounding_gt.py` are not
included either.

The one data file that is included is the annotated taxonomy,
`data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx`. It maps WordNet synsets onto
the three axes and is what makes the ImageNet, COCO and Places365 labels usable
as ground truth.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Needs a CUDA GPU for the VLM (served via vLLM) and for SAM3.
