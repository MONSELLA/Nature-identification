# Rejection-sampling fine-tuning (RFT) of the VLM pipeline

Everything needed to fine-tune the pipeline's **language decoder** on BIG-5
lives in this folder. Nothing here changes the pipeline itself — the fine-tuned
model is evaluated by the exact same `run_vlm_pipeline.py` path as every
baseline in the thesis, with `--lora_adapter_path` (or, for a merged
checkpoint, `--model_name`) as the only difference.

```
make_splits.py        70/10/20 train/val/test, grouped by post
build_rft_dataset.py  artifacts -> accepted images -> training examples
rft_common.py         acceptance rule + prompt/target reconstruction (shared)
vision_cache.py       cached frozen-vision embeddings + the equivalence check
train_lora.py         the LoRA/DoRA fine-tune
merge_adapter.py      merge into a standalone checkpoint (DoRA / non-vLLM use only)
job_finetune.sh       all five steps, end to end, on SLURM
tests/                torch-free tests for the data pipeline
```

## LoRA, not DoRA, and why

`train_lora.py` trains **plain LoRA by default** (`--use_dora` is opt-in).
That's a serving decision, not a training one: this pipeline needs the
fine-tuned adapter applied to only *some* calls in a run — extraction and
labeling, never captioning (see below) — on one resident vLLM engine, in a
single pass over the dataset. vLLM's native LoRA serving supports exactly that
for plain LoRA (`--enable-lora`, per-request `lora_request`, sub-millisecond
hot-swap), but **does not support DoRA weights**
([vllm-project/vllm#10849](https://github.com/vllm-project/vllm/issues/10849)
is open; the one PR that attempted it, #14389, is closed and unmerged).
Passing a DoRA-trained adapter through that path wouldn't error — vLLM would
silently apply plain-LoRA math to DoRA-trained weights, which is wrong, not
merely unsupported.

DoRA usually edges out plain LoRA on quality at the same rank, so `--use_dora`
stays available for a training-only comparison or if you're willing to give up
selective serving: merge it (`merge_adapter.py`) and serve every call
adapted, captioning included, same as any other checkpoint.

**Selective serving, no merge, single pass** (the default path):

```bash
python scripts/run_vlm_pipeline.py --dataset big5_twitter \
  --model_family gemma --model_name google/gemma-4-12B-it \
  --lora_adapter_path runs/lora_gemma12b_balanced/adapter \
  --lora_max_rank 16 \
  --split_file /home/pmonserrat/datasets/big_5/rft/splits/test_images.txt ...
```

`--lora_adapter_path` builds the vLLM engine once with the adapter attached
(`src.models.vlm_models.VLLMBackedVLM`), and `src.vlm_pipeline.run_inference`
routes `use_lora=True` to the extraction and labeling calls only —
`caption_batch`/`summarize_caption_batch` take no `use_lora` parameter at all,
so there's no flag to get wrong: the caption call structurally cannot apply
the adapter, the same way it structurally never receives a system prompt.
`--lora_max_rank` must be `>=` the adapter's own training rank
(`train_lora.py --lora_r`, default 16 for both).

## The idea

BIG-5 has only **image-level** annotations, so there is no per-entity
supervision for the extraction and labeling calls. Rejection sampling
manufactures it: run the pipeline, keep the images whose final image-level
prediction matches the human annotation, and treat every VLM call that produced
that answer as a correct demonstration.

The model's own greedy outputs are already on disk
(`results/vlm_pipeline/baseline/big5_*/responses/vlm_responses_*.jsonl`), so
building a training set needs **no new inference**.

## Two things to know before running

### 1. Rejection sampling is biased in the direction you care about

Measured on the `gemma-4-12B-it` BIG-5 artifacts:

| GT | images | accepted | rate |
|---|---|---|---|
| nature | 3634 | 3397 | **93.5%** |
| non-nature | 3029 | 1672 | **55.2%** |
| total | 6663 | 5069 | 76.1% |

The gap is the model's failure mode showing through the filter. The pipeline
over-predicts nature, so the images it most often gets wrong — and therefore
discards — are exactly the ones whose correct answer is "no nature here".
Training on the raw accepted set means training on a 67%-nature mixture drawn
from a 55%-nature dataset, pushing the model further in the direction it is
already wrong.

`--balance` in `build_rft_dataset.py` offers three treatments and the first run
uses `downsample_nature`:

- `none` — every accepted image (~3578 train images, 67% nature).
- `downsample_nature` — subsample accepted nature images to match the
  non-nature count (~2368 train images, 50/50). Costs data, most direct fix.
- `loss_weight` — every image, each example weighted inversely to its GT-nature
  class frequency (normalized to mean 1.0, so the effective learning rate does
  not change with the mixture).

Whichever you pick, **compare against the same baseline on the same test
split** — `job_finetune.sh` step 6 re-scores the existing baseline artifact
through `--split_file` so the comparison is like for like.

### 2. Splits are grouped by POST, and are write-once

A BIG-5 image is one slot of a post (`<platform_id>_<slot>.<ext>`, 3.4 images
per post on average), and images within a post are frequently near-duplicates.
Splitting at image level would put slot 0 in train and slot 1 in test, making
the test score partly a memorization score. Every image of a post therefore
lands in the same split.

The split is derived from image paths and ground truth only — no model output —
so one split file is valid for the self-training run and for every future
distillation run alike. Regenerating it with a different `--seed` invalidates
every number already reported against it.

Observed on the real data (exactly on target despite whole-post assignment):

```
train   4664 images (70.0%)  1377 posts  nature 54.8%
val      666 images (10.0%)   196 posts  nature 54.4%
test    1333 images (20.0%)   395 posts  nature 53.9%
```

## What counts as "correct" (the acceptance rule)

Same thing the evaluation scores — an image-level verdict against an
image-level annotation:

- **GT non-nature** → accepted iff the pipeline predicted **no nature entity at
  all**.
- **GT nature** → accepted iff at least one predicted nature entity carries
  **both** a life-category and a tangibility matching the GT (`--accept_rule
  strict`, the default). `--accept_rule lenient` instead satisfies the two axes
  independently, which is the rule `run_vlm_pipeline.py`'s BIG-5 branch already
  uses for reporting. On the real data the two barely differ (3397 vs 3366
  accepted); strict is the default because a training demonstration should be
  one coherent reading of the scene, not two half-correct ones that add up.
- Coder-disagreement images (`gt_material == [True, False]`) count as **both**
  labels, per the project convention, so an entity matching either direction
  satisfies that axis.

## Which calls become training examples

Default `--stages extraction label_full label_material`, giving ~5 examples per
accepted image (one extraction call plus one labeling call per extracted
entity) — **11803 examples from 2368 balanced images** on the first build.

The free-form **caption** stage is deliberately excluded: the acceptance test
says nothing about caption quality, it is the longest generation in the chain,
and training on it risks dragging the neutral descriptive caption toward
taxonomy vocabulary, which the project keeps out of that call on purpose. It is
not silently dropped — passing `--stages caption` errors out with what would
need implementing (the caption call is the one stage that runs with no system
prompt, so it needs its own example shape).

Reconstruction is **exact**, not approximate: both prompt builders are imported
from `src.models.prompts`, so a prompt change cannot desynchronize the training
data from what inference sends. The one subtlety is that `TaxonomyResponse`'s
two reasoning fields were stored space-joined; `rft_common.split_taxonomy_
reasoning` recovers the boundary from the model's own regular phrasing and
**drops** anything it cannot split rather than inventing one (recovery rate on
the real artifacts: 27213/27213). Runs made from now on store the two fields
separately, so that path is back-compatibility only.

## The visual-embedding cache

The fine-tune trains the language decoder only; the vision tower and projector
are frozen, so an image's soft-token embeddings are **constant for the whole
run**. They are computed once, on first use, and reused — across epochs (the
intended win) and *within* epoch 1, since one image feeds ~5 examples.

`train_lora.py` asserts the premise instead of trusting it: after wrapping the
model it walks every trainable parameter and aborts if any lives outside the
language decoder, because an adapter in the vision path would make every cached
vector go stale after the first optimizer step.

Injecting precomputed embeddings means bypassing the model's own multimodal
forward, and a mistake there (wrong image-token id, off-by-one scatter, dtype
downcast) does not crash — it just trains something worse, indistinguishable
from bad hyperparameters. So before any optimizer step, `--verify_cache N`
runs real batches through **both** the normal `pixel_values` forward and the
cached-embedding forward and aborts unless the losses agree to tolerance. That
check is what makes the cache safe rather than merely plausible. Disable the
whole thing with `--no_vision_cache`.

What the cache does **not** skip is CPU image preprocessing: the processor is
what inserts the right number of image tokens into `input_ids`, and predicting
that count ourselves would break on any model with dynamic image resolution.
Preprocessing runs in dataloader workers and overlaps with the GPU anyway.

Training images go through `load_training_image`, which reproduces inference's
resize **and its JPEG quality-90 re-encode** — otherwise the model would be
fine-tuned on pristine pixels and served images carrying JPEG artifacts.

## Running it

```bash
sbatch fine_tuning/job_finetune.sh          # balanced (default)
BALANCE=none sbatch fine_tuning/job_finetune.sh
```

Or step by step (steps 1-2 are CPU-only and take seconds):

```bash
python fine_tuning/make_splits.py \
  --artifact results/vlm_pipeline/baseline/big5_twitter/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
  --artifact results/vlm_pipeline/baseline/big5_weibo/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
  --out /home/pmonserrat/datasets/big_5/rft/splits
```

```bash
python fine_tuning/build_rft_dataset.py \
  --artifact results/vlm_pipeline/baseline/big5_twitter/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
  --artifact results/vlm_pipeline/baseline/big5_weibo/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
  --splits /home/pmonserrat/datasets/big_5/rft/splits/splits.json --balance downsample_nature \
  --out /home/pmonserrat/datasets/big_5/rft/rft_gemma12b_downsample_nature
```

```bash
python fine_tuning/train_lora.py \
  --model google/gemma-4-12B-it \
  --dataset_dir /home/pmonserrat/datasets/big_5/rft/rft_gemma12b_downsample_nature \
  --output_dir runs/lora_gemma12b_balanced
```

Then evaluate on the held-out test split with the normal pipeline — the
adapter is served **directly**, no merge step, applied only to extraction and
labeling (captioning always runs base weights):

```bash
python scripts/run_vlm_pipeline.py --dataset big5_twitter \
  --model_family gemma --model_name google/gemma-4-12B-it \
  --lora_adapter_path runs/lora_gemma12b_balanced/adapter --lora_max_rank 16 \
  --split_file /home/pmonserrat/datasets/big_5/rft/splits/test_images.txt ...
```

Only needed if you passed `--use_dora` to `train_lora.py` (see "LoRA, not
DoRA, and why" above) or want a standalone checkpoint outside this pipeline:

```bash
python fine_tuning/merge_adapter.py \
  --base google/gemma-4-12B-it \
  --adapter runs/lora_gemma12b_balanced/adapter \
  --out runs/lora_gemma12b_balanced/merged
# then: --model_name runs/lora_gemma12b_balanced/merged, no --lora_adapter_path
```

Hyperparameters are all flags on `train_lora.py` and start at their base
values (`--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --learning_rate 1e-4
--num_train_epochs 2`, cosine schedule, LoRA on every Linear in the language
decoder, `--use_dora` off).

## Distillation (later)

Already supported, no code change: point `build_rft_dataset.py --artifact` at a
heavier model's responses (`vlm_responses_google_gemma-4-31B-it.jsonl`,
`vlm_responses_OpenGVLab_InternVL3_5-38B.jsonl`, …) and train `gemma-4-12B-it`
on the result. Prompts are rebuilt from the artifact's own caption and object
list, so the teacher's caption correctly conditions the teacher's extraction
target. Use the **same** `splits.json` so the test set stays untouched and the
self-training and distilled runs are directly comparable. Every example records
its `source_model`; the only thing that changes is that the examples become
off-policy.

## Tests

```bash
python -m unittest discover -s fine_tuning/tests
```

22 checks over the acceptance rule (including the strict/lenient divergence and
coder disagreement), reasoning recovery (including that it refuses to guess a
boundary), target reconstruction (each target's key order is asserted against
its own pydantic schema, since these schemas are chains of thought and the
reasoning must precede the verdict it justifies), and the guarantee that no
post is ever split across two splits. Torch-free, so the part that decides what
the model learns is checkable without a GPU.
