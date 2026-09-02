#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=ground_coco
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2
#SBATCH --output=/dev/null
#
# COCO GROUNDING EVALUATION, END TO END IN ONE JOB:
#
#     VLM inference  ->  SAM3 grounding  ->  scoring
#
# ...the same shape as the VLM-pipeline job (job_vlm_pipeline.sh): one command,
# resumable, writing into the standard <results_dir>/<run_name>/ layout. It
# REPLACES the old two-step job_coco_infer_ground.sh + job_evaluate_grounding.sh
# coco pass, where the artifact had to be produced by one job and scored by a
# second, and a half-finished run could not be continued.
#
# Each stage is its own OS SUBPROCESS (run_pipeline.py), so the VLM's VRAM is
# fully reclaimed before SAM3 loads, and SAM3's before CLIP loads for scoring.
#
# NO-CAPTION CONFIGURATION (--no_caption), matching the current VLM benchmark:
# Stage 1 is skipped and entities are extracted from the image alone. COCO is
# not a ClipMatch dataset, so this composes with no extra flags. The artifact
# header records caption_stage=false, and the run_name keeps these artifacts in
# their own tree so they can never be confused with captioned ones (the model
# slug in the filename is identical either way).
#
# RESUMABLE. --resume means a re-submission after a timeout/preemption picks up
# where the artifact left off instead of re-running inference from scratch:
# run_vlm_pipeline.py skips images already present and appends the rest. The
# grounding stage is likewise incremental — it only grounds records that carry
# no masks yet. Safe to re-run at any point; a completed run costs a model load
# and exits.
#
# OUTPUT LAYOUT (identical to the VLM pipeline's — see README):
#   <RESULTS_DIR>/<RUN_NAME>/
#     vlm_pipeline_coco_results.json         merged results, keyed dataset->model
#     responses/vlm_responses_<slug>.jsonl   the artifact (VLM + grounding)
#     predictions/..._predictions.csv        one browsable row per image
#
# WHAT THE SCORING PRODUCES for COCO (run_vlm_pipeline.py --stage score's COCO
# block): mask-IoU detection against COCO's own instance segmentation — the IoU
# sweep (@0.50/@0.75/@[.50:.95], which reads out mask tightness), the
# small/medium/large size split, label scoring (exact + hierarchical), and
# biotic/material axis agreement. Alongside the usual axis metrics.
#
#   sbatch scripts/job_grounding_coco.sh              # whole model array
#   sbatch --array=1 scripts/job_grounding_coco.sh    # one model
#
# ALWAYS QUOTE excluded_predictions NEXT TO COCO PRECISION. COCO annotates 80
# curated classes, so a correctly-segmented tree is exempted rather than charged
# as a false positive (measured: 76% of predictions on the gemma run). COCO
# precision therefore describes only the minority COCO can adjudicate. Recall
# carries no such caveat. The BIG-5 dense GT (job_grounding_big5.sh) is
# exhaustive and its precision IS directly meaningful — do not compare the two.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# facebook/sam3 is a GATED HuggingFace repo, so grounding needs authentication.
# Either run `hf auth login` once (cached in ~/.cache/huggingface/token and
# picked up automatically), or export HF_TOKEN before `sbatch`. Never hardcode.
#
# UNSET IT IF EMPTY, deliberately: an empty HF_TOKEN is WORSE than none at all —
# huggingface_hub sends a literal "Authorization: Bearer " header (httpx then
# raises `Illegal header value b'Bearer '`) instead of falling back to the
# cached login token.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

# family|hf_name|max_model_len|batch_cap
MODELS=(
  "gemma|google/gemma-4-E4B-it|8192|96"
  "gemma|google/gemma-4-12B-it|8192|64"   # 24 GB of weights on a 48 GB card
  "gemma|google/gemma-4-26B-A4B-it|8192|64"
)

MODEL_IDX=$SLURM_ARRAY_TASK_ID
if [ "$MODEL_IDX" -ge "${#MODELS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$MODEL_IDX is out of range (valid --array 0-$(( ${#MODELS[@]} - 1 )))."
  exit 1
fi
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"
# Matches run_vlm_pipeline.py's own _model_slug() exactly
# (model_name.replace("/", "_")), so the path echoed at the end is the real one.
MODEL_SLUG="${MODEL_NAME//\//_}"

# --- LoRA adapter (optional) --------------------------------------------------
# LORA_ADAPTER=<run dir or its adapter/ subdir> evaluates a fine_tuning/
# train_lora.py adapter served on top of the base model, no merge needed.
#
# THE OUTPUTS MUST BE SEPARATED BY HAND. An adapter does not change
# --model_name, so run_vlm_pipeline.py names its artifact
# vlm_responses_<base slug>.jsonl either way — the path is the ONLY thing that
# keeps an adapter's predictions from being written over (or resumed onto) the
# base model's. Hence the label folded into RUN_NAME below.
# POSITIONAL FIRST, env second. This cluster has been confirmed to drop
# variables passed via `VAR=... sbatch` / `--export=ALL,VAR=...` — the job then
# either silently falls back to the default or, as seen here, Slurm fails to
# retrieve the environment at all and holds the job:
#     (user env retrieval failed requeued held)
# A positional argument always arrives, unaffected by any export policy. The
# env var is kept as a fallback so existing invocations still work.
#   sbatch --array=1 scripts/job_grounding_coco.sh /path/to/lora_run
LORA_ADAPTER="${1:-${LORA_ADAPTER:-}}"
LORA_ARGS=""
LORA_SUFFIX=""
if [ -n "$LORA_ADAPTER" ]; then
    # Accept either the run directory or its adapter/ subdirectory: train_lora.py
    # writes the peft output to <output_dir>/adapter, and both spellings get
    # typed in practice.
    [ -d "$LORA_ADAPTER/adapter" ] && LORA_ADAPTER="$LORA_ADAPTER/adapter"
    if [ ! -f "$LORA_ADAPTER/adapter_config.json" ]; then
        echo "ABORT: $LORA_ADAPTER is not a peft adapter directory (no adapter_config.json)."
        exit 1
    fi
    # Default label = the run directory's own name, which already encodes
    # teacher/configuration (e.g. lora_gemma12b_from_gemma26b_a4b_no_caption_balanced).
    LORA_LABEL="${LORA_LABEL:-$(basename "$(dirname "$LORA_ADAPTER")")}"
    LORA_ARGS="--lora_adapter_path $LORA_ADAPTER --lora_max_rank ${LORA_RANK:-16}"
    LORA_SUFFIX="_${LORA_LABEL}"
    echo "LoRA adapter : $LORA_ADAPTER  (outputs labelled ${LORA_LABEL})"
fi

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR="$CODE_DIR/results/"
RUN_NAME="vlm_pipeline/grounding_no_caption${LORA_SUFFIX}/coco/"

COCO_IMAGES_DIR=/home/pmonserrat/datasets/coco/images/val2017
COCO_INSTANCES_JSON=/home/pmonserrat/datasets/coco/annotations/instances_val2017.json

mkdir -p "$CODE_DIR/logs"
exec > "$CODE_DIR/logs/out_ground_coco_${MODEL_SLUG}${LORA_SUFFIX}.log" 2>&1
cd "$CODE_DIR/scripts" || exit 1

# COCO images are pre-resized benchmark images, not raw social-media
# resolutions, so this needs neither a lowered --batch_size nor --max_num_seqs
# to avoid the BIG-5 vision-encoder OOM (recap v18/v19), and the grounding
# stage's own defaults (--batch_size 8, --max_pairs_per_forward 16) are left
# alone — they exist for exactly this already-modest-image case.
DS_BATCH=96
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))

echo "=============================================================="
echo "COCO grounding evaluation (infer -> ground -> score)"
echo "  model      : $MODEL_NAME  (slug=$MODEL_SLUG)"
echo "  batch_size : $BATCH   max_model_len: $MAX_LEN"
echo "  output     : ${RESULTS_DIR}${RUN_NAME}"
echo "=============================================================="

# --score adds the third subprocess, so this one command produces the finished
# results JSON + predictions CSV rather than leaving scoring as a manual step.
# --instances_json is REQUIRED by the detection block, not optional: COCO's
# per-instance segmentation is deliberately never stored in the artifact (it
# would bloat every record for a scoring-only use), and without it detection is
# skipped outright rather than silently degrading to box matching.
# --instance_grounding stays at its "auto" default: on for a COCO artifact,
# read from the dataset name in the header.
python run_pipeline.py \
    --dataset coco \
    --data_dir "$COCO_IMAGES_DIR" \
    --instances_json "$COCO_INSTANCES_JSON" \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --max_model_len "$MAX_LEN" \
    --batch_size "$BATCH" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --output_file "vlm_pipeline_coco_results.json" \
    --clipscore_model longclip \
    --dtype bfloat16 \
    --trust_remote_code \
    $LORA_ARGS \
    --no_caption \
    --score \
    --resume \
    --verbose

STATUS=$?
echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Results:"
    echo "  ${RESULTS_DIR}${RUN_NAME}vlm_pipeline_coco_results.json"
    echo "  ${RESULTS_DIR}${RUN_NAME}responses/vlm_responses_${MODEL_SLUG}.jsonl"
    echo "  ${RESULTS_DIR}${RUN_NAME}predictions/"
    echo "Read detection_iou_sweep as the headline, and quote excluded_predictions"
    echo "next to precision (see the header of this file for why)."
else
    echo "FAILED (exit $STATUS) — see the subprocess output above for which stage."
    echo "Re-submitting this same task resumes from the finished records."
fi
exit $STATUS
