#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=ground_coco
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3
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
  "gemma|google/gemma-4-12B-it|8192|96"
  "gemma|google/gemma-4-26B-A4B-it|8192|64"
  "qwen|Qwen/Qwen3.6-27B|8192|96"
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

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR="$CODE_DIR/results/"
RUN_NAME="vlm_pipeline/grounding/coco/"

COCO_IMAGES_DIR=/home/pmonserrat/datasets/coco/images/val2017
COCO_INSTANCES_JSON=/home/pmonserrat/datasets/coco/annotations/instances_val2017.json

mkdir -p "$CODE_DIR/logs"
exec > "$CODE_DIR/logs/out_ground_coco_${MODEL_SLUG}.log" 2>&1
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
