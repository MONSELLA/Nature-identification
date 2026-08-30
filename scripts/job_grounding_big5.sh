#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=ground_big5
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3
#SBATCH --output=/dev/null
#
# BIG-5 DENSE-GT GROUNDING EVALUATION, END TO END IN ONE JOB:
#
#     VLM inference  ->  SAM3 grounding  ->  scoring vs the hand-drawn masks
#
# ...the same shape as job_grounding_coco.sh and the VLM-pipeline job: one
# command, resumable, writing into the standard <results_dir>/<run_name>/
# layout. It REPLACES the big5 half of job_evaluate_grounding.sh, which assumed
# a full-dataset artifact already existed and subset it after the fact.
#
# THE 340-IMAGE SUBSET IS SELECTED BEFORE INFERENCE, not after. The GT covers
# 340 hand-drawn images (170 nature + 170 no-nature) out of ~6663 BIG-5 images,
# so make_grounding_split_file.py writes those basenames and --split_file
# restricts BOTH inference and grounding to them. That is ~20x less VLM and
# SAM3 work than infer-everything-then-subset, on exactly the raw social-media
# resolutions that make the vision encoder OOM-prone. It also means this job
# needs no pre-existing artifact — a freshly fine-tuned adapter can be
# evaluated from nothing.
#
# ONE DATASET, BOTH PLATFORMS. --dataset big5 pools Twitter and Weibo
# (dataset_loader.BIG5_DATASETS), which is required here: the annotations span
# both (84 twitter / 86 weibo), and running one platform alone would leave the
# other's GT regions unmatched and silently halve recall.
#
# RESUMABLE, at both stages: --resume skips images already in the artifact, and
# grounding only touches records without masks. Re-run freely.
#
# OUTPUT LAYOUT (identical to the VLM pipeline's — see README):
#   <RESULTS_DIR>/<RUN_NAME>/
#     grounding_gt_results.json                merged, keyed dataset->model
#     responses/vlm_responses_<slug>.jsonl     artifact (VLM + grounding)
#     predictions/..._per_image.csv            one browsable row per image
#
# PRECISION HERE IS THE STRICT ONE. These annotations are EXHAUSTIVE — every
# nature entity in the image was drawn — so nothing is exempted and every
# unmatched prediction is a false positive. COCO's precision is not comparable
# (it exempts its 79 unannotated-class predictions); see job_grounding_coco.sh.
#
#   sbatch scripts/job_grounding_big5.sh              # whole model array
#   sbatch --array=1 scripts/job_grounding_big5.sh    # one model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# See job_grounding_coco.sh for why an EMPTY HF_TOKEN must be unset rather than
# exported (huggingface_hub would send a literal "Authorization: Bearer ").
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

set -o pipefail

# family|hf_name|max_model_len|batch_cap
MODELS=(
  "gemma|google/gemma-4-E4B-it|8192|64"
  "gemma|google/gemma-4-12B-it|8192|64"
  "gemma|google/gemma-4-26B-A4B-it|8192|48"
  "qwen|Qwen/Qwen3.6-27B|8192|64"
)

MODEL_IDX=$SLURM_ARRAY_TASK_ID
if [ "$MODEL_IDX" -ge "${#MODELS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$MODEL_IDX is out of range (valid --array 0-$(( ${#MODELS[@]} - 1 )))."
  exit 1
fi
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"
MODEL_SLUG="${MODEL_NAME//\//_}"

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR="$CODE_DIR/results/"
RUN_NAME="vlm_pipeline/grounding/big5/"
OUT_ROOT="${RESULTS_DIR}${RUN_NAME}"

# BIG-5 data
BIG5_TWITTER_IMAGES=/home/pmonserrat/datasets/big_5/twitter
BIG5_WEIBO_IMAGES=/home/pmonserrat/datasets/big_5/weibo
ANNOT_DIR=/home/pmonserrat/datasets/big_5/annotations

# Hand-drawn grounding GT — the folder holding nature/, no_nature/,
# manifest.json and coco_instances.json (output of
# convert_grounding_annotations.py). Accepts either layout, because what gets
# copied to the cluster has varied: the converter writes these under
# `processed/`, but the tree has also been copied with that level flattened
# away. Whichever actually contains nature/ wins.
BIG5_GROUNDING_ROOT=/home/pmonserrat/datasets/big_5/grounding
if [ -d "${BIG5_GROUNDING_ROOT}/processed/nature" ]; then
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}/processed"
else
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}"
fi

# BIG-5 images are uncapped phone-camera/screenshot resolutions, so both the
# VLM's and SAM3's concurrency stay modest here — what is comfortable on
# pre-resized COCO images is not (recap v18/v19).
DS_BATCH=64
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))
MAX_NUM_SEQS=32
GROUND_BATCH=4
GROUND_MAX_PAIRS=8

mkdir -p "$CODE_DIR/logs" "$OUT_ROOT"
exec > "$CODE_DIR/logs/out_ground_big5_${MODEL_SLUG}.log" 2>&1
cd "$CODE_DIR/scripts" || exit 1

SPLIT_FILE="${OUT_ROOT}grounding_gt_split.txt"
ARTIFACT="${OUT_ROOT}responses/vlm_responses_${MODEL_SLUG}.jsonl"

echo "=============================================================="
echo "BIG-5 dense-GT grounding evaluation (infer -> ground -> score)"
echo "  model      : $MODEL_NAME  (slug=$MODEL_SLUG)"
echo "  GT dir     : $BIG5_GT_DIR"
echo "  batch_size : $BATCH  max_num_seqs: $MAX_NUM_SEQS"
echo "  output     : $OUT_ROOT"
echo "=============================================================="

if [ ! -d "$BIG5_GT_DIR/nature" ]; then
    echo "ABORT: no nature/ directory under ${BIG5_GROUNDING_ROOT}/processed/ or ${BIG5_GROUNDING_ROOT}/"
    echo "       Copy convert_grounding_annotations.py's processed/ output to the cluster first."
    exit 1
fi

# --- 1. Which images the GT covers -------------------------------------------
# Regenerated every run: it is derived purely from the GT directory, costs
# milliseconds, and a stale split file would silently evaluate the wrong subset.
echo
echo "--- building the split file from the GT ---"
python make_grounding_split_file.py --gt_dir "$BIG5_GT_DIR" --out "$SPLIT_FILE" || exit 1
echo "  $(wc -l < "$SPLIT_FILE") annotated images"

# --- 2. Inference + grounding, restricted to those images ---------------------
# --score is deliberately NOT passed: run_vlm_pipeline.py's score stage
# computes the IMAGE-LEVEL axis metrics, which is a different question from the
# dense mask evaluation below and would write a second results file into the
# same tree. The dense scoring is step 3, and it is the point of this job.
echo
echo "--- VLM inference + SAM3 grounding (340 images) ---"
python run_pipeline.py \
    --dataset big5 \
    --big_5_twitter_images_dir "$BIG5_TWITTER_IMAGES" \
    --big_5_weibo_images_dir "$BIG5_WEIBO_IMAGES" \
    --twitter_en_gt_csv "$ANNOT_DIR/twitter-en-6_majority.csv" \
    --twitter_es_gt_csv "$ANNOT_DIR/twitter-es-6_majority.csv" \
    --weibo_ch0_gt_csv "$ANNOT_DIR/weibo-ch-6-B-0_majority.csv" \
    --weibo_ch1_gt_csv "$ANNOT_DIR/weibo-ch-6-B-1_majority.csv" \
    --split_file "$SPLIT_FILE" \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --max_model_len "$MAX_LEN" \
    --batch_size "$BATCH" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --grounding_batch_size "$GROUND_BATCH" \
    --max_pairs_per_forward "$GROUND_MAX_PAIRS" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --dtype bfloat16 \
    --trust_remote_code \
    --resume \
    --verbose \
|| { echo "INFERENCE/GROUNDING FAILED — not scoring an incomplete artifact"; exit 1; }

# --- 3. Score against the hand-drawn masks -----------------------------------
# Void handling is ON by default and should stay on: the annotation draws
# `cloud` on top of `sky` (pairwise GT IoU up to 0.747), and voiding the
# contested pixels is what makes the score independent of whether SAM3's "sky"
# includes the clouds in front of it. Measured on the worst such image: a
# perfect-but-other-convention sky scores 0.436 without voiding (FAILING the
# 0.50 threshold) and 1.000 with it. Pass --no_void for the comparison run.
echo
echo "--- scoring against the dense GT ---"
python score_grounding_gt.py \
    --artifact "$ARTIFACT" \
    --gt_dir "$BIG5_GT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --output_file "grounding_gt_results.json" \
    --dataset big5_grounding

STATUS=$?
echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Results:"
    echo "  ${OUT_ROOT}grounding_gt_results.json   (keyed big5_grounding -> $MODEL_NAME)"
    echo "  ${OUT_ROOT}responses/vlm_responses_${MODEL_SLUG}.jsonl"
    echo "  ${OUT_ROOT}predictions/"
else
    echo "SCORING FAILED (exit $STATUS). The artifact is intact — re-running"
    echo "this job resumes and re-scores without repeating inference."
fi
exit $STATUS
