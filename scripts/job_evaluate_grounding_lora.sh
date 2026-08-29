#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=eval_grounding_lora
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
#
# Evaluate a LoRA-fine-tuned model against the 340 hand-drawn BIG-5 grounding
# annotations, END TO END: VLM inference -> grounding (SAM3) -> scoring.
#
# DIFFERENT FROM job_evaluate_grounding.sh, which explicitly never runs
# inference — it assumes a full-dataset responses artifact already exists and
# just subsets/grounds/scores it. A LoRA adapter has never been evaluated at
# all yet, so there is nothing to subset. This script runs inference too.
#
# COST CONTROL: inference is restricted to exactly the 340 annotated images
# FROM THE START, via `--split_file` (make_grounding_split_file.py + run_vlm_
# pipeline.py's own basename-matched --split_file), never the full ~6663-
# image BIG-5 platforms. This is strictly cheaper than the infer-then-subset
# path the other job script uses, because there is no wasted inference on the
# other ~6300 images in the first place.
#
# OUTPUT NAMING is keyed on ADAPTER_SLUG (the adapter directory's own parent
# folder name — e.g. "lora_gemma12b_from_gemma26b_a4b_balanced"), NOT the base
# model's slug alone. The base model's slug ALONE would collide with the
# vanilla (non-adapted) model's own baseline artifacts and results, since a
# LoRA adapter is served on top of the SAME base model/architecture.
#
# USAGE — all three positional (see job_evaluate_grounding.sh's own note on
# why env-var-through-sbatch is unreliable and positional args are not):
#   sbatch job_evaluate_grounding_lora.sh
#       # uses the adapter you just trained, as the default below
#   sbatch job_evaluate_grounding_lora.sh <adapter_dir> <base_model_name> <model_family>
#       # e.g. .../adapter  google/gemma-4-12B-it  gemma
#
# Run it from this scripts/ directory, like every other job_*.sh here.

# facebook/sam3 is a GATED HuggingFace repo, so grounding needs an authenticated
# token. Export HF_TOKEN in your shell before `sbatch`, or put it in ~/.env —
# never hardcode it here (this file is tracked in git).
export HF_TOKEN="${HF_TOKEN:-}"

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

ADAPTER_PATH="${1:-/home/pmonserrat/code/runs/lora_gemma12b_from_gemma26b_a4b_balanced/adapter}"
BASE_MODEL_NAME="${2:-google/gemma-4-12B-it}"
MODEL_FAMILY="${3:-gemma}"

# The adapter's OWN run-folder name (train_lora.py's output directory),
# e.g. "lora_gemma12b_from_gemma26b_a4b_balanced" — already a unique,
# self-describing identifier (base model + data source + balance mode), so it
# is used as-is for every output filename below rather than inventing a
# separate slug.
ADAPTER_SLUG="$(basename "$(dirname "$ADAPTER_PATH")")"
BASE_SLUG="${BASE_MODEL_NAME//\//_}"

mkdir -p ../logs
exec > "../logs/out_eval_grounding_lora_${ADAPTER_SLUG}.log" 2>&1

echo "=============================================================="
echo "LoRA grounding evaluation"
echo "  ADAPTER_PATH    = $ADAPTER_PATH"
echo "  ADAPTER_SLUG    = $ADAPTER_SLUG    (keys every output filename below)"
echo "  BASE_MODEL_NAME = $BASE_MODEL_NAME"
echo "  MODEL_FAMILY    = $MODEL_FAMILY"
echo "  SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-unset}"
echo "=============================================================="

# GPU visibility check, BEFORE the ~minutes-long vLLM engine load. vLLM
# auto-detects its device from torch's own CUDA platform state and gives no
# useful error when that detection comes back empty (a raw pydantic
# "Device string must not be empty" deep inside DeviceConfig) — confirmed on
# a real run to be a NODE-level GPU/NVML visibility problem, not anything
# this script or the LoRA-serving code controls (VLLMBackedVLM never passes
# `device=`; nothing about --lora_adapter_path touches device selection).
# This turns that failure mode into a 5-second, legible check instead of
# discovering it after --split_file filtering and model download have
# already run.
echo "--- GPU visibility check ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || \
    echo "WARNING: nvidia-smi failed — this node may not actually have a GPU allocated."
python -c "
import torch
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
" || echo "WARNING: torch CUDA check itself failed."
echo "-----------------------------"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

RESULTS_DIR="/home/pmonserrat/code/results/"
# Each PLATFORM gets its OWN run_name subfolder (mirroring the baseline
# vlm_pipeline/baseline/big5_twitter//big5_weibo/ convention) — both platforms
# sharing one folder would have both inference calls try to write the SAME
# default responses filename (vlm_responses_<BASE_SLUG>.jsonl, since the
# default is keyed on --model_name alone, not the adapter) and clobber each
# other.
RUN_ROOT="vlm_pipeline/lora/${ADAPTER_SLUG}"

BIG5_GROUNDING_ROOT="/home/pmonserrat/datasets/big_5/grounding"
if [ -d "${BIG5_GROUNDING_ROOT}/processed/nature" ]; then
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}/processed"
else
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}"
fi
if [ ! -d "$BIG5_GT_DIR/nature" ]; then
    echo "ERROR: no nature/ directory found under either"
    echo "         ${BIG5_GROUNDING_ROOT}/processed/  or  ${BIG5_GROUNDING_ROOT}/"
    exit 1
fi

OUT_DIR="/home/pmonserrat/code/results/grounding_eval"
mkdir -p "$OUT_DIR"

# =============================================================================
# 1. Split file — the 340 annotated image basenames, one per line
# =============================================================================
SPLIT_FILE="${OUT_DIR}/grounding_gt_split_images.txt"
python make_grounding_split_file.py --gt_dir "$BIG5_GT_DIR" --out "$SPLIT_FILE" \
    || exit 1
N_SPLIT=$(wc -l < "$SPLIT_FILE")
echo "split file: $N_SPLIT images -> $SPLIT_FILE"

# =============================================================================
# 2. VLM inference, LoRA-adapted, restricted to the split — ONE call per
#    platform (BIG-5 Twitter/Weibo are separate --dataset values with
#    separate image folders; --split_file matches whichever of the 340 basenames
#    actually live in the platform being loaded and is a no-op for the rest —
#    see run_vlm_pipeline._filter_to_split).
# =============================================================================
run_infer () {
    local dataset="$1"; shift
    local run_name="${RUN_ROOT}/${dataset}"
    local responses_file="${RESULTS_DIR}${run_name}/responses/vlm_responses_${BASE_SLUG}.jsonl"
    echo
    echo "--- VLM inference (LoRA, $dataset) -> $responses_file ---"
    if [ -s "$responses_file" ]; then
        echo "  already exists and is non-empty — skipping re-inference. Delete it"
        echo "  first (or use --resume manually) if you want to force a re-run."
        return 0
    fi
    # --stage infer ONLY: this evaluation needs the raw predictions for
    # grounding, not the VLM pipeline's own CLIP-based axis metrics — skipping
    # --stage score here avoids loading CLIP for no reason.
    #
    # --lora_adapter_path applies the adapter SELECTIVELY on the SAME resident
    # vLLM engine as the base weights (extraction/labeling run adapted, the
    # caption call always runs base — see CLAUDE.md's fine-tuning section).
    # Requires a PLAIN LoRA adapter (train_lora.py's default, --use_dora NOT
    # passed) — vLLM does not support DoRA natively.
    python run_vlm_pipeline.py --stage infer \
        --dataset "$dataset" \
        "$@" \
        --model_family "$MODEL_FAMILY" \
        --model_name "$BASE_MODEL_NAME" \
        --lora_adapter_path "$ADAPTER_PATH" \
        --lora_adapter_name "$ADAPTER_SLUG" \
        --split_file "$SPLIT_FILE" \
        --results_dir "$RESULTS_DIR" \
        --run_name "$run_name" \
        --max_model_len 8192 \
        --batch_size 32 \
        --gpu_memory_utilization 0.80 \
        --dtype bfloat16 \
        --trust_remote_code \
        --verbose
}

run_infer big5_twitter \
    --big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
    --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
    --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
    || { echo "TWITTER INFERENCE FAILED"; exit 1; }

run_infer big5_weibo \
    --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
    --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
    --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
    || { echo "WEIBO INFERENCE FAILED"; exit 1; }

TWITTER_RESPONSES="${RESULTS_DIR}${RUN_ROOT}/big5_twitter/responses/vlm_responses_${BASE_SLUG}.jsonl"
WEIBO_RESPONSES="${RESULTS_DIR}${RUN_ROOT}/big5_weibo/responses/vlm_responses_${BASE_SLUG}.jsonl"

# =============================================================================
# 3. Subset+merge the two freshly-inferred artifacts, ground, score — the
#    SAME flow job_evaluate_grounding.sh's run_big5() uses for an existing
#    model, just pointed at these new files. Since inference was ALREADY
#    restricted to the 340 images, subsetting here is close to a no-op on
#    content — its real job is MERGING the two platform artifacts into one
#    file, which is exactly what score_grounding_gt.py --artifact expects to
#    read.
# =============================================================================
SUBSET="${OUT_DIR}/big5_grounding_subset_${ADAPTER_SLUG}.jsonl"
SUBSET_TMP="${SUBSET}.tmp"
SUBSET_INFO=$(python subset_artifact_for_gt.py \
    --artifact "$TWITTER_RESPONSES" "$WEIBO_RESPONSES" \
    --gt_dir "$BIG5_GT_DIR" \
    --out "$SUBSET_TMP") || exit 1
mv "$SUBSET_TMP" "$SUBSET"
echo "$SUBSET_INFO"

counts=$(echo "$SUBSET_INFO" | grep -o 'SUBSET_GROUNDED=[0-9]*/[0-9]*' | cut -d= -f2)
n_grounded="${counts%%/*}"
n_total="${counts##*/}"

if [ "$n_grounded" -lt "$n_total" ]; then
    echo
    echo "--- grounding $((n_total - n_grounded)) of $n_total records (SAM3) ---"
    # Same modest batch knobs job_evaluate_grounding.sh uses for BIG-5's
    # uncapped social-media resolutions (recap v18/v19).
    python run_grounding_pipeline.py \
        --responses_file "$SUBSET" \
        --in_place \
        --batch_size "${GROUND_BATCH:-4}" \
        --max_pairs_per_forward "${GROUND_MAX_PAIRS:-8}" \
        --verbose \
        || { echo "GROUNDING FAILED — not scoring an ungrounded artifact"; exit 1; }
else
    echo "  all $n_total subset records already grounded — skipping SAM3"
fi

echo
echo "--- scoring ---"
python score_grounding_gt.py \
    --artifact "$SUBSET" \
    --gt_dir "$BIG5_GT_DIR" \
    --out "${OUT_DIR}/big5_grounding_${ADAPTER_SLUG}"

STATUS=$?
echo
echo "=============================================================="
echo "Done (exit status $STATUS)."
echo "  BIG-5 -> ${OUT_DIR}/big5_grounding_${ADAPTER_SLUG}_results.json"
echo "           + _per_image.csv (one browsable row per image)"
echo "=============================================================="
exit $STATUS
