#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=rtx3090
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=coco_gemma
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# facebook/sam3 is a GATED HuggingFace repo (grounding needs an authenticated
# token whose account has accepted its license). run_pipeline.py's grounding
# stage falls back to this env var automatically if --hf_token isn't passed
# (src/grounding_pipeline.py's SAM3Grounder), so setting it here is enough —
# no code-side change needed.
# facebook/sam3 is a GATED HuggingFace repo, so grounding needs an authenticated
# token. Export HF_TOKEN in your shell before `sbatch`, or put it in ~/.env —
# never hardcode it here (this file is tracked in git).
export HF_TOKEN="${HF_TOKEN:-}"

# family|hf_name|max_model_len|batch_cap
MODELS=(
  "gemma|google/gemma-4-E4B-it|8192|96"
  "gemma|google/gemma-4-12B-it|8192|96"
)

# --array=2-2 above -> MODEL_IDX=2 -> whatever MODELS[2] currently says — the
# array must stay in sync with whichever slot in MODELS you actually want to
# run. 0=mistral, 1=qwen, 2=gemma, 3=llava, 4=internvl. Currently:
# 2=gemma|google/gemma-4-E4B-it.
MODEL_IDX=$SLURM_ARRAY_TASK_ID
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"

# Filesystem-safe slug identifying the VLM, matching run_vlm_pipeline.py's own
# _model_slug() EXACTLY (model_name.replace("/", "_")) — this is what every
# model's .jsonl artifact is actually named after (vlm_responses_<slug>.jsonl,
# _resolve_responses_file). Building --responses_file from THIS instead of a
# hardcoded literal is the actual fix here: the old hardcoded path silently
# re-scored google_gemma-4-12B-it's artifact under whatever model this array
# slot said, regardless of which model MODELS[MODEL_IDX] actually named.
MODEL_SLUG="${MODEL_NAME//\//_}"

mkdir -p ../logs
exec > "../logs/out_coco_${MODEL_FAMILY}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Scoring $MODEL_NAME (slug=$MODEL_SLUG) on coco"

# --- COCO paths --------------------------------------------------------------
# Confirmed working against the Qwen run already completed on this data.
COCO_IMAGES_DIR="/home/pmonserrat/datasets/coco/images/val2017"
COCO_INSTANCES_JSON="/home/pmonserrat/datasets/coco/annotations/instances_val2017.json"

# This artifact must ALREADY exist — grounded, with instance_grounding on for
# the detection block to run — before this script does anything useful. This
# script runs ONLY --stage score (see the single python call below); it was
# never actually running "VLM + grounding + score" as its old echo line
# claimed, so it cannot produce this file itself. If it's missing, run
# --stage infer (+ grounding) for $MODEL_NAME on coco first.
RESPONSES_FILE="/home/pmonserrat/code/results/vlm_pipeline/baseline/coco/responses/vlm_responses_${MODEL_SLUG}.jsonl"
if [ ! -e "$RESPONSES_FILE" ]; then
    echo "ERROR: no COCO artifact for $MODEL_NAME at $RESPONSES_FILE"
    echo "       Run VLM inference (+ grounding) for this model on coco first —"
    echo "       this script only scores an artifact that already exists."
    exit 1
fi

# COCO images are pre-resized benchmark images (like imagenet/places), not raw
# social-media resolutions — so unlike big5_weibo this needs neither a lowered
# --batch_size nor --max_num_seqs to avoid the BIG-5-specific vision-encoder
# OOM (recap v18/v19). SAM3 itself never competes with the VLM for VRAM either
# way: run_pipeline.py runs infer -> ground -> score as three SEPARATE OS
# subprocesses, so each model's VRAM is fully reclaimed by the OS before the
# next one loads.
DS_BATCH=96
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))

echo "Config: batch_size=$BATCH gpu_util=0.8"
echo "Artifact: $RESPONSES_FILE"

# --instance_grounding is left at its default ("auto"): it turns on SAM3's
# instance head automatically for a COCO artifact, read from the artifact's
# own header, so it doesn't need to be passed here explicitly.
#
# --detection_iou_threshold / --instance_score_threshold are both left at
# their defaults (0.5 / SAM3's own 0.3) rather than overridden.
#
# No --max_samples — this is the REAL full run over all of val2017, not the
# smoke test.
python run_vlm_pipeline.py --stage score \
    --dataset coco \
    --responses_file "$RESPONSES_FILE" \
    --instances_json "$COCO_INSTANCES_JSON" \
    --clipscore_model longclip \
    --results_dir /home/pmonserrat/code/results/ \
    --run_name vlm_pipeline/baseline/coco/ \
    --output_file vlm_pipeline_coco_results.json \
    --verbose