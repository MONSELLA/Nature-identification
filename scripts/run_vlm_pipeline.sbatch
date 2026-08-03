#!/bin/bash
# =============================================================================
# VLM pipeline — SLURM array launcher (RTX PRO 6000 Blackwell, 96 GB)
#
# ARRAY INDEXING: MODEL_IDX = TASK_ID / 4, DATASET_IDX = TASK_ID % 4, so
# CONSECUTIVE ids sweep the four datasets for ONE model:
#   --array=8-11   -> mistral only, across all 4 datasets  (one model)
#   --array=0-19   -> the full 5-model x 4-dataset grid
#
# NOTE ON --array=0-19 (full grid): ids 0/4/8/12/16 are all DATASET_IDX=0, so
# five different models would write the SAME shared results store
# (<results_dir>/<run_name>/<output_file>, merged in place by
# src/utils.update_results_store). That read-modify-write is not locked, so two
# jobs finishing within the same few milliseconds can lose one model's entry.
# Throttling with `--array=0-19%4` keeps concurrent tasks mostly on DISTINCT
# datasets and shrinks the window; running one model at a time (8-11 style)
# avoids it entirely.
# =============================================================================
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -p a40
#SBATCH --qos=normal
#SBATCH --account=a40
#SBATCH --job-name=vlm_pipeline
#SBATCH --gres=gpu:1
#SBATCH --array=8-11
# Real file, NOT /dev/null: anything that fails BEFORE the `exec` redirect
# below (conda activate, a bad array index) would otherwise vanish silently.
# ../logs must already exist at SUBMIT time — `mkdir -p ../logs` once first.
#SBATCH --output=../logs/slurm_%A_%a.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# family|hf_name|max_model_len|batch_cap
# batch_cap is a PER-MODEL ceiling applied on top of the per-dataset batch
# size below (final batch = min of the two). Only the 12B needs it: its bf16
# weights take ~24 GB before any KV cache or vision activations are allocated,
# vs ~16 GB for the 8-9B models.
MODELS=(
  "qwen|Qwen/Qwen3.5-9B|8192|128"
  "gemma|google/gemma-4-12B-it|8192|64"
  "mistral|mistralai/Ministral-3-8B-Instruct-2512-BF16|8192|128"
  "llava|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct|8192|128"
  "internvl|OpenGVLab/InternVL3_5-8B|8192|128"
)

DATASETS=("imagenet" "places" "big5_twitter" "big5_weibo")

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

mkdir -p ../logs

# Log name carries MODEL as well as dataset: on the full grid five models share
# a dataset, and a dataset-only name would have them all truncate/interleave
# into one file.
exec > "../logs/out_${DATASET}_${MODEL_FAMILY}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Running $MODEL_NAME on $DATASET"

# --- Per-dataset GPU budget -------------------------------------------------
# --gpu_memory_utilization reserves WEIGHTS + KV CACHE; the vision encoder's
# activations come out of what is LEFT OVER. That leftover is where the BIG-5
# OOM lived (recap v18/v19), so the two dataset families want opposite tuning:
#
#   imagenet/places : pre-resized benchmark images, vision encoder barely
#                     stresses the leftover -> push utilization UP (0.9) and
#                     give the KV cache more room for a bigger batch.
#   big5_*          : raw social-media resolutions (arbitrary phone/screenshot
#                     sizes) -> keep utilization at 0.8 so the leftover can
#                     absorb the spikes, and cap CONCURRENT vision-encoding
#                     with --max_num_seqs (the actual v19 lever) rather than
#                     sacrificing --batch_size or --max_image_side.
#
# --batch_size only bounds how many prompts THIS SCRIPT submits per call; vLLM
# still schedules concurrency itself, so past ~128 it stops buying throughput.
case "$DATASET" in
  "imagenet")
    DS_BATCH=128; GPU_UTIL=0.9; MAX_NUM_SEQS=""
    EXTRA_ARGS="--dataset imagenet \
      --data_dir /home/pmonserrat/datasets/imagenet/extracted_data \
      --run_name vlm_pipeline/imagenet/ \
      --output_file vlm_pipeline_imagenet_results.json"
    ;;
  "places")
    DS_BATCH=128; GPU_UTIL=0.9; MAX_NUM_SEQS=""
    EXTRA_ARGS="--dataset places365 \
      --data_dir /home/pmonserrat/datasets/places/val_formatted \
      --places_categories_txt /home/pmonserrat/datasets/places/categories_places365.txt \
      --run_name vlm_pipeline/places/ \
      --output_file vlm_pipeline_places_results.json"
    ;;
  "big5_twitter")
    DS_BATCH=64; GPU_UTIL=0.8; MAX_NUM_SEQS=32
    EXTRA_ARGS="--dataset big5_twitter \
      --big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
      --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
      --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
      --run_name vlm_pipeline/big5_twitter/ \
      --output_file vlm_pipeline_big5_twitter_results.json"
    ;;
  "big5_weibo")
    DS_BATCH=64; GPU_UTIL=0.8; MAX_NUM_SEQS=32
    EXTRA_ARGS="--dataset big5_weibo \
      --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
      --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
      --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
      --run_name vlm_pipeline/big5_weibo/ \
      --output_file vlm_pipeline_big5_weibo_results.json"
    ;;
esac

# Final batch = min(per-dataset budget, per-model ceiling)
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))
SEQS_ARG=""
[ -n "$MAX_NUM_SEQS" ] && SEQS_ARG="--max_num_seqs $MAX_NUM_SEQS"

echo "Config: batch_size=$BATCH gpu_util=$GPU_UTIL ${SEQS_ARG:-(max_num_seqs unset)}"

# --max_samples caps EVERY dataset — this is the smoke-test knob. REMOVE it
# (or raise it) for the real full-scale runs, e.g. the 50k ImageNet val set.
python run_vlm_pipeline.py \
  $EXTRA_ARGS \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --max_model_len "$MAX_LEN" \
  --batch_size "$BATCH" \
  $SEQS_ARG \
  --gpu_memory_utilization "$GPU_UTIL" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir /home/pmonserrat/code/results/ \
  --max_samples 500 \
  --max_new_tokens_caption 248 \
  --summary_max_new_tokens 77 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --dtype bfloat16 \
  --verbose
