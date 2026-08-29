#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=a40
#SBATCH --job-name=vlm_pipeline_heavy
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=/dev/null

export VLLM_USE_FLASHINFER_SAMPLER=0

# family|hf_name|max_model_len|batch_size
MODELS=(
  "qwen|Qwen/Qwen3.6-27B|8192|16"
  "qwen|Qwen/Qwen3.6-35B-A3B|8192|12"
  "gemma|google/gemma-4-31B-it|8192|16"
  "gemma|google/gemma-4-26B-A4B-it|8192|16"
  "internvl|OpenGVLab/InternVL3_5-38B|8192|8"
  "internvl|OpenGVLab/InternVL3_5-30B-A3B|8192|16"
)

DATASETS=("big5_twitter" "big5_weibo")

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_SIZE <<< "${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

mkdir -p ../logs

# Log name carries MODEL as well as dataset: on the full grid six models share
# a dataset, and a dataset-only name would have them all truncate/interleave
# into one file.
exec > "../logs/out_${DATASET}_${MODEL_FAMILY}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Running $MODEL_NAME on $DATASET"

case "$DATASET" in
  "big5_twitter")
    EXTRA_ARGS="--dataset big5_twitter \
      --big_5_twitter_images_dir ../../datasets/big_5/twitter \
      --twitter_es_gt_csv ../../datasets/big_5/annotations/twitter-es-6_majority.csv \
      --twitter_en_gt_csv ../../datasets/big_5/annotations/twitter-en-6_majority.csv \
      --run_name vlm_pipeline/heavy_tier/big5_twitter/ \
      --output_file vlm_pipeline_big5_twitter_results.json"
    ;;
  "big5_weibo")
    EXTRA_ARGS="--dataset big5_weibo \
      --big_5_weibo_images_dir ../../datasets/big_5/weibo \
      --weibo_ch0_gt_csv ../../datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
      --weibo_ch1_gt_csv ../../datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
      --run_name vlm_pipeline/heavy_tier/big5_weibo/ \
      --output_file vlm_pipeline_big5_weibo_results.json"
    ;;
esac

echo "Config: batch_size=$BATCH_SIZE"

python run_vlm_pipeline.py \
  $EXTRA_ARGS \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --max_model_len "$MAX_LEN" \
  --batch_size "$BATCH_SIZE" \
  --gpu_memory_utilization 0.8 \
  --longclip_repo_path ../../Long-CLIP \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir ../results/ \
  --max_new_tokens_caption 248 \
  --summary_max_new_tokens 77 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose \
  --resume