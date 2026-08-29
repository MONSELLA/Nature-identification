#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=a40
#SBATCH --job-name=vlm_pipeline
#SBATCH --gres=gpu:1
#SBATCH --array=14-15
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# List of models (5 items)
MODELS=(
  "mistral|mistralai/Ministral-3-8B-Instruct-2512-BF16|8192|32"
  "qwen|Qwen/Qwen3.5-9B|8192|32"
  "gemma|google/gemma-4-12B-it|8192|32"
  "llava|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct|8192|32"
  "internvl|OpenGVLab/InternVL3_5-8B|8192|32"
)

# List of datasets (4 items)
DATASETS=("imagenet" "places" "big5_twitter" "big5_weibo")

# Calculate Model and Dataset indices from SLURM_ARRAY_TASK_ID
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

# Extract Model metadata
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH <<< "${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

# Ensure logs directory exists
mkdir -p ../logs

# Dynamically redirect stdout & stderr to out_<dataset>.log
exec > "../logs/out_${DATASET}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Running $MODEL_NAME on $DATASET"

# Configure Dataset-specific arguments
case "$DATASET" in
  "imagenet")
    EXTRA_ARGS="--dataset imagenet \
      --data_dir /home/pmonserrat/datasets/imagenet/extracted_data \
      --run_name vlm_pipeline/baseline/imagenet/ \
      --output_file vlm_pipeline_imagenet_results.json"
    ;;
  "places")
    EXTRA_ARGS="--dataset places365 \
      --data_dir /home/pmonserrat/datasets/places/val_formatted \
      --places_categories_txt /home/pmonserrat/datasets/places/categories_places365.txt \
      --run_name vlm_pipeline/baseline/places/ \
      --output_file vlm_pipeline_places_results.json"
    ;;
  "big5_twitter")
    EXTRA_ARGS="--dataset big5_twitter \
      --big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
      --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
      --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
      --run_name vlm_pipeline/baseline/big5_twitter/ \
      --output_file vlm_pipeline_big5_twitter_results.json"
    ;;
  "big5_weibo")
    EXTRA_ARGS="--dataset big5_weibo \
      --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
      --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
      --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
      --run_name vlm_pipeline/baseline/big5_weibo/ \
      --output_file vlm_pipeline_big5_weibo_results.json"
    ;;
esac

python run_vlm_pipeline.py \
  $EXTRA_ARGS \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --max_model_len "$MAX_LEN" \
  --batch_size "$BATCH" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir /home/pmonserrat/code/results/ \
  --max_new_tokens_caption 248 \
  --summary_max_new_tokens 77 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --gpu_memory_utilization 0.80 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose 