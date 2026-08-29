#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH -p a40
#SBATCH --qos=q_a40
#SBATCH --account=ga40
#SBATCH --job-name=taxonomy_labeling
#SBATCH --output=../logs/out_taxonomy.log
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# Tell vLLM to skip JIT-compiling the FlashInfer sampler
export VLLM_USE_FLASHINFER_SAMPLER=0

# family | model_name | max_model_len | batch_size
MODELS=(
  "qwen|Qwen/Qwen3.5-9B|8192|32"
  "gemma|google/gemma-4-E4B-it|8192|32"
  "mistral|mistralai/Ministral-3-8B-Instruct-2512|8192|32"
  "llava|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct|8192|32"
  "internvl|OpenGVLab/InternVL3_5-8B|8192|32"
)

IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_')

echo "Running $MODEL_FAMILY / $MODEL_NAME (len=$MAX_LEN, batch=$BATCH)"

#--data_dir /home/pmonserrat/datasets/imagenet/extracted_data \

#--data_dir /home/pmonserrat/datasets/places/val_formatted \

python evaluate_taxonomy_labeling.py \
  --dataset imagenet \
  --data_dir /home/pmonserrat/datasets/imagenet/extracted_data \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --max_model_len "$MAX_LEN" \
  --batch_size "$BATCH" \
  --results_dir /home/pmonserrat/code/results/ \
  --run_name "taxonomy_labeling/" \
  --output_file evaluate_taxonomy_labeling.json \
  --num_preds_to_store 1000 \
  --trust_remote_code \
  --verbose
