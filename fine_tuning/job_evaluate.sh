#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=rft_eval
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
#
# EVALUATION-ONLY: re-runs job_finetune.sh's step 4/5 (pooled big5 test-split
# comparison) against an ARBITRARY already-saved LoRA checkpoint, without
# repeating splits/dataset-build/training. For when you want to score a
# checkpoint OTHER than the one $RUN/adapter currently holds -- e.g. an
# intermediate checkpoint-<step> directory, kept on disk by
# --save_total_limit before the final adapter was written, rather than
# retraining or waiting for a from-scratch run's own step 4/5.
#
# ADAPTER_PATH (env, REQUIRED): the LoRA adapter directory to evaluate --
# either the final "<run_dir>/adapter" a training job wrote, or one of its
# intermediate "<run_dir>/checkpoint-<step>" directories (Trainer's periodic
# save calls model.save_pretrained() on the PeftModel, so a checkpoint dir is
# structurally identical to the final adapter dir -- both load fine via
# --lora_adapter_path, no conversion needed).
#
# NOTE ON CHECKPOINT ALIGNMENT: --save_steps and --eval_steps are NOT
# guaranteed to be aligned unless a run used --load_best_model_at_end (which
# forces them to match) -- an unaligned run's checkpoints will NOT land on
# the exact same step as any particular eval pass. If you're trying to
# recover "the checkpoint at eval step X", the closest surviving
# checkpoint-<step> (given --save_total_limit only keeps the N most recent)
# is the best available approximation, not necessarily an exact match --
# confirm what's actually on disk with `ls <run_dir>` first.
#
# RUN_LABEL (env, REQUIRED): a short, distinct name for this evaluation --
# used in run_name/output_file so it lands in its own results path rather
# than overwriting the "vlm_pipeline/rft/big5/" results from evaluating
# $RUN/adapter directly (job_finetune.sh's own step 4).
#
# MODEL (env, default google/gemma-4-12B-it) / LORA_R (env, default 16):
# override only if evaluating a checkpoint trained with different values.
#
# Usage:
#   ADAPTER_PATH=/home/pmonserrat/code/runs/lora_gemma12b_balanced/checkpoint-1200 \
#     RUN_LABEL=checkpoint1200 \
#     sbatch fine_tuning/job_evaluate.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

set -euo pipefail

if [ -z "${ADAPTER_PATH:-}" ]; then
  echo "ERROR: ADAPTER_PATH is required -- point it at the LoRA adapter or " \
       "checkpoint-<step> directory to evaluate." >&2
  exit 1
fi
if [ -z "${RUN_LABEL:-}" ]; then
  echo "ERROR: RUN_LABEL is required -- a short name distinguishing this " \
       "evaluation's results from any other (e.g. 'checkpoint1200')." >&2
  exit 1
fi
if [ ! -d "$ADAPTER_PATH" ]; then
  echo "ERROR: ADAPTER_PATH=$ADAPTER_PATH does not exist." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/home/pmonserrat/code
RESULTS=$CODE/results/vlm_pipeline/baseline
DATA=/home/pmonserrat/datasets/big_5/rft
MODEL=${MODEL:-google/gemma-4-12B-it}
SLUG=${MODEL//\//_}
LORA_R=${LORA_R:-16}

mkdir -p ../logs
exec > "../logs/out_rft_eval_${RUN_LABEL}.log" 2>&1

cd "$CODE/fine_tuning"

if [ ! -f "$DATA/splits/splits.json" ]; then
  echo "ERROR: $DATA/splits/splits.json not found -- run job_finetune.sh " \
       "(or make_splits.py standalone) at least once first, this script " \
       "reuses that split rather than regenerating it." >&2
  exit 1
fi

BIG5_ARGS="--big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
  --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
  --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
  --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
  --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
  --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv"

cd "$CODE/scripts"
echo ""
echo "=========================================================================="
echo "EVALUATION -- LoRA adapter $ADAPTER_PATH on top of $MODEL"
echo "        label: $RUN_LABEL"
echo "        (pooled big5 test split, both platforms together)"
echo "=========================================================================="
python run_vlm_pipeline.py \
  --dataset big5 $BIG5_ARGS \
  --split_file "$DATA/splits/test_images.txt" \
  --model_family gemma \
  --model_name "$MODEL" \
  --lora_adapter_path "$ADAPTER_PATH" \
  --lora_max_rank "$LORA_R" \
  --max_model_len 8192 \
  --batch_size 62 \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir "$CODE/results/" \
  --run_name "vlm_pipeline/rft_eval_${RUN_LABEL}/big5/" \
  --output_file "vlm_pipeline_big5_rft_eval_${RUN_LABEL}_results.json" \
  --max_new_tokens_caption 248 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --gpu_memory_utilization 0.80 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose
