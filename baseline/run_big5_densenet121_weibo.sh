#!/bin/bash
# Runs JUST the BIG-5 Weibo multitask_direct (DenseNet121) evaluation and
# stores, for every evaluated image, its ground truth + prediction
# (nature/biotic/material) in baseline/results/ -- via evaluate_big5.py's
# --per_image_csv, which dumps the FULL per-image result set (not just the
# small fixed --diagnostic_sample_file/--comparison_file cross-model sample).
# Also appends the usual aggregate metrics into the shared results CSV, same
# as run_big5_multitask.sh.
#
# USAGE:
#   sbatch run_big5_densenet121_weibo.sh
# or:
#   bash run_big5_densenet121_weibo.sh
#
# Same VERIFIED/UNVERIFIED path caveat as run_all_experiments.sh -- see that
# script's header. The BIG-5 Weibo *_GT_CSV filenames and images dir below are
# VERIFIED against scripts/job_vlm_pipeline.sh / job_vlm_pipeline_heavy.sh
# (which already run successfully against these exact paths). MULTITASK_CHECKPOINT
# is still an UNVERIFIED guess; confirm it against `ls ~/code/weights/`.

#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=closed_set_big5_densenet121_weibo
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=../logs/closed_set_big5_densenet121_weibo.log

set -euo pipefail
# Same spool-dir issue as run_all_experiments.sh -- see that script's
# comment on this exact block for why BASH_SOURCE alone isn't enough.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
if [[ ! -f evaluate_big5.py && -f baseline/evaluate_big5.py ]]; then
    cd baseline
fi
if [[ ! -f evaluate_big5.py ]]; then
    echo "FATAL: could not locate baseline/evaluate_big5.py from $(pwd) " \
         "(SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}). Submit this job from the " \
         "repo root or from baseline/ itself." >&2
    exit 1
fi

# ============================================================================
# CONFIG -- edit to match the cluster (same values as run_big5_multitask.sh)
# ============================================================================
CODE_ROOT="/home/pmonserrat/code"
DATA_ROOT="/home/pmonserrat/datasets"
RESULTS_DIR="${CODE_ROOT}/results/closed_set_baseline"
PER_IMAGE_RESULTS_DIR="results"
RESULTS_CSV="${RESULTS_DIR}/closed_set_baseline_results.csv"
EXCEL_PATH="${CODE_ROOT}/data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx"

WANDB_FLAG="--wandb"   # set to "" to disable W&B logging

BIG5_WEIBO_IMAGES_DIR="${DATA_ROOT}/big_5/weibo"

# VERIFIED against scripts/job_vlm_pipeline.sh / job_vlm_pipeline_heavy.sh.
WEIBO_CH0_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-0_majority.csv"
WEIBO_CH1_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-1_majority.csv"

# UNVERIFIED -- confirm against `ls ~/code/weights/`.
MULTITASK_CHECKPOINT="${CODE_ROOT}/weights/trained_DenseNet121_100epochs.pth"
MULTITASK_BACKBONE="DenseNet121"

mkdir -p "${RESULTS_DIR}" "${PER_IMAGE_RESULTS_DIR}"
echo "=== Results CSV: ${RESULTS_CSV}"
echo "=== Per-image GT/prediction CSV: ${PER_IMAGE_RESULTS_DIR}/big5_weibo_densenet121_per_image.csv"

python evaluate_big5.py \
    --dataset big5_weibo \
    --weibo_ch0_gt_csv "${WEIBO_CH0_GT_CSV}" \
    --weibo_ch1_gt_csv "${WEIBO_CH1_GT_CSV}" \
    --weibo_images_dir "${BIG5_WEIBO_IMAGES_DIR}" \
    --excel_path "${EXCEL_PATH}" \
    --model_family multitask_direct \
    --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" \
    --multitask_backbone_choice "${MULTITASK_BACKBONE}" \
    --diagnostic_sample_file "${RESULTS_DIR}/big5_big5_weibo_diagnostic_sample.json" \
    --comparison_file "${RESULTS_DIR}/big5_big5_weibo_model_comparison.json" \
    --output_file "${RESULTS_DIR}/big5_big5_weibo_multitask_densenet121.json" \
    --per_image_csv "${PER_IMAGE_RESULTS_DIR}/big5_weibo_densenet121_per_image.csv" \
    --results_csv "${RESULTS_CSV}" \
    ${WANDB_FLAG} --verbose

echo ""
echo "=== Done. Per-image GT/predictions: ${PER_IMAGE_RESULTS_DIR}/big5_weibo_densenet121_per_image.csv"
echo "=== Aggregate metrics appended to: ${RESULTS_CSV} (filter model_type=multitask_direct, dataset=big5_weibo)."
