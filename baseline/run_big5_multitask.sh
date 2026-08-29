#!/bin/bash
# Runs JUST the BIG-5 multitask_direct (DenseNet121) evaluation, for BOTH
# platforms (Twitter and Weibo), appending into the SAME shared results CSV
# run_all_experiments.sh uses. Split out from that script so you can get
# these two numbers without waiting on ImageNet/Places/COCO (or re-running
# them after a failure like the COCO/Q2L one).
#
# USAGE:
#   sbatch run_big5_multitask.sh
# or:
#   bash run_big5_multitask.sh
#
# Same VERIFIED/UNVERIFIED path caveat as run_all_experiments.sh -- see that
# script's header. The BIG-5 *_GT_CSV filenames and images dirs below are
# VERIFIED against scripts/job_vlm_pipeline.sh / job_vlm_pipeline_heavy.sh
# (which already run successfully against these exact paths). MULTITASK_CHECKPOINT
# is still an UNVERIFIED guess; confirm it against `ls ~/code/weights/`.

#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=closed_set_big5_multitask
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=../logs/closed_set_big5_multitask.log

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
# CONFIG -- edit to match the cluster (same values as run_all_experiments.sh)
# ============================================================================
CODE_ROOT="/home/pmonserrat/code"
DATA_ROOT="/home/pmonserrat/datasets"
RESULTS_DIR="${CODE_ROOT}/results/closed_set_baseline"
RESULTS_CSV="${RESULTS_DIR}/closed_set_baseline_results.csv"
EXCEL_PATH="${CODE_ROOT}/data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx"

WANDB_FLAG="--wandb"   # set to "" to disable W&B logging

BIG5_TWITTER_IMAGES_DIR="${DATA_ROOT}/big_5/twitter"
BIG5_WEIBO_IMAGES_DIR="${DATA_ROOT}/big_5/weibo"

# VERIFIED against scripts/job_vlm_pipeline.sh / job_vlm_pipeline_heavy.sh.
TWITTER_EN_GT_CSV="${DATA_ROOT}/big_5/annotations/twitter-en-6_majority.csv"
TWITTER_ES_GT_CSV="${DATA_ROOT}/big_5/annotations/twitter-es-6_majority.csv"
WEIBO_CH0_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-0_majority.csv"
WEIBO_CH1_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-1_majority.csv"

# UNVERIFIED -- confirm against `ls ~/code/weights/`.
MULTITASK_CHECKPOINT="${CODE_ROOT}/weights/trained_DenseNet121_100epochs.pth"
MULTITASK_BACKBONE="DenseNet121"

mkdir -p "${RESULTS_DIR}"
echo "=== Results CSV: ${RESULTS_CSV}"

for BIG5_DATASET in big5_twitter big5_weibo; do
    echo ""
    echo "############################################################"
    echo "# BIG-5 (${BIG5_DATASET}): multitask_direct (${MULTITASK_BACKBONE})"
    echo "############################################################"
    python evaluate_big5.py \
        --dataset "${BIG5_DATASET}" \
        --twitter_en_gt_csv "${TWITTER_EN_GT_CSV}" \
        --twitter_es_gt_csv "${TWITTER_ES_GT_CSV}" \
        --twitter_images_dir "${BIG5_TWITTER_IMAGES_DIR}" \
        --weibo_ch0_gt_csv "${WEIBO_CH0_GT_CSV}" \
        --weibo_ch1_gt_csv "${WEIBO_CH1_GT_CSV}" \
        --weibo_images_dir "${BIG5_WEIBO_IMAGES_DIR}" \
        --excel_path "${EXCEL_PATH}" \
        --model_family multitask_direct \
        --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" \
        --multitask_backbone_choice "${MULTITASK_BACKBONE}" \
        --diagnostic_sample_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_diagnostic_sample.json" \
        --comparison_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_model_comparison.json" \
        --output_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_multitask_densenet121.json" \
        --results_csv "${RESULTS_CSV}" \
        ${WANDB_FLAG} --verbose
done

echo ""
echo "=== Done. Pivot ${RESULTS_CSV} (filter model_type=multitask_direct, dataset in [big5_twitter, big5_weibo])."
