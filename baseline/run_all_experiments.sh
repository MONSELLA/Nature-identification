#!/bin/bash
# Launches every closed-set baseline experiment (Tables 1-4: ImageNet,
# Places365, COCO, BIG-5-Twitter, and BIG-5-Weibo) back to back, on the
# cluster, all appending into ONE shared results CSV that can be pivoted
# straight into the thesis tables.
#
# USAGE (submit as a SLURM job from baseline/):
#   sbatch run_all_experiments.sh
# or run directly (e.g. on an interactive GPU node):
#   bash run_all_experiments.sh
#
# Edit the "CONFIG" section below to match actual checkpoint/data paths on
# the cluster before running. Paths below fall into two groups:
#   VERIFIED   -- confirmed against another file that actually RUNS
#                 successfully against these exact paths on the cluster
#                 (scripts/job_vlm_pipeline.sh, scripts/job_vlm_pipeline_heavy.sh,
#                 scripts/job_coco_infer_ground.sh, src/loaders/excel_loader.py's
#                 own default, scripts/evaluate_taxonomy_labeling.py's argparse
#                 defaults): EXCEL_PATH, IMAGENET_DIR, PLACES_DIR,
#                 places_categories_txt (count_classes call), COCO_IMAGES_DIR,
#                 COCO_INSTANCES_JSON, BIG5_*_IMAGES_DIR, and all four BIG-5
#                 *_GT_CSV paths.
#   UNVERIFIED -- no other file in this repo references these; they're
#                 carried over from baseline/closed_set_job_test.sh's
#                 commented-out example invocations (which may themselves be
#                 stale) or are plain guesses at a plausible layout: every
#                 *_WEIGHTS/*_CHECKPOINT/*_CONFIG path and Q2L_REPO_PATH.
#                 CONFIRM these against the actual cluster filesystem
#                 (e.g. `ls ~/code/weights/`) before trusting a run.

#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=closed_set_all
#SBATCH --gres=gpu:l40s:1 
#SBATCH --output=../logs/closed_set_all.log

set -euo pipefail
# Always run from baseline/. Under `sbatch`, SLURM copies this script into a
# transient spool directory (/var/spool/slurm/...) before executing it, so
# `dirname "${BASH_SOURCE[0]}"` resolves to THAT spool dir, not this file's
# real location on disk -- `python count_classes.py` then fails with "No such
# file or directory" because count_classes.py never existed there. SLURM sets
# SLURM_SUBMIT_DIR to the directory `sbatch` was actually invoked FROM, which
# is what we want; fall back to BASH_SOURCE's dirname for a plain
# `bash run_all_experiments.sh` invocation (no SLURM_SUBMIT_DIR set there).
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
# Land in baseline/ itself regardless of whether sbatch was invoked FROM
# baseline/ (SLURM_SUBMIT_DIR already correct) or from the repo root
# (SLURM_SUBMIT_DIR is the repo root instead) -- verified by count_classes.py
# actually being present, rather than assumed from the path shape.
if [[ ! -f count_classes.py && -f baseline/count_classes.py ]]; then
    cd baseline
fi
if [[ ! -f count_classes.py ]]; then
    echo "FATAL: could not locate baseline/count_classes.py from $(pwd) " \
         "(SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}). Submit this job from the " \
         "repo root or from baseline/ itself." >&2
    exit 1
fi

# ============================================================================
# CONFIG -- edit these to match the cluster
# ============================================================================
CODE_ROOT="/home/pmonserrat/code"
DATA_ROOT="/home/pmonserrat/datasets"
RESULTS_DIR="${CODE_ROOT}/results/closed_set_baseline"
RESULTS_CSV="${RESULTS_DIR}/closed_set_baseline_results.csv"
EXCEL_PATH="${CODE_ROOT}/data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx"

WANDB_FLAG="--wandb"   # set to "" to disable W&B logging for every run below

# Dataset roots -- IMAGENET_DIR/PLACES_DIR VERIFIED against
# scripts/job_vlm_pipeline.sh's own working --data_dir values.
IMAGENET_DIR="${DATA_ROOT}/imagenet/extracted_data"
PLACES_DIR="${DATA_ROOT}/places/val_formatted"
PLACES_WEIGHTS="${CODE_ROOT}/weights/resnet50_places365.pth.tar"
COCO_IMAGES_DIR="${DATA_ROOT}/coco/images/val2017"
COCO_INSTANCES_JSON="${DATA_ROOT}/coco/annotations/instances_val2017.json"
BIG5_TWITTER_IMAGES_DIR="${DATA_ROOT}/big_5/twitter"
BIG5_WEIBO_IMAGES_DIR="${DATA_ROOT}/big_5/weibo"

# BIG-5 majority-vote ground-truth CSVs -- VERIFIED against
# scripts/job_vlm_pipeline.sh / scripts/job_vlm_pipeline_heavy.sh, which
# already run successfully against these exact paths on the cluster.
TWITTER_EN_GT_CSV="${DATA_ROOT}/big_5/annotations/twitter-en-6_majority.csv"
TWITTER_ES_GT_CSV="${DATA_ROOT}/big_5/annotations/twitter-es-6_majority.csv"
WEIBO_CH0_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-0_majority.csv"
WEIBO_CH1_GT_CSV="${DATA_ROOT}/big_5/annotations/weibo-ch-6-B-1_majority.csv"

# Query2Labels checkpoint (COCO)
Q2L_REPO_PATH="/home/pmonserrat/query2labels"
Q2L_CONFIG="${CODE_ROOT}/weights/query2labels/config_r101.json"
Q2L_CHECKPOINT="${CODE_ROOT}/weights/query2labels/checkpoint_r101.pkl"

# Paula Feliu's TFG multitask-direct checkpoint (used for every dataset's
# "DenseNet121 (trained on BIG-5)" row)
MULTITASK_CHECKPOINT="${CODE_ROOT}/weights/trained_DenseNet121_100epochs.pth"
MULTITASK_BACKBONE="DenseNet121"

BATCH_SIZE=128

mkdir -p "${RESULTS_DIR}"
echo "=== Results dir: ${RESULTS_DIR}"
echo "=== Results CSV: ${RESULTS_CSV}"

# ============================================================================
# 0. DIAGNOSTIC: taxonomy class-coverage tables (not part of the results CSV)
# ============================================================================
echo ""
echo "############################################################"
echo "# 0. count_classes.py -- taxonomy coverage diagnostics"
echo "############################################################"
python count_classes.py \
    --dataset all \
    --excel_path "${EXCEL_PATH}" \
    --imagenet_dir "${IMAGENET_DIR}" \
    --coco_instances_json "${COCO_INSTANCES_JSON}" \
    --places_categories_txt "${DATA_ROOT}/places/categories_places365.txt" \
    --output_file "${RESULTS_DIR}/count_classes_report.txt"

python count_classes.py \
    --dataset coco_dense \
    --excel_path "${EXCEL_PATH}" \
    --coco_instances_json "${COCO_INSTANCES_JSON}" \
    --output_file "${RESULTS_DIR}/count_classes_report.txt"

# ============================================================================
# 1. TABLE 1/2: IMAGENET
# ============================================================================
echo ""
echo "############################################################"
echo "# 1. evaluate_imagenet.py"
echo "############################################################"
for MODEL in convnext_base vit_b_16 swin_v2_b; do
    echo "--- ImageNet: ${MODEL} ---"
    python evaluate_imagenet.py \
        --data_dir "${IMAGENET_DIR}" \
        --excel_path "${EXCEL_PATH}" \
        --model_name "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --output_file "${RESULTS_DIR}/imagenet_${MODEL}.json" \
        --results_csv "${RESULTS_CSV}" \
        ${WANDB_FLAG} --verbose
done

echo "--- ImageNet: DenseNet121 (multitask_direct) ---"
python evaluate_imagenet.py \
    --data_dir "${IMAGENET_DIR}" \
    --excel_path "${EXCEL_PATH}" \
    --model_type multitask_direct \
    --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" \
    --multitask_backbone_choice "${MULTITASK_BACKBONE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_file "${RESULTS_DIR}/imagenet_multitask_densenet121.json" \
    --results_csv "${RESULTS_CSV}" \
    ${WANDB_FLAG} --verbose

# ============================================================================
# 2. TABLE 1/2: PLACES365
# ============================================================================
echo ""
echo "############################################################"
echo "# 2. evaluate_places.py"
echo "############################################################"
python evaluate_places.py \
    --data_dir "${PLACES_DIR}" \
    --excel_path "${EXCEL_PATH}" \
    --model_name resnet50 \
    --places_weights "${PLACES_WEIGHTS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_file "${RESULTS_DIR}/places_resnet50.json" \
    --results_csv "${RESULTS_CSV}" \
    --allow-unresolved ${WANDB_FLAG} --verbose

echo "--- Places365: DenseNet121 (multitask_direct) ---"
python evaluate_places.py \
    --data_dir "${PLACES_DIR}" \
    --excel_path "${EXCEL_PATH}" \
    --model_type multitask_direct \
    --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" \
    --multitask_backbone_choice "${MULTITASK_BACKBONE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_file "${RESULTS_DIR}/places_multitask_densenet121.json" \
    --results_csv "${RESULTS_CSV}" \
    --allow-unresolved ${WANDB_FLAG} --verbose

# ============================================================================
# 3. TABLE 3: COCO
# ============================================================================
echo ""
echo "############################################################"
echo "# 3. evaluate_coco.py"
echo "############################################################"
python evaluate_coco.py \
    --images_dir "${COCO_IMAGES_DIR}" \
    --instances_json "${COCO_INSTANCES_JSON}" \
    --excel_path "${EXCEL_PATH}" \
    --model_type q2l \
    --q2l_repo_path "${Q2L_REPO_PATH}" \
    --q2l_config "${Q2L_CONFIG}" \
    --checkpoint_path "${Q2L_CHECKPOINT}" \
    --output_file "${RESULTS_DIR}/coco_q2l.json" \
    --results_csv "${RESULTS_CSV}" \
    ${WANDB_FLAG} --verbose

echo "--- COCO: DenseNet121 (multitask_direct) ---"
python evaluate_coco.py \
    --images_dir "${COCO_IMAGES_DIR}" \
    --instances_json "${COCO_INSTANCES_JSON}" \
    --excel_path "${EXCEL_PATH}" \
    --model_type multitask_direct \
    --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" \
    --multitask_backbone_choice "${MULTITASK_BACKBONE}" \
    --output_file "${RESULTS_DIR}/coco_multitask_densenet121.json" \
    --results_csv "${RESULTS_CSV}" \
    ${WANDB_FLAG} --verbose

# ============================================================================
# 4. TABLE 4: BIG-5 (Twitter and Weibo, kept as SEPARATE runs/rows -- see
#    CLAUDE.md: pooling the two platforms silently averages them together)
# ============================================================================
run_big5_family () {
    local BIG5_DATASET="$1"   # big5_twitter | big5_weibo
    local FAMILY="$2"
    local TAG="$3"            # short, filename-safe identifier for this specific model
    shift 3
    echo "--- BIG-5 (${BIG5_DATASET}): ${FAMILY} (${TAG}) ---"
    python evaluate_big5.py \
        --dataset "${BIG5_DATASET}" \
        --twitter_en_gt_csv "${TWITTER_EN_GT_CSV}" \
        --twitter_es_gt_csv "${TWITTER_ES_GT_CSV}" \
        --twitter_images_dir "${BIG5_TWITTER_IMAGES_DIR}" \
        --weibo_ch0_gt_csv "${WEIBO_CH0_GT_CSV}" \
        --weibo_ch1_gt_csv "${WEIBO_CH1_GT_CSV}" \
        --weibo_images_dir "${BIG5_WEIBO_IMAGES_DIR}" \
        --excel_path "${EXCEL_PATH}" \
        --model_family "${FAMILY}" \
        --diagnostic_sample_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_diagnostic_sample.json" \
        --comparison_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_model_comparison.json" \
        --output_file "${RESULTS_DIR}/big5_${BIG5_DATASET}_${TAG}.json" \
        --results_csv "${RESULTS_CSV}" \
        "$@" \
        ${WANDB_FLAG} --verbose
}

for BIG5_DATASET in big5_twitter big5_weibo; do
    echo ""
    echo "############################################################"
    echo "# 4. evaluate_big5.py -- ${BIG5_DATASET}"
    echo "############################################################"

    run_big5_family "${BIG5_DATASET}" imagenet convnext_base --model_name convnext_base
    run_big5_family "${BIG5_DATASET}" imagenet vit_b_16 --model_name vit_b_16
    run_big5_family "${BIG5_DATASET}" imagenet swin_v2_b --model_name swin_v2_b

    run_big5_family "${BIG5_DATASET}" places resnet50 \
        --places_model_name resnet50 --places_weights "${PLACES_WEIGHTS}"

    run_big5_family "${BIG5_DATASET}" coco_q2l q2l \
        --q2l_repo_path "${Q2L_REPO_PATH}" --q2l_config "${Q2L_CONFIG}" --checkpoint_path "${Q2L_CHECKPOINT}"

    run_big5_family "${BIG5_DATASET}" multitask_direct multitask_densenet121 \
        --multitask_checkpoint_path "${MULTITASK_CHECKPOINT}" --multitask_backbone_choice "${MULTITASK_BACKBONE}"
done

echo ""
echo "=== All closed-set baseline experiments finished."
echo "=== Pivot ${RESULTS_CSV} (columns: dataset, model, category, granularity, accuracy, precision, recall, f1, support, ...) into Tables 1-4."
