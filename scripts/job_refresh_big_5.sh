#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=refresh_big5_gt
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pmonserrat/code/logs/slurm_refresh_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# 1 = dry run only (nothing written, just prints patched/unmatched counts)
# 0 = actually patch the artifacts in place
DRY_RUN=0

ANNOT_DIR=~/datasets/big_5/annotations
TW_EN_GT="$ANNOT_DIR/twitter-en-6_majority.csv"
TW_ES_GT="$ANNOT_DIR/twitter-es-6_majority.csv"
WEIBO_CH0_GT="$ANNOT_DIR/weibo-ch-6-B-0_majority.csv"
WEIBO_CH1_GT="$ANNOT_DIR/weibo-ch-6-B-1_majority.csv"

TW_IMG_DIR=~/datasets/big_5/twitter
WEIBO_IMG_DIR=~/datasets/big_5/weibo

DRY_FLAG=""
if [ "$DRY_RUN" -eq 1 ]; then
    DRY_FLAG="--dry_run"
fi

refresh_dir () {
    local resp_dir="$1"; shift
    for f in "$resp_dir"/vlm_responses_*.jsonl; do
        [ -e "$f" ] || continue
        echo "=== $f ==="
        python refresh_big5_gt.py \
            --responses_file "$f" \
            "$@" \
            $DRY_FLAG
    done
}

echo "########## TWITTER ##########"
refresh_dir "../results/vlm_pipeline/baseline/big5_twitter/responses" \
    --twitter_en_gt_csv "$TW_EN_GT" \
    --twitter_es_gt_csv "$TW_ES_GT" \
    --big_5_twitter_images_dir "$TW_IMG_DIR"

echo "########## WEIBO ##########"
refresh_dir "../results/vlm_pipeline/baseline/big5_weibo/responses" \
    --weibo_ch0_gt_csv "$WEIBO_CH0_GT" \
    --weibo_ch1_gt_csv "$WEIBO_CH1_GT" \
    --big_5_weibo_images_dir "$WEIBO_IMG_DIR"