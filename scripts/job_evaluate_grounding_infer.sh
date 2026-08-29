#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=ground_infer
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
#
# Produces the GROUNDED artifacts job_evaluate_grounding.sh needs to score a
# FINE-TUNED ADAPTER — that script only ever reads an existing artifact, it
# never runs VLM inference or (for COCO) grounding itself. This is the
# missing "make the artifact" half; job_evaluate_grounding.sh is the
# "score the artifact" half.
#
# COCO: full val2017, via run_pipeline.py (VLM infer -> grounding, each its
# own OS subprocess so VRAM is fully reclaimed between them — see that
# script's own module docstring). This is a REAL, multi-hour run — COCO's
# 5000 images, unlike BIG-5, have no cheaper GT-restricted subset to target.
# PARTITION NOTE: scripts/job_coco_infer_ground.sh (the template this reuses)
# assumed --partition=rtx6000 but is itself flagged as never having actually
# been run. This script uses l40s instead, matching what has ACTUALLY been
# proven working in this project's fine-tuning jobs; COCO images are
# pre-resized/modest (unlike BIG-5's raw social-media resolutions), so l40s's
# lower VRAM is not expected to be the tighter constraint here — but this is
# reasoned, not independently re-verified for the infer+ground combination
# specifically. Override at the sbatch command line if it OOMs.
#
# BIG-5: restricted to just the 340 hand-annotated images (nature/ +
# no_nature/ under BIG5_GT_DIR), not the full ~6663-image dataset --
# grounding only ever scores those 340 anyway (job_evaluate_grounding.sh's
# own big5 half subsets down to them), so running inference on the other
# ~6300 would be pure waste. The split file is regenerated from the GT dir on
# every invocation (cheap, deterministic, always in sync with whatever is
# actually on disk there) via --split_file's own basename-matching (CLAUDE.md).
# NO grounding here for BIG-5 -- job_evaluate_grounding.sh's own big5 half
# does that itself (subset_artifact_for_gt.py + run_grounding_pipeline.py
# --in_place), so this script only needs to produce the VLM-inference-only
# artifact it expects to find.
#
# USAGE — three POSITIONAL arguments, same reasoning as
# job_evaluate_grounding.sh's own $1/$2 (sbatch env-var propagation into the
# job's actual shell is not guaranteed on this cluster):
#   sbatch job_evaluate_grounding_infer.sh <mode> <adapter_path> <label>
#     mode         : coco | big5 | both
#     adapter_path : the LoRA adapter directory (a run's .../adapter, or a
#                    checkpoint-<step> dir -- see train_lora.py/PeftModel,
#                    both are structurally identical adapter dirs)
#     label        : short name for THIS adapter's output paths -- reuse the
#                    SAME label you will later pass as job_evaluate_grounding.sh's
#                    $4, so the artifacts this script writes are exactly what
#                    that script's $3 (RESPONSES_ROOT) will look for.
#
# Example (the self-distilled adapter):
#   sbatch job_evaluate_grounding_infer.sh both \
#     /home/pmonserrat/code/runs/lora_gemma12b_balanced/adapter \
#     rft_selfdistill
# Then, once this finishes:
#   sbatch job_evaluate_grounding.sh both google/gemma-4-12B-it \
#     /home/pmonserrat/code/results/vlm_pipeline/grounding_eval_rft_selfdistill \
#     rft_selfdistill
#
# Run from the scripts/ directory, like every other job_*.sh here.

# facebook/sam3 is a GATED HuggingFace repo, so grounding needs an authenticated
# token. Export HF_TOKEN in your shell before `sbatch`, or put it in ~/.env —
# never hardcode it here (this file is tracked in git).
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

MODE="${1:-both}"
ADAPTER_PATH="${2:-}"
LABEL="${3:-}"

if [ -z "$ADAPTER_PATH" ] || [ -z "$LABEL" ]; then
    echo "ERROR: both <adapter_path> and <label> are required positional args." >&2
    echo "  usage: sbatch job_evaluate_grounding_infer.sh <mode> <adapter_path> <label>" >&2
    exit 2
fi
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: ADAPTER_PATH=$ADAPTER_PATH does not exist." >&2
    exit 1
fi

MODEL_NAME="google/gemma-4-12B-it"
LORA_R="${LORA_R:-16}"

CODE="/home/pmonserrat/code"
RESULTS_DIR="${CODE}/results/"
RUN_ROOT="vlm_pipeline/grounding_eval_${LABEL}"

mkdir -p ../logs
exec > "../logs/out_ground_infer_${LABEL}_${MODE}.log" 2>&1

cd "$CODE/scripts"

echo "=============================================================="
echo "Grounding artifact production"
echo "  MODEL_NAME   = $MODEL_NAME"
echo "  ADAPTER_PATH = $ADAPTER_PATH"
echo "  LABEL        = $LABEL"
echo "  mode         = $MODE"
echo "=============================================================="

# =============================================================================
# 1. COCO
# =============================================================================
run_coco () {
    echo
    echo "################## COCO infer + ground ##################"
    COCO_IMAGES_DIR="/home/pmonserrat/datasets/coco/images/val2017"
    COCO_INSTANCES_JSON="/home/pmonserrat/datasets/coco/annotations/instances_val2017.json"

    # run_pipeline.py shares run_vlm_pipeline.py's FULL argument parser
    # (build_arg_parser()), so --lora_adapter_path/--lora_max_rank are valid
    # here even though run_pipeline.py never re-declares them itself.
    #
    # --run_name MUST match job_evaluate_grounding.sh's $3 (RESPONSES_ROOT)
    # + "/coco/" exactly, since that script derives COCO_ARTIFACT from
    # RESPONSES_ROOT with no other coupling between the two scripts.
    #
    # No --skip_grounding / --score: default behavior is infer -> ground,
    # nothing more (matches job_coco_infer_ground.sh's own template) --
    # job_evaluate_grounding.sh's run_coco() does the SCORING afterward, as
    # its own separate step.
    python run_pipeline.py \
        --dataset coco \
        --data_dir "$COCO_IMAGES_DIR" \
        --instances_json "$COCO_INSTANCES_JSON" \
        --model_family gemma \
        --model_name "$MODEL_NAME" \
        --lora_adapter_path "$ADAPTER_PATH" \
        --lora_max_rank "$LORA_R" \
        --max_model_len 8192 \
        --batch_size 62 \
        --results_dir "$RESULTS_DIR" \
        --run_name "${RUN_ROOT}/coco/" \
        --trust_remote_code \
        --verbose
}

# =============================================================================
# 2. BIG-5 (VLM inference only, restricted to the 340 GT images)
# =============================================================================
run_big5 () {
    echo
    echo "################## BIG-5 (340-image subset) infer ##################"
    BIG5_GROUNDING_ROOT="/home/pmonserrat/datasets/big_5/grounding"
    if [ -d "${BIG5_GROUNDING_ROOT}/processed/nature" ]; then
        BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}/processed"
    else
        BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}"
    fi
    if [ ! -d "$BIG5_GT_DIR/nature" ]; then
        echo "ERROR: no nature/ directory found under either" >&2
        echo "  ${BIG5_GROUNDING_ROOT}/processed/  or  ${BIG5_GROUNDING_ROOT}/" >&2
        return 1
    fi

    # Split file regenerated fresh every run -- matches subset_artifact_for_gt.py's
    # OWN image-id derivation exactly (gt_image_ids: strip ".json" off each
    # nature/no_nature filename, keep the rest verbatim as the image basename)
    # so this is guaranteed consistent with what that script will later match
    # against, not a second, independently-drifting definition of the GT set.
    GT_SPLIT_FILE="${BIG5_GROUNDING_ROOT}/gt_split_${LABEL}.txt"
    : > "$GT_SPLIT_FILE"
    for f in "$BIG5_GT_DIR"/nature/*.json "$BIG5_GT_DIR"/no_nature/*.json; do
        [ -e "$f" ] || continue
        basename "$f" .json >> "$GT_SPLIT_FILE"
    done
    N_GT=$(wc -l < "$GT_SPLIT_FILE")
    echo "GT split file: $GT_SPLIT_FILE ($N_GT images)"
    if [ "$N_GT" -eq 0 ]; then
        echo "ERROR: 0 images found under ${BIG5_GT_DIR}/{nature,no_nature} -- nothing to infer." >&2
        return 1
    fi

    # --dataset big5 (pooled, not big5_twitter/big5_weibo separately): the
    # GT spans both platforms, and job_evaluate_grounding.sh's own
    # BIG5_ARTIFACTS collection already accepts a pooled "big5" artifact
    # (it loops over big5_twitter/big5_weibo/big5 and includes whichever
    # exist). --split_file restricts to exactly the 340 GT images
    # (basename-matched, CLAUDE.md) -- this is what keeps this run to
    # minutes instead of the hours a full-dataset run would cost.
    #
    # --stage infer only: this script's job is just to produce the raw
    # artifact; job_evaluate_grounding.sh's own big5 half does the
    # subsetting+SAM3-grounding+scoring from here, not this script.
    python run_vlm_pipeline.py \
        --stage infer \
        --dataset big5 \
        --big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
        --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
        --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
        --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
        --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
        --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
        --split_file "$GT_SPLIT_FILE" \
        --model_family gemma \
        --model_name "$MODEL_NAME" \
        --lora_adapter_path "$ADAPTER_PATH" \
        --lora_max_rank "$LORA_R" \
        --max_model_len 8192 \
        --batch_size 62 \
        --results_dir "$RESULTS_DIR" \
        --run_name "${RUN_ROOT}/big5/" \
        --max_new_tokens_caption 248 \
        --max_new_tokens_extraction 512 \
        --max_new_tokens_label 512 \
        --gpu_memory_utilization 0.80 \
        --dtype bfloat16 \
        --trust_remote_code \
        --verbose
}

# =============================================================================
# Dispatch
# =============================================================================
STATUS=0
case "$MODE" in
    coco) run_coco || STATUS=1 ;;
    big5) run_big5 || STATUS=1 ;;
    both)
        run_coco || { echo "COCO infer+ground FAILED — continuing to BIG-5"; STATUS=1; }
        run_big5 || { echo "BIG-5 infer FAILED"; STATUS=1; }
        ;;
    *)
        echo "Unknown mode '$MODE' — expected one of: both, coco, big5"
        exit 2
        ;;
esac

echo
echo "=============================================================="
echo "Done (exit status $STATUS)."
echo "  COCO artifact  -> ${RESULTS_DIR}${RUN_ROOT}/coco/responses/vlm_responses_google_gemma-4-12B-it.jsonl"
echo "  BIG-5 artifact -> ${RESULTS_DIR}${RUN_ROOT}/big5/responses/vlm_responses_google_gemma-4-12B-it.jsonl"
echo "  Next: sbatch job_evaluate_grounding.sh both $MODEL_NAME ${RESULTS_DIR}${RUN_ROOT} $LABEL"
echo "=============================================================="
exit $STATUS
