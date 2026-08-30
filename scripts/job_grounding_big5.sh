#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=ground_big5
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=/dev/null
#
# BIG-5 DENSE-GT GROUNDING EVALUATION, END TO END IN ONE JOB:
#
#     VLM inference  ->  SAM3 grounding  ->  scoring vs the hand-drawn masks
#
# ...the same shape as job_grounding_coco.sh and the VLM-pipeline job: one
# command, resumable, writing into the standard <results_dir>/<run_name>/
# layout. It REPLACES the big5 half of job_evaluate_grounding.sh, which assumed
# a full-dataset artifact already existed and subset it after the fact.
#
# IT REUSES EXISTING PREDICTIONS WHENEVER THEY EXIST. If this model already has
# full BIG-5 artifacts (the normal VLM benchmark writes them per platform), the
# 340 annotated images are SUBSET out of those and the VLM is never loaded — only
# SAM3 runs. Re-inferring predictions that already exist would be pure waste, and
# on a large model it is also what makes this job need a big GPU it otherwise
# does not (gemma-4-26B-A4B is ~52 GB of weights: it cannot load on a 48 GB card
# at all, while SAM3 alone fits comfortably).
#
# WHEN NO ARTIFACT EXISTS it falls back to running inference itself, restricted
# to the 340 images via --split_file (make_grounding_split_file.py) so a fresh
# model still costs ~20x less than infer-everything-then-subset. That path DOES
# load the VLM, so it needs a GPU that fits the model.
#
# WHICH PATH RAN IS PRINTED. $2 overrides where existing artifacts are looked
# for (default: the no-caption benchmark tree), e.g. to evaluate a fine-tuned
# adapter whose own artifacts live elsewhere.
#
# ONE DATASET, BOTH PLATFORMS. --dataset big5 pools Twitter and Weibo
# (dataset_loader.BIG5_DATASETS), which is required here: the annotations span
# both (84 twitter / 86 weibo), and running one platform alone would leave the
# other's GT regions unmatched and silently halve recall.
#
# RESUMABLE, at both stages: --resume skips images already in the artifact, and
# grounding only touches records without masks. Re-run freely.
#
# NO-CAPTION CONFIGURATION (--no_caption), matching the current VLM benchmark:
# Stage 1 is skipped and entities are extracted from the image alone. BIG-5 runs
# no ClipMatch, so nothing else changes. The run_name keeps these artifacts in
# their own tree — the model slug in the filename is the same with or without a
# caption stage, so only the path separates the two configurations.
#
# OUTPUT LAYOUT (identical to the VLM pipeline's — see README):
#   <RESULTS_DIR>/<RUN_NAME>/
#     grounding_gt_results.json                merged, keyed dataset->model
#     responses/vlm_responses_<slug>.jsonl     artifact (VLM + grounding)
#     predictions/..._per_image.csv            one browsable row per image
#
# PRECISION HERE IS THE STRICT ONE. These annotations are EXHAUSTIVE — every
# nature entity in the image was drawn — so nothing is exempted and every
# unmatched prediction is a false positive. COCO's precision is not comparable
# (it exempts its 79 unannotated-class predictions); see job_grounding_coco.sh.
#
#   sbatch scripts/job_grounding_big5.sh              # whole model array
#   sbatch --array=1 scripts/job_grounding_big5.sh    # one model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# See job_grounding_coco.sh for why an EMPTY HF_TOKEN must be unset rather than
# exported (huggingface_hub would send a literal "Authorization: Bearer ").
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

set -o pipefail

# family|hf_name|max_model_len|batch_cap
MODELS=(
  "gemma|google/gemma-4-E4B-it|8192|64"
  "gemma|google/gemma-4-12B-it|8192|64"
  "gemma|google/gemma-4-26B-A4B-it|8192|48"
)

MODEL_IDX=$SLURM_ARRAY_TASK_ID
if [ "$MODEL_IDX" -ge "${#MODELS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$MODEL_IDX is out of range (valid --array 0-$(( ${#MODELS[@]} - 1 )))."
  exit 1
fi
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"
MODEL_SLUG="${MODEL_NAME//\//_}"

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR="$CODE_DIR/results/"
RUN_NAME="vlm_pipeline/grounding_no_caption/big5/"
OUT_ROOT="${RESULTS_DIR}${RUN_NAME}"

# BIG-5 data
BIG5_TWITTER_IMAGES=/home/pmonserrat/datasets/big_5/twitter
BIG5_WEIBO_IMAGES=/home/pmonserrat/datasets/big_5/weibo
ANNOT_DIR=/home/pmonserrat/datasets/big_5/annotations

# Hand-drawn grounding GT — the folder holding nature/, no_nature/,
# manifest.json and coco_instances.json (output of
# convert_grounding_annotations.py). Accepts either layout, because what gets
# copied to the cluster has varied: the converter writes these under
# `processed/`, but the tree has also been copied with that level flattened
# away. Whichever actually contains nature/ wins.
BIG5_GROUNDING_ROOT=/home/pmonserrat/datasets/big_5/grounding
if [ -d "${BIG5_GROUNDING_ROOT}/processed/nature" ]; then
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}/processed"
else
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}"
fi

# BIG-5 images are uncapped phone-camera/screenshot resolutions, so both the
# VLM's and SAM3's concurrency stay modest here — what is comfortable on
# pre-resized COCO images is not (recap v18/v19).
DS_BATCH=64
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))
MAX_NUM_SEQS=32
GROUND_BATCH=4
GROUND_MAX_PAIRS=8

mkdir -p "$CODE_DIR/logs" "$OUT_ROOT"
exec > "$CODE_DIR/logs/out_ground_big5_${MODEL_SLUG}.log" 2>&1
cd "$CODE_DIR/scripts" || exit 1

SPLIT_FILE="${OUT_ROOT}grounding_gt_split.txt"
ARTIFACT="${OUT_ROOT}responses/vlm_responses_${MODEL_SLUG}.jsonl"

# Where to look for THIS model's existing full-dataset predictions. The VLM
# benchmark writes one artifact per platform, so all three names are tried and
# every match is merged (the GT spans both platforms — 84 twitter / 86 weibo).
SOURCE_ROOT="${2:-${RESULTS_DIR}vlm_pipeline/baseline_no_caption}"
SOURCE_ARTIFACTS=()
for ds in big5_twitter big5_weibo big5; do
    f="${SOURCE_ROOT}/${ds}/responses/vlm_responses_${MODEL_SLUG}.jsonl"
    [ -e "$f" ] && SOURCE_ARTIFACTS+=("$f")
done

echo "=============================================================="
echo "BIG-5 dense-GT grounding evaluation (infer -> ground -> score)"
echo "  model      : $MODEL_NAME  (slug=$MODEL_SLUG)"
echo "  GT dir     : $BIG5_GT_DIR"
echo "  batch_size : $BATCH  max_num_seqs: $MAX_NUM_SEQS"
echo "  output     : $OUT_ROOT"
echo "  source     : $SOURCE_ROOT  (existing artifacts found: ${#SOURCE_ARTIFACTS[@]})"
echo "=============================================================="

if [ ! -d "$BIG5_GT_DIR/nature" ]; then
    echo "ABORT: no nature/ directory under ${BIG5_GROUNDING_ROOT}/processed/ or ${BIG5_GROUNDING_ROOT}/"
    echo "       Copy convert_grounding_annotations.py's processed/ output to the cluster first."
    exit 1
fi

# --- 1. Which images the GT covers -------------------------------------------
# Regenerated every run: it is derived purely from the GT directory, costs
# milliseconds, and a stale split file would silently evaluate the wrong subset.
echo
echo "--- building the split file from the GT ---"
python make_grounding_split_file.py --gt_dir "$BIG5_GT_DIR" --out "$SPLIT_FILE" || exit 1
echo "  $(wc -l < "$SPLIT_FILE") annotated images"

# --- 2. Predictions for those images: REUSE, or infer ------------------------
mkdir -p "${OUT_ROOT}responses"

if [ ${#SOURCE_ARTIFACTS[@]} -gt 0 ]; then
    # ---- REUSE PATH: no VLM is loaded at all --------------------------------
    echo
    echo "--- reusing existing predictions (${#SOURCE_ARTIFACTS[@]} artifact(s)) ---"
    for f in "${SOURCE_ARTIFACTS[@]}"; do echo "    $f"; done

    # An ALREADY-GROUNDED subset from a previous run is passed LAST so its
    # records win the merge (subset_artifact_for_gt.py keeps the last occurrence
    # of each image name). Without this, every run would rebuild a fresh
    # ungrounded copy over the previous one and SAM3 would re-run from scratch
    # each time — even for an invocation whose only intent was to re-score.
    SUBSET_SOURCES=("${SOURCE_ARTIFACTS[@]}")
    [ -e "$ARTIFACT" ] && SUBSET_SOURCES+=("$ARTIFACT")
    # Via a temp file: subset_artifact_for_gt.py reads every input fully before
    # opening the output, so writing over an input would technically work, but
    # this matches the project's crash-safety convention and costs nothing.
    SUBSET_INFO=$(python subset_artifact_for_gt.py \
        --artifact "${SUBSET_SOURCES[@]}" \
        --gt_dir "$BIG5_GT_DIR" \
        --out "${ARTIFACT}.tmp") || exit 1
    mv "${ARTIFACT}.tmp" "$ARTIFACT"
    echo "$SUBSET_INFO"

    # "<grounded>/<total>" — how many subset records already carry SAM3 masks.
    counts=$(echo "$SUBSET_INFO" | grep -o 'SUBSET_GROUNDED=[0-9]*/[0-9]*' | cut -d= -f2)
    n_grounded="${counts%%/*}"
    n_total="${counts##*/}"

    if [ "${n_grounded:-0}" -lt "${n_total:-0}" ]; then
        echo
        echo "--- grounding $(( n_total - n_grounded )) of $n_total records (SAM3 only) ---"
        # --in_place on THIS subset, never on the production artifacts: the
        # subset is this evaluation's own working copy. run_grounding_pipeline.py
        # writes via a temp file and swaps on success, so a crash cannot corrupt it.
        python run_grounding_pipeline.py \
            --responses_file "$ARTIFACT" \
            --in_place \
            --batch_size "$GROUND_BATCH" \
            --max_pairs_per_forward "$GROUND_MAX_PAIRS" \
            --verbose \
        || { echo "GROUNDING FAILED — not scoring an ungrounded artifact"; exit 1; }
    else
        echo "  all $n_total records already grounded — skipping SAM3"
    fi
else
    # ---- INFER PATH: no existing predictions, so run the VLM ----------------
    # Restricted to the 340 annotated images via --split_file. NOTE this loads
    # the model, so the GPU in the SBATCH header must fit it.
    echo
    echo "--- no existing artifact under $SOURCE_ROOT — running inference ---"
    echo "--- building the split file from the GT ---"
    python make_grounding_split_file.py --gt_dir "$BIG5_GT_DIR" --out "$SPLIT_FILE" || exit 1
    echo "  $(wc -l < "$SPLIT_FILE") annotated images"

    echo
    echo "--- VLM inference + SAM3 grounding (340 images) ---"
    # --score is deliberately NOT passed: run_vlm_pipeline.py's score stage
    # computes IMAGE-LEVEL axis metrics, a different question from the dense
    # mask evaluation below, and would write a second results file into this
    # same tree. The dense scoring is step 3, and it is the point of this job.
    python run_pipeline.py \
        --dataset big5 \
        --big_5_twitter_images_dir "$BIG5_TWITTER_IMAGES" \
        --big_5_weibo_images_dir "$BIG5_WEIBO_IMAGES" \
        --twitter_en_gt_csv "$ANNOT_DIR/twitter-en-6_majority.csv" \
        --twitter_es_gt_csv "$ANNOT_DIR/twitter-es-6_majority.csv" \
        --weibo_ch0_gt_csv "$ANNOT_DIR/weibo-ch-6-B-0_majority.csv" \
        --weibo_ch1_gt_csv "$ANNOT_DIR/weibo-ch-6-B-1_majority.csv" \
        --split_file "$SPLIT_FILE" \
        --model_family "$MODEL_FAMILY" \
        --model_name "$MODEL_NAME" \
        --max_model_len "$MAX_LEN" \
        --batch_size "$BATCH" \
        --max_num_seqs "$MAX_NUM_SEQS" \
        --grounding_batch_size "$GROUND_BATCH" \
        --max_pairs_per_forward "$GROUND_MAX_PAIRS" \
        --results_dir "$RESULTS_DIR" \
        --run_name "$RUN_NAME" \
        --dtype bfloat16 \
        --trust_remote_code \
        --no_caption \
        --resume \
        --verbose \
    || { echo "INFERENCE/GROUNDING FAILED — not scoring an incomplete artifact"; exit 1; }
fi

# --- 3. Score against the hand-drawn masks -----------------------------------
# Void handling is ON by default and should stay on: the annotation draws
# `cloud` on top of `sky` (pairwise GT IoU up to 0.747), and voiding the
# contested pixels is what makes the score independent of whether SAM3's "sky"
# includes the clouds in front of it. Measured on the worst such image: a
# perfect-but-other-convention sky scores 0.436 without voiding (FAILING the
# 0.50 threshold) and 1.000 with it. Pass --no_void for the comparison run.
echo
echo "--- scoring against the dense GT ---"
python score_grounding_gt.py \
    --artifact "$ARTIFACT" \
    --gt_dir "$BIG5_GT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --output_file "grounding_gt_results.json" \
    --dataset big5_grounding

STATUS=$?
echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Results:"
    echo "  ${OUT_ROOT}grounding_gt_results.json   (keyed big5_grounding -> $MODEL_NAME)"
    echo "  ${OUT_ROOT}responses/vlm_responses_${MODEL_SLUG}.jsonl"
    echo "  ${OUT_ROOT}predictions/"
else
    echo "SCORING FAILED (exit $STATUS). The artifact is intact — re-running"
    echo "this job resumes and re-scores without repeating inference."
fi
exit $STATUS
