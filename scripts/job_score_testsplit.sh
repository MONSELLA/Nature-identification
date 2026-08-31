#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=score_testsplit
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#
# SCORE AN ALREADY-INFERRED MODEL ON THE HELD-OUT TEST SPLIT. No inference,
# no VLM: --stage score only, over artifacts that already exist.
#
# WHAT IT IS FOR: the un-finetuned baseline number that a fine-tuning run is
# compared against. The fine-tuning jobs evaluate their adapter on
# $DATA/splits/test_images.txt; this produces the SAME model's un-adapted
# number on the SAME images, so the pair is directly comparable.
#
# WHY NO INFERENCE IS NEEDED. The full-dataset BIG-5 artifacts already contain
# every image's predictions, and --split_file simply restricts scoring to the
# test basenames. Re-running the VLM over a subset of images it has already
# processed would produce the same records at GPU cost.
#
# THE GPU IS FOR CLIP, NOT THE VLM. --stage score loads the CLIPScore model
# (longclip) to compute the reference-free metrics; nothing else here needs a
# GPU, which is why an a40 is plenty regardless of how big the scored model is.
#
# GT COMES FROM THE ARTIFACT, NOT FROM THE DATASET. phase_score never calls
# load_dataset — each record carries its own `targets` — so no image dirs and
# no GT CSVs are passed below. NOTE the consequence: this scores the
# annotations AS THEY WERE AT INFERENCE TIME. If the BIG-5 CSVs have been
# revised since, patch the artifact first with scripts/refresh_big5_gt.py,
# otherwise the numbers silently reflect stale labels.
#
#   sbatch scripts/job_score_testsplit.sh                          # gemma-4-12B
#   sbatch scripts/job_score_testsplit.sh google/gemma-4-26B-A4B-it
#
# $2 overrides the artifact root (default: search the *no_caption* trees).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

set -o pipefail

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR="$CODE_DIR/results/"
SPLITS_DIR=/home/pmonserrat/datasets/big_5/rft/splits
TEST_SPLIT="$SPLITS_DIR/test_images.txt"

MODEL_NAME="${1:-google/gemma-4-12B-it}"
MODEL_SLUG="${MODEL_NAME//\//_}"

mkdir -p "$CODE_DIR/logs"
exec > "$CODE_DIR/logs/out_score_testsplit_${MODEL_SLUG}.log" 2>&1
cd "$CODE_DIR/scripts" || exit 1

# Same artifact search as job_grounding_big5.sh, and for the same reason:
# no-caption runs have landed in more than one tree (baseline_no_caption/,
# ablation_no_caption/), so pinning one root silently finds nothing. Only
# caption_stage=false artifacts qualify — the filename is identical with or
# without a caption stage, so the header is the only thing that tells them
# apart, and scoring a captioned artifact as if it were caption-free would
# quietly compare two different configurations.
if [ -n "${2:-}" ]; then
    SOURCE_ROOTS=("$2")
else
    SOURCE_ROOTS=()
    for d in "${RESULTS_DIR}vlm_pipeline"/*no_caption*/; do
        [ -d "$d" ] && SOURCE_ROOTS+=("${d%/}")
    done
fi

ARTIFACTS=()
for root in "${SOURCE_ROOTS[@]}"; do
    # Both platforms are needed: the test split spans Twitter and Weibo, and a
    # missing platform would score its images as absent rather than as wrong.
    for ds in big5_twitter big5_weibo; do
        f="${root}/${ds}/responses/vlm_responses_${MODEL_SLUG}.jsonl"
        [ -e "$f" ] || continue
        head -1 "$f" | grep -q '"caption_stage": *false' && ARTIFACTS+=("$f")
    done
done

echo "=============================================================="
echo "Test-split scoring (no inference)"
echo "  model     : $MODEL_NAME  (slug=$MODEL_SLUG)"
echo "  split     : $TEST_SPLIT"
echo "  searched  : ${SOURCE_ROOTS[*]:-(none)}"
echo "  artifacts : ${#ARTIFACTS[@]}"
for f in "${ARTIFACTS[@]}"; do echo "      $f"; done
echo "=============================================================="

if [ ${#ARTIFACTS[@]} -eq 0 ]; then
    echo "ABORT: no no-caption artifact found for $MODEL_SLUG under the searched roots."
    exit 1
fi
if [ ! -f "$TEST_SPLIT" ]; then
    echo "ABORT: $TEST_SPLIT not found — run fine_tuning/make_splits.py once first."
    exit 1
fi

# Concatenating the per-platform artifacts is safe: _read_artifact tags every
# line by its own record_type, so the interleaved headers/footers a plain `cat`
# produces are ignored rather than misread as image records. mktemp (not a
# fixed name) so two of these running at once cannot race on one path.
MERGED=$(mktemp --suffix=.jsonl) || exit 1
trap 'rm -f "$MERGED"' EXIT
cat "${ARTIFACTS[@]}" > "$MERGED"
echo "merged $(grep -c '"record_type": *"image"' "$MERGED") image records"

# --dataset big5 pools both platforms into ONE number, matching how the
# fine-tuning jobs evaluate their adapters. No --clipmatch_model: BIG-5 never
# runs ClipMatch, so naming one would load a second CLIP for nothing.
python run_vlm_pipeline.py \
    --stage score \
    --dataset big5 \
    --split_file "$TEST_SPLIT" \
    --responses_file "$MERGED" \
    --results_dir "$RESULTS_DIR" \
    --run_name "vlm_pipeline/baseline_testsplit_no_caption/big5/" \
    --output_file "vlm_pipeline_big5_baseline_testsplit_no_caption_results.json" \
    --clipscore_model longclip \
    --longclip_repo_path ../../Long-CLIP \
    --verbose

STATUS=$?
echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Results:"
    echo "  ${RESULTS_DIR}vlm_pipeline/baseline_testsplit_no_caption/big5/vlm_pipeline_big5_baseline_testsplit_no_caption_results.json"
    echo "  (keyed big5 -> gemma/$MODEL_NAME; directly comparable to the fine-tuned"
    echo "   adapter's number, which uses this same test split)"
else
    echo "FAILED (exit $STATUS)."
fi
exit $STATUS
