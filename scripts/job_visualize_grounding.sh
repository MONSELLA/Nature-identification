#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=viz_grounding
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/home/pmonserrat/code/logs/slurm_viz_grounding_%j.out
#
# Render ONE image's SAM3 nature masks as a figure (scripts/visualize_grounding.py)
# — the qualitative "final output of the pipeline" picture for the thesis.
#
# Needs a GPU only because SAM3 has to run: the artifacts this points at were
# produced by --stage infer and were never enriched by the Grounding pipeline,
# so no masks are stored in them yet. An A40 is far more than this needs (one
# forward pass per entity, seconds of work) but it is the smallest thing in the
# queue that fits SAM3 comfortably. If these artifacts are ever grounded for
# real, the script reuses their stored masks and loads no model at all — at
# which point this job can drop --gres entirely and run on CPU.
#
#   sbatch scripts/job_visualize_grounding.sh
#   sbatch scripts/job_visualize_grounding.sh <image> <responses_file>
#
# Output lands in $OUT_DIR (results/figures/grounding/ by default):
#   <stem>_<model>_overlay.png     the figure
#   <stem>_<model>_panels.png      one panel per entity
#   <stem>_<model>_grounding.json  provenance + both nature-relevance scores

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# facebook/sam3 is a GATED HuggingFace repo, so grounding needs an authenticated
# token. Export HF_TOKEN in your shell before `sbatch`, or put it in ~/.env —
# never hardcode it here (this file is tracked in git).
export HF_TOKEN="${HF_TOKEN:-}"

CODE_DIR=/home/pmonserrat/code
RESULTS_DIR=$CODE_DIR/results

# --- What to render -----------------------------------------------------------
# Defaults are the method-figure example: the corgi + mushroom-plush image,
# predicted by Gemma-4-26B-A4B on the NO-CAPTION run (the configuration the
# current benchmark uses). Override by passing both as arguments.
#
# NOTE the artifact choice is a real decision, not a formality: on this image
# the captioned baseline extracted "corgi"/"mushroom plushie"/"couch" while the
# no-caption run extracted "dog"/"mushroom plush"/"sofa", so the two produce
# visibly different SAM3 prompts and therefore different figures. Point this at
# the run the surrounding text is actually describing.
IMAGE="${1:-/home/pmonserrat/datasets/big_5/twitter/1703862454143304161_0.jpg}"
RESPONSES="${2:-$RESULTS_DIR/vlm_pipeline/ablation_no_caption/big5_twitter/responses/vlm_responses_google_gemma-4-26B-A4B-it.jsonl}"

OUT_DIR="$RESULTS_DIR/figures/grounding"
ALPHA=0.45           # mask opacity over the photograph
MASK_THRESHOLD=0.5   # SAM3's own default — the pipeline's DEFAULT_MASK_THRESHOLD

mkdir -p "$CODE_DIR/logs" "$OUT_DIR"
cd "$CODE_DIR/scripts" || exit 1

echo "image     : $IMAGE"
echo "artifact  : $RESPONSES"
echo "out_dir   : $OUT_DIR"
[ -z "$HF_TOKEN" ] && echo "WARNING: HF_TOKEN is unset — SAM3 (facebook/sam3) is gated and the load will fail."

python visualize_grounding.py \
  --image "$IMAGE" \
  --responses_file "$RESPONSES" \
  --out_dir "$OUT_DIR" \
  --alpha "$ALPHA" \
  --mask_threshold "$MASK_THRESHOLD" \
  --device cuda \
  --panels
