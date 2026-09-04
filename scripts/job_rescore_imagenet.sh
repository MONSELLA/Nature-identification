#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=rescore_imagenet
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=11-11
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# =============================================================================
# RE-SCORE ONLY — no VLM inference. Regenerates every ImageNet metric from the
# EXISTING artifacts after the taxonomy Excel was corrected.
# =============================================================================
# WHY THIS IS SAFE TO RE-RUN: `--stage score` never loads a VLM and never
# re-reads the image dataset. It reads the artifact
# (`responses/vlm_responses_<model>.jsonl`), the candidate vocab off the
# artifact's own header, and the Excel; then it recomputes every metric and
# rewrites the results JSON + predictions CSV. Only CLIP touches the GPU
# (longclip + metaclip2), which is why this asks for a modest card rather than
# the tier the inference job needed — same resource shape as the existing
# job_score_testsplit.sh, which is also a --stage score-only job.
#
# NOTE ON --gres: the a40 partition here takes the UNTYPED form `gpu:1`. The
# typed form (`gpu:a40:1`) is only valid on l40s/rtx6000 and fails on a40 with
# "Requested node configuration is not available".
#
# PREREQUISITE — DO NOT SKIP: the GT labels (`targets[].gt_*`) AND the
# mapping-derived half of the predictions (`object_finals`) are baked into the
# artifact at INFER time and are NOT recomputed here. Correcting the Excel
# alone changes nothing. Run scripts/job_resync_gt.sh FIRST (no compute is
# allowed on the login node, so it is a batch job too):
#
#   sbatch scripts/job_resync_gt.sh            # audit; READ the log
#   APPLY=1 sbatch scripts/job_resync_gt.sh    # write
#
# It reports exactly which classes moved and by how many images, refuses to
# touch an artifact a live inference job still holds the .lock on, flags any
# artifact with no footer record (an unfinished run), and leaves every
# model-produced field byte-identical.
#
# INCOMPLETE ARTIFACTS: a model whose inference never finished must be left OUT
# of the MODELS list below (comment it out and shrink --array to match).
# Scoring it "succeeds" and emits a normal-looking results row computed over
# whatever fraction of the dataset exists, which is not comparable with the
# other models — a silent way to publish a number over 20% of ImageNet.
#
# RUN_NAME picks which arm to re-score. Set it, submit, repeat per arm — the
# artifacts of the different arms are independent files.
#
# SEQUENTIAL BY DESIGN: `--array=0-10%1` runs one task at a time. Kept as an
# array rather than a single loop so each model gets its own wall-clock
# allowance — a lone job scoring 11 models back to back risks hitting a time
# limit midway and losing the tail. All tasks APPEND to one shared log,
# logs/out_rescore_imagenet.log, which is only safe because of the %1 throttle.
# =============================================================================

RUN_NAME=${RUN_NAME:-vlm_pipeline/baseline_no_caption/imagenet/}
RESULTS_DIR=${RESULTS_DIR:-/home/pmonserrat/code/results/}

# Same 12 entries, in the same order, as job_vlm_pipeline.sh's MODELS — so a
# task id here refers to the same model it did there. Only the model NAME is
# needed at scoring time (it selects the artifact filename via _model_slug and
# names the predictions CSV); the serving fields are irrelevant without a VLM.
MODELS=(
  "mistralai/Ministral-3-8B-Instruct-2512-BF16"   #0
  "Qwen/Qwen3.5-9B"                               #1
  "google/gemma-4-12B-it"                         #2
  "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct" #3
  "OpenGVLab/InternVL3_5-8B"                      #4
  "Qwen/Qwen3.5-0.8B"                             #5
  "OpenGVLab/InternVL3_5-2B"                      #6
  "mistralai/Ministral-3-3B-Instruct-2512-BF16"   #7
  "google/gemma-4-E4B-it"                         #8
  "google/gemma-4-26B-A4B-it"                     #9
  "Qwen/Qwen3.6-35B-A3B"                          #10
  "OpenGVLab/InternVL3_5-30B-A3B"                 #11
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#MODELS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (${#MODELS[@]} models, valid --array 0-$(( ${#MODELS[@]} - 1 )))."
  exit 1
fi
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_TAG=$(basename "$MODEL_NAME")

# Absolute paths, never $0: under SLURM $0 is the SPOOLED COPY of this script
# (/var/spool/slurmd/job<id>/slurm_script), so `cd $(dirname $0)` lands in the
# spool directory and a relative "../logs/..." redirect then fails — the job
# dies in under a second having written no log at all. Same convention as
# job_score_testsplit.sh.
CODE_DIR=${CODE_DIR:-/home/pmonserrat/code}
mkdir -p "$CODE_DIR/logs"
# ONE SHARED LOG, APPENDED. Safe only because of the `%1` throttle on --array
# above: tasks run strictly one at a time, so there is never more than one
# writer. `>>` (append), never `>` — a truncating redirect would wipe the
# previous task's output every time a task started.
exec >> "$CODE_DIR/logs/out_rescore_imagenet.log" 2>&1
echo ""
echo "================================================================"
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  $(date '+%F %T')"
echo "================================================================"
cd "$CODE_DIR/scripts" || exit 1

echo "Task $SLURM_ARRAY_TASK_ID: re-scoring $MODEL_NAME  (run_name=$RUN_NAME)"

# --model_name is what resolves the artifact path
# (results_dir/run_name/responses/vlm_responses_<slug>.jsonl) — the same rule
# the infer job used to WRITE it, so the two always agree. --model_family is
# not passed: no model is constructed at this stage.
#
# The CLIP checkpoints MUST match the original run (longclip for
# CLIPScore-family, metaclip2 for ClipMatch), or the new numbers are not
# comparable with anything previously reported.
python run_vlm_pipeline.py \
  --stage score \
  --dataset imagenet \
  --model_name "$MODEL_NAME" \
  --run_name "$RUN_NAME" \
  --results_dir "$RESULTS_DIR" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --trust_remote_code \
  --verbose

echo "done: $MODEL_NAME"
