#!/bin/bash
# =============================================================================
# VLM pipeline — SCORE-ONLY SLURM array launcher, ImageNet + Places
#
# Runs `--stage score` ONLY (never infer) against artifacts that inference
# has ALREADY finished writing. Use this once you've confirmed the matching
# `--stage infer` runs are done — this script does NOT check that for you
# (see the dependency note below).
#
# ARRAY INDEXING: MODEL_IDX = TASK_ID / 2, DATASET_IDX = TASK_ID % 2, so
# CONSECUTIVE ids sweep {imagenet, places} for ONE model:
#   --array=0-1   -> model 0 (qwen), both datasets
#   --array=0-9   -> the full 5-model x 2-dataset grid
#
# CONCURRENCY: safe to run several of these at once now (2026-08 fix) —
# update_results_store/update_dataset_image_stats (src/utils.py) hold a real
# flock across their whole read-modify-write, so multiple models' scoring
# runs finishing close together can no longer clobber each other's entry in
# the shared <run_name>/<output_file> JSON (this is what silently lost the
# qwen/gemma ImageNet results before the fix). No need to throttle with
# `%N` for that reason anymore. phase_score also refuses to start if a
# phase_infer for the SAME artifact is still actively writing it (raises
# ArtifactLockHeld) rather than reading a torn file.
#
# WHAT THIS DOESN'T PROTECT AGAINST: submitting this before infer has
# actually finished. phase_score doesn't require the artifact to have a
# footer — it'll happily score whatever records exist so far, silently
# giving you numbers over a partial run. Two ways to avoid that:
#   (a) check the artifact ends in a footer yourself first:
#       tail -c 200 <responses_file> | grep -q '"record_type": "footer"'
#   (b) submit this with a SLURM dependency on the infer array job, e.g.:
#       sbatch --dependency=afterok:<infer_job_id> run_vlm_pipeline_score.sbatch
# =============================================================================
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -p a40
#SBATCH --qos=normal
#SBATCH --account=a40
#SBATCH --job-name=vlm_score
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
# Real file, NOT /dev/null — anything failing before the `exec` redirect
# below (conda activate, a bad array index) would otherwise vanish silently.
# ../logs must already exist at SUBMIT time — `mkdir -p ../logs` once first.
#SBATCH --output=../logs/slurm_score_%A_%a.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# MUST match the models actually run at --stage infer time — family|hf_name
# pairs only (no max_model_len/batch_cap: those are inference-engine knobs,
# irrelevant to scoring). Keep this list's ORDER identical to the infer
# launcher's MODELS array, since MODEL_IDX below indexes into it the same way.
MODELS=(
  "qwen|Qwen/Qwen3.5-9B"
  "gemma|google/gemma-4-12B-it"
  "mistral|mistralai/Ministral-3-8B-Instruct-2512-BF16"
  "llava|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
  "internvl|OpenGVLab/InternVL3_5-8B"
)

DATASETS=("imagenet" "places")

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

IFS='|' read -r MODEL_FAMILY MODEL_NAME <<< "${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

mkdir -p ../logs

# Log name carries the SLURM job id (%A), not just the array task id (%a):
# resubmitting this same array later would otherwise reuse the exact same
# log filename and truncate/interleave with a previous run's still-live
# process — this is what produced the corrupted, cross-spliced tracebacks
# seen earlier. model+dataset in the name too, so five models scoring the
# same dataset don't share one file either.
exec > "../logs/score_${DATASET}_${MODEL_FAMILY}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: scoring $MODEL_NAME on $DATASET"
echo "commit: $(git -C /home/pmonserrat/code rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --run_name/--output_file/--results_dir here MUST be byte-identical to
# whatever the matching --stage infer run used — that's what makes
# _resolve_responses_file rebuild the SAME artifact path (see _model_slug:
# only --model_name, not --dataset, drives the filename) rather than
# pointing at a different/nonexistent file.
case "$DATASET" in
  "imagenet")
    EXTRA_ARGS="--dataset imagenet \
      --run_name vlm_pipeline/imagenet/ \
      --output_file vlm_pipeline_imagenet_results.json"
    ;;
  "places")
    EXTRA_ARGS="--dataset places365 \
      --run_name vlm_pipeline/places/ \
      --output_file vlm_pipeline_places_results.json"
    ;;
esac

# --max_samples: if the matching --stage infer run used --max_samples N,
# pass the SAME N here — phase_score's `records[:args.max_samples]` slices
# whatever the artifact holds, so a mismatch silently scores the wrong
# subset instead of erroring. Omit entirely (as below) only if infer ALSO
# ran with no --max_samples (the full dataset).
python run_vlm_pipeline.py \
  --stage score \
  $EXTRA_ARGS \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir /home/pmonserrat/code/results/ \
  --device cuda \
  --verbose
