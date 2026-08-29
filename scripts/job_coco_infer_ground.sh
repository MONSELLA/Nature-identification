#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=coco_gemma_infer
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --array=3-3
#SBATCH --output=/dev/null
#
# Produces the artifact job_coco_score.sh needs to exist before it can do
# anything: runs the VLM inference stage, then the Grounding (SAM3) stage,
# for one model on COCO, via scripts/run_pipeline.py (VLM infer -> grounding,
# each stage its own OS subprocess so VRAM is fully reclaimed between them —
# see run_pipeline.py's own module docstring). Deliberately does NOT run
# --stage score itself (no --score flag below) — scoring stays the separate,
# already-working job (job_coco_score.sh), so this script's only job is to
# get a grounded artifact onto disk. Run job_coco_score.sh afterward to get
# the actual results JSON / predictions CSV.
#
# THIS SCRIPT DOES NOT EXIST YET ON THE CLUSTER — reconstructed from this
# repo's run_pipeline.py / run_vlm_pipeline.py / run_grounding_pipeline.py CLI
# definitions plus job_coco_score.sh's already-proven paths and conventions
# (same --results_dir/--run_name, same RESPONSES_SUBDIR layout, same
# MODEL_SLUG derivation), since no prior COCO infer+ground job was available
# to copy from. Sanity-check the flags below against your actual cluster
# before trusting a 12+ hour run to it — nothing here has been run for real.
#
# USAGE
#   sbatch job_coco_infer_ground.sh          # runs array index 0 (see MODELS)
#
# Run it from the scripts/ directory, like every other job_*.sh here — the
# taxonomy/definition-file flags below rely on their defaults, which are
# relative paths ("../data/big5_taxonomy/...").

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

export VLLM_USE_FLASHINFER_SAMPLER=0

# facebook/sam3 is a GATED HuggingFace repo (grounding needs an authenticated
# token whose account has accepted its license). run_pipeline.py's grounding
# stage falls back to this env var automatically if --hf_token isn't passed
# (src/grounding_pipeline.py's SAM3Grounder), so setting it here is enough —
# no code-side change needed.
# facebook/sam3 is a GATED HuggingFace repo, so grounding needs authentication.
# Either run `hf auth login` once (the token is then cached in
# ~/.cache/huggingface/token and picked up automatically), or export HF_TOKEN in
# your shell before `sbatch`. Never hardcode it here — this file is tracked.
#
# UNSET IT IF EMPTY, deliberately: an empty HF_TOKEN is WORSE than none at all.
# huggingface_hub would send a literal "Authorization: Bearer " header (httpx
# then raises `Illegal header value b'Bearer '`) instead of falling back to the
# cached login token. So only export it when it actually has a value.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

# family|hf_name|max_model_len|batch_cap — SAME array as job_coco_score.sh, so
# an index here and in that script always refer to the same model.
MODELS=(
  "gemma|google/gemma-4-E4B-it|8192|96"
  "gemma|google/gemma-4-12B-it|8192|96"
  "gemma|google/gemma-4-26B-A4B-it|8192|64"
  "qwen|Qwen/Qwen3.6-27B|8192|256"
)

MODEL_IDX=$SLURM_ARRAY_TASK_ID
IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP <<< "${MODELS[$MODEL_IDX]}"

# Matches run_vlm_pipeline.py's own _model_slug() EXACTLY
# (model_name.replace("/", "_")) — same derivation job_coco_score.sh uses, so
# the artifact this job writes is the exact file that job looks for.
MODEL_SLUG="${MODEL_NAME//\//_}"

mkdir -p ../logs
exec > "../logs/out_coco_infer_${MODEL_FAMILY}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Inference + grounding for $MODEL_NAME (slug=$MODEL_SLUG) on coco"

# --- COCO paths --------------------------------------------------------------
COCO_IMAGES_DIR="/home/pmonserrat/datasets/coco/images/val2017"
COCO_INSTANCES_JSON="/home/pmonserrat/datasets/coco/annotations/instances_val2017.json"

# --results_dir/--run_name MUST match job_coco_score.sh's RESPONSES_FILE
# construction exactly: with no explicit --responses_file, run_vlm_pipeline.py
# writes to '<results_dir>/<run_name>/responses/vlm_responses_<slug>.jsonl'
# (_resolve_responses_file) — the same path job_coco_score.sh's
# RESPONSES_FILE variable already points at.
RESULTS_DIR="/home/pmonserrat/code/results/"
RUN_NAME="vlm_pipeline/baseline/coco/"

# COCO images are pre-resized benchmark images (like imagenet/places), not raw
# social-media resolutions — so unlike big5_weibo this needs neither a lowered
# --batch_size nor --max_num_seqs to avoid the BIG-5-specific vision-encoder
# OOM (recap v18/v19), and the Grounding stage's own defaults (--batch_size 8,
# --max_pairs_per_forward 16) are left alone for the same reason — those
# defaults exist for exactly this "already-modest, pre-resized image" case.
DS_BATCH=96
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))

echo "Config: batch_size=$BATCH max_model_len=$MAX_LEN gpu_util=0.9 (default)"

# --skip_grounding / --score are BOTH omitted: default behavior is exactly
# infer -> ground, nothing more (see run_pipeline.py's own docstring). This
# script's whole job is to leave a grounded artifact on disk for
# job_coco_score.sh to pick up afterward.
#
# --instance_grounding is left at its "auto" default on the grounding side:
# it turns on SAM3's instance head automatically for a COCO artifact (read
# from the dataset name written into the header during infer), same as
# job_coco_score.sh already relies on.
#
# No --max_samples — this is the REAL full run over all of val2017, not a
# smoke test.
python run_pipeline.py \
    --dataset coco \
    --data_dir "$COCO_IMAGES_DIR" \
    --instances_json "$COCO_INSTANCES_JSON" \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --max_model_len "$MAX_LEN" \
    --batch_size "$BATCH" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --verbose

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo
    echo "Done — artifact should now exist at:"
    echo "  ${RESULTS_DIR}${RUN_NAME}responses/vlm_responses_${MODEL_SLUG}.jsonl"
    echo "Next: run job_coco_score.sh (array index $MODEL_IDX) to score it."
else
    echo "FAILED (exit $STATUS) — see subprocess output above for which stage."
fi
exit $STATUS
