#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=rft_lora_distill
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
# 
# DISTILLATION variant of job_finetune.sh: LoRA fine-tune the SAME 12B
# STUDENT (google/gemma-4-12B-it) as that script, but on rejection-sampled
# TEACHER responses from a bigger sibling -- DEFAULT google/gemma-4-31B-it,
# overridable via env (see TEACHER/TEACHER_LABEL below) -- instead of its
# own. Everything about HOW training runs (--no_vision_cache, gradient
# checkpointing, --qlora, --auto_find_batch_size, the checkpoint/resume fix,
# the PYTORCH_CUDA_ALLOC_CONF fragmentation mitigation) is identical to
# job_finetune.sh and lives in train_lora.py — see that script's comments for
# the full history/rationale behind each of those; this file only documents
# what's DIFFERENT about a distillation run.
#
# TEACHER (env, default google/gemma-4-31B-it): which model's rejection-
# sampled responses to train the 12B student on. TEACHER_SLUG is derived
# automatically (TEACHER with "/" -> "_", matching this project's existing
# vlm_responses_<slug>.jsonl naming convention exactly -- STUDENT_SLUG below
# is built the identical way) -- do NOT set TEACHER_SLUG separately, it
# would drift from TEACHER. TEACHER_LABEL (env, default "gemma31b") is a
# SEPARATE, short human name used only in output directory/run names
# ($DATA/rft_gemma12b_from_${TEACHER_LABEL}_*, run_name, etc.) -- it is NOT
# auto-derived, because a clean short label can't be reliably parsed out of
# an arbitrary model id (e.g. google/gemma-4-26B-A4B-it). If you override
# TEACHER, ALSO set TEACHER_LABEL, or every output path will still say
# "gemma31b" while actually holding a different teacher's data -- e.g.:
#   TEACHER=google/gemma-4-26B-A4B-it TEACHER_LABEL=gemma26b_a4b \
#     sbatch fine_tuning/job_finetune_distill.sh
# Requires that teacher's response artifacts already exist under
# $RESULTS/{big5_twitter,big5_weibo}/responses/vlm_responses_<slug>.jsonl
# (a prior run_vlm_pipeline.py inference pass) -- checked below before step 2.
#
# PARTITION: rtx6000 (96GB/card, per Pau) -- --qos/--account here are COPIED
# from the l40s submission that's already been run successfully
# (`--partition=l40s --qos=normal --account=acct_gen --gres=gpu:l40s:1`),
# on the assumption this cluster uses one shared qos/account across GPU
# partitions and only --partition/--gres are hardware-specific (typed gres,
# `gpu:<type>:N`, matching the l40s job's own `--gres=gpu:l40s:1` pattern) --
# NOT independently confirmed for rtx6000 specifically. Check these before
# relying on the job actually landing on the right queue; override at the
# sbatch command line if they're wrong, e.g.:
#   sbatch --qos=<real> --account=<real> fine_tuning/job_finetune_distill.sh
#
# WHAT'S DIFFERENT FROM job_finetune.sh:
#   1. SPLITS ARE REUSED, NOT REGENERATED. Per CLAUDE.md's fine_tuning section:
#      "For DISTILLATION later, point --artifact at a heavier model's
#      responses ... Nothing else changes: the splits stay the same file, so
#      the test set is untouched." Re-deriving splits.json here (even
#      deterministically, same seed) would risk drifting from the
#      already-completed self-distillation run's test set the moment anything
#      upstream changes, and there is no reason to pay that risk when the
#      file already exists. This script therefore has NO "step 1" and instead
#      asserts $DATA/splits/splits.json already exists (i.e. job_finetune.sh
#      has been run, or make_splits.py has been run standalone, at least
#      once) -- see the check right before step 2 below.
#   2. --artifact (step 2, build_rft_dataset.py) points at the TEACHER's
#      responses (vlm_responses_google_gemma-4-31B-it.jsonl) instead of the
#      student's own. Everything about HOW those artifacts get filtered
#      (image_verdict acceptance, --balance) is identical -- the artifact is
#      just a different model's predictions over the same images.
#   3. --model (step 3, train_lora.py) is STILL google/gemma-4-12B-it -- the
#      STUDENT being trained never changes; only whose captions/labels it
#      trains on changes. This is why $RUN/$DATA/rft_gemma12b_from_gemma31b_*
#      below are named "gemma12b_from_gemma31b", not "gemma31b" -- the 31B
#      model itself is never loaded by this script at all, it only supplied
#      training targets earlier (a separate run_vlm_pipeline.py inference
#      pass that already produced the artifact files this reads).
#   4. Step 4/5 (evaluation) run_name/output_file are suffixed "_distill" so
#      they land next to, not overwrite, job_finetune.sh's own self-
#      distillation results -- both can be compared once both exist, since
#      they share the identical --split_file test set (point 1 above).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

set -euo pipefail

# Same fragmentation mitigation as job_finetune.sh -- see that script's
# comment for why (PyTorch's own suggestion on a real OOM message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/home/pmonserrat/code
RESULTS=$CODE/results/vlm_pipeline/baseline
DATA=/home/pmonserrat/datasets/big_5/rft
STUDENT=google/gemma-4-12B-it
STUDENT_SLUG=${STUDENT//\//_}
TEACHER=${TEACHER:-google/gemma-4-31B-it}
TEACHER_SLUG=${TEACHER//\//_}
TEACHER_LABEL=${TEACHER_LABEL:-gemma31b}
RUN=$CODE/runs/lora_gemma12b_from_${TEACHER_LABEL}_balanced
BALANCE=${BALANCE:-downsample_nature}   # none | downsample_nature | loss_weight
LORA_R=16

QLORA_FLAG=""
if [ "${QLORA:-0}" = "1" ]; then
  QLORA_FLAG="--qlora"
fi

mkdir -p ../logs
exec > "../logs/out_rft_lora_distill_${TEACHER_LABEL}.log" 2>&1

cd "$CODE/fine_tuning"

# --- (no step 1) Splits must already exist — reused verbatim, never --------
# regenerated here. See the file-header note above for why.
if [ ! -f "$DATA/splits/splits.json" ]; then
  echo "ERROR: $DATA/splits/splits.json not found. This distillation run " \
       "reuses the splits from the self-distillation run (job_finetune.sh) " \
       "so both runs share the identical test set -- run job_finetune.sh " \
       "(or make_splits.py standalone) at least once first." >&2
  exit 1
fi

# --- 2. Rejection-sampled training set, from the TEACHER's responses -------
# Same acceptance rule (rft_common.image_verdict) and --balance handling as
# job_finetune.sh -- only WHICH model's predictions get filtered changes.
for DS in big5_twitter big5_weibo; do
  if [ ! -f "$RESULTS/$DS/responses/vlm_responses_$TEACHER_SLUG.jsonl" ]; then
    echo "ERROR: $RESULTS/$DS/responses/vlm_responses_$TEACHER_SLUG.jsonl not found. " \
         "TEACHER=$TEACHER needs a prior run_vlm_pipeline.py inference pass over " \
         "$DS before it can supply distillation targets here." >&2
    exit 1
  fi
done
python build_rft_dataset.py \
  --artifact "$RESULTS/big5_twitter/responses/vlm_responses_$TEACHER_SLUG.jsonl" \
  --artifact "$RESULTS/big5_weibo/responses/vlm_responses_$TEACHER_SLUG.jsonl" \
  --splits "$DATA/splits/splits.json" \
  --balance "$BALANCE" \
  --out "$DATA/rft_gemma12b_from_${TEACHER_LABEL}_$BALANCE"

# --- 3. LoRA fine-tune the STUDENT (12B) on the TEACHER's responses --------
# Same hyperparameters/flags as job_finetune.sh's step 3 (--no_vision_cache,
# gradient checkpointing, resume-from-checkpoint, etc. -- all in
# train_lora.py, nothing distillation-specific needed there). QLORA=1 and
# AUTO_FIND=1 work exactly as documented in job_finetune.sh; on a 96GB
# rtx6000 AUTO_FIND=1 is likely the more useful default here, but left
# opt-in (not forced) for the same reason job_finetune.sh leaves it opt-in --
# a plain rerun should stay reproducible unless you deliberately ask for
# auto-sizing.
#
# --eval_steps / --per_device_eval_batch_size / --eval_on_start ported over
# from job_finetune.sh after that run's real OOM: HF's per_device_eval_batch_size
# defaults independently of --per_device_train_batch_size (hardcoded 8), which
# crashed eval well past a stable multi-hundred-step training run there.
# --eval_on_start now catches that at step 0 (before any real training time is
# spent) instead of hours in. --eval_steps 246 was tuned to job_finetune.sh's
# OWN total-step count (1476, for ~6 even eval passes) -- carried over as a
# reasonable starting point, but NOT re-derived for this run's actual dataset
# size: the teacher's (31B) acceptance rate through rft_common.image_verdict
# can differ from the student's own, so this run's total step count (only
# known after step 2 above actually writes train.jsonl) may not be exactly
# 1476, and 246 may land on a different number of eval passes than 6. Not
# worth hand-computing here -- --per_device_eval_batch_size 2 already has real
# headroom evidence behind it (see the earlier a40 OOM-message math), and
# --eval_on_start means a bad --eval_steps choice costs at most one
# ill-timed extra eval pass, not a crash risk.
if [ "${AUTO_FIND:-0}" = "1" ]; then
  AUTO_FIND_FLAG="--auto_find_batch_size"
  BATCH_SIZE=${START_BATCH:-8}
  ACCUM_STEPS=$((16 / BATCH_SIZE))
elif [ "${QLORA:-0}" = "1" ]; then
  AUTO_FIND_FLAG=""
  BATCH_SIZE=${QLORA_BATCH:-1}
  ACCUM_STEPS=$((16 / BATCH_SIZE))
else
  AUTO_FIND_FLAG=""
  BATCH_SIZE=1
  ACCUM_STEPS=16
fi
python train_lora.py \
  --model "$STUDENT" \
  --dataset_dir "$DATA/rft_gemma12b_from_${TEACHER_LABEL}_$BALANCE" \
  --output_dir "$RUN" \
  --lora_r "$LORA_R" \
  --eval_steps 246 \
  --per_device_eval_batch_size 2 \
  --eval_on_start \
  --load_best_model_at_end \
  --no_vision_cache \
  $QLORA_FLAG \
  $AUTO_FIND_FLAG \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$ACCUM_STEPS" \
  --nature_definition_path "$CODE/data/big5_taxonomy/big5_nature_definition.txt" \
  --biotic_definition_path "$CODE/data/big5_taxonomy/big5_biotic_definition.txt" \
  --material_definition_path "$CODE/data/big5_taxonomy/big5_material_definition.txt" \
  --max_image_side 1024 \
  --wandb_project TFM_VLM \
  --run_name "rft_lora_gemma12b_from_${TEACHER_LABEL}_$BALANCE"

# --- 4/5. Evaluate the DISTILLED student on the HELD-OUT TEST SPLIT --------
# POOLED across platforms (--dataset big5), matching job_finetune.sh's own
# step 4/5 restructure -- ONE fine-tuned number and ONE baseline number, not
# per-platform pairs. Same test set as job_finetune.sh's own step 4 (point 1
# above) -- directly comparable to that run's "vlm_pipeline/rft/big5/"
# results, and to any other teacher's "vlm_pipeline/rft_distill_<label>/big5/"
# results, since all of them share the identical --split_file test set.
BIG5_ARGS="--big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
  --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
  --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
  --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
  --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
  --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv"

cd "$CODE/scripts"
echo ""
echo "=========================================================================="
echo "STEP 4: DISTILLED-STUDENT EVALUATION -- LoRA adapter $RUN/adapter"
echo "        ($STUDENT trained on $TEACHER's responses)"
echo "        (pooled big5 test split, both platforms together)"
echo "=========================================================================="
python run_vlm_pipeline.py \
  --dataset big5 $BIG5_ARGS \
  --split_file "$DATA/splits/test_images.txt" \
  --model_family gemma \
  --model_name "$STUDENT" \
  --lora_adapter_path "$RUN/adapter" \
  --lora_max_rank "$LORA_R" \
  --max_model_len 8192 \
  --batch_size 62 \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir "$CODE/results/" \
  --run_name "vlm_pipeline/rft_distill_${TEACHER_LABEL}/big5/" \
  --output_file "vlm_pipeline_big5_rft_distill_${TEACHER_LABEL}_results.json" \
  --max_new_tokens_caption 248 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --gpu_memory_utilization 0.80 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose

# --- 5. The BASELINE (un-finetuned STUDENT) on the SAME pooled test split --
# Same comparison point as job_finetune.sh's own step 5 -- re-SCORE only, no
# inference. Deliberately the STUDENT's own baseline (not the teacher's): the
# question this answers is "did distilling from $TEACHER improve $STUDENT",
# not "how much worse is the student than the teacher". Two separate
# per-platform artifacts get merged first -- see job_finetune.sh's step 5
# comment for exactly why this concatenation is safe (_read_artifact tags
# records by their own record_type, not position). NO $BIG5_ARGS on the
# --stage score call, matching job_finetune.sh -- phase_score never calls
# load_dataset; GT comes from each record's own embedded targets field.
echo ""
echo "=========================================================================="
echo "STEP 5: BASELINE EVALUATION -- $STUDENT, NO adapter (un-finetuned)"
echo "        (pooled big5 test split, both platforms together -- for direct"
echo "        comparison against STEP 4 above)"
echo "=========================================================================="
# Unique per-process temp path (mktemp), NOT a fixed shared filename -- see
# job_finetune.sh's identical step 5 comment for why (concurrent teacher
# runs would otherwise race on the same merged-baseline path).
BASELINE_MERGED=$(mktemp --suffix=.jsonl)
cat "$RESULTS/big5_twitter/responses/vlm_responses_$STUDENT_SLUG.jsonl" \
    "$RESULTS/big5_weibo/responses/vlm_responses_$STUDENT_SLUG.jsonl" \
    > "$BASELINE_MERGED"
python run_vlm_pipeline.py \
  --stage score \
  --dataset big5 \
  --split_file "$DATA/splits/test_images.txt" \
  --responses_file "$BASELINE_MERGED" \
  --results_dir "$CODE/results/" \
  --run_name "vlm_pipeline/baseline_testsplit/big5/" \
  --output_file "vlm_pipeline_big5_baseline_testsplit_results.json" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2
rm -f "$BASELINE_MERGED"
