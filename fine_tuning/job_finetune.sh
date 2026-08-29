#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=rft_lora
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
#
# BACK TO 1 GPU — a 2-card allocation (the naive-model-parallelism approach
# tried here previously, splitting the decoder's layers across 2 a40s via
# device_map="auto" to pool VRAM) is not available on this cluster's queue.
# --per_device_train_batch_size is back to 1, the only value confirmed
# working on a single 48GB card; see step 3 for what's left to try to raise
# throughput within that single-GPU constraint.
#
# Rejection-sampling fine-tune of gemma-4-12B-it on BIG-5, end to end, with
# STANDARD (unmodified default) hyperparameters — every train_lora.py flag
# below is either required (paths) or already the default, EXCEPT
# --no_vision_cache; nothing else here is a hyperparameter override. To
# sweep something, add the flag explicitly to step 3 rather than editing
# train_lora.py's own defaults.
#
# --no_vision_cache IS DELIBERATE, not a stopgap left in by accident. The
# cache (vision_cache.py) is a pure SPEED optimization — it does not change
# what gets trained, only how many times an image's own vision-tower forward
# gets recomputed. On a real run against google/gemma-4-12B-it, its
# injection path hit FIVE real, distinct bugs in a row (a chat-template
# whitespace assumption, a HuggingFace ModelOutput-wrapper return shape, and
# finally a genuine, unresolved embedding-VALUE mismatch — max |Δlogits| ~31
# — that survived a full rewrite to capture real, not reimplemented, feature
# values, meaning the remaining cause is most likely an architecture-specific
# embedding-scaling step this hasn't been able to pin down without further
# live access). Given the cache buys throughput, not correctness, and
# --no_vision_cache routes through the completely standard
# `model(pixel_values=..., ...)` HF training path — the one path NONE of
# those five failures ever touched — continuing to debug the cache further
# was worse effort-for-value than just training without it. See
# vision_cache.py's own module docstring for the full history if this is
# ever revisited.
#
# PARTITION: a40 (48GB/card) — this is TIGHTER than the 96GB card this job
# was originally sized for, and unlike that case this has NOT been verified
# by a comfortable margin, only reasoned through:
#   - INFERENCE of this EXACT model already runs successfully on a40 in this
#     repo (scripts/job_vlm_pipeline.sh's own MODELS array includes
#     "gemma|google/gemma-4-12B-it|...", --gpu_memory_utilization 0.9,
#     --partition=a40 — the very artifact this fine-tune trains from,
#     results/vlm_pipeline/baseline/big5_twitter/responses/
#     vlm_responses_google_gemma-4-12B-it.jsonl, was produced by that run).
#     That's real evidence the ~26GB bf16 base weights + vLLM's OWN KV-cache
#     reservation fit in 48GB.
#   - TRAINING here should need LESS than that KV-cache-inclusive inference
#     footprint, not more: LoRA-trainable params (328 rank-16 adapters) are a
#     small fraction of the 11.9B decoder, --gradient_checkpointing defaults
#     ON (recomputes activations during backward instead of storing them
#     across all 48 layers — the real lever against OOM here), and
#     --per_device_train_batch_size defaults to 1 (already the minimum).
#     Real text-token lengths measured on the actual built dataset run
#     ~1900-2100 tokens typical (p50-p99), far under --max_seq_len's 8192
#     ceiling — that flag is a safety cap, not something driving typical
#     memory use, so lowering it would not meaningfully help if this DOES
#     run tight.
#   - What is NOT yet verified: this SPECIFIC combination (LoRA training +
#     gradient checkpointing + the vision-cache's --verify_cache pass, which
#     runs BOTH a normal forward AND a cached-embedding forward back to back)
#     has never actually been run on this hardware. --verify_cache runs
#     FIRST, before any real training step, specifically so an OOM here (if
#     one happens) is a fast, cheap failure in the first couple of minutes —
#     not a multi-hour training run cut short.
# If it DOES OOM: there is no cheap dial left to turn (batch size is already
# 1, gradient checkpointing already on, max_seq_len isn't the driver) — the
# next step would be requesting a bigger card, not tuning this job further.
#
# WANDB: --wandb_project below assumes this cluster account is ALREADY
# authenticated (no job script in this repo sets WANDB_API_KEY/WANDB_MODE
# explicitly, and W&B logging already works elsewhere per CLAUDE.md's
# Environment section) — if this hangs instead of failing, that assumption
# was wrong; run `wandb login` once outside a batch job.
#
# STAGES 1 and 2 (splits + dataset) are CPU-only and take seconds — they are
# run here for reproducibility, but you can just as well run them on a login
# node and start this job at stage 3.
#
# Steps 1-2 are also SEED-DETERMINISTIC, so re-running this script does not
# quietly change which images are in the test split. If you ever regenerate
# the splits with a different --seed, every previously reported number becomes
# incomparable — treat $DATA/splits (see below) as write-once.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

set -euo pipefail

# PyTorch's OWN suggestion on the real OOM this job hit at batch_size=2
# under --qlora ("Tried to allocate 7.13 GiB ... 8.68 GiB is reserved by
# PyTorch but unallocated ... try setting
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"): a fragmented allocator
# can leave several GB "reserved" but too fragmented to serve one large
# alloc, even though the total is nominally free. expandable_segments lets
# the allocator grow one physical segment instead of hunting for a
# contiguous new block, directly targeting that failure mode. Free,
# standard, and touches nothing about what gets trained.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/home/pmonserrat/code
RESULTS=$CODE/results/vlm_pipeline/baseline
# Splits + the reconstructed RFT training sets are DATA, not code — they live
# alongside the raw BIG-5 images/annotations, not under $CODE/data. Only the
# trained adapter (a model artifact, $RUN below) stays under $CODE.
DATA=/home/pmonserrat/datasets/big_5/rft
MODEL=google/gemma-4-12B-it
SLUG=google_gemma-4-12B-it
RUN=$CODE/runs/lora_gemma12b_balanced
BALANCE=${BALANCE:-downsample_nature}   # none | downsample_nature | loss_weight
# QLORA=1 sbatch job_finetune.sh opts into 4-bit (bitsandbytes NF4) base
# weights instead of full bf16 -- shrinks the ~24GB frozen decoder to ~6GB,
# freeing per-GPU headroom for a bigger --per_device_train_batch_size on one
# 48GB a40 now that a 2-GPU allocation isn't available on this queue. OFF by
# default: unlike every other setting in this script, it has NOT been run on
# this cluster yet (needs `bitsandbytes` installed in the `tfm` env, and is a
# genuine base-weight-loading change, not just a flag flip) -- opt in
# deliberately for the first test rather than silently changing the
# already-proven batch_size=1/bf16 run. See train_lora.py's --qlora help and
# _load_model's docstring for exactly what it does and doesn't change.
QLORA_FLAG=""
if [ "${QLORA:-0}" = "1" ]; then
  QLORA_FLAG="--qlora"
fi
# ONE source of truth for the LoRA rank, used by BOTH step 3 (training) and
# step 4 (--lora_max_rank at serving time) — those two MUST agree, since
# vLLM allocates its LoRA buffers sized to --lora_max_rank at engine
# startup, and a rank higher than what the adapter was trained with fails to
# load while a rank lower than it silently truncates. Change training rank
# here, not by editing --lora_r into step 3 directly, so step 4 can never
# drift out of sync with it.
LORA_R=16

mkdir -p ../logs
exec > "../logs/out_rft_lora.log" 2>&1

cd "$CODE/fine_tuning"

# --- 1. Splits (70/10/20, grouped by post, Twitter+Weibo pooled) ------------
python make_splits.py \
  --artifact "$RESULTS/big5_twitter/responses/vlm_responses_$SLUG.jsonl" \
  --artifact "$RESULTS/big5_weibo/responses/vlm_responses_$SLUG.jsonl" \
  --out "$DATA/splits"

# --- 2. Rejection-sampled training set -------------------------------------
# For DISTILLATION later, point --artifact at a heavier model's responses
# (e.g. vlm_responses_google_gemma-4-31B-it.jsonl) and change --out. Nothing
# else changes: the splits stay the same file, so the test set is untouched.
python build_rft_dataset.py \
  --artifact "$RESULTS/big5_twitter/responses/vlm_responses_$SLUG.jsonl" \
  --artifact "$RESULTS/big5_weibo/responses/vlm_responses_$SLUG.jsonl" \
  --splits "$DATA/splits/splits.json" \
  --balance "$BALANCE" \
  --out "$DATA/rft_gemma12b_$BALANCE"

# --- 3. LoRA fine-tune (base hyperparameters; --use_dora NOT passed, so ----
#        the adapter stays servable directly by vLLM in step 5 — see
#        train_lora.py's module docstring for why DoRA is opt-in here.
#
# HISTORY: batch_size=4 (and, before that, 2) OOM'd on a SINGLE a40 even
# after the batching/collator bugs above were fixed — the 12B decoder's own
# activation memory for >1 image at once genuinely doesn't fit in 48GB with
# only one card. A 2-GPU naive-model-parallelism attempt (splitting the
# decoder's layers across 2 a40s to pool VRAM, which would have let batch
# size go back up) is not usable here — this cluster's queue does not have a
# 2-GPU allocation available. Back to batch_size=1 / accum=16 (effective
# batch 16, matching the project's original default), the only configuration
# actually confirmed to run to completion on one 48GB card. GPU memory is
# still logged every --logging_steps ("[gpu-mem] ... device 0: ..." — see
# train_lora.py's _build_memory_logger_callback) so real headroom (or lack
# of it) at batch_size=1 is visible rather than assumed.
#
# QLORA_FLAG (set above from $QLORA) IS IMPLEMENTED: --qlora loads the
# frozen base in 4-bit via bitsandbytes (train_lora.py's _load_model +
# prepare_model_for_kbit_training). REAL RESULTS so far, in order:
#   batch_size=1 (QLoRA): loads/trains fine, but no faster than plain bf16
#   at the same batch_size=1 -- expected, dequantizing weights every forward
#   pass is pure overhead with no bigger batch yet to offset it.
#   batch_size=4 (QLoRA): never tried -- superseded by the batch_size=2
#   result below before it was run.
#   batch_size=2 (QLoRA): GENUINE OOM -- "Tried to allocate 7.13 GiB ...
#   this process has 40.59 GiB in use ... 8.68 GiB is reserved by PyTorch
#   but unallocated" on a 44.42 GiB-visible a40. That "reserved but
#   unallocated" figure is allocator fragmentation, not truly-unusable
#   memory -- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True above (set
#   per PyTorch's OWN suggestion in that exact error) targets it directly,
#   and gradient_checkpointing_kwargs={"use_reentrant": False} (train_lora.py)
#   switches to the lower-overhead non-reentrant checkpointing implementation
#   -- both applied since that failure, neither yet re-tested.
# QLORA_BATCH (env, default 1) lets you retry a specific batch size WITHOUT
# another edit here -- e.g. `QLORA=1 QLORA_BATCH=2 sbatch job_finetune.sh`
# to see whether the two mitigations above actually rescued batch_size=2.
# Defaults to 1 (not 2) because 2 is the one value already CONFIRMED to OOM
# on this card as of the last run -- an unconditional default of 2 would
# silently re-attempt a known failure.
#
# AUTO_FIND=1 (independent of QLORA -- can combine with it or not) switches
# to --auto_find_batch_size instead of a fixed guess: START_BATCH (env,
# default 8) is the STARTING --per_device_train_batch_size, and the
# framework itself halves it and retries on OOM until it fits, rather than
# spending one cluster job per guess the way the a40 tuning above did. 8 is
# a reasoned, not verified, starting point for a 96GB rtx6000 (2x the a40's
# usable ~44GB, and batch_size=2 there needed roughly ~48GB by itself going
# by the OOM message, so 4x that card's headroom -> 8 is the same kind of
# conservative-but-ambitious single step used for every batch-size change in
# this file's history, not a confident calculation from first principles --
# treat whatever it actually lands on as the real number, not this guess).
# Per --auto_find_batch_size's own help text (train_lora.py), it does NOT
# keep --gradient_accumulation_steps in lockstep if it backs off, so the
# EFFECTIVE batch size may end up below 16 -- acceptable for finding the
# ceiling, revisit gradient_accumulation_steps once the real final batch
# size is known (train_config.json records it either way).
#
# To actually submit TO rtx6000 once it's free, override the SBATCH
# directives at the sbatch command line (they take precedence over the
# #SBATCH lines above) -- confirm the real --qos/--account names for that
# partition on this cluster before running, e.g.:
#   AUTO_FIND=1 sbatch --partition=rtx6000 --qos=<confirm> --account=<confirm> job_finetune.sh
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
  --model "$MODEL" \
  --dataset_dir "$DATA/rft_gemma12b_$BALANCE" \
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
  --run_name "rft_lora_gemma12b_$BALANCE" \
  
# --- 4/5. Evaluate on the HELD-OUT TEST SPLIT, POOLED across both platforms
# (--dataset big5, not big5_twitter/big5_weibo run separately) so there is
# ONE fine-tuned number and ONE baseline number to compare, not two pairs of
# per-platform numbers -- per Pau, splitting by platform for this final
# comparison was more confusing than useful. --lora_adapter_path serves the
# adapter DIRECTLY on top of the base model, on one resident vLLM engine — no
# merge step. --lora_max_rank reuses $LORA_R from step 3 (see its definition
# above for why they must match). --split_file keeps both calls on test
# images only, so the fine-tuned and baseline numbers stay comparable.
BIG5_ARGS="--big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
  --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
  --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
  --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
  --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
  --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv"

cd "$CODE/scripts"
echo ""
echo "=========================================================================="
echo "STEP 4: FINE-TUNED EVALUATION -- LoRA adapter $RUN/adapter on top of $MODEL"
echo "        (pooled big5 test split, both platforms together)"
echo "=========================================================================="
python run_vlm_pipeline.py \
  --dataset big5 $BIG5_ARGS \
  --split_file "$DATA/splits/test_images.txt" \
  --model_family gemma \
  --model_name "$MODEL" \
  --lora_adapter_path "$RUN/adapter" \
  --lora_max_rank "$LORA_R" \
  --max_model_len 8192 \
  --batch_size 62 \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir "$CODE/results/" \
  --run_name "vlm_pipeline/rft/big5/" \
  --output_file "vlm_pipeline_big5_rft_results.json" \
  --max_new_tokens_caption 248 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --gpu_memory_utilization 0.80 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose

# --- 5. The BASELINE (no adapter) on the SAME pooled test split ------------
# The existing per-platform whole-dataset artifacts already cover these
# images, so this is a re-SCORE only (no inference) -- but they are two
# SEPARATE files (one per platform, from the original baseline inference
# runs), and --responses_file only ever reads ONE file. Concatenated into one
# merged artifact first: _read_artifact (run_vlm_pipeline.py) tags records by
# their own "record_type" field rather than by position, and simply keeps
# whichever "header" record appears LAST when there are several -- so a plain
# concatenation of two already-valid artifacts is safe to read as one, no
# custom merge logic needed.
# NO $BIG5_ARGS here (images_dir/GT csv) -- deliberately, matching the
# original per-platform score-only call, which never passed them either:
# phase_score (unlike phase_infer) never calls load_dataset -- GT is read
# straight off each record's own embedded `targets` field, written into the
# artifact back when it was first inferred. --dataset big5 here is only used
# to pick the right scoring/axis logic, not to reload images or CSVs.
echo ""
echo "=========================================================================="
echo "STEP 5: BASELINE EVALUATION -- $MODEL, NO adapter (un-finetuned)"
echo "        (pooled big5 test split, both platforms together -- for direct"
echo "        comparison against STEP 4 above)"
echo "=========================================================================="
# Unique per-process temp path (mktemp), NOT a fixed shared filename -- this
# job and job_finetune_distill.sh (possibly several teachers, run
# concurrently) all merge the SAME student baseline artifacts at roughly the
# same pipeline stage; a fixed shared path would let two concurrent runs
# truncate/overwrite each other's in-progress merge, corrupting whichever one
# reads it. This is a transient scoring input, not a persistent artifact --
# nothing downstream needs it to live at a predictable path.
BASELINE_MERGED=$(mktemp --suffix=.jsonl)
cat "$RESULTS/big5_twitter/responses/vlm_responses_$SLUG.jsonl" \
    "$RESULTS/big5_weibo/responses/vlm_responses_$SLUG.jsonl" \
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
