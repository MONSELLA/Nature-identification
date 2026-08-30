#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=vlm_pipeline
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --array=30-35
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm
 
export VLLM_USE_FLASHINFER_SAMPLER=0
 
# =============================================================================
# ONE TIER PER SUBMISSION — set the #SBATCH lines above and --array accordingly
# =============================================================================
# The three tiers target different GPUs, so this file is submitted three times.
# Task id = model_index * 3 + dataset_index, so each tier is a contiguous range
# (the per-model ranges are the trailing comments in MODELS below):
#
#   tier    models     --array    GPU               #SBATCH to set above
#   base    idx 0-4    0-14       L40S     48 GB    --partition=l40s --gres=gpu:l40s:1
#   light   idx 5-8    15-26      A40      48 GB    --partition=a40  --gres=gpu:a40:1
#   moe     idx 9-11   27-35      RTX6000  96 GB    (whatever the queue calls it)
#
# Nothing in the script reads the tier — it only needs the right --array range
# on the right GPU. A range submitted to the wrong card still runs; it just
# uses a batch cap sized for a different one.
#
# --- BATCH CAPS: where these numbers come from -------------------------------
# --batch_size only bounds how many prompts THIS SCRIPT submits per
# generate_batch call. It is NOT the memory lever: vLLM reserves weights + KV
# cache up front (--gpu_memory_utilization 0.8) and its own scheduler decides
# how many of the submitted prompts actually run together. So the cap wants to
# sit comfortably ABOVE the concurrency the KV cache can sustain — enough to
# keep the engine saturated through a batch's tail — without being so large
# that host-side image prep balloons. Roughly 2x achievable concurrency.
#
# Concurrency per model at 0.8 utilization, for a realistic ~2000-token request
# (~1.5k prompt including image tokens + <=512 generated):
#
#   KV bytes/token = 2 (K,V) * layers * kv_heads * head_dim * 2 (bf16),
#   with Gemma's sliding-window layers capped at their window rather than the
#   full sequence (~5 windowed layers per global one).
#
#   model                GPU    weights   KV budget   MB/req   concurrency  cap
#   Ministral-3-8B       48GB     16 GB     19.4 GB     279         70      128
#   Qwen3.5-9B           48GB     18 GB     17.4 GB     262         66      128
#   gemma-4-12B-it       48GB     24 GB     11.4 GB     467         24       64
#   LLaVA-OV-2-8B        48GB     16 GB     19.4 GB     295         66      128
#   InternVL3_5-8B       48GB     16 GB     19.4 GB     262         74      128
#   Qwen3.5-0.8B         48GB      2 GB     33.8 GB      98        344      512
#   InternVL3_5-2B       48GB      4 GB     31.4 GB     115        274      512
#   Ministral-3-3B       48GB      6 GB     29.4 GB     213        138      256
#   gemma-4-E4B-it       48GB     16 GB     19.4 GB      65        297      384
#   gemma-4-26B-A4B      96GB     52 GB     21.8 GB     292         75      128
#   InternVL3_5-30B-A3B  96GB     60 GB     13.8 GB     192         72      128
#   Qwen3.6-35B-A3B      96GB     70 GB      3.8 GB     196         19       64
#
# Two models are genuinely tight and their caps reflect it, not timidity:
#   - gemma-4-12B on a 48 GB card: 24 GB of weights plus a 256-wide head_dim
#     leaves room for only ~24 concurrent sequences.
#   - Qwen3.6-35B-A3B on the 96 GB card: 70 GB of weights against a 76.8 GB
#     budget leaves under 4 GB of KV cache. It will START (one 8192-token
#     sequence needs ~0.8 GB) but runs ~19 sequences at a time, so expect it to
#     be the slowest job in the grid by some margin. If that hurts, the levers
#     are --gpu_memory_utilization 0.85-0.9 for THIS model only, or an FP8
#     checkpoint — not a bigger --batch_size, which cannot create KV cache.
# MoE saves FLOPs, not parameter memory: the 26B MoE's KV budget on a 96 GB
# card is no better than an 8B dense model's on a 48 GB one.
#
# HYBRID-ATTENTION MODELS NEED --max_num_seqs. Qwen3.6-35B-A3B interleaves
# Gated-DeltaNet (Mamba-style) layers with normal attention, and vLLM allocates
# ONE MAMBA STATE BLOCK PER CONCURRENT SEQUENCE out of the same leftover memory
# as the KV cache. With 65.5 GiB of weights against a 76.8 GiB budget only
# ~5.5 GiB is left, i.e. ~274 blocks — and vLLM's DEFAULT max_num_seqs is 1024,
# so it refuses to start at all:
#     ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (274)
# --batch_size does NOT bound this: it only caps how many prompts this script
# SUBMITS per call, while max_num_seqs caps what the engine schedules. A purely
# transformer model has no such per-sequence allocation and is unaffected, which
# is why only the hybrid entry sets the field.
#
# family|hf_name|max_model_len|batch_cap|max_num_seqs (empty = vLLM default)
MODELS=(
  # Base tier — L40S 48 GB (--array 0-14)
  "mistral|mistralai/Ministral-3-8B-Instruct-2512-BF16|8192|128" #0-2
  "qwen|Qwen/Qwen3.5-9B|8192|128" #3-5
  "gemma|google/gemma-4-12B-it|8192|64" #6-8
  "llava|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct|8192|128" #9-11
  "internvl|OpenGVLab/InternVL3_5-8B|8192|128" #12-14
  # Lightweight tier — A40 48 GB (--array 15-26)
  "qwen|Qwen/Qwen3.5-0.8B|8192|512" #15-17
  "internvl|OpenGVLab/InternVL3_5-2B|8192|512" #18-20
  "mistral|mistralai/Ministral-3-3B-Instruct-2512-BF16|8192|256" #21-23
  "gemma|google/gemma-4-E4B-it|8192|384" #24-26
  # MoE tier — RTX 6000 96 GB (--array 27-35)
  "gemma|google/gemma-4-26B-A4B-it|8192|128" #27-29
  "qwen|Qwen/Qwen3.6-35B-A3B|8192|64|128" #30-32  hybrid: see note above
  "internvl|OpenGVLab/InternVL3_5-30B-A3B|8192|128" #33-35
)

DATASETS=("imagenet" "big5_twitter" "big5_weibo")

# The divisor/modulus MUST equal ${#DATASETS[@]}. With 3 datasets and a /4,%4
# split, task ids 3, 7, 11 ... produced DATASET_IDX=3 -> an empty DATASET, an
# unset EXTRA_ARGS, and a python invocation with no --dataset at all; the other
# ids silently addressed the wrong model/dataset pair. Derived from the array
# length so the two can never drift apart again.
N_DATASETS=${#DATASETS[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))

N_TASKS=$(( ${#MODELS[@]} * N_DATASETS ))
if [ "$MODEL_IDX" -ge "${#MODELS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range: ${#MODELS[@]} models x $N_DATASETS datasets = $N_TASKS tasks (valid --array 0-$(( N_TASKS - 1 )))."
  exit 1
fi

IFS='|' read -r MODEL_FAMILY MODEL_NAME MAX_LEN BATCH_CAP MODEL_SEQS <<< "${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

mkdir -p ../logs

# Log name carries MODEL as well as dataset: on the full grid five models share
# a dataset, and a dataset-only name would have them all truncate/interleave
# into one file.
exec > "../logs/out_${DATASET}_${MODEL_FAMILY}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID: Running $MODEL_NAME on $DATASET"

# --- Per-dataset GPU budget -------------------------------------------------
# --gpu_memory_utilization reserves WEIGHTS + KV CACHE; the vision encoder's
# activations come out of what is LEFT OVER. That leftover is where the BIG-5
# OOM lived (recap v18/v19), so the two dataset families want opposite tuning:
#
#   imagenet/places : pre-resized benchmark images, vision encoder barely
#                     stresses the leftover.
#   big5_*          : raw social-media resolutions (arbitrary phone/screenshot
#                     sizes) -> the leftover has to absorb the spikes, so cap
#                     CONCURRENT vision-encoding with --max_num_seqs (the
#                     actual v19 lever) rather than sacrificing --batch_size
#                     or --max_image_side.
#
# Utilization itself is held at 0.8 for BOTH families (see GPU_UTIL below) —
# the safe figure for BIG-5, applied everywhere so the whole grid runs on one
# engine config.
#
# --batch_size only bounds how many prompts THIS SCRIPT submits per call; vLLM
# still schedules concurrency itself, so past ~128 it stops buying throughput.
case "$DATASET" in
  "imagenet")
    DS_BATCH=512
    EXTRA_ARGS="--dataset imagenet \
      --data_dir /home/pmonserrat/datasets/imagenet/extracted_data \
      --run_name vlm_pipeline/baseline_no_caption/imagenet/ \
      --output_file vlm_pipeline_imagenet_results.json"
    ;;
  "big5_twitter")
    DS_BATCH=512
    EXTRA_ARGS="--dataset big5_twitter \
      --big_5_twitter_images_dir /home/pmonserrat/datasets/big_5/twitter \
      --twitter_es_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-es-6_majority.csv \
      --twitter_en_gt_csv /home/pmonserrat/datasets/big_5/annotations/twitter-en-6_majority.csv \
      --run_name vlm_pipeline/baseline_no_caption/big5_twitter/ \
      --output_file vlm_pipeline_big5_twitter_results.json"
    ;;
  "big5_weibo")
    DS_BATCH=512
    EXTRA_ARGS="--dataset big5_weibo \
      --big_5_weibo_images_dir /home/pmonserrat/datasets/big_5/weibo \
      --weibo_ch0_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-0_majority.csv \
      --weibo_ch1_gt_csv /home/pmonserrat/datasets/big_5/annotations/weibo-ch-6-B-1_majority.csv \
      --run_name vlm_pipeline/baseline_no_caption/big5_weibo/ \
      --output_file vlm_pipeline_big5_weibo_results.json \
      --max_num_seqs 32"
    ;;
esac

# Final batch = min(per-dataset budget, per-model ceiling)
BATCH=$(( DS_BATCH < BATCH_CAP ? DS_BATCH : BATCH_CAP ))

# --gpu_memory_utilization: 0.8 for EVERY dataset, deliberately. It reserves
# WEIGHTS + KV CACHE, and the vision encoder's activations come out of what is
# LEFT OVER — which is where the BIG-5 OOM lived (recap v18/v19). ImageNet's
# pre-resized images could in principle afford a higher figure, but one value
# across the grid keeps every model/dataset pair on identical engine settings,
# so a difference in the numbers is never a difference in the serving config.
GPU_UTIL=0.8

# --max_num_seqs comes from two places, deliberately: per DATASET inside
# EXTRA_ARGS (BIG-5 Weibo, to bound concurrent vision-encoding — recap v19) and
# per MODEL via the 5th MODELS field (hybrid-attention models, which cannot
# start without it). The model-level one is only added when the dataset did not
# already set one, so a dataset-specific cap is never silently overridden.
SEQS_ARG=""
if [ -n "${MODEL_SEQS:-}" ] && [[ "$EXTRA_ARGS" != *"--max_num_seqs"* ]]; then
  SEQS_ARG="--max_num_seqs $MODEL_SEQS"
fi

echo "Config: dataset=$DATASET model=$MODEL_NAME batch_size=$BATCH gpu_util=$GPU_UTIL ${SEQS_ARG:-(max_num_seqs: vLLM default)} (no-caption run)"

# NO-CAPTION RUN (--no_caption): Stage 1 is skipped entirely, so
# --max_new_tokens_caption below is inert and is kept only so this file stays
# diff-able against the captioned baseline job. On ImageNet the summary call
# SURVIVES and becomes the run's one short image description — the text
# ClipMatch scores — so --summary_max_new_tokens still matters there.
# CLIPScore/F-CLIPScore will report n/a for every dataset in this grid.
python run_vlm_pipeline.py \
  $EXTRA_ARGS \
  --no_caption \
  --model_family "$MODEL_FAMILY" \
  --model_name "$MODEL_NAME" \
  --max_model_len "$MAX_LEN" \
  --batch_size "$BATCH" \
  $SEQS_ARG \
  --gpu_memory_utilization "$GPU_UTIL" \
  --clipscore_model longclip \
  --clipmatch_model metaclip2 \
  --results_dir /home/pmonserrat/code/results/ \
  --max_new_tokens_caption 248 \
  --summary_max_new_tokens 77 \
  --max_new_tokens_extraction 512 \
  --max_new_tokens_label 512 \
  --dtype bfloat16 \
  --trust_remote_code \
  --verbose \
  --resume