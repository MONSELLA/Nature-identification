#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=rescore_big5_imagenet
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

mkdir -p ../logs
exec > "../logs/out_rescore_big5_imagenet.log" 2>&1

RESULTS_DIR=/home/pmonserrat/code/results/

# --output_file is the SHARED results JSON for the whole run_name — each
# --stage score call below only updates its own model's entry inside it
# (update_results_store merges under a lock), so it's safe to reuse the same
# file across every model in a loop like this.
score_dataset () {
    local dataset="$1"
    local resp_dir="$2"
    local run_name="$3"
    local output_file="$4"
    shift 4
    
    local target_files=("$@")

    for filename in "${target_files[@]}"; do
        local f="$resp_dir/$filename"
        [ -e "$f" ] || continue
        echo "=== [$dataset] $f ==="
        python run_vlm_pipeline.py \
            --stage score \
            --dataset "$dataset" \
            --responses_file "$f" \
            --results_dir "$RESULTS_DIR" \
            --run_name "$run_name" \
            --output_file "$output_file" \
            --batch_size 512 \
            --clipscore_model longclip \
            --verbose \
        || echo "❌ FAILED: $f — see error above, continuing with next model"
    done
}

echo "########## IMAGENET ##########"
score_dataset imagenet \
    "../results/vlm_pipeline/baseline/imagenet/responses" \
    "vlm_pipeline/baseline/imagenet/" \
    "vlm_pipeline_imagenet_results.json" \
    "vlm_responses_google_gemma-4-26B-A4B-it.jsonl" \
    "vlm_responses_google_gemma-4-31B-it.jsonl" \
    "vlm_responses_OpenGVLab_InternVL3_5-30B-A3B.jsonl" \
    "vlm_responses_OpenGVLab_InternVL3_5-38B.jsonl" \
    "vlm_responses_Qwen_Qwen3.6-27B.jsonl" \
    "vlm_responses_Qwen_Qwen3.6-35B-A3B-FP8.jsonl"