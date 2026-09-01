#!/bin/bash
# preflight_check.sh — validate models, data and code BEFORE queueing a job.
#
# Run this on the login node. It touches no GPU and loads no model; every check
# is a file test or a fast CLI call, so the whole thing takes seconds. It exits
# non-zero if anything would make the Slurm job fail, so it can also be chained:
#
#   ./preflight_check.sh && sbatch job_vlm_pipeline_big5_datasets.sh
#
# WHY: every failed run so far has been one of four things, none of which needs
# a GPU to detect —
#   1. the checkout is older than the job script (a flag like --no_caption is
#      not recognised, and the job dies seconds after the model finishes
#      loading),
#   2. a model directory is missing or its download was interrupted (config.json
#      present, shards absent),
#   3. a dataset path or GT CSV is missing/renamed,
#   4. the venv is not where the script thinks it is.
#
#   ./preflight_check.sh                    # checks everything below
#   ./preflight_check.sh --models-only      # or --data-only / --code-only
#
# Paths default to the BSC layout and can be overridden by environment:
#   MODELS_DIR, DATA_DIR, VENV, CODE_DIR

set -o pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-/gpfs/projects/bsc100/big5/models}"
DATA_DIR="${DATA_DIR:-$CODE_DIR/../datasets}"
VENV="${VENV:-$CODE_DIR/.venv}"

WHAT="${1:-all}"
FAILURES=0
WARNINGS=0

ok()   { printf "  \033[32mOK\033[0m    %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAILURES=$((FAILURES+1)); }
warn() { printf "  \033[33mWARN\033[0m  %s\n" "$1"; WARNINGS=$((WARNINGS+1)); }

# Every model the two BSC job scripts reference, relative to MODELS_DIR.
MODEL_PATHS=(
  "Qwen/Qwen3.6-27B"
  "google/gemma-4-31B-it"
  "OpenGVLab/InternVL3_5-38B"
  "OpenGVLab/InternVL3_5-30B-A3B"
)

# -----------------------------------------------------------------------------
# 1. CODE — is this checkout new enough for the job scripts?
# -----------------------------------------------------------------------------
check_code () {
  echo "CODE ($CODE_DIR)"
  [ -d "$VENV" ] && ok "venv present: $VENV" || { bad "venv MISSING: $VENV"; return; }
  # shellcheck disable=SC1091
  . "$VENV/bin/activate" 2>/dev/null || { bad "cannot activate $VENV"; return; }

  local help
  if ! help=$(cd "$CODE_DIR/scripts" && python run_vlm_pipeline.py --help 2>&1); then
    bad "run_vlm_pipeline.py --help failed — the environment is broken:"
    echo "$help" | tail -3 | sed 's/^/          /'
    return
  fi
  ok "run_vlm_pipeline.py imports and parses"

  # The exact class of failure Ramin hit: the checkout predates the flags the
  # job script passes, so the job dies AFTER the model has loaded.
  local missing=()
  for flag in --no_caption --dataset_task --max_num_seqs --resume --split_file; do
    grep -q -- "$flag" <<<"$help" || missing+=("$flag")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    ok "every flag the job scripts use is recognised"
  else
    bad "checkout is STALE — these flags are not recognised: ${missing[*]}"
    echo "          run: cd $CODE_DIR && git pull"
  fi

  if command -v git >/dev/null && [ -d "$CODE_DIR/.git" ]; then
    local behind
    behind=$(cd "$CODE_DIR" && git rev-list --count HEAD..@{u} 2>/dev/null)
    if [ -n "$behind" ] && [ "$behind" -gt 0 ]; then
      warn "$behind commit(s) behind the remote — consider git pull"
    else
      ok "checkout is up to date with the remote"
    fi
  fi
}

# -----------------------------------------------------------------------------
# 2. MODELS — present, and actually complete?
# -----------------------------------------------------------------------------
# A config.json alone proves nothing: an interrupted download leaves the small
# files and none of the shards. The index file states how many shards there
# should be, so that is what gets counted.
check_models () {
  echo
  echo "MODELS ($MODELS_DIR)"
  [ -d "$MODELS_DIR" ] || { bad "MODELS_DIR does not exist: $MODELS_DIR"; return; }
  for m in "${MODEL_PATHS[@]}"; do
    local d="$MODELS_DIR/$m"
    if [ ! -d "$d" ]; then bad "$m — directory missing"; continue; fi
    if [ ! -f "$d/config.json" ]; then bad "$m — no config.json"; continue; fi

    local idx="$d/model.safetensors.index.json"
    local have want
    have=$(find "$d" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)
    if [ -f "$idx" ]; then
      # Distinct shard filenames named in the index.
      want=$(grep -o '"model-[0-9]*-of-[0-9]*\.safetensors"' "$idx" | sort -u | wc -l)
    else
      want=$have
    fi
    if [ "$have" -eq 0 ]; then
      bad "$m — config present but NO .safetensors shards (interrupted download)"
    elif [ "$want" -gt 0 ] && [ "$have" -lt "$want" ]; then
      bad "$m — only $have of $want shards present (incomplete download)"
    else
      local sz; sz=$(du -sh "$d" 2>/dev/null | cut -f1)
      [ -f "$d/tokenizer_config.json" ] || warn "$m — no tokenizer_config.json"
      ok "$m — $have shard(s), $sz"
    fi
  done
}

# -----------------------------------------------------------------------------
# 3. DATA — image dirs and GT CSVs
# -----------------------------------------------------------------------------
check_data () {
  echo
  echo "DATA ($DATA_DIR)"
  local ann="$DATA_DIR/big_5/annotations"

  for pair in "big_5/twitter:BIG-5 Twitter images" "big_5/weibo:BIG-5 Weibo images"; do
    # SEPARATE statements, deliberately: `local a=$x b=$a` expands every
    # argument BEFORE the builtin assigns any, so `d` would take the previous
    # loop iteration's `rel` (empty on the first pass) — silently checking the
    # wrong directory and reporting it under the right name.
    local rel desc d
    rel="${pair%%:*}"
    desc="${pair##*:}"
    d="$DATA_DIR/$rel"
    if [ ! -d "$d" ]; then bad "$desc — missing: $d"; continue; fi
    local n; n=$(find "$d" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | head -20000 | wc -l)
    [ "$n" -gt 0 ] && ok "$desc — $n image files" || bad "$desc — directory is EMPTY: $d"
  done

  for csv in twitter-en-6_majority.csv twitter-es-6_majority.csv \
             weibo-ch-6-B-0_majority.csv weibo-ch-6-B-1_majority.csv; do
    local f="$ann/$csv"
    if [ ! -f "$f" ]; then bad "GT CSV missing: $f"; continue; fi
    # A GT CSV must carry the per-slot nature columns the loader reads; a
    # truncated or wrong-format file is worse than an absent one, because the
    # run completes and scores against nothing.
    if head -1 "$f" | grep -q "nature_visual_0"; then
      ok "$csv — $(($(wc -l < "$f") - 1)) rows"
    else
      bad "$csv — no 'nature_visual_0' column (wrong file or truncated header)"
    fi
  done

  local imnet="$DATA_DIR/ImageNet/extracted_data"
  if [ -d "$imnet" ]; then
    local nclass; nclass=$(find "$imnet" -maxdepth 1 -mindepth 1 -type d | wc -l)
    [ "$nclass" -gt 0 ] && ok "ImageNet — $nclass class directories" \
                        || bad "ImageNet — $imnet has no class subdirectories"
  else
    warn "ImageNet not found at $imnet (only needed by the imagenet job)"
  fi
}

# -----------------------------------------------------------------------------
# 4. OUTPUTS — writable, and is there room?
# -----------------------------------------------------------------------------
check_outputs () {
  echo
  echo "OUTPUTS"
  for d in "$CODE_DIR/results" "$CODE_DIR/logs"; do
    mkdir -p "$d" 2>/dev/null
    [ -w "$d" ] && ok "writable: $d" || bad "NOT writable: $d"
  done
  local avail; avail=$(df -BG --output=avail "$CODE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$avail" ]; then
    [ "$avail" -lt 20 ] && warn "only ${avail}G free under $CODE_DIR — artifacts need room" \
                        || ok "${avail}G free under $CODE_DIR"
  fi
}

echo "=========================================================="
echo "PREFLIGHT  $(date '+%Y-%m-%d %H:%M')"
echo "=========================================================="
case "$WHAT" in
  all)          check_code; check_models; check_data; check_outputs ;;
  --code-only)  check_code ;;
  --models-only) check_models ;;
  --data-only)  check_data ;;
  *) echo "usage: $0 [all|--code-only|--models-only|--data-only]"; exit 2 ;;
esac

echo
echo "=========================================================="
if [ "$FAILURES" -eq 0 ]; then
  echo "PASSED${WARNINGS:+ with $WARNINGS warning(s)} — safe to submit."
  exit 0
fi
echo "$FAILURES CHECK(S) FAILED — fix these before submitting; the Slurm job"
echo "would hit the same problem after burning queue time."
exit 1
