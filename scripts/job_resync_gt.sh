#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=resync_gt
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

# Unbuffered: stdout is block-buffered when redirected to a file, so without
# this the log stays empty until the process exits and a 2-minute job looks hung.
export PYTHONUNBUFFERED=1

# =============================================================================
# Re-sync artifact GT + mapping-derived predictions with the CURRENT Excel.
# This job is CPU-bound: resolve_hybrid_label and TaxonomyGraph import nothing
# heavier than networkx/nltk/pandas (verified — neither torch nor vllm is
# pulled in), so the GPU below is NEVER touched. It is requested only because
# this cluster requires --gres on every job; keep it to the smallest card the
# queue will grant. Runs as a batch job for the same reason: no compute is
# allowed on the login node.
#
# TWO-STEP BY DESIGN. Submit once to AUDIT (default), read the log, and only
# then submit again with APPLY=1. The audit prints a per-synset cause table,
# and it must be read rather than skipped: prediction changes reflect the
# CURRENT taxonomy, which can differ from the one a run was made with for
# reasons beyond the latest edit (rows appended to the sheet after a run
# resolve phrases that were unmapped at infer time).
#
#   sbatch scripts/job_resync_gt.sh                         # audit
#   less logs/out_resync_gt_<jobid>.log                     # READ IT
#   APPLY=1 sbatch scripts/job_resync_gt.sh                 # write
#
# STOP if the log contains a "changed final_NATURE" warning: grounding only
# ever ran on final_nature==True entities, so the stored masks would no longer
# cover the same entity set and the grounding stage must be re-run — a
# re-score alone would silently report groundings for the wrong entity set.
#
# ARTIFACTS is a space-separated list of responses/ directories. Override to
# point at another arm or several at once:
#   ARTIFACTS="results/vlm_pipeline/baseline/imagenet/responses" sbatch ...
# =============================================================================

CODE_DIR=${CODE_DIR:-$HOME/code}
cd "$CODE_DIR" || exit 1

ARTIFACTS=${ARTIFACTS:-"results/vlm_pipeline/baseline_no_caption/imagenet/responses"}
APPLY=${APPLY:-0}
BACKUP=${BACKUP:-1}

mkdir -p logs
exec > "logs/out_resync_gt_${SLURM_JOB_ID}.log" 2>&1

FLAGS=""
[ "$APPLY" = "1" ] && FLAGS="$FLAGS --apply"
[ "$BACKUP" = "0" ] && FLAGS="$FLAGS --no-backup"

echo "mode: $([ "$APPLY" = 1 ] && echo APPLY || echo 'AUDIT (dry run)')"
echo "dirs: $ARTIFACTS"
echo

for d in $ARTIFACTS; do
  if [ ! -d "$d" ]; then
    echo "!! missing directory: $d"; exit 1
  fi
  echo "================================================================"
  echo "$d"
  echo "================================================================"
  python -u scripts/resync_artifact_gt.py --dir "$d" $FLAGS || exit 1
done

echo
if [ "$APPLY" = "1" ]; then
  echo "DONE. Next: sbatch scripts/job_rescore_imagenet.sh"
else
  echo "AUDIT ONLY — nothing written. Read the tables above, then:"
  echo "  APPLY=1 sbatch scripts/job_resync_gt.sh"
fi
