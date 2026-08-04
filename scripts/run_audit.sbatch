#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --account=a40
#SBATCH --job-name=audit
#SBATCH --output=../logs/audit_%j.log
#SBATCH --time=00:10:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

mkdir -p ../logs
# NOT `cd "$(dirname "$0")"` — SLURM copies the submitted script into its own
# spool dir before running it, so $0 points at
# /var/spool/slurm/.../job<id>/run_audit.sbatch, not the real repo checkout.
# dirname on that resolves to the spool dir, which has no audit_artifacts.py
# in it at all ("No such file or directory"). Use the real absolute path
# instead.
cd /home/pmonserrat/code/scripts

python audit_artifacts.py --results_dir /home/pmonserrat/code/results
