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
cd "$(dirname "$0")"

python audit_artifacts.py --results_dir /home/pmonserrat/code/results
