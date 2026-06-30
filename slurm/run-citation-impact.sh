#!/bin/bash

#SBATCH -J citation-impact
#SBATCH -o logs/citation-impact_%j.txt
#SBATCH -e logs/citation-impact_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH -A r00272
#SBATCH --chdir=/N/slate/gpanayio/scisci-roles

module load python/3.12.4

echo "===== Citation impact analysis started at $(date) ====="

echo "[1/2] Extracting works and building citation pairs..."
python -u extract_works.py

echo "[2/2] Aggregating to author level and running regression..."
python -u aggregate_and_regress.py

echo "===== Done at $(date) ====="
