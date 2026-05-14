#!/bin/bash

#SBATCH -J brokerage-roles
#SBATCH -o logs/brokerage-roles_%j.txt
#SBATCH --chdir=/N/slate/gpanayio/scisci-roles
#SBATCH -e logs/brokerage-roles_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH -A r00272

module load python/3.12.4

BASE="/N/slate/gpanayio/scisci-roles"

echo "===== Brokerage role analysis started at $(date) ====="

python -u brokerage_roles.py

echo "===== Brokerage role analysis finished at $(date) ====="