#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=adv
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=tomer.erez@campus.technion.ac.il
#SBATCH --mail-type=ALL

# === Conda setup ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# === Run experiment ===
python main.py --batch_size 16 --num_epochs 8 --run_all_norms