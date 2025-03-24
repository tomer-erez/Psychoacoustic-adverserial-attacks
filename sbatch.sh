#!/bin/bash

# List of norm types to try
norm_types=("l2" "linf" "snr" "fletcher_munson" "leakage" "min_max_freqs")

# Loop over each norm type
for norm in "${norm_types[@]}"; do
  # Run twice: once with --loss_ascent, once without
  for loss_ascent in true false; do

    # If loss_ascent is true, include the flag; otherwise, omit it
    loss_flag=""
    suffix=""
    if [ "$loss_ascent" == "true" ]; then
      loss_flag="--loss_ascent"
      suffix="_la"  # optional suffix to distinguish jobs
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=adv_${norm}${suffix}
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=tomer.erez@campus.technion.ac.il
#SBATCH --mail-type=ALL

# === Conda setup ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# === Run experiment ===
python main.py --norm_type ${norm} --batch_size 64 --num_epochs 8 ${loss_flag}
EOF

  done
done
