#!/bin/bash

# Define arrays of norm types and corresponding size flags/values
norms=("linf" "fletcher_munson" "min_max_freqs" "snr" "l2")

# Define associative array for size arguments per norm
declare -A norm_args
norm_args["linf"]="--linf_size 0.05"
norm_args["l2"]="--l2_size 0.33"
norm_args["snr"]="--snr_db 16"
norm_args["fletcher_munson"]="--fm_epsilon 9.3"
norm_args["min_max_freqs"]="--min_freq_attack 2500 --max_freq_attack 20000"

for norm in "${norms[@]}"; do

  size_args=${norm_args[$norm]}

  sbatch <<EOF
#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=high_adv_${norm}
#SBATCH --output=logs/high_adv_${norm}_%j.out
#SBATCH --mail-user=tomer.erez@campus.technion.ac.il
#SBATCH --mail-type=ALL

# === Conda setup ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# === Run experiment for norm: $norm ===
python main.py --batch_size 22 --num_epochs 10 --norm_type ${norm} ${size_args}

EOF

done
