import subprocess
import os

logs_dir='/home/tomer.erez/psychoacoustic_attacks/logs'
os.makedirs(logs_dir, exist_ok=True)


def generate_sbatch_job(norm_type, size_value,attack_mode=None, target_word=None,target_reps=5):
    """
    Generate a sbatch job script for a specific norm and size

    Args:
    norm_type (str): Type of norm to use
    size_value (float): Size parameter for the norm

    Returns:
    str: Filename of the generated sbatch script
    """

    # Determine the correct argument based on norm type
    size_args = {
        "snr": f"--snr_db {int(size_value)}",
        "min_max_freqs": f"--min_freq_attack {size_value}",
        "fletcher_munson": f"--fm_epsilon {size_value}",
        "l2": f"--l2_size {size_value}",
        "linf": f"--linf_size {size_value}",
    }

    # Check if norm type is valid
    if norm_type not in size_args:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    if size_value is None:
        return
    name_to_use = f"{norm_type}_{size_value}_{attack_mode}"
    if attack_mode=="targeted" and target_word is not None:
        name_to_use = f"{name_to_use}_{target_word}"
    # Generate unique filename
    script_filename = f"{name_to_use}.sh"

    base_args = f"--batch_size 28 --num_epochs 30 --norm_type {norm_type} {size_args[norm_type]}"


    safe_target = target_word.replace(" ", "_") if target_word else "none"


    if attack_mode == "targeted":
        base_args += f" --attack_mode targeted --target_reps {target_reps} --target \"{target_word}\""

    # Create the sbatch script
    sbatch_script = f'''#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --job-name={name_to_use}
#SBATCH --output={logs_dir}/{name_to_use}_%j.out
#SBATCH --mail-user=tomer.erez@campus.technion.ac.il
#SBATCH --mail-type=ALL

# === Conda setup ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# === Run experiment ===
python main.py {base_args}
'''

    # Write the script to file
    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    return script_filename


def submit_jobs():
    """
    Submit sbatch jobs iterleaved across norm types
    """
    norm_ranges = {
        "snr": [],
        "min_max_freqs": [],
        "fletcher_munson": [],
        "l2": [0.4],
        "linf": [0.0001],
    }

    attack_mode = "untargeted"  # "untargeted" or "targeted"
    target_words = ["delete"]  # Sweep over these
    target_reps = 5

    # Find the max number of sizes among all norms
    max_len = max(len(sizes) for sizes in norm_ranges.values())
    norm_types = list(norm_ranges.keys())

    submitted_jobs = []

    for target in target_words if attack_mode == "targeted" else [None]:
        for i in range(max_len):
            for norm_type in norm_types:
                size_list = norm_ranges[norm_type]
                if i < len(size_list):
                    size_value = size_list[i]
                    try:
                        # Generate sbatch script with optional target word
                        script_filename = generate_sbatch_job(
                            norm_type, size_value,
                            attack_mode=attack_mode,
                            target_word=target,
                            target_reps=target_reps
                        )

                        # Submit
                        result = subprocess.run(['sbatch', script_filename],
                                                capture_output=True,
                                                text=True)

                        print(f"Submitted job for {norm_type}, size {size_value}, target {target}")
                        print(result.stdout.strip())

                        submitted_jobs.append({
                            'norm_type': norm_type,
                            'size_value': size_value,
                            'target_word': target,
                            'script': script_filename,
                            'job_id': result.stdout.strip().split()[-1]
                        })

                    except Exception as e:
                        print(f"Error submitting job for {norm_type} with size {size_value}: {e}")

    return submitted_jobs


# Run the job submission
if __name__ == "__main__":
    submitted_jobs = submit_jobs()

    # Optional: You can do something with the submitted jobs if needed
    # For example, write to a log file
    with open('submitted_jobs.log', 'w') as f:
        for job in submitted_jobs:
            f.write(f"{job['job_id']}: {job['norm_type']} - {job['size_value']}\n")