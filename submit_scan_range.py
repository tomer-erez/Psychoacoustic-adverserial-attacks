import subprocess
import os

os.makedirs('logs', exist_ok=True)


def generate_sbatch_job(norm_type, size_value):
    """
    Generate an sbatch job script for a specific norm and size

    Args:
    norm_type (str): Type of norm to use
    size_value (float): Size parameter for the norm

    Returns:
    str: Filename of the generated sbatch script
    """
    # Determine the correct argument based on norm type
    size_args = {
        "linf": f"--linf_size {size_value}",
        "l2": f"--l2_size {size_value}",
        "snr": f"--snr_db {int(size_value)}",
        "fletcher_munson": f"--fm_epsilon {size_value}",
        "min_max_freqs": f"--min_freq_attack {size_value}",
    }

    # Check if norm type is valid
    if norm_type not in size_args:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    # Generate unique filename
    script_filename = f"job_{norm_type}_{size_value}.sh"

    # Create the sbatch script
    sbatch_script = f'''#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=adv_{norm_type}_{size_value}
#SBATCH --output=logs/adv_{norm_type}_{size_value}_%j.out
#SBATCH --mail-user=tomer.erez@campus.technion.ac.il
#SBATCH --mail-type=ALL

# === Conda setup ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# === Run experiment ===
python main.py --batch_size 22 --num_epochs 10 --norm_type {norm_type} {size_args[norm_type]}
'''

    # Write the script to file
    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    return script_filename


def submit_jobs():
    """
    Submit sbatch jobs for different norms and their size ranges
    """
    # Define norms and their parameter lists
    norm_ranges = {
        "linf": [0.001, 0.008, 0.01, 0.06],
        "l2": [0.01, 0.04, 0.09, 0.15],
        "snr": [19,32,44,57],
        "fletcher_munson": [2, 4, 8, 12],
        "min_max_freqs": [100,750,1250,1800],  # Customize if needed
    }

    # Track submitted jobs
    submitted_jobs = []

    # Loop through norms and their ranges
    for norm_type, size_range in norm_ranges.items():
        for size_value in size_range:
            try:
                # Generate sbatch script
                script_filename = generate_sbatch_job(norm_type, size_value)

                # Submit the job
                result = subprocess.run(['sbatch', script_filename],
                                        capture_output=True,
                                        text=True)

                # Print job submission result
                print(f"Submitted job for {norm_type} with size {size_value}")
                print(result.stdout.strip())

                # Optional: Store submitted job information
                submitted_jobs.append({
                    'norm_type': norm_type,
                    'size_value': size_value,
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