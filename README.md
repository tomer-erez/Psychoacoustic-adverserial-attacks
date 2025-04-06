

# Psychoacoustically-Informed Adversarial Attacks on Speech Recognition Systems

This project explores adversarial attacks on automatic speech recognition (ASR) models, with a focus on crafting **perturbations that are difficult for humans to perceive** but significantly degrade model performance.

We investigate how different norm types (e.g., L2, Linf, perceptual/Fletcher-Munson weighted) and perturbation sizes affect both the model's predictions and the human perceptibility of the perturbation. Our goal is to develop attacks that are **effective yet stealthy** — confusing the model while remaining inaudible to the human ear.

---

## 📦 Requirements

First, install the environment using the provided Conda file:

```bash
conda env create -f cs236207.yaml
conda activate cs236207
```

---

## 🚀 Running the Project

To run the full training + attack pipeline:

```bash
python main.py
```

You can specify various command-line arguments to control the attack (see **Hyperparameters** below).

---

## 🎧 Dataset: Common Voice

The project uses the [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) dataset. The script will automatically download it on first run.

If prompted, you may need to authenticate with your Hugging Face account. You can do this by visiting:  
https://huggingface.co/settings/tokens

---

Here's an improved and cleaner **Hyperparameters** section for your `README.md`, rewritten to:

- Highlight only the **important and attack-specific arguments**
- Give examples and context
- Use a well-organized table
- Provide a practical CLI example

---

## ⚙️ Key Hyperparameters

You can control the behavior of the attack using the following command-line arguments:

| Argument              | Description                                                                                         | Example                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--norm_type`         | Type of norm used to constrain the perturbation. Choices: `l2`, `linf`, `snr`, `fletcher_munson`, `min_max_freqs`. | `--norm_type fletcher_munson`                |
| `--attack_mode`       | Type of attack: `targeted` (force a word) or `untargeted` (increase error).                         | `--attack_mode targeted`                      |
| `--target`            | Target phrase for targeted attacks. Only used in `targeted` mode.                                   | `--target "delete"`                           |
| `--target_reps`       | How many times to repeat the target word in the label (e.g., `delete delete delete`).               | `--target_reps 3`                             |
| `--loss_fn`           | Loss used to optimize the perturbation. Choices: `ctc`, `wer`.                                      | `--loss_fn wer`                               |
| `--num_epochs`        | Number of training epochs.                                                                           | `--num_epochs 10`                             |
| `--resume_from`       | Path to a saved perturbation `.pt` file to resume training from.                                    | `--resume_from path/to/pert.pt`               |
| `--small_data`        | Use only a small subset (~1%) of the dataset for fast debugging.                                    | `--small_data`                                |
| `--dataset`           | Which dataset to use. Prefer `CommonVoice` if attacking Wav2Vec2.                                   | `--dataset CommonVoice`                       |
| `--fm_epsilon`        | Max perceptual norm for Fletcher-Munson constraint.                                                 | `--fm_epsilon 2.0`                            |
| `--l2_size`           | L2 constraint (ε) for `l2`-norm attack.                                                              | `--l2_size 0.09`                              |
| `--linf_size`         | Linf constraint (ε) for `linf`-norm attack.                                                          | `--linf_size 0.0001`                          |
| `--snr_db`            | Desired SNR (in dB) for `snr`-based attack.                                                          | `--snr_db 60`                                 |
| `--min_freq_attack`   | Minimum frequency to perturb (used in `min_max_freqs` mode).                                         | `--min_freq_attack 300`                       |
| `--max_freq_attack`   | Maximum frequency to perturb.                                                                        | `--max_freq_attack 20000`                     |
| `--num_items_to_inspect` | Number of samples to visualize with transcription comparison (GT, clean, perturbed).            | `--num_items_to_inspect 12`                   |

---

### 🔁 Example Command

Here's how to run a targeted Fletcher-Munson attack for 5 epochs:

```bash
python main.py \
  --attack_mode targeted \
  --target "delete" \
  --norm_type fletcher_munson \
  --fm_epsilon 2.0 \
  --num_epochs 5
```

Or a debug run with minimal data:

```bash
python main.py --small_data --norm_type linf --linf_size 0.0002
```


---

## 🧪 Evaluation & Visualization

- Perturbations are saved as `.wav` files
- Loss and WER plots are saved per epoch
- Clean vs perturbed transcription differences are logged
- Random "suspicious" examples are inspected and saved

---

## 👤 Author

**Tomer Erez**  
MSc Student, Technion – Israel Institute of Technology  
📧 tomer.erez@campus.technion.ac.il

---

## 📄 License

This project is for academic and research purposes only. Feel free to use and adapt it with credit.
```

