# Psychoacoustic Adversarial Attacks on speech recognition models

This project explores adversarial attacks on automatic speech recognition (ASR) models, 
with a focus on crafting **perturbations that are difficult for humans to perceive** but significantly degrade model performance.

We investigate how different norm types (e.g. Total variation, Signal to noise ratio, equal loudness contours) 
and perturbation sizes affect both the model's predictions and the human perceptibility of the perturbation. 
Our goal is to develop attacks that are **effective yet stealthy** — confusing the model while remaining inaudible to the human ear.

targeted(encourage a malicious command, e.g. "delete all files") and untargeted attacks can be run.


---

## 🎧 Model and Dataset

attacking the Wav2vec2 model by meta, which was trained on libreelight and libreespeech.

supporting attacks vs the test set of libreespeech(clean english, Audio-book readings), CommonVoice(heavy accents included), Tedlium(Ted lectures).

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
#targeted
python main.py --attack_mode targeted --target "delete" --norm_type snr --snr_db 40 
#untargeted
python main.py --attack_mode untargeted  --norm_type max_phon --max_phon_level 20 
```
You can specify various command-line arguments to control the attack (see **Hyperparameters** below, or the [`training_utils/parser.py`](training_utils/parser.py) file).

---


## ⚙️ Key Hyperparameters

You can control the behavior of the attack using the following command-line arguments:

| Argument            | Description                                                                                                                           | Example                                       |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--norm_type`       | Type of norm used to constrain the perturbation. Choices: `max_phon`,`tv`, `l2`, `linf`, `snr`, `fletcher_munson`, `min_max_freqs`... | `--norm_type fletcher_munson`                |
| `--attack_mode`     | Type of attack: `targeted` (force a word) or `untargeted` (increase error).                                                           | `--attack_mode targeted`                      |
| `--target`          | Target phrase for targeted attacks. Only used in `targeted` mode.                                                                     | `--target "delete"`                           |
| `--target_reps`     | How many times to repeat the target word in the label (e.g., `delete delete delete`).                                                 | `--target_reps 3`                             |
| `--small_data`      | Use only a small subset (~1%) of the dataset for fast debugging.                                                                      | `--small_data`                                |
| `--dataset`         | Which dataset to use. Prefer `CommonVoice` if attacking Wav2Vec2.                                                                     | `--dataset CommonVoice`                       |
| `--fm_epsilon`      | Max perceptual norm for Fletcher-Munson constraint.                                                                                   | `--fm_epsilon 2.0`                            |
| `--l2_size`         | L2 constraint (ε) for `l2`-norm attack.                                                                                               | `--l2_size 0.09`                              |
| `--linf_size`       | Linf constraint (ε) for `linf`-norm attack.                                                                                           | `--linf_size 0.0001`                          |
| `--snr_db`          | Desired SNR (in dB) for `snr`-based attack.                                                                                           | `--snr_db 60`                                 |
| `--min_freq_attack` | maximum frequency to perturb (used in `min_max_freqs` mode). frequenices above it will be set to zero                                 | `--min_freq_attack 300`                       |
| `--tv_epsilon`      | maximum total variation allowed                                                                                                       | `--min_freq_attack 300`                       |
| `--max_phon_level`  | maximum phon allowed                                                                                                                  | `--min_freq_attack 300`                       |

---
### Hidden attack Results vs LibriSpeech:

**Metric**: Word Error Rate (WER), based on edit distance between predicted and reference transcripts.

a standard metric for evaluating (ASR) systems. It quantifies the difference between the predicted transcript and the ground truth using edit distance:

WER = (S + D + I) / N

Where:
- **S**: Number of substitutions  
- **D**: Number of deletions  
- **I**: Number of insertions  
- **N**: Total number of words in the reference transcript  

A **higher WER** after applying perturbations indicates that the ASR model is more confused by the audio, which is the goal in adversarial settings. The **clean WER** serves as a baseline, and the **perturbed WER** shows the impact of the attack under different imperceptibility constraints.

**Task**: Un-targeted hidden attacks on the union of LibriSpeech test-clean, test-other, dev-clean, dev-other
set using various imperceptibility constraints.

| Constraint (`norm`)          | Norm size | WER   | Optimizer |
|------------------------------|-----------|-------|-----------|
| clean sound, no perturbation | 0         | 0.41  |
| `--fm`                       | -         |       | 
| `--l2`                       | 0.1       |       | Adam 
| `--linf`                     | 0.0015    |       |
| `--snr`                      | 64        |       | 
| `--min_freq_attack`          | 120 Hz    |       |
| `--max_phon`                 | 25        |       |
| `--tv`                       | 0.001     |       | 



---

## 🧪 Evaluation & Visualization

the logs dir will save perturbation results, including the .wav file, stft images, loss plots etc.

you can also generate aggregate results for analysis

```bash
#tensorboard
python training_utils/tensor_board_logging
#graphs
python data_plots/grab_results.py
```

---

## 👤 Author

**Tomer Erez**  
MSc Computer science, Technion – Israel Institute of Technology  
📧 tomer.erez@campus.technion.ac.il

---

## 📄 License

This project is for academic and research purposes only. Feel free to use and adapt it with credit.


