import torch
import torchaudio
import matplotlib.pyplot as plt
from core import fourier_transforms, loss_helpers
import random
import shutil
import json
import os


def save_audio(filename, tensor, sample_rate=16000, amplify=1.0):
    tensor = tensor.detach().cpu()
    # Optional amplification (e.g. if you want to play it and the original perturbation is completely un-detectable)
    tensor = tensor * amplify
    # Clamp to [-1.0, 1.0]
    tensor = torch.clamp(tensor, -1.0, 1.0)
    # Convert to int16 PCM format
    int_tensor = (tensor * 32767).to(torch.int16)
    # Save as WAV
    torchaudio.save(filename, int_tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)


def plot_pert(path, tensor, sample_rate=16000, title="Perturbation waveform"):
    """
    Plots a waveform of the perturbation tensor and saves it as a PNG.

    Args:
        path (str): Path to save the PNG.
        tensor (Tensor): 1D or 2D waveform tensor (e.g., [T] or [B, T]).
        sample_rate (int): Sample rate for x-axis time scaling.
        title (str): Plot title.
    """
    tensor = tensor.squeeze()  # This removes the batch dim if it exists
    tensor = tensor.detach().cpu()
    time_axis = torch.arange(tensor.shape[-1]) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, tensor.numpy(), linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def inspect_random_samples(args, test_data_loader, p, model, processor, epoch):
    model.eval()
    all_samples = list(test_data_loader)
    chosen_samples = random.sample(all_samples, args.num_items_to_inspect)

    for i, (audio_batch, text_batch) in enumerate(chosen_samples):
        k = random.randint(0, len(audio_batch) - 1)
        audio = audio_batch[k].to(args.device)
        clean = audio.clone().unsqueeze(0)
        perturbed = (audio + p.squeeze(0)).unsqueeze(0)

        clean_logits = loss_helpers.get_logits(clean, processor, args, model)
        pert_logits = loss_helpers.get_logits(perturbed, processor, args, model)

        clean_pred = loss_helpers.decode(clean_logits, processor)[0]
        pert_pred = loss_helpers.decode(pert_logits, processor)[0]
        ground_truth = text_batch[k]
        # Decide if sample is suspicious
        is_sus = False
        if args.attack_mode == "targeted":
            is_sus = args.target in pert_pred
        elif args.attack_mode == "untargeted":
            is_sus = clean_pred != pert_pred

        # Paths
        sample_dir = os.path.join(args.save_dir, f"sample_{i}")
        sus_sample_dir = os.path.join(args.save_dir, f"sus_sample_{i}")

        # Clear and choose correct directory
        if is_sus:
            if os.path.exists(sample_dir):
                shutil.rmtree(sample_dir)
            if os.path.exists(sus_sample_dir):
                shutil.rmtree(sus_sample_dir)
            os.makedirs(sus_sample_dir)
            out_dir = sus_sample_dir
        else:
            if os.path.exists(sus_sample_dir):
                shutil.rmtree(sus_sample_dir)
            if os.path.exists(sample_dir):
                shutil.rmtree(sample_dir)
            os.makedirs(sample_dir)
            out_dir = sample_dir

        # Save audio
        save_audio(os.path.join(out_dir, "clean.wav"), clean)
        save_audio(os.path.join(out_dir, "perturbed.wav"), perturbed)

        name_tr = "sus_transcription.txt" if is_sus else "transcription.txt"
        with open(os.path.join(out_dir, name_tr), "w") as f:
            f.write(f"{'Ground Truth:'.ljust(28)}{ground_truth.lower()}\n\n")
            f.write(f"{'Clean Prediction:'.ljust(28)}{clean_pred.lower()}\n\n")
            f.write(f"{'Perturbed Prediction:'.ljust(28)}{pert_pred.lower()}\n\n")



def stft_plot(path, tensor, args, sample_rate=16000, title="STFT Magnitude"):
    stft = fourier_transforms.compute_stft(tensor.squeeze(0), args)
    magnitude = stft.abs().detach().cpu()

    # Compute frequency bins in Hz
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / sample_rate).numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(
        20 * torch.log10(magnitude + 1e-8),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[0, magnitude.shape[1], freqs[0], freqs[-1]]  # Set frequency axis (y) in Hz
    )
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_by_epoch(args, p,test_data_loader,model, processor,epoch_num):
    save_audio(filename=os.path.join(args.save_dir, "perturbation.wav"), tensor=p)
    save_audio(filename=os.path.join(args.save_dir, "perturbation_5x.wav"), tensor=p, amplify=5)
    plot_pert(path=os.path.join(args.save_dir, "perturbation.png"), tensor=p)
    stft_plot(path=os.path.join(args.save_dir, "perturbation_stft.png"), tensor=p, args=args)

    inspect_random_samples(
        args=args,
        test_data_loader=test_data_loader,
        p=p,
        model=model,
        processor=processor,
        epoch=epoch_num
    )

def save_loss_plot(train_scores, eval_scores_perturbed, eval_scores_clean, save_dir, norm_type,
                   clean_test_loss=None, perturbed_test_loss=None):
    os.makedirs(save_dir, exist_ok=True)

    x = list(range(len(train_scores["ctc"])))  # Epoch indices

    for loss_type in ["ctc", "wer"]:
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_scores[loss_type], label='Train', marker='o', linestyle='-', color='blue')
        plt.plot(x, eval_scores_clean[loss_type], label='Eval Clean', marker='^', linestyle='-', color='orange')
        plt.plot(x, eval_scores_perturbed[loss_type], label='Eval Perturbed', marker='x', linestyle='-', color='purple')

        if clean_test_loss is not None:
            plt.axhline(y=clean_test_loss[loss_type], linestyle='-', color='green', label='Clean Test')

        if perturbed_test_loss is not None:
            plt.axhline(y=perturbed_test_loss[loss_type], linestyle='-', color='red', label='Perturbed Test')

        plt.xlabel("Epoch")
        plt.ylabel(f"{loss_type.upper()} Loss")
        plt.title(f"{loss_type.upper()} Loss Curve — Norm Type: {norm_type}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"loss_plot_{loss_type}.png")
        plt.savefig(plot_path)
        plt.close()



def plot_fm_weights(freqs, weights, path="fm_weights.png"):
    """
    Plots the Fletcher-Munson perceptual sensitivity curve.
    """
    freqs_cpu = freqs.detach().cpu().numpy()
    weights_cpu = weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_cpu, weights_cpu, label="Interpolated FM Weights", color='purple')
    plt.title("Fletcher-Munson Perceptual Sensitivity Curve")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Perceptual Sensitivity (Normalized)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()





def save_json_results(save_dir,
                      norm_type,
                      attack_size,
                      epoch=None,
                      train_score=None,
                      eval_score_clean=None,
                      eval_score_perturbed=None,
                      best_train_score=None,
                      best_eval_score=None,
                      final_test_clean=None,
                      final_test_perturbed=None):
    json_path = os.path.join(save_dir, "results.json")

    def safe_to_float(v):
        return {k: float(v[k]) for k in v} if isinstance(v, dict) else float(v)

    results = {
        "norm_type": norm_type,
        "attack_size": float(attack_size),
    }

    if epoch is not None:
        results["epoch"] = epoch
    if train_score is not None:
        results["train_score"] = safe_to_float(train_score)
    if eval_score_clean is not None:
        results["eval_score_clean"] = safe_to_float(eval_score_clean)
    if eval_score_perturbed is not None:
        results["eval_score_perturbed"] = safe_to_float(eval_score_perturbed)
    if best_train_score is not None:
        results["best_train_score"] = safe_to_float(best_train_score)
    if best_eval_score is not None:
        results["best_eval_score"] = safe_to_float(best_eval_score)
    if final_test_clean is not None:
        results["test_loss_clean"] = safe_to_float(final_test_clean)
    if final_test_perturbed is not None:
        results["test_loss_perturbed"] = safe_to_float(final_test_perturbed)
        if final_test_clean:
            if isinstance(final_test_clean, dict):
                results["perturbation_efficiency"] = {
                    k: final_test_perturbed[k] / final_test_clean[k] for k in final_test_clean
                }
            else:
                results["perturbation_efficiency"] = final_test_perturbed / final_test_clean

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

