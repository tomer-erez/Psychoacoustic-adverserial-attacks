import torch
import torchaudio
import matplotlib.pyplot as plt
import eval as eval_
from core import fourier_transforms
import os
import random

def save_audio(filename, tensor, sample_rate=16000, amplify=1.0):
    tensor = tensor.detach().cpu()
    # Optional amplification (e.g. for perturbation)
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



def inspect_random_samples(args, test_data_loader, p, model, processor):
    model.eval()
    # Collect all test data into a list (only needed once)
    all_samples = list(test_data_loader)
    chosen_samples = random.sample(all_samples, args.num_items_to_inspect)

    for i, (audio_batch, text_batch) in enumerate(chosen_samples):
        audio = audio_batch[0].to(args.device)  # [T]
        clean = audio.clone().unsqueeze(0)
        perturbed = (audio + p.squeeze(0)).unsqueeze(0)  # assuming p is [1, T]

        # Predict both
        clean_logits = eval_.get_logits(clean, processor, args, model)
        pert_logits = eval_.get_logits(perturbed, processor, args, model)

        clean_pred = eval_.decode(clean_logits, processor)[0]
        pert_pred = eval_.decode(pert_logits, processor)[0]

        ground_truth = text_batch[0]

        # Create folder
        sample_dir = os.path.join(args.save_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save audio
        save_audio(os.path.join(sample_dir, "clean.wav"), clean)
        save_audio(os.path.join(sample_dir, "perturbed.wav"), perturbed)

        # Save plots
        plot_pert(os.path.join(sample_dir, "clean.png"), clean)
        plot_pert(os.path.join(sample_dir, "perturbed.png"), perturbed)

        # Save transcription
        with open(os.path.join(sample_dir, "transcription.txt"), "w") as f:
            f.write(f"Ground Truth:   {ground_truth}\n")
            f.write(f"Clean Pred:     {clean_pred}\n")
            f.write(f"Perturbed Pred: {pert_pred}\n")

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
    save_audio(filename=f"{args.save_dir}\perturbation.wav",tensor=p)
    save_audio(filename=f"{args.save_dir}\perturbation_5x.wav",tensor=p,amplify=5)
    plot_pert(path=f"{args.save_dir}\perturbation.png",tensor=p)
    stft_plot(path=f"{args.save_dir}\perturbation_stft.png", tensor=p, args=args)

    inspect_random_samples(
        args=args,
        test_data_loader=test_data_loader,
        p=p,
        model=model,
        processor=processor
    )

def save_loss_plot(train_scores, eval_scores, save_dir):
    plt.figure()

    plt.plot(train_scores, label='Train Loss', marker='o')
    plt.plot(eval_scores, label='Eval Loss', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, "loss_plot.png")
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
