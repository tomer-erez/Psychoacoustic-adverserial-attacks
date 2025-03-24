import torch
import numpy as np
import scipy.interpolate

def project_snr(clean, perturbation, snr_db):
    """
    Rescales the perturbation so that it has a target SNR vs the clean signal.
    """
    signal_power = torch.mean(clean ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Desired L2 norm of perturbation
    target_perturbation_norm = torch.sqrt(noise_power * clean.numel())

    current_norm = torch.norm(perturbation.view(-1), p=2)
    if current_norm < 1e-8:
        return perturbation  # nothing to rescale

    # Scale perturbation to have the desired L2 norm
    return perturbation * (target_perturbation_norm / current_norm)

def project_linf(p,min_val,max_val):
    return torch.clamp(p,min_val,max_val)

def project_l2(p, epsilon):
    norm = torch.norm(p, p=2)
    if norm < 1e-8:
        return p
    return p * (epsilon / norm)


def project_min_max_freqs(args,stft_p, min_freq, max_freq):

    # Get frequencies for STFT bins
    freq_bins = stft_p.shape[-2]
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sr).to(stft_p.device)

    # Mask: 1 where freqs < min or freqs > max
    mask = ((freqs < min_freq) | (freqs > max_freq)).float()
    mask = mask.view(1, -1, 1)  # Match (batch, freq, time)

    # Apply mask
    stft_p = stft_p * mask

    # Inverse STFT to time domain
    return stft_p


def compute_fm_weighted_norm(stft_p, args):
    """
    Computes perceptual (FM-weighted) norm of the STFT of the perturbation.
    Uses precomputed frequency weights in args.weights.
    """

    # Assume: args.weights is a [F] tensor matching the STFT's frequency bins
    weights = args.weights  # [F]
    weights = weights.view(1, -1, 1)   # [1, F, 1] for broadcasting with [B, F, T]

    # Apply FM-weighted norm
    power = stft_p.abs() ** 2          # [B, F, T]
    weighted_power = power * weights  # [B, F, T]
    fm_norm = torch.sqrt(weighted_power.sum())  # scalar

    return fm_norm


def project_fm_norm(stft_p, args):
    """
    Scales the STFT of a perturbation so that its FM-weighted norm is ≤ epsilon.
    Input `stft_p` is expected to be in the frequency domain.
    """
    norm = compute_fm_weighted_norm(stft_p, args)
    scale = args.fm_epsilon / norm.clamp(min=1e-8)
    return stft_p * scale