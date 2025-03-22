import torch
import fourier_transforms

def project_snr(orig_p, new_p, snr_db):
    signal_power = torch.mean(orig_p ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    new_norm = torch.sqrt(noise_power * orig_p.numel())

    actual_norm = torch.norm(new_p.view(-1), p=2)
    if actual_norm < 1e-8:
        return new_p
    return new_p * (new_norm / actual_norm)

def project_l2(p, epsilon):
    norm = torch.norm(p, p=2)
    if norm < 1e-8:
        return p
    return p * (epsilon / norm)

def project_fm_norm(p, args):
    fm_weight = get_fm_time_weight(p, args.sample_rate).to(p.device)
    weighted = p * fm_weight
    norm = torch.norm(weighted, p=2)
    factor = args.epsilon / norm.clamp(min=1e-8)
    return p * factor


def get_leakage_mask(stft_shape, sample_rate):
    """
    Generate a frequency mask that keeps only the perceptible human range (e.g. 20Hz–16kHz)
    and removes very high/low leakage noise.
    """
    num_freq_bins = stft_shape[-2]
    freqs = torch.fft.rfftfreq(n=(num_freq_bins - 1) * 2, d=1/sample_rate)

    mask = ((freqs >= 20) & (freqs <= 16000)).float()
    return mask.view(1, 1, -1, 1)  # Match STFT shape (batch, channel, freq, time)


def apply_leakage_mask(p, args):
    stft_p = fourier_transforms.compute_stft(p, args)
    leakage_mask = get_leakage_mask(stft_p.shape, args.sample_rate).to(p.device)
    masked_stft = stft_p * leakage_mask
    return fourier_transforms.compute_istft(masked_stft, args)

def apply_fletcher_munson_weighting(stft, args):
    """
    Apply inverse Fletcher-Munson weighting to downplay frequencies humans hear well.
    """
    freq_bins = stft.shape[-2]
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sample_rate).to(stft.device)

    # Generate perceptual sensitivity curve (peaks at 1–5 kHz)
    sensitivity = 1.0 / (0.1 + (freqs / 1000 - 4.3) ** 2)
    sensitivity = sensitivity / sensitivity.max()

    # Invert it — now low and high freqs are prioritized
    weights = 1.0 - sensitivity  # range: 0 (sensitive) to 1 (hard to hear)

    return stft * weights.view(1, -1, 1)

def remove_fletcher_munson_weighting(stft, args):
    freq_bins = stft.shape[-2]
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sample_rate).to(stft.device)

    sensitivity = 1.0 / (0.1 + (freqs / 1000 - 4.3) ** 2)
    sensitivity = sensitivity / sensitivity.max()

    weights = 1.0 - sensitivity

    return stft / (weights.view(1, -1, 1) + 1e-8)

def get_fm_time_weight(p, sample_rate):
    """
    Approximate a time-domain FM weighting by averaging the FM weights
    across time.
    """
    n_fft = 2048
    freqs = torch.fft.rfftfreq(n=n_fft, d=1/sample_rate).to(p.device)
    weights = 1.0 / (0.1 + (freqs / 1000 - 4.3) ** 2)
    weights = weights / weights.max()

    # Convert weights to time domain using IFFT
    weight_time = torch.fft.irfft(weights, n=n_fft)
    weight_time = torch.roll(weight_time, shifts=-(n_fft // 2))  # center window

    # Pad or crop to match input
    if weight_time.numel() > p.numel():
        weight_time = weight_time[:p.numel()]
    elif weight_time.numel() < p.numel():
        weight_time = torch.nn.functional.pad(weight_time, (0, p.numel() - weight_time.numel()))

    return weight_time
