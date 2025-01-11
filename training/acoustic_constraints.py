import torch
min_val, max_val= None, None


def apply_JND(perturbation):
    """
    Applies psychoacoustic constraints to ensure perturbation remains imperceptible.
    Leverages Just Noticeable Difference (JND) for amplitude and frequency.
    limit the perturbation's amplitude, frequency shifts, or duration changes to stay below JND thresholds. For example:

    Limit amplitude perturbations to be smaller than the JND for loudness (~1 dB).
    Add frequency perturbations that are smaller than the frequency JND (~0.5%).
    Scale the perturbation at each frequency bin to ensure that the perturbation remains imperceptible.

    """
    # JND threshold for loudness (in decibels)
    JND_loudness_db = 1.0  # ~1 dB is generally imperceptible

    # Convert to linear scale for calculations
    JND_loudness_factor = 10 ** (JND_loudness_db / 20)  # Convert dB to linear factor

    # **Amplitude Constraint**: Ensure the perturbation does not exceed the JND loudness
    max_amplitude = torch.max(torch.abs(perturbation))
    if max_amplitude > JND_loudness_factor:
        perturbation = perturbation / max_amplitude * JND_loudness_factor

    # **Frequency JND Constraint** (optional): Limit shifts in frequency bins
    # Example: Only allow small frequency perturbations (e.g., 0.5% shift)
    frequency_jnd_threshold = 0.005  # 0.5% JND for frequency perturbation
    perturbation_frequency_shift = torch.fft.fftshift(perturbation)
    perturbation_frequency_shift = torch.clamp(perturbation_frequency_shift, -frequency_jnd_threshold, frequency_jnd_threshold)

    # Combine and return the updated perturbation
    return perturbation

def compute_imperceptibility_penalty(perturbed_spectrogram, clean_spectrogram,a=0.01,b=0.01,c=0.01):
    """
    Bonuses to the loss to encourage imperceptibility of the perturbation w.r.t the original sound
    """
    # encourage attacks which are make noise in close time steps to the human speaker
    temp_mask_score = get_temporal_masking(perturbed_spectrogram, clean_spectrogram) 
    # encourage attacks which act in similar frequencies, with lower amplitude as the speaker
    spec_mask_score = get_spectral_masking(perturbed_spectrogram, clean_spectrogram)
    # overall deviation from the original sound
    deviation_score = get_deviation_score(perturbed_spectrogram, clean_spectrogram)
    return a * temp_mask_score + b * spec_mask_score + c * deviation_score

def perturbation_psychoacoustic_constraints(p):
    """
    Apply psychoacoustic constraints to the perturbation to ensure it remains imperceptible.
    Returns:
        Constrained perturbation.
    """
    # Step 1: Clip values to the valid audio range allowed (min and max amplitudes).
    p = clip(p, min_val, max_val)
    # Step 2: Frequency Sensitivity - Scale perturbations based on frequency sensitivity 
    # ensure most of the perturbation is outside of the human speech frequency 2-4 khz
    p = apply_frequency_sensitivity(p)
    # Step 3: Critical Bands - Distribute perturbations across different frequency bands.
    p = apply_critical_bands(p)
    p = apply_JND(p)
    # ensuere norm
    p = Lp_norm(p)
    return p
