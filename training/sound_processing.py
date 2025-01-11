import random
import numpy as np
from scipi.signal import stft, istft

def apply_rir(perturbed_audio, rirs):
    r = random.choice(rirs)
    perturbed_audio = convolve(perturbed_audio, r, mode='same')[:len(x)]
    return perturbed_audio

def dynamic_mic_compressor(audio, threshold, ratio):
    above_thresh = np.abs(audio) > threshold
    audio[above_thresh] = threshold + (audio[above_thresh] - threshold) / ratio
    return audio


def preprocess_sound(args, x, p, rirs):
    clean_spectrogram = stft(x, nperseg=args.n_fft, noverlap=args.hop_length)  # STFT of clean audio
    # Step 1: Detect start of speech
    speech_start_frame = np.argmax(np.sum(np.abs(clean_spectrogram), axis=0) > args.silence_threshold)
    # Step 2: Align perturbation `p` to start when the speech begins
    aligned_p = np.zeros_like(clean_spectrogram)
    aligned_p[:, speech_start_frame:] = p[:, :clean_spectrogram.shape[1] - speech_start_frame]
    # Step 3: Add aligned perturbation
    perturbed_spectrogram = clean_spectrogram + aligned_p
    # Convert back to time-domain audio
    perturbed_audio = istft(perturbed_spectrogram)
    # Apply room effects
    if rirs is not None:
        perturbed_audio = sounds_preprocessing.apply_rirs(perturbed_audio, rirs, mode='same')[:len(x)]
    # Simulate microphone behavior
    perturbed_audio = mic_filter(perturbed_audio, -args.mic_threshold, args.mic_threshold)
    return perturbed_audio, perturbed_spectrogram, clean_spectrogram