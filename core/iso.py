from typing import Union
from types import MappingProxyType
import os
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
import matplotlib.pyplot as plt
import torch

"""
This script is an implementation of 'equal loudness contours'.
the concept of equal loudness contours is that for different frequency levels, we have different amplitude sensitivity.
you can briefly read https://en.wikipedia.org/wiki/Equal-loudness_contour
key terms:

phon: estimation for human percieved loudness. 0 is the lower threshold of hearing,
60 is about the speech range, 120 is maybe a rock concert, and 150 is painful.

SPL:  physical measurment unit for effective pressure(amplitude) of a sound relative to a reference value.

frequency: self explanatory.

so equal loudness contours are over a map of SPL, frequency, showing for different frequency levels, what SPL level is needed to achieve a phon level X.

for example:
at frequency 100hz, we need ~25 SPL to achieve 0 phon level of loudness(lower threshold), 
but at 1000hz, we need ~5 SPL to achieve 0 phon level of loudness(lower threshold), 
"""



class ISO226:
    reference = MappingProxyType({
        'frequencies': (
            20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
            400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
            5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
        ),
        'alpha': (
            0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301,
            0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242,
            0.242, 0.245, 0.254, 0.271, 0.301,
        ),
        'l_u': (
            -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1,
            -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7,
            -3.1,
        ),
        't_f': (
            78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2,
            4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
        )
    })

    def __init__(self, phon: Union[int, float]) -> None:
        if phon < 0 or phon > 90:
            raise ValueError('Phon must be in range [0, 90]')
        self._phon = phon

        # Extend to 20kHz
        f = np.array(self.reference['frequencies'] + (20000.0,))
        self._alpha = PchipInterpolator(f, np.array(self.reference['alpha'] + (self.reference['alpha'][0],)))
        self._lu = PchipInterpolator(f, np.array(self.reference['l_u'] + (self.reference['l_u'][0],)))
        self._tf = PchipInterpolator(f, np.array(self.reference['t_f'] + (self.reference['t_f'][0],)))

    def __call__(self, frequencies) -> np.ndarray:
        frequencies = np.asarray(frequencies)
        if np.any(frequencies < 20.0) or np.any(frequencies > 20000.0):
            raise ValueError("Frequency must be in [20, 20000] Hz")

        out = np.zeros_like(frequencies)
        for i, f in np.ndenumerate(frequencies):
            a = 0.00447 * ((10.0 ** (0.025 * self._phon)) - 1.15)
            b = (0.4 * (10.0 ** (((self._tf(f) + self._lu(f)) / 10.0) - 9.0))) ** self._alpha(f)
            out[i] = ((10.0 / self._alpha(f)) * np.log10(a + b)) - self._lu(f) + 94.0
        return out


def compute_iso226_weight_matrix():
    phons = np.arange(0, 100, 10)
    freqs = np.array(ISO226.reference['frequencies'] + (20000.0,))
    spl_matrix = np.array([ISO226(phon)(freqs) for phon in phons])
    return freqs, phons, spl_matrix


def perceptual_weight(spl_matrix):
    max_spl = spl_matrix.max()
    weights = (1 - (spl_matrix / spl_matrix.max())) ** 2  # gamma > 1
    # weights = spl_matrix / max_spl # higher weights where more preciptible, so we later project back by the integral
    # weights=1-weights
    return np.clip(weights, 0, 1)


def build_weight_interpolator():
    freqs, phons, spl_matrix = compute_iso226_weight_matrix()
    weights = perceptual_weight(spl_matrix)
    return RegularGridInterpolator((phons, freqs), weights, bounds_error=False, fill_value=1.0)


def make_fm_weight_tensor(args, phon_level: float, interp):
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sr).to(args.device)
    freqs_np = freqs.detach().cpu().numpy()
    query_pts = np.stack([np.full_like(freqs_np, phon_level), freqs_np], axis=-1)
    weights_np = interp(query_pts)
    weights = torch.tensor(weights_np, device=args.device, dtype=torch.float32)
    weights /= weights.max()
    return freqs, weights


# Optional plotting functions
def plot_equal_loudness_contours(freqs, phons, spl_matrix,save_dir):
    for i, phon in enumerate(phons):
        plt.semilogx(freqs, spl_matrix[i], label=f'{phon} phon')
    plt.gca().invert_yaxis()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SPL (dB)")
    plt.title("Equal-Loudness Contours")
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'equal_loudness_contours.png'))


def plot_perceptual_weight_surface(freqs, phons, weights, save_dir):
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(freqs, phons)

    plt.contourf(X, Y, weights, levels=100, cmap='magma')
    plt.colorbar(label='Perceptual Sensitivity (Weight)')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phon')
    plt.title('Perceptual Sensitivity (Higher = More Perceptible)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perceptual_weights_heatmap_better.png'))
