####took the script as is from equakl loudness contours
#### credit to??????? #TODO
####
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
    #industry standard approximation to ISO 226 equal-loudness contours.
    """
    ISO 226 equal-loudness contours (2013/2003): given a target loudness level in phons
    and a frequency in Hz, compute the sound pressure level (SPL, in dB) that humans
    perceive as equally loud at that frequency.

    Usage:
        spls = ISO226(phon=60)(frequencies_hz_array)

    Design notes:
    - The ISO tables below (alpha, l_u, t_f) are frequency-dependent parameters used in
      the standard's closed-form approximation (see equation in __call__).
    - We extend/interpolate the tables to a continuous function over [20, 20000] Hz
      using PCHIP (monotone, shape-preserving) rather than e.g. cubic splines, which
      can overshoot and produce physically implausible SPLs between tabulated points.
    - The class is callable so you can treat an ISO226(phon) instance as a function f↦SPL.

    Units:
    - Frequency inputs/outputs: Hz
    - SPL outputs: dB re 20 µPa
    - Loudness input: phon (by definition, 1 kHz SPL in dB for that contour)
    """

    # Read-only “reference” dictionary holding ISO226 tabulated parameters at 29 1/3-octave bands.
    # MappingProxyType prevents accidental mutation at runtime.
    reference = MappingProxyType({
        # Center frequencies (Hz) for the standard 1/3-octave grid, covering 20 Hz..12.5 kHz
        'frequencies': (
            20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
            400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
            5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
        ),
        # α(f): frequency-dependent exponent shaping how loudness grows with SPL
        #how quickly perceived loudness grows as you increase sound pressure level (SPL), at a given frequency.
        'alpha': (
            0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301,
            0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242,
            0.242, 0.245, 0.254, 0.271, 0.301,
        ),
        # L_u(f): low-SPL upward spread correction term (dB)
        #tweaks the model at quiet levels so it better matches how our ears actually hear soft sounds
        'l_u': (
            -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1,
            -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7,
            -3.1,
        ),
        # T_f(f): absolute threshold of hearing (dB SPL) at frequency f
        't_f': (
            78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2,
            4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
        )
    })

    def __init__(self, phon: Union[int, float]) -> None:
        """
        Construct an equal-loudness “contour” parameterized by loudness level (phon).

        Args:
            phon: Loudness level in phons (0..90). By definition, at 1 kHz the SPL equals `phon` dB.

        Raises:
            ValueError: if `phon` is outside the canonical 0-90 range used by ISO 226.
        """
        # Sanity-check the phon range. ISO 226 tables/fit are typically validated 0..90 phon.
        if phon < 0 or phon > 90:
            raise ValueError('Phon must be in range [0, 90]')
        self._phon = phon

        # === Build continuous interpolants for α(f), L_u(f), and T_f(f) over [20, 20000] Hz ===
        # We extend the frequency axis to 20 kHz by appending a single endpoint (20000 Hz) so the
        # interpolators cleanly cover the closed interval [20, 20000] without extrapolation.
        f = np.array(self.reference['frequencies'] + (20000.0,))

        # IMPORTANT NOTE:
        # Below, we append a *duplicate* value at 20 kHz to match the new grid length.
        # The code uses reference['...'][0] (the 20 Hz entry) as the appended value. This makes the
        # parameter curve wrap to the 20 Hz value at 20 kHz—effectively a constant-end boundary at
        # 20 kHz equal to the 20 Hz level. That avoids out-of-range errors but may not be physically
        # intended. Many implementations instead repeat the *last* tabulated value at 12.5 kHz.
        # Kept as-is to preserve behavior; consider revisiting if you want more accurate 20 kHz tails.

        #PchipInterpolator creates a Piecewise Cubic Hermite Interpolating Polynomial
        #fills in the gaps between tabulated data points using smooth, stable cubic curves, without introducing fake wiggles
        self._alpha = PchipInterpolator(
            f,
            np.array(self.reference['alpha'] + (self.reference['alpha'][0],))
        )
        self._lu = PchipInterpolator(
            f,
            np.array(self.reference['l_u'] + (self.reference['l_u'][0],))
        )
        self._tf = PchipInterpolator(
            f,
            np.array(self.reference['t_f'] + (self.reference['t_f'][0],))
        )

    def __call__(self, frequencies) -> np.ndarray:
        """
        Evaluate the equal-loudness contour at one or many frequencies.

        Args:
            frequencies: scalar or array-like of Hz (any shape). Accepts vectors, matrices, etc.

        Returns:
            np.ndarray of SPL values (dB), shape matches `frequencies`.a NumPy array of SPLs (in dB)—one value per input frequency—where
              each value is the sound-pressure level required at that frequency to be perceived as loud as the instance's phon level.

        Raises:
            ValueError: if any frequency is outside the supported [20, 20000] Hz range.

        Math (ISO 226 closed-form, using a(f), L_u(f), T_f(f)):
            Let N = phon (loudness level, phons). Define
              A = 0.00447 * (10^(0.025*N) - 1.15)
              B = [0.4 * 10^((T_f(f) + L_u(f))/10- 9)]^{a(f)}
            Then the required SPL is:
              L_p(f; N) = (10/a(f)) * log10(A + B) - L_u(f) + 94  [dB SPL]
            This implementation follows that form, with a, L_u, T_f provided by smooth PCHIP interpolants.
        """
        # Convert to ndarray for robust broadcasting; preserve shape for output.
        frequencies = np.asarray(frequencies)

        # Guardrail: the interpolants are only guaranteed on [20, 20000] Hz, matching ISO tables.
        if np.any(frequencies < 20.0) or np.any(frequencies > 20000.0):
            raise ValueError("Frequency must be in [20, 20000] Hz")

        # Prepare output array with same shape/dtype as input
        out = np.zeros_like(frequencies)

        # Iterate with np.ndenumerate so arbitrary-shaped inputs (e.g., 2D grids) are supported
        # without flattening/reshaping logic. This is simple and explicit; for very large arrays,
        # you could vectorize by evaluating α/L_u/T_f on the whole array at once.
        for freq_idx, freq in np.ndenumerate(frequencies):
            # a) Loudness-dependent term A (depends only on phon N but we compute inline for clarity)
            a = 0.00447 * ((10.0 ** (0.025 * self._phon)) - 1.15)  # dimensionless

            # b) Frequency-dependent base term raised to α(freq).
            #    tf+lu appears inside a 10^(x/10) factor to convert from dB to a linear power-like quantity.
            b_base = 0.4 * (10.0 ** (((self._tf(freq) + self._lu(freq)) / 10.0) - 9.0))
            b = b_base ** self._alpha(freq)  # dimensionless

            # c) Final SPL per ISO 226: combine A and B, scale by 10/α(f), subtract L_u(f), add 94 dB reference.
            out[freq_idx] = ((10.0 / self._alpha(freq)) * np.log10(a + b)) - self._lu(freq) + 94.0

        #out at each index is the spl needed at the frequency[index] to reach the requested phon(how you instantiated the class with phon x ...)
        return out


def compute_iso226_weight_matrix():
    """
    Build a small grid (phon x frequency) of ISO226 SPL values.

    Returns:
        freqs: (F,) array of frequencies (Hz), using ISO grid extended to 20 kHz
        phons: (P,) array of phon levels [0, 10, ..., 90]
        spl_matrix: (P, F) SPLs (dB), where spl_matrix[p, f] = SPL needed at freq to reach p loudness. wher ep is phone

    Why:
    - Precomputing this grid lets us later convert SPLs to perceptual weights (how "sensitive"
      the ear is at (phon, freq)) and build a fast interpolator over the 2D domain.
    """
    # Discrete (human perceptually)loudness levels (0..90 by 10). densify if need smoother phon dependence.
    phons = np.arange(0, 100, 10)

    # Frequency axis: use the ISO frequencies and append 20 kHz so downstream interpolators cover the full band. interpolate between 12.5k to 20k as it does not exist in the iso 226 starndard
    freqs = np.array(ISO226.reference['frequencies'] + (20000.0,)) #interoplate all weh way to 20000

    # Evaluate ISO226 for each phon level over the same frequency grid.
    # Resulting shape: (P, F) where each entry is  spl needed at freq 
    spl_matrix = np.array([ISO226(phon)(freqs) for phon in phons])

    return freqs, phons, spl_matrix


def perceptual_weight(spl_matrix: np.ndarray) -> np.ndarray:
    #converts the ISO-226 SPL grid into a perceptual penalty map
    """
    Map SPL (dB) requirements to *penalty weights* in [0, 1] for perceptual optimization.

    Args:
        spl_matrix: (P, F) SPLs in dB as returned by compute_iso226_weight_matrix()

    Returns:
        weights: (P, F) penalty weights in [0,1], larger where the ear is *more sensitive*.

    Rationale:
    - Equal-loudness SPL is the *required* dB to achieve a fixed perceived loudness.
      If the required SPL is *low* at (phon, freq), the ear is *sensitive* there.
      We therefore want a *higher penalty* (discourage adding energy) in sensitive regions.
    - We implement that by normalizing SPLs to [0,1] by max, inverting (1 - x), and squaring (y=2)
      for a smoother emphasis: weights = (1 - (SPL / SPL_max))^2.
      → low SPL ⇒ large weight (more penalty), high SPL ⇒ small weight (less penalty).
    - Clip to [0,1] to be robust to numerical noise.

    Notes:
    - Alternative mappings are possible depending on your cost function. E.g., using SPL/max directly
      would *favor* adding energy where SPL is high (i.e., less sensitive) if you interpret weights
      as *allowance* instead of *penalty*. Here we choose the penalty interpretation explicitly.
    """
    # Global max across the grid; used to normalize SPLs to [0, 1].
    max_spl = spl_matrix.max()

    # Penalty weight: emphasize regions where the ear is sensitive (low SPL requirement).
    # The exponent (γ>1) makes the penalty grow nonlinearly; γ=2 is a common gentle choice.
    weights = (1 - (spl_matrix / max_spl)) ** 2

    # Defensive clipping (e.g., if max_spl == 0 is impossible here, but rounding could push tiny negatives)
    return np.clip(weights, 0, 1)


def build_weight_interpolator():
    """
    Construct a fast 2D interpolator w(phon, freq) → penalty weight in [0,1].

    Returns:
        interpolator: callable such that interpolator([[phon, freq], ...]) → weights
                      (supports vectorized queries; see RegularGridInterpolator doc)

    Implementation details:
    - We first compute a coarse grid of SPLs (PxF), convert them to penalty weights (PxF),
      then wrap with RegularGridInterpolator over the axes (phons, freqs).
    - bounds_error=False: queries slightly outside the grid won't raise; we fall back to fill_value.
    - fill_value=1.0: if outside, we assign maximum penalty (conservative), i.e., “dont put energy there”.
      Adjust if you prefer a different behavior (e.g., nearest).
    """
    # Build the (phon × freq) grid of SPLs
    #in the spl mat, each entry spl[f][p] means how much spl(phisical measurment in DB) is needed at frequency f to create loudness level p 
    freqs, phons, spl_matrix = compute_iso226_weight_matrix()

    # Convert SPL grid to perceptual penalty weights, high weight for loud areas in the spl matrix
    weights = perceptual_weight(spl_matrix)

    # Create an interpolator for continuous queries that are not specified in the descrete phons, freqs sets(very possible that we use freqs that are not in the reference map of the iso class but are in between values). Axes order must match the weights shape (P, F).
    return RegularGridInterpolator(
        (phons, freqs),
        weights,
        bounds_error=False,  # out-of-bounds returns fill_value rather than raising
        fill_value=1.0       # max penalty outside the calibrated domain
    )
