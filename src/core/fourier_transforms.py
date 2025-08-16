import torch


def compute_stft(p, args):
    """
    maybe switch to meL???????

    input perturbation shaped:
                                (batch_size, length)
    the stft returns:
                                (batch_size, freq_bins, time_frames)
    freq_bins from 0 to nyquist which is (n_fft//2)+1
    time_frames number of windows slid over the signal

    Each value in stft[b, f, t]:

    Is a complex number (since return_complex=True), representing:
        The amplitude and phase of frequency f during time frame t, for batch sample b.
    """
    window = torch.hann_window(args.n_fft).to(p.device)
    return torch.stft(
        p,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=window,
        return_complex=True,
        center=True
    )

def compute_istft(stft_p, args):
    window = torch.hann_window(args.n_fft).to(stft_p.device)

    return torch.istft(
        stft_p,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=window,
        center=True
    )