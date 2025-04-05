import torch

def compute_stft(p, args):
    """
    maybe switch to meL???????

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