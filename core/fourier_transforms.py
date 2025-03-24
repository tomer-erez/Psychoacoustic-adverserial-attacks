import torch

def compute_stft(p, args):
    window = torch.hann_window(args.n_fft).to(p.device)
    hop_length = args.n_fft // 4
    win_length = args.n_fft
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sr)

    return torch.stft(
        p,
        n_fft=args.n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True
    )

def compute_istft(stft_p, args):
    window = torch.hann_window(args.n_fft).to(stft_p.device)
    hop_length = args.n_fft // 4
    win_length = args.n_fft

    return torch.istft(
        stft_p,
        n_fft=args.n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )