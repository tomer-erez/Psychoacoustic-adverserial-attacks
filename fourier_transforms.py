import torch

def compute_stft(p, args):
    return torch.stft(
        p,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        return_complex=True
    )

def compute_istft(stft_p, args):
    return torch.istft(
        stft_p,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )
