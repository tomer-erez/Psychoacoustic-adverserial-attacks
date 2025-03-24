import torch
import torch.optim as optim
import shutil
import logging
import os
import json
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset
import numpy as np
import scipy.interpolate
import save

def make_collate_fn(audio_length):
    def collate_fn(batch):
        waveforms = []
        texts = []
        for waveform, sample_rate, transcript, *_ in batch:
            waveform = waveform.squeeze()  # [T]

            # Clamp or pad to audio_length
            if waveform.shape[0] > audio_length:
                waveform = waveform[:audio_length]
            else:
                pad_size = audio_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            waveforms.append(waveform)  # [T] of fixed length
            texts.append(transcript)

        waveforms = torch.stack(waveforms)  # [B, T]
        return waveforms, texts

    return collate_fn

class SafeLibriSpeech(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.valid_indices = self._find_valid_indices()

    def _find_valid_indices(self):
        valid = []
        for i in range(len(self.base_dataset)):
            try:
                _ = self.base_dataset[i]
                valid.append(i)
            except Exception as e:
                print(f"[WARN] Skipping invalid file at index {i}: {e}")
        return valid

    def __getitem__(self, index):
        return self.base_dataset[self.valid_indices[index]]

    def __len__(self):
        return len(self.valid_indices)



def create_data_loaders(args):
    """
    Create DataLoaders using torchaudio's built-in LIBRISPEECH dataset,
    and batch them with waveform clamping/padding to fixed audio_length.
    """

    base_dataset = LIBRISPEECH(
        root=args.dataset_path,
        url="train-clean-100",
        folder_in_archive="LibriSpeech",
        download=args.download_ds
    )
    dataset = SafeLibriSpeech(base_dataset)

    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_size=999
    if args.small_data:
        subset_size = 23
        args.num_items_to_inspect=1
        indices = indices[:subset_size]

    # === Estimate 75th percentile waveform length ===
    sample_indices = indices[:min(200,subset_size)]  # Sample 200 examples for efficiency
    sample_lengths = []
    for idx in sample_indices:
        waveform, sample_rate, *_ = dataset[idx]
        sample_lengths.append(waveform.shape[1])  # [1, T]

    length_tensor = torch.tensor(sample_lengths)
    quantile = 0.85
    audio_length = int(length_tensor.float().quantile(quantile).item())
    # === Create collate_fn using that length ===
    collate_fn = make_collate_fn(audio_length)


    num_train = int(0.8 * len(indices))
    num_eval = int(0.1 * len(indices))

    train_subset = torch.utils.data.Subset(dataset, indices[:num_train])
    eval_subset = torch.utils.data.Subset(dataset, indices[num_train:num_train + num_eval])
    test_subset = torch.utils.data.Subset(dataset, indices[num_train + num_eval:])

    # === DataLoaders with fixed-length waveforms ===
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    x=(f"training size: {len(train_loader)*args.batch_size}\t"
          f"eval size: {len(eval_loader)*args.batch_size}\t"
          f"test size: {len(test_loader)*args.batch_size}\t"
          f"audio length: {audio_length}\n")
    print(x)
    return train_loader, eval_loader, test_loader, audio_length



def get_model_size_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    total_size_gb = (param_size_bytes + buffer_size_bytes) / (1024 ** 3)
    return total_size_gb

def load_model(args):
    """
    Load the ASR model from `args.model_path`.
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(args.device)
    print(f"Model size: {get_model_size_gb(model):.2f} GB")
    return model,processor


def create_logger(args):
    """
    Creates a save directory and logger that writes to `args.log_file`.
    If the log file already exists, it will be overwritten.
    """
    size=''
    if args.norm_type in ["fletcher_munson", "leakage","min_max_freqs"]:
        if args.norm_type == "leakage":
            size = f'{args.min_freq_leakage}_{args.max_freq_leakage}'
        elif args.norm_type == "min_max_freqs":
            size = f'{args.min_freq_attack}_{args.max_freq_attack}'
        elif args.norm_type == "fletcher_munson":
            size=f'{args.fm_epsilon}'
    else:
        if args.norm_type == "l2":
            size= f"{args.l2_size}"
        elif args.norm_type=="linf":
            size= f"+-{args.linf_size}"
        elif args.norm_type=="snr":
            size= f"{args.snr_db}"
    args.attack_size_string=size

    print(f"\nnorm type: {args.norm_type}, attack size: {args.attack_size_string}\n")

    args.save_dir = os.path.join("logs", args.device, f"{args.norm_type}_{args.attack_size_string}")
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    args.log_file = os.path.join(args.save_dir, 'train_log.txt')

    logger = logging.getLogger("ASR_Training")
    logger.setLevel(logging.INFO)

    # Open file in write mode to overwrite existing logs
    file_handler = logging.FileHandler(args.log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    # Log all args cleanly
    logger.info("=== Experiment Args ===")
    logger.info(json.dumps(vars(args), indent=4))
    logger.info(f"Running on device: {args.device}")
    logger.info("========================")

    return logger





def make_fm_spline(args):
    """
    Returns interpolated Fletcher-Munson weights for the STFT frequency bins.
    """
    # ISO 226 frequency sensitivity data (normalized)
    ISO226_FREQS = np.array([
        0, 5, 10, 15,
        20, 25, 31.5, 40, 50, 63, 80, 100,
        125, 160, 200, 250, 315, 400, 500,
        630, 800, 1000, 1250, 1600, 2000, 2500,
        3150, 4000, 5000, 6300, 8000
    ])

    ISO226_SENSITIVITY = np.array([
        0, 0, 0, 0.001,
        0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18,
        0.26, 0.35, 0.45, 0.55, 0.66, 0.75, 0.82,
        0.88, 0.93, 1.0, 0.99, 0.95, 0.9, 0.85,
        0.75, 0.68, 0.58, 0.45, 0.32
    ])

    # STFT frequency bins in Hz
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sr).to(args.device)

    # Interpolate sensitivity weights to match your STFT frequency bins
    interp = scipy.interpolate.interp1d(
        ISO226_FREQS, ISO226_SENSITIVITY,
        kind="quadratic", bounds_error=False, fill_value=0.0
    )

    freqs_np = freqs.detach().cpu().numpy()
    weights_np = interp(freqs_np)
    weights = torch.tensor(weights_np, device=freqs.device, dtype=torch.float32)

    # Normalize (optional)
    weights /= weights.max()

    save.plot_fm_weights(freqs, weights, path=f"{args.save_dir}/fm_weights.png")
    return freqs,weights

def init_perturbation(args,length):
    """
    Initializes a universal perturbation `p` with the same shape as input audio.
    Assumes args.dim = (num_channels, num_samples) or similar.
    """
    p = torch.zeros(1, length, device=args.device).detach().requires_grad_()
    freqs,weights=make_fm_spline(args)

    if args.resume_from is not None and os.path.isfile(args.resume_from):
        print(f"[INFO] Resuming perturbation from: {args.resume_from}")
        loaded = torch.load(args.resume_from, map_location=args.device)
        p = loaded.detach().to(args.device).requires_grad_()
        if p.shape[-1] != length:
            raise ValueError(f"Loaded perturbation length {p.shape[-1]} does not match expected length {length}")
    else:
        p = torch.zeros(1, length, device=args.device).detach().requires_grad_()

    return p, freqs, weights


def create_optimizer(args,p):
    if args.optimize_type!='pgd':
        return optim.Adam([p], lr=args.lr)
    return None