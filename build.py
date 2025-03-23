import torch
import torch.optim as optim
import logging
import os
import json
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset


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



def create_data_loaders(args,logger):
    """
    Create DataLoaders using torchaudio's built-in LIBRISPEECH dataset,
    and batch them with waveform clamping/padding to fixed audio_length.
    """
    dataset = LIBRISPEECH(
        root=args.dataset_path,
        url="train-clean-100",
        folder_in_archive="LibriSpeech",
        download=False
    )

    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # === Estimate 75th percentile waveform length ===
    sample_indices = indices[:200]  # Sample 200 examples for efficiency
    sample_lengths = []
    for idx in sample_indices:
        waveform, sample_rate, *_ = dataset[idx]
        sample_lengths.append(waveform.shape[1])  # [1, T]

    length_tensor = torch.tensor(sample_lengths)
    audio_length = int(length_tensor.float().quantile(0.75).item())  # 75th percentile length
    logger.info(f'75% audio length {audio_length}')
    # === Create collate_fn using that length ===
    collate_fn = make_collate_fn(audio_length)

    # === Dataset split ===
    num_train = int(0.8 * len(indices))
    num_eval = int(0.1 * len(indices))

    train_subset = torch.utils.data.Subset(dataset, indices[:num_train])
    eval_subset = torch.utils.data.Subset(dataset, indices[num_train:num_train + num_eval])
    test_subset = torch.utils.data.Subset(dataset, indices[num_train + num_eval:])

    # === DataLoaders with fixed-length waveforms ===
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"training data set size: {len(train_loader)}")

    return train_loader, eval_loader, test_loader, audio_length




def load_model(args):
    """
    Load the ASR model from `args.model_path`.
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.to(args.device)
    return model,processor


def create_logger(args):
    """
    Creates a save directory and logger that writes to `args.log_file`.
    If the log file already exists, it will be overwritten.
    """
    args.save_dir = os.path.join("logs", args.device, f"{args.norm_type}_{args.pert_size}_{args.jobid}")
    os.makedirs(args.save_dir, exist_ok=True)

    args.log_file = os.path.join(args.save_dir, 'train_log.txt')

    logger = logging.getLogger("ASR_Training")
    logger.setLevel(logging.INFO)

    # Open file in write mode to overwrite existing logs
    file_handler = logging.FileHandler(args.log_file, mode='w')  # 👈 This is the key line
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


def init_perturbation(args,length):
    """
    Initializes a universal perturbation `p` with the same shape as input audio.
    Assumes args.dim = (num_channels, num_samples) or similar.
    """
    return (torch.randn(1, length, device=args.device) / 100).detach().requires_grad_()

    # return (torch.randn(1, length, device=args.device) / 50).detach().requires_grad_()

def create_perturbation_optimizer(args, p):
    """
    Creates an optimizer that only updates the adversarial perturbation `p`.
    """
    return optim.AdamW([p], lr=args.lr)
