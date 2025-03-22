import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import logging
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset
from torchaudio.transforms import MelSpectrogram


class SpeechDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Custom dataset for speech recordings.
        :param file_paths: List of file paths for the dataset
        :param transform: Transformation to apply (e.g., MelSpectrogram)
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate


def create_data_loaders(args):
    """
    Create train, eval, and test DataLoaders.
    Assumes `args.data_dir` contains audio recordings.
    """
    random.seed(args.seed)
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.wav')]

    # Shuffle and split into 80-10-10 train-eval-test
    random.shuffle(all_files)
    num_train = int(0.8 * len(all_files))
    num_eval = int(0.1 * len(all_files))

    train_files = all_files[:num_train]
    eval_files = all_files[num_train:num_train + num_eval]
    test_files = all_files[num_train + num_eval:]

    transform = MelSpectrogram(sample_rate=16000, n_mels=80)

    train_dataset = SpeechDataset(train_files, transform)
    eval_dataset = SpeechDataset(eval_files, transform)
    test_dataset = SpeechDataset(test_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader


def create_loss_fn(args):
    """
    Creates a loss function for ASR models.
    Connectionist Temporal Classification (CTC) loss is commonly used in ASR.
    """
    return nn.CTCLoss()

def load_model(args):
    """
    Load the ASR model from `args.model_path`.
    """
    model = torch.load(args.model_path)  # Assuming it's a PyTorch model checkpoint
    return model


def create_logger(args):
    """
    creates a save directory which
    Create a logger that writes to `args.log_file`.
    """
    args.save_dir = f"/home/tomer.erez/psychoacoustic_attacks/logs/{args.device}/{args.norm_type}_{args.pert_size}_{args.jobid}"
    args.log_file = os.path.join(args.save_dir, 'train_log.txt')

    logger = logging.getLogger("ASR_Training")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def init_perturbation(args):
    """
    Initializes a universal perturbation `p` with the same shape as input audio.
    Assumes args.dim = (num_channels, num_samples) or similar.
    """
    p = torch.randn(args.dim) * 1e-3  # Small Gaussian noise around 0
    p = p.to(args.device)
    p.requires_grad = True
    p = torch.nn.Parameter(p)

    return p

def create_perturbation_optimizer(args, p):
    """
    Creates an optimizer that only updates the adversarial perturbation `p`.
    """
    return optim.AdamW([p], lr=args.lr)
