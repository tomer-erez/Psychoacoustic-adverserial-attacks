import torch
import torch.optim as optim
import shutil
import logging
import os
import sys
import json
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset
import numpy as np
import scipy.interpolate
from torch.utils.data import Subset, DataLoader
from core.train import perturbation_constraint
import sys
from datasets import load_dataset, Audio


def make_collate_fn(audio_length):
    def collate_fn(batch):
        waveforms = []
        texts = []
        for waveform, sample_rate, transcript, *_ in batch:
            waveform = waveform.squeeze()  # [T]
            waveform = waveform.to(dtype=torch.float32)  # <-- ensure float32
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

class SafeDatasetWrapper(Dataset):
    def __init__(self, base_dataset, dataset_type="CommonVoice"):
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type
        self.valid_indices = self._find_valid_indices()

    def _find_valid_indices(self):
        valid = []
        for i in range(len(self.base_dataset)):
            try:
                raw = self.base_dataset[i]  # avoid calling self[i]!
                if self.dataset_type == "LibreeSpeech":
                    _, _, transcript, *_ = raw
                    _ = transcript.strip()  # check it's a string and not empty
                elif self.dataset_type == "CommonVoice":
                    _ = raw["audio"]["array"]  # ensure audio is decoded
                    _ = raw["audio"]["sampling_rate"]
                    _ = raw["sentence"].strip()
                valid.append(i)
            except Exception as e:
                print(f"[WARN] Skipping invalid file at index {i}: {e}")
        return valid

    def __getitem__(self, index):
        raw = self.base_dataset[self.valid_indices[index]]

        if self.dataset_type == "LibreeSpeech":
            waveform, sample_rate, transcript, *_ = raw
        elif self.dataset_type == "CommonVoice":
            waveform, sample_rate = raw["audio"]["array"], raw["audio"]["sampling_rate"]
            waveform = torch.tensor(waveform).unsqueeze(0)  # make it [1, T] to match LibriSpeech format
            transcript = raw["sentence"]
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        return waveform, sample_rate, transcript

    def __len__(self):
        return len(self.valid_indices)


def create_data_loaders(args):
    """
    Load and prepare DataLoaders for either LibriSpeech or CommonVoice datasets.
    Handles fixed-length audio batching without SafeDatasetWrapper.
    """
    random.seed(args.seed)

    if args.dataset == "LibreeSpeech":
        base_dataset = LIBRISPEECH(
            root=args.LibriSpeech_path,
            url="train-clean-100",
            folder_in_archive="LibriSpeech",
            download=args.download_ds
        )
        dataset = list(base_dataset)  # already returns (waveform, sr, text)

    elif args.dataset == "CommonVoice":
        data_split = "test[:1%]" if args.small_data else "test" #test is about 16k samples
        base_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split=data_split, trust_remote_code=True)
        base_dataset = base_dataset.cast_column("audio", Audio(sampling_rate=16_000))

        def to_tuple(example):
            waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)
            sample_rate = example["audio"]["sampling_rate"]
            transcript = example["sentence"]
            return waveform, sample_rate, transcript
        dataset = list(map(to_tuple, base_dataset))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Shuffle and optionally reduce
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    if args.small_data:
        indices = indices[:20]
        args.num_items_to_inspect = 1
    # Estimate fixed waveform length
    sample_lengths = [dataset[i][0].shape[1] for i in indices[:min(200, len(indices))]]
    audio_length = int(torch.tensor(sample_lengths).float().quantile(0.85).item())

    # clip audios to be length
    collate_fn = make_collate_fn(audio_length)

    num_train = int(0.8 * len(indices))
    num_eval = int(0.1 * len(indices))

    train_subset = Subset(dataset, indices[:num_train])
    eval_subset = Subset(dataset, indices[num_train:num_train + num_eval])
    test_subset = Subset(dataset, indices[num_train + num_eval:])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"training size: {len(train_loader) * args.batch_size}\t"
          f"eval size: {len(eval_loader) * args.batch_size}\t"
          f"test size: {len(test_loader) * args.batch_size}\t"
          f"audio length: {audio_length}")

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
    Creates a logger that writes to the console if `small_data` is True,
    or to a log file otherwise.
    """
    # Determine attack size string
    size = ''
    if args.norm_type in ["fletcher_munson", "leakage", "min_max_freqs"]:
        if args.norm_type == "leakage":
            size = f'{args.min_freq_leakage}'
        elif args.norm_type == "min_max_freqs":
            size = f'{args.min_freq_attack}'
        elif args.norm_type == "fletcher_munson":
            size = f'{args.fm_epsilon}'
    else:
        if args.norm_type == "l2":
            size = f"{args.l2_size}"
        elif args.norm_type == "linf":
            size = f"{args.linf_size}"
        elif args.norm_type == "snr":
            size = f"{args.snr_db}"
    args.attack_size_string = size


    # Set save_dir and log_file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_logs_dir = os.path.join(os.path.dirname(current_dir), "logs")
    args.save_dir = os.path.join(all_logs_dir, args.device, f"{args.norm_type}_{args.attack_size_string}_{args.attack_mode}")
    args.was_preempted=False
    args.had_checkpoint=False

    os.makedirs(args.save_dir, exist_ok=True)

    args.log_file = os.path.join(args.save_dir, 'train_log.txt')

    if os.path.exists(args.save_dir):
        print('was preempted or save directory already existed')
        args.was_preempted = True
        perturbation_path = os.path.join(args.save_dir, "perturbation.pt")


        if os.path.exists(perturbation_path):
            args.had_checkpoint = True
            args.resume = True
            args.resume_from = perturbation_path
            x = f"this job was preempted and has a checkpoint in {args.resume_from}\n"
            if args.small_data:
                print(x)
            else:
                with open(args.log_file, 'w') as f:
                    f.write(x)

        else:
            with open(args.log_file, 'w') as f:
                f.write("this job was preempted but has no checkpoint, so we are training it from scratch\n")
                args.resume = False

        # Delete everything except "perturbation.pt"
        for item in os.listdir(args.save_dir):
            item_path = os.path.join(args.save_dir, item)
            if not(item.endswith(".pt")):
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)



    if not args.small_data:
        log_file = open(args.log_file, 'a')
        sys.stdout = log_file
        sys.stderr = log_file

    print("========= Args =========")
    print(json.dumps(vars(args), indent=4))
    print("========================")

    print(f"\nnorm type: {args.norm_type}, attack size: {args.attack_size_string}")
    sys.stdout.flush()
    sys.stderr.flush()
    return None


def init_perturbation(args,length,interp,first_batch_data):
    """
    Initializes a universal perturbation `p` with the same shape as input audio.
    Assumes args.dim = (num_channels, num_samples) or similar.
    """
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        print(f"Resuming perturbation from checkpoint: {args.resume_from}")
        loaded = torch.load(args.resume_from, map_location=args.device)
        p = loaded.detach().to(args.device).requires_grad_()
        if p.shape[-1] != length:
            raise ValueError(f"Loaded perturbation length {p.shape[-1]} does not match expected length {length}")
    else:
        p = torch.randn(1, length, device=args.device).detach().requires_grad_()
        p = perturbation_constraint(p=p,clean_audio=first_batch_data,args=args,interp=interp).detach().requires_grad_()
        p.retain_grad()
    return p


def create_optimizer(args,p):
    if args.optimize_type!='pgd':
        return optim.Adam([p], lr=args.lr)
    return None