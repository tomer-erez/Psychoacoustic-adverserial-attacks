import torch
import shutil
from torch.nn import Parameter
import os
import json
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset
from torch.utils.data import Subset, DataLoader, Dataset
from training_utils.train import perturbation_constraint
import sys
from datasets import load_dataset, Audio
import numpy as np
from core import fourier_transforms
from core.iso import ISO226
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(save_dir: str, log_name: str = "train.log", console: bool = True) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("asr_attack")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if re-called

    # File handler (rotating)
    fh = RotatingFileHandler(os.path.join(save_dir, log_name), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(ch)

    return logger


def make_collate_fn(audio_length):
    def collate_fn(batch):
        waveforms = []
        texts = []
        for waveform, _, transcript, *_ in batch:
            waveform = waveform.squeeze()  # [T]
            waveform = waveform.to(dtype=torch.float32)  # <-- ensure float32
            # Clamp or pad to audio_length
            if waveform.shape[0] > audio_length:
                waveform = waveform[:audio_length]
            else:
                pad_size = audio_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            waveforms.append(waveform)  # [T] of fixed length
            texts.append(transcript) #labels are the transcirpts

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
    Load and prepare DataLoaders for LibriSpeech, CommonVoice, or TEDLIUM.
    Enforces a target dataset size early to optimize memory and speed.

    this chunk is built on the huggingface page of each of the selected speech datasets
    """
    random.seed(args.seed)


    target_size = 30_000

    print(f"loading dataset: {args.dataset}")

    if args.dataset == "LibreeSpeech":
        splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
        all_samples = []
        os.makedirs("librispeech_data", exist_ok=True)
        for split in splits:
            ds = LIBRISPEECH(
                root="librispeech_data",
                url=split,
                folder_in_archive="LibriSpeech",
                download=True
            )
            all_samples += list(ds)

        random.shuffle(all_samples)
        selected = all_samples[:target_size]
        dataset = [(w, sr, t) for (w, sr, t, *_) in selected]

    elif args.dataset == "CommonVoice":
        data_split = "train"  # always use train for size, we'll subselect
        base_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split=data_split, trust_remote_code=True)
        base_dataset = base_dataset.shuffle(seed=args.seed)
        base_dataset = base_dataset.select(range(min(target_size, len(base_dataset))))  # buffer for filtering
        base_dataset = base_dataset.cast_column("audio", Audio(sampling_rate=16_000))

        def to_tuple(example):
            waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)
            sample_rate = example["audio"]["sampling_rate"]
            transcript = example["sentence"]
            return waveform, sample_rate, transcript

        dataset = list(map(to_tuple, base_dataset))

    elif args.dataset == "tedlium":
        data_split = "train"
        base_dataset = load_dataset("sanchit-gandhi/tedlium-data", split=data_split, trust_remote_code=True)
        base_dataset = base_dataset.shuffle(seed=args.seed)
        base_dataset = base_dataset.select(range(min(target_size, len(base_dataset))))  # buffer for filtering
        base_dataset = base_dataset.cast_column("audio", Audio(sampling_rate=16_000))

        def to_tuple(example):
            waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)
            sample_rate = example["audio"]["sampling_rate"]
            transcript = example["text"]
            return waveform, sample_rate, transcript

        dataset = list(map(to_tuple, base_dataset))

    # elif args.dataset == "speech_commands":
    #     data_split = "train"  # or "validation" / "test" for evaluation only
    #     base_dataset = load_dataset("speech_commands", split=data_split)
    #     base_dataset = base_dataset.shuffle(seed=args.seed)
    #     base_dataset = base_dataset.select(range(min(target_size, len(base_dataset))))
    #     base_dataset = base_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    #
    #     def to_tuple(example):
    #         waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)
    #         sample_rate = example["audio"]["sampling_rate"]
    #         transcript = example["label"]
    #         return waveform, sample_rate, transcript
    #
    #     dataset = list(map(to_tuple, base_dataset))

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")



    # Measure lengths
    sample_lengths = [x[0].shape[1] for x in dataset[:min(300, len(dataset))]]
    sample_lengths_tensor = torch.tensor(sample_lengths, dtype=torch.float32)
    min_len = int(sample_lengths_tensor.quantile(0.10).item())
    audio_length = int(sample_lengths_tensor.quantile(args.relative_audio_length).item())

    dataset = [(w, sr, t) for (w, sr, t) in dataset if min_len <= w.shape[1] <= audio_length]
    dataset = dataset[:target_size]

    print(  f"Filtered dataset size: {len(dataset)}"
            f"Kept samples in range: [{min_len}, {audio_length}], "
            f"seconds [{min_len/args.sr:.1f}, {audio_length/args.sr:.1f}]")

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    collate_fn = make_collate_fn(audio_length)

    # Split
    num_train = int(0.8 * len(indices))
    num_eval = int(0.1 * len(indices))

    train_subset = Subset(dataset, indices[:num_train])
    eval_subset = Subset(dataset, indices[num_train:num_train + num_eval])
    test_subset = Subset(dataset, indices[num_train + num_eval:])

    # Loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train size: {len(train_loader)*args.batch_size}, "
          f"Eval size: {len(eval_loader)*args.batch_size}, "
          f"Test size: {len(test_loader)*args.batch_size}, "
          )

    return train_loader, eval_loader, test_loader, audio_length




def load_model(args):
    """
    Load the ASR model
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(args.device)
    return model,processor

def create_logger(args):
    # Pick an attack-size string
    sizes = {
        "min_max_freqs": f"{args.min_freq_attack}",
        "fletcher_munson": f"{args.fm_epsilon}",
        "max_phon": f"{args.max_phon_level}",
        "l2": f"{args.l2_size}",
        "linf": f"{args.linf_size}",
        "snr": f"{args.snr_db}",
        "tv": f"{args.tv_epsilon}",
    }
    if args.norm_type not in sizes:
        raise ValueError(f"Unsupported norm_type: {args.norm_type}")
    args.attack_size_string = sizes[args.norm_type]

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_logs_dir = os.path.join(os.path.dirname(current_dir), "logs")
    args.save_dir = os.path.join(
        all_logs_dir, args.attack_mode, args.dataset,
        f"{args.norm_type}_{args.attack_size_string}_{args.attack_mode}_{args.optimizer_type}"
    )
    args.tensorboard_logger = os.path.join(all_logs_dir, "tensor_board_log_dir")
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup logger
    logger = setup_logging(args.save_dir, log_name="train.log", console=not getattr(args, "silent", False))
    logger.info("========= Args =========")
    logger.info(json.dumps(vars(args), indent=4))
    logger.info("========================")
    logger.info(f"norm_type={args.norm_type} | attack_size={args.attack_size_string}")

    # Resume / checkpoint discovery (non-destructive)
    args.was_preempted = os.path.exists(os.path.join(args.save_dir, "perturbation.pt"))
    args.had_checkpoint = args.was_preempted
    chkpt_epoch = 0

    results_path = os.path.join(args.save_dir, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
                chkpt_epoch = int(results.get("epoch", 0))
        except Exception as e:
            logger.warning(f"Failed to read results.json: {e}")

    if args.was_preempted and not getattr(args, "small_data", False):
        args.resume = True
        args.resume_from = os.path.join(args.save_dir, "perturbation.pt")
        logger.info(f"Resuming from checkpoint: {args.resume_from} (epoch={chkpt_epoch})")
    else:
        args.resume = False

    return logger, chkpt_epoch

def init_perturbation(args, length, spl_thresh, interp, first_batch_data):
    """
    Initialize a universal perturbation p of shape (1, length).
    If resuming, loads from checkpoint and validates length.
    """
    logger = logging.getLogger("asr_attack")
    
    #check if resuming or train from scratch
    ckpt_path = getattr(args, "resume_from", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        logger.info(f"Resuming perturbation from: {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=args.device)
        p = loaded.detach().to(args.device)
    else:
        p = torch.randn(1, length, device=args.device)
        # Project to constraints once at init
        p = perturbation_constraint(p=p, clean_audio=first_batch_data, args=args,
                                    interp=interp, spl_thresh=spl_thresh).detach()

    if args.optimizer_type == "adam":
        p = Parameter(p)
    elif args.optimizer_type == "pgd":
        p.requires_grad_()
        p.retain_grad()
    else:
        raise NotImplementedError(f"Unsupported optimizer_type: {args.optimizer_type}")

    if p.shape[-1] != length:
        raise ValueError(f"Loaded perturbation length {p.shape[-1]} != expected {length}")

    logger.info(
        "Perturbation waveform shape: %s; STFT shape: %s",
        tuple(p.shape), tuple(fourier_transforms.compute_stft(p, args).shape)
    )
    return p



def init_phon_threshold_tensor(args):
    """
    Build a per-frequency SPL threshold (in dB) for a single phon contour,
    aligned to your STFT rFFT bins. Shape: (1, F, 1) for easy broadcasting.
    """
    # rFFT bin frequencies in Hz: [0, sr/n_fft, ..., sr/2], length F = n_fft//2 + 1
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1.0 / args.sr).cpu().numpy()  # (F,)

    # ISO226 is only defined on [20, 20000] Hz. rfftfreq includes 0 Hz (DC),
    #    and sr/2 might exceed 20 kHz for high sample rates.
    #    We clamp to the valid range so the ISO226 call won't raise.
    freqs_clamped = np.clip(freqs, 20.0, 20000.0)

    # Use your ISO226 class to get the SPL (dB) required at each frequency
    #    to be as loud as `max_phon_level` phons.
    iso = ISO226(phon=float(args.max_phon_level))
    spl_thresh_np = iso(freqs_clamped)  # (F,) in dB SPL

    # for bins <20 Hz, we reused the 20 Hz value

    #Torch tensor shaped for broadcasting over (B, F, T) STFTs: (1, F, 1)
    spl_thresh = torch.tensor(spl_thresh_np, dtype=torch.float32, device=args.device)
    spl_thresh = spl_thresh.view(1, -1, 1)
    return spl_thresh



def create_optimizer(args, p):
    """
    Adam optimizer on the perturbation parameter with a StepLR scheduler.
    Note: call scheduler.step() at your chosen cadence (per-epoch is common).
    """
    optimizer = torch.optim.Adam([p], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return optimizer, scheduler
