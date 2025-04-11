import torch
import torch.optim as optim
import shutil
from torch.nn import Parameter
import os
import json
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH  # Example ASR dataset
from torch.utils.data import Subset, DataLoader
from training_utils.train import perturbation_constraint
import sys
from datasets import load_dataset, Audio
from core import elc,fourier_transforms

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

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
    Load and prepare DataLoaders for LibriSpeech, CommonVoice, or TEDLIUM.
    Enforces a target dataset size early to optimize memory and speed.
    """
    random.seed(args.seed)
    flush()

    target_size = 24 if args.small_data else 30_000

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

    flush()

    # Optional truncation again (after filtering)
    if args.small_data:
        args.num_items_to_inspect = 1

    # Measure lengths
    sample_lengths = [x[0].shape[1] for x in dataset[:min(300, len(dataset))]]
    sample_lengths_tensor = torch.tensor(sample_lengths, dtype=torch.float32)
    min_len = int(sample_lengths_tensor.quantile(0.10).item())
    audio_length = int(sample_lengths_tensor.quantile(args.relative_audio_length).item())

    dataset = [(w, sr, t) for (w, sr, t) in dataset if min_len <= w.shape[1] <= audio_length]
    dataset = dataset[:target_size]

    print(f"Filtered dataset size: {len(dataset)}")
    print(f"Kept samples in range: [{min_len}, {audio_length}], "
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


def get_model_size_gb(model):
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_gb = (param_size_bytes + buffer_size_bytes) / (1024 ** 3)
    return total_size_gb

def load_model(args):
    """
    Load the ASR model
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(args.device)
    print(f"Model size: {get_model_size_gb(model):.2f} GB")
    return model,processor

def create_logger(args):
    """
    Creates a logger that writes to the console if `small_data` is True,
    or to a log file otherwise.
    """
    # Determine attack size string

    if args.norm_type == "min_max_freqs":
        size = f'{args.min_freq_attack}'
    elif args.norm_type == "fletcher_munson":
        size = f'{args.fm_epsilon}'
    elif args.norm_type == "max_phon":
        size = f'{args.max_phon_level}'
    elif args.norm_type == "l2":
        size = f"{args.l2_size}"
    elif args.norm_type == "linf":
        size = f"{args.linf_size}"
    elif args.norm_type == "snr":
        size = f"{args.snr_db}"
    elif args.norm_type == "tv":
        size = f"{args.tv_epsilon}"
    else:
        raise ValueError(f"Unsupported norm_type: {args.norm_type}")
    args.attack_size_string = size


    # Set save_dir and log_file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_logs_dir = os.path.join(os.path.dirname(current_dir), "logs")
    args.save_dir = os.path.join(all_logs_dir, args.attack_mode,args.dataset,f"{args.norm_type}_{args.attack_size_string}_{args.attack_mode}_{args.optimizer_type}")
    args.tensorboard_logger = os.path.join(all_logs_dir, 'tensor_board_log_dir')
    os.makedirs(args.save_dir, exist_ok=True)
    args.was_preempted=False
    args.had_checkpoint=False

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    args.log_file = os.path.join(args.save_dir, 'train_log.txt')

    chkpt_epoch = 0

    if os.path.exists(args.save_dir):
        print('was preempted or save directory already existed')
        args.was_preempted = True
        perturbation_path = os.path.join(args.save_dir, "perturbation.pt")

        if (os.path.exists(perturbation_path)) and (not args.small_data):
            args.had_checkpoint = True
            args.resume = True
            args.resume_from = perturbation_path
            x = f"this job was preempted and has a checkpoint in {args.resume_from}\n"
            with open(args.log_file, 'w') as f:
                f.write(x)
            results_path = os.path.join(args.save_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        chkpt_epoch = int(results.get("epoch", 0))
                except Exception as e:
                    print(f"Failed to read results.json: {e}")

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
    if chkpt_epoch>0:
        print(f"resuming checkpoint from epoch: {chkpt_epoch}")

    sys.stdout.flush()
    sys.stderr.flush()
    return chkpt_epoch

def init_perturbation(args,length,spl_thresh,interp,first_batch_data):
    """
    Initializes a universal perturbation `p` with the same shape as input audio.
    Assumes args.dim = (num_channels, num_samples) or similar.
    """
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        print(f"Resuming perturbation from checkpoint: {args.resume_from}")
        if args.optimizer_type=="pgd":
            loaded = torch.load(args.resume_from, map_location=args.device)
            p = loaded.detach().to(args.device).requires_grad_()
        elif args.optimizer_type=="adam":
            loaded = torch.load(args.resume_from, map_location=args.device)
            p = Parameter(loaded.to(args.device))
        else:
            raise NotImplementedError(f"Unsupported optimizer_type: {args.optimizer_type}")
        if p.shape[-1] != length:
            raise ValueError(f"Loaded perturbation length {p.shape[-1]} does not match expected length {length}")

    else:
        p = torch.randn(1, length, device=args.device)
        p = perturbation_constraint(p=p, clean_audio=first_batch_data, args=args, interp=interp,
                                    spl_thresh=spl_thresh).detach().requires_grad_()
        if args.optimizer_type=="pgd":
            p = p.detach().requires_grad_()
            p.retain_grad()
        elif args.optimizer_type=="adam":
            p = Parameter(p)
        else:
            raise NotImplementedError(f"Unsupported optimizer_type: {args.optimizer_type}")

    print(f"the waveform shape of the perturbation is :{p.shape}, as in (1,sr*time)\n"
          f"the stft shape of the perturbation is: {fourier_transforms.compute_stft(p, args).shape}, as in (1,frequency bins,time frames)\n")
    return p


def init_phon_threshold_tensor(args):
    freqs = torch.fft.rfftfreq(n=args.n_fft, d=1 / args.sr).cpu().numpy()
    spl_thresh_np = elc.elc(args.max_phon_level, freqs)
    spl_thresh = torch.tensor(spl_thresh_np, dtype=torch.float32, device=args.device)
    spl_thresh = spl_thresh.view(1, -1, 1)
    return spl_thresh



def create_optimizer(args,p):
    o,s=None,None
    if args.optimizer_type=="adam":
        o = torch.optim.Adam([p], lr=args.lr)
        s = torch.optim.lr_scheduler.StepLR(o, step_size=args.step_size, gamma=args.gamma)
    return o,s