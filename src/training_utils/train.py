import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Tuple
import time
import torch

from core import fourier_transforms, projections, loss_helpers
from training_utils import log_helpers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainEpochResult:
    p: torch.Tensor
    avg_ctc: float
    avg_wer: float


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def _align_to(clean_audio: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pad/crop x to match last-dim of clean_audio (post-iSTFT length drift)."""
    target_len = clean_audio.shape[-1]
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] < target_len:
        pad_size = target_len - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad_size))
    return x[..., :target_len]


def _project_frequency_domain(
    p: torch.Tensor,
    args,
    interp,
    spl_thresh,
    clean_audio: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply frequency-domain constraints by projecting STFT(p) and returning iSTFT result.
    """
    # STFT
    stft_p = fourier_transforms.compute_stft(p, args=args)

    if args.norm_type == "min_max_freqs":
        stft_p = projections.project_min_max_freqs(
            args, stft_p=stft_p, min_freq=args.min_freq_attack, max_freq=args.max_freq_attack
        )
    elif args.norm_type == "fletcher_munson":
        stft_p = projections.project_fm_norm(stft_p=stft_p, args=args, interp=interp)
    elif args.norm_type == "max_phon":
        stft_p = projections.project_phon_level(stft_p=stft_p, args=args, spl_thresh=spl_thresh)
    else:
        raise ValueError(f"Unsupported frequency-domain norm_type: {args.norm_type!r}")

    # iSTFT (can change length slightly)
    p_time = fourier_transforms.compute_istft(stft_p, args=args)
    if clean_audio is not None:
        p_time = _align_to(clean_audio, p_time)
    return p_time


def perturbation_constraint(
    p: torch.Tensor,
    clean_audio: torch.Tensor | None,
    args,
    interp,
    spl_thresh,
) -> torch.Tensor:
    """
    Project perturbation p into the feasible set specified by args.norm_type.
    Keeps p as a Tensor (not Parameter). No grads through the projection.
    """
    with torch.no_grad():
        #frequency based norms
        if args.norm_type in ["fletcher_munson", "min_max_freqs", "max_phon"]:
            p = _project_frequency_domain(p, args=args, interp=interp, spl_thresh=spl_thresh, clean_audio=clean_audio)
        #not frequency based norms
        elif args.norm_type == "l2":
            p = projections.project_l2(p, args.l2_size)
        elif args.norm_type == "linf":
            p = projections.project_linf(p, -args.linf_size, args.linf_size)
        elif args.norm_type == "snr":
            if clean_audio is None:
                raise ValueError("SNR projection requires clean_audio ro compare to")
            p = projections.project_snr(clean=clean_audio, perturbation=p, snr_db=args.snr_db)
        elif args.norm_type == "tv":
            if clean_audio is None:
                raise ValueError("TV projection can benefit from clean_audio for bounds")
            p = projections.project_tv(p=p, args=args, clean_audio=clean_audio)
        else:
            raise ValueError(f"Unknown norm_type: {args.norm_type!r}")
    return p



def train_epoch(
    args,train_data_loader,p: torch.Tensor,
    model,epoch: int,processor,interp,
    wer_metric,spl_thresh,optimizer: torch.optim.Optimizer | None,
) -> TrainEpochResult:
    """
    One epoch optimizing a *universal* perturbation `p` over the train loader.

    PGD: sign ascent/descent on `p` with projection.
    Adam: treat `p` as a parameter (we'll modify `p.data` and project).
    """
    #TODO : cleaner calls to the deatach, requires_grad, no grad ...calls
    ctc_scores= []
    wer_scores= []
    times= []

    model.eval()  # ASR model in eval; we only optimize `p`


    logger.info("timestamp: %s | starting epoch: %d", datetime.now(), epoch)

    # Untargeted: increase loss (+1); Targeted: reduce loss (-1)
    direction = +1 if args.attack_mode == "untargeted" else -1

    for clean_audio, target_texts in train_data_loader:
        t0 = time.perf_counter()

        clean_audio = clean_audio.to(args.device, non_blocking=True)
        clean_audio.requires_grad_(False)

        # Prepare `p` for grad
        p.requires_grad_(True)

        # Compose and clamp to audio range ([-1, 1] typical) to match micrhophone's filters/sotware filters
        perturbed = (clean_audio + p).clamp_(-1.0, 1.0)

        # Forward + loss
        loss, logits = loss_helpers.get_loss_for_training(
            model=model,
            data=perturbed,
            target_texts=target_texts,
            processor=processor,
            args=args,
        )
        ctc_scores.append(float(loss.item()))

        # WER measurement (no grad)
        with torch.inference_mode():
            wer = loss_helpers.compute_wer(
                logits=logits, target_texts=target_texts, processor=processor, wer_metric=wer_metric
            )
        wer_scores.append(float(wer))

        # === Optimize p ======================================================
        if args.optimizer_type == "pgd":
            # maximize loss for untargeted; minimize for targeted
            (direction * loss).backward()
            #update without grads - MANUAL UPDATE OF THE GRAD ACCORDING TO THE PGD LOGIC, backwards() call is not needed
            with torch.no_grad(): 
                p.add_(args.lr * p.grad.sign()) #step
                p = perturbation_constraint(p=p, clean_audio=clean_audio, args=args, interp=interp, spl_thresh=spl_thresh) # project to the allowed pert size
            # reset leaf for next batch (clears .grad)
            p = p.detach()

        elif args.optimizer_type == "adam":
            if optimizer is None:
                raise ValueError("Adam optimizer selected but optimizer is None")
            optimizer.zero_grad(set_to_none=True)
            # gradient descent on ( - direction * loss ) ⇒ ascent if untargeted, descent if targeted
            (-1 * direction * loss).backward()
            optimizer.step()
            with torch.no_grad():
                p.data = perturbation_constraint(
                    p=p.data, clean_audio=clean_audio, args=args, interp=interp, spl_thresh=spl_thresh
                )
        else:
            raise NotImplementedError(f"Optimization type not implemented: {args.optimizer_type!r}")
        # =====================================================================

        times.append(time.perf_counter() - t0)

    return TrainEpochResult(p=p, avg_ctc=_avg(ctc_scores), avg_wer=_avg(wer_scores))
