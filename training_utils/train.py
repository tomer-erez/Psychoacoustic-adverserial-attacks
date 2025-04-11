import torch
from core import fourier_transforms,projections,loss_helpers
from training_utils import log_helpers
import time
from datetime import datetime


def avg_scores(scores):
    return sum(scores) / len(scores)


def perturbation_constraint(p,clean_audio, args,interp,spl_thresh):
    """
    Projects the perturbation to the allowed constraint set.
    Assumes input is a Tensor (not a Parameter).
    Applies constraint based on the selected norm type.
    """
    # --- frequency domain norms:
    if args.norm_type in ["fletcher_munson" ,"min_max_freqs","max_phon"]:
        p = fourier_transforms.compute_stft(p, args=args)#to freq

        if args.norm_type == "min_max_freqs":
            p = projections.project_min_max_freqs(args, stft_p=p,min_freq=args.min_freq_attack,max_freq=args.max_freq_attack)
        elif args.norm_type == "fletcher_munson":
            p = projections.project_fm_norm(stft_p=p, args=args, interp=interp)
        elif args.norm_type == "max_phon":
            p =projections.project_phon_level(stft_p=p, args=args, spl_thresh=spl_thresh)
        p= fourier_transforms.compute_istft(p, args=args)

        if clean_audio is not None:
            if p.shape[-1] < clean_audio.shape[-1]:
                # Pad to match
                pad_amt = clean_audio.shape[-1] - p.shape[-1]
                p = torch.nn.functional.pad(p, (0, pad_amt))
            elif p.shape[-1] > clean_audio.shape[-1]:
                # Crop to match
                p = p[..., :clean_audio.shape[-1]]
    # time domain norms
    else:
        if args.norm_type == "l2":
            p = projections.project_l2(p, args.l2_size)
        elif args.norm_type=="linf":
            p = projections.project_linf(p, -args.linf_size, args.linf_size)
        elif args.norm_type=="snr":
            p = projections.project_snr(clean=clean_audio, perturbation=p, snr_db=args.snr_db)
        elif args.norm_type=="tv":
            p=projections.project_tv(p=p,args=args,clean_audio=clean_audio)
    return p






def train_epoch(args, train_data_loader, p, model, epoch, processor, interp, wer_metric,spl_thresh,optimizer):
    ctc_scores, wer_scores, times = [], [], []
    model.eval()

    print(f"\n\n{'=' * 60}")
    print(f'timestamp: {datetime.now()}\tstarting epoch: {epoch}')

    total_batches = len(train_data_loader)
    report_points = set([int(r * total_batches) for r in [0.0, 0.25, 0.50, 0.75, 1]])
    direction = +1 if args.attack_mode == "untargeted" else -1
    #untargeted-> increase loss -> positive (+1)
    #targeted-> reduce loss -> negative (-1)

    for batch_idx, (clean_audio, target_texts) in enumerate(train_data_loader):
        a = time.time()
        # Prepare audio + perturbation
        clean_audio = clean_audio.to(args.device)

        clean_audio.requires_grad = False
        p.requires_grad_()
        perturbed_data = clean_audio + p

        # Forward pass + compute loss
        loss, logits = loss_helpers.get_loss_for_training(model=model, data=perturbed_data, target_texts=target_texts, processor=processor, args=args)
        ctc_scores.append(loss.item())

        #compute wer
        wer = loss_helpers.compute_wer(logits=logits, target_texts=target_texts, processor=processor, wer_metric=wer_metric)
        wer_scores.append(wer)

        # untargeted-> increase loss -> positive (+1)
        # targeted-> reduce loss -> negative (-1)
        if args.optimizer_type == "pgd":
            (direction * loss).backward()
            with torch.no_grad():
                p = p +  args.lr * p.grad.sign()
                p = perturbation_constraint(p=p, clean_audio=clean_audio, args=args, interp=interp,spl_thresh=spl_thresh)
            p = p.detach().requires_grad_()

        elif args.optimizer_type == "adam":
            optimizer.zero_grad()
            (-1 * direction * loss).backward()
            optimizer.step()
            with torch.no_grad():
                p.data = perturbation_constraint(p=p.data, clean_audio=clean_audio, args=args, interp=interp, spl_thresh=spl_thresh)
        else:
            raise NotImplementedError("Optimization type not implemented")

        times.append(time.time() - a)
        if batch_idx in report_points:
            log_helpers.log_train_progress(batch_idx, total_batches, ctc_scores, wer_scores, times)

    return p, avg_scores(ctc_scores), avg_scores(wer_scores)
