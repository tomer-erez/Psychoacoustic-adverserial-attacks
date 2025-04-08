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






def train_epoch(args, train_data_loader, p, model, epoch, processor, optimizer, interp, wer_metric,spl_thresh):
    ctc_scores, wer_scores, times = [], [], []
    model.eval()

    print(f"\n\n{'=' * 60}")
    print(f'timestamp: {datetime.now()}\tstarting epoch: {epoch}')

    total_batches = len(train_data_loader)
    report_points = set([int(r * total_batches) for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1]])

    for batch_idx, (data, target_texts) in enumerate(train_data_loader):
        a = time.time()

        data = data.to(args.device)
        data.requires_grad = False
        p.requires_grad_()
        data = data + p

        loss, logits = loss_helpers.get_loss_for_training(model=model, data=data, target_texts=target_texts, processor=processor, args=args)
        ctc_scores.append(loss.item())

        wer = loss_helpers.compute_ctc_and_wer_loss(logits=logits, target_texts=target_texts, processor=processor, wer_metric=wer_metric)
        wer_scores.append(wer)

        if args.optimize_type == "pgd":
            loss.backward()
            with torch.no_grad():
                step = -args.lr if args.attack_mode == "targeted" else args.lr
                p = p + step * p.grad.sign()
                p = perturbation_constraint(p=p, clean_audio=data, args=args, interp=interp,spl_thresh=spl_thresh)

            p = p.detach().requires_grad_()
        else:
            raise NotImplementedError("Optimization type not implemented")

        times.append(time.time() - a)

        if batch_idx in report_points:
            log_helpers.log_train_progress(batch_idx, total_batches, ctc_scores, wer_scores, times)

    return p, avg_scores(ctc_scores), avg_scores(wer_scores)
