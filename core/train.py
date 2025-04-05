import torch
from core import fourier_transforms,projections
import time
from datetime import datetime

def avg_scores(scores):
    return sum(scores) / len(scores)

def perturbation_constraint(p,clean_audio, args,interp):
    """
    Projects the perturbation to the allowed constraint set.
    Assumes input is a Tensor (not a Parameter).
    Applies constraint based on the selected norm type.
    """
    # --- frequency domain norms:
    if args.norm_type in ["fletcher_munson", "leakage","min_max_freqs"]:
        p = fourier_transforms.compute_stft(p, args=args)#to freq

        if args.norm_type == "leakage":
            p = projections.project_min_max_freqs(args, stft_p=p,
                                                  min_freq=args.min_freq_leakage,
                                                  max_freq=args.max_freq_leakage)
        elif args.norm_type == "min_max_freqs":
            p = projections.project_min_max_freqs(args, stft_p=p,
                                                  min_freq=args.min_freq_attack,
                                                  max_freq=args.max_freq_attack)
        elif args.norm_type == "fletcher_munson":
            p = projections.project_fm_norm(stft_p=p, args=args, interp=interp)

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
    return p


def get_loss_for_training(model, data, target_texts, processor, args):
    if args.attack_mode == "targeted":
        repeated_target = " ".join([args.target] * args.target_reps)
        target_texts = [repeated_target] * len(data)


    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=data, labels=labels)
    return outputs.loss, outputs.logits



def train_epoch(args, train_data_loader, p, model, epoch,processor,optimizer,interp):
    scores = []
    times= []
    model.eval()
    print(f"\n\n{'=' * 50}")
    print(f'timestamp: {datetime.now()}\tstarting epoch: {epoch}')

    for batch_idx,(data, target_texts) in enumerate(train_data_loader):
        a=time.time()
        data=data.to(args.device)
        data.requires_grad=False
        p.requires_grad_()
        data = data + p
        loss, _ = get_loss_for_training(model=model, data=data,target_texts=target_texts,processor=processor,args=args)
        scores.append(loss.item())

        if args.optimize_type == "pgd":
            (loss).backward()  # <-- YES, when doing gradient ascent
            with torch.no_grad():
                if args.attack_mode=="targeted":
                    p = p - args.lr * p.grad.sign()  # gradient descent
                else:
                    p = p + args.lr * p.grad.sign()  # gradient ascent

                p = perturbation_constraint(p=p, clean_audio=data, args=args,interp=interp)
            p = p.detach().requires_grad_()  # reset for next step
        else:
            raise NotImplementedError("not implemented other optimization types")
        b=time.time()
        times.append(b-a)

        if batch_idx % args.report_interval == 0:
            print(f"batch: {batch_idx}/{len(train_data_loader)},\tavg_score {avg_scores(scores)},\tavg batch time {avg_scores(times)}")

    avg_score = avg_scores(scores)
    print(f"Train epoch number: {epoch}, avg score: {avg_score:.4f}")
    # print(f"first train score was {scores[0]:.4f}\t last train score was {scores[-1]:.4f}")
    return p, avg_score
