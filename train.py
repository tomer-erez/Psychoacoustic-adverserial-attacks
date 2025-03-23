import torch
import projections
import fourier_transforms

def avg_scores(scores):
    return sum(scores) / len(scores)

def perturbation_constraint(p_tensor, args):
    """
    Projects the perturbation to the allowed constraint set.
    Assumes input is a Tensor (not a Parameter).
    Applies constraint based on the selected norm type.
    """

    # --- Frequency-domain update (used by perceptual norms) ---
    if args.norm_type in ['fletcher_munson', 'leakage', 'freq_l2']:
        # STFT
        stft_p = fourier_transforms.compute_stft(p_tensor, args=args)
        # Perceptual emphasis
        stft_p = projections.apply_fletcher_munson_weighting(stft_p, args)
        # Remove weighting
        stft_p = projections.remove_fletcher_munson_weighting(stft_p, args)
        # ISTFT back to time domain
        p_tensor = fourier_transforms.compute_istft(stft_p, args)

    # --- Time-domain projection constraints ---

    if args.norm_type in ['freq_l2', 'time_l2']:
        p_tensor = projections.project_l2(p_tensor, args.pert_size)
    elif args.norm_type=="l2":
        p_tensor = projections.project_l2(p_tensor, args.pert_size)

    elif args.norm_type == 'linf':
        p_tensor = torch.clamp(p_tensor, -args.pert_size, args.pert_size)

    elif args.norm_type == 'snr':
        # Project to target SNR vs zero signal (pure noise constraint)
        orig_p = torch.zeros_like(p_tensor)
        p_tensor = projections.project_snr(orig_p=orig_p, new_p=p_tensor, snr_db=args.snr_db)

    elif args.norm_type == 'fletcher_munson':
        p_tensor = projections.project_fm_norm(p_tensor, args)

    elif args.norm_type == 'leakage':
        # If not handled above, apply leakage mask now (time-domain or freq-domain inside the function)
        p_tensor = projections.apply_leakage_mask(p_tensor, args)
    return p_tensor


def get_loss_for_training(model, p, original_waveforms, target_texts, processor, args):
    p=perturbation_constraint(p, args)
    perturbed_waveforms = (original_waveforms + p).requires_grad_()
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=perturbed_waveforms, labels=labels)
    return outputs.loss, outputs.logits



def train_epoch(args, train_data_loader, p, model, optimizer, epoch, logger,processor):
    scores = []
    model.eval()
    for batch_idx,(data, target_texts) in enumerate(train_data_loader):
        optimizer.zero_grad()
        data=data.to(args.device)
        data.requires_grad=False
        loss, _ = get_loss_for_training(model=model,p=p, original_waveforms=data,target_texts=target_texts,processor=processor,args=args)
        scores.append(loss.item())
        (-loss).backward()  # maximize loss
        optimizer.step()

        if batch_idx % args.report_interval == 0:
            logger.info(f"batch: {batch_idx}/{len(train_data_loader)},avg_score {avg_scores(scores)}")

    avg_score = avg_scores(scores)
    logger.info(f"Train epoch number: {epoch}, avg score: {avg_score:.4f}")
    return p, scores
