import torch
import projections
import fourier_transforms


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
        p_tensor = projections.project_l2(p_tensor, args.epsilon)

    elif args.norm_type == 'linf':
        p_tensor = torch.clamp(p_tensor, -args.epsilon, args.epsilon)

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


def train_epoch(args, train_data_loader, p, model, optimizer, criterion, epoch, logger):
    scores = []
    model.train()

    for data, labels in train_data_loader:
        optimizer.zero_grad()

        x_pert = data + p
        y_pred = model(x_pert)

        loss = criterion(y_pred, labels)
        scores.append(loss.item())

        loss.backward()
        optimizer.step()

        # Apply constraint to p in-place
        p.data = perturbation_constraint(p.data, args)

    avg_score = sum(scores) / len(scores) if scores else float('inf')
    logger.info(f"Train epoch number: {epoch}, avg score: {avg_score:.4f}")
    return p, scores
