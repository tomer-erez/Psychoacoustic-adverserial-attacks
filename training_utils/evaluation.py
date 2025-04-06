import torch
from core import loss_helpers


def evaluate(args, eval_data_loader, p, model, processor, wer_metric, perturbed=False, epoch_number=-1):
    model.eval()
    ctc_scores, wer_scores = [], []

    with torch.no_grad():
        for data, target_texts in eval_data_loader:
            data = data.to(args.device)
            data.requires_grad = False

            if perturbed:
                data = data + p

            loss, logits = loss_helpers.get_loss(
                batch_waveforms=data,
                target_texts=target_texts,
                processor=processor,
                args=args,
                model=model
            )
            ctc_scores.append(loss.item())
            wer = loss_helpers.compute_ctc_and_wer_loss(logits, target_texts, processor, wer_metric)
            wer_scores.append(wer)
    avg_ctc = sum(ctc_scores) / len(ctc_scores) if ctc_scores else float('inf')
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else float('inf')

    return avg_ctc, avg_wer

