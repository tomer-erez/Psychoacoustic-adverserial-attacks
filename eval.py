import torch


# === 5. Helper functions
def get_logits(batch_waveforms,processor,args,model):
    inputs = processor(
        batch_waveforms.cpu().tolist(),
        sampling_rate=args.sr,
        return_tensors="pt",
        padding=True
    ).to(args.device)

    with torch.no_grad():
        return model(input_values=inputs.input_values).logits


def get_loss(batch_waveforms, target_texts, processor, args, model):
    input_values = batch_waveforms.to(args.device)
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=input_values, labels=labels)
    return outputs.loss, outputs.logits


def decode(logits,processor):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)

def evaluate(args, eval_data_loader, p, model, processor, logger, perturbed=False, epoch_number=-1):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_idx, (data, target_texts) in enumerate(eval_data_loader):
            data = data.to(args.device)
            data.requires_grad = False
            # if p.shape[-1]!=data.shape[-1]:
            #     raise ValueError(f"data shape {data.shape} does not match p shape {p.shape}")
            if perturbed:
                data = data + p

            loss, logits = get_loss(
                batch_waveforms=data,
                target_texts=target_texts,
                processor=processor,
                args=args,
                model=model
            )

            scores.append(loss.item())

            # # === Print predictions for first batch only
            # if epoch_number==-1 and batch_idx == 0:
            #     predictions = decode(logits, processor)
            #     for pred, gt in zip(predictions, target_texts):
            #         logger.info(f"→ Predicted: {pred}\n  Ground truth: {gt}\n")


    avg_score = sum(scores) / len(scores) if scores else float('inf')

    if epoch_number == -1:
        msg = f"[TEST set - {'PERTURBED' if perturbed else 'CLEAN'}] Avg CTC Loss: {avg_score:.4f}\n"
    else:
        msg = f"[EVAL set - Epoch {epoch_number}] Avg CTC Loss: {avg_score:.4f}\n"

    logger.info(msg)
    return avg_score

