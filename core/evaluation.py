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
    if args.attack_mode == "targeted":
        repeated_target = " ".join([args.target] * args.target_reps)
        target_texts = [repeated_target] * len(batch_waveforms)


    input_values = batch_waveforms.to(args.device)
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=input_values, labels=labels)
    return outputs.loss, outputs.logits


def decode(logits,processor):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)

def evaluate(args, eval_data_loader, p, model, processor, perturbed=False, epoch_number=-1):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_idx, (data, target_texts) in enumerate(eval_data_loader):
            data = data.to(args.device)
            data.requires_grad = False

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


    avg_score = sum(scores) / len(scores) if scores else float('inf')

    return avg_score

