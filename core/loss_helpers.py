import torch

def get_loss_for_training(model, data, target_texts, processor, args):
    if args.attack_mode == "targeted":
        repeated_target = " ".join([args.target] * args.target_reps)
        target_texts = [repeated_target] * len(data)


    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=data, labels=labels)
    return outputs.loss, outputs.logits

def compute_ctc_and_wer_loss(logits, target_texts, processor, wer_metric):
    """Compute decoded predictions and WER from logits and targets."""
    pred_ids = torch.argmax(logits, dim=-1)
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    pred_texts = [p.strip().lower() for p in pred_texts]
    ref_texts = [t.strip().lower() for t in target_texts]

    wer = wer_metric.compute(predictions=pred_texts, references=ref_texts)
    return wer

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