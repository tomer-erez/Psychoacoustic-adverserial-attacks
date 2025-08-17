###all funcs from hugging face page for interfacing the asr models wav2vec2 etc


import torch
import re

def clean_transcripts(texts):
    """Removes <unk> and normalizes whitespace from a list of transcripts."""
    return [re.sub(r'\s+', ' ', t.replace("<unk>", "").lower()).strip() for t in texts]


def get_loss_for_training(model, data, target_texts, processor, args):
    if args.attack_mode == "targeted":
        repeated_target = " ".join([args.target] * args.target_reps)
        target_texts = [repeated_target] * len(data)
    # print('tt before cleaning:\n', target_texts)
    target_texts = clean_transcripts(target_texts)
    # print('tt after cleaning:\n', target_texts)
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=data, labels=labels)
    # print("loss: ", outputs.loss.item())
    return outputs.loss, outputs.logits

def compute_wer(logits, target_texts, processor, wer_metric):
    pred_ids = torch.argmax(logits, dim=-1)
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    pred_texts = [p.strip().lower() for p in pred_texts]
    ref_texts = clean_transcripts(target_texts)
    ref_texts = [t.lower() for t in ref_texts]
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

    target_texts = clean_transcripts(target_texts)

    input_values = batch_waveforms.to(args.device)
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    outputs = model(input_values=input_values, labels=labels)
    return outputs.loss, outputs.logits


def decode(logits,processor):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)