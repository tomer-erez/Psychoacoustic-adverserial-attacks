import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running on device: {device}')
def collate_fn(batch):
    waveforms = []
    texts = []
    for waveform, sample_rate, transcript, *_ in batch:
        waveforms.append(waveform.squeeze())  # [T]
        texts.append(transcript)

    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = torch.zeros(len(waveforms), max_len)  # [B, T]

    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w

    return padded_waveforms, lengths, texts

# === 1. Load dataset
print("Loading dataset...")
LIBRISPEECH_PATH = r"../train-clean-100"
ds = torchaudio.datasets.LIBRISPEECH(
    root=LIBRISPEECH_PATH,
    url="train-clean-100",
    folder_in_archive="LibriSpeech",
    download=False
)

print(f"Dataset size: {len(ds)}")
BATCH_SIZE = 2
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# === 2. Load model
print("Loading Wav2Vec2 model...\n")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# === 3. Load batch
waveforms, lengths, texts = next(iter(loader))  # waveforms: [B, T]
waveforms = waveforms.to(device)
texts = list(texts)
B, T = waveforms.shape

# === 4. Init perturbation: [1, T]
waveforms.requires_grad = False
perturbation = (torch.randn(1, T, device=device) / 150).detach().requires_grad_()
print(f"\nwaveform device: {waveforms.device}\t waveform shape {waveforms.shape}\t requires grad: {waveforms.requires_grad}")
print(f"perturbation device: {perturbation.device} perturbation shape {perturbation.shape} requires grad: {perturbation.requires_grad}")

# === 5. Helper functions
def get_logits(batch_waveforms):
    inputs = processor(
        batch_waveforms.cpu().tolist(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        return model(input_values=inputs.input_values).logits


def get_loss(batch_waveforms, target_texts):
    input_values = processor.feature_extractor(
        batch_waveforms.cpu().tolist(),  # clean list of raw samples
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(device).float()


    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    outputs = model(input_values=input_values, labels=labels)
    return outputs.loss, outputs.logits



def decode(logits):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


def get_loss_for_training(perturbation, original_waveforms, target_texts):
    # Clamp here (still differentiable)
    perturbation = torch.clamp(perturbation, -0.03, 0.03)
    perturbed_waveforms = original_waveforms + perturbation
    # Tokenize labels
    labels = processor(text=target_texts, return_tensors="pt", padding=True).input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Forward through model
    outputs = model(input_values=perturbed_waveforms, labels=labels)
    return outputs.loss, outputs.logits



# === 6. Clean predictions
print("\n=== Clean Audio ===")
loss_clean, logits_clean = get_loss(waveforms, texts)
print(f"Loss (clean): {loss_clean.item():.4f}")

# === 7. Optimize Perturbation
print("\n=== Optimizing Perturbation ===")
optimizer = torch.optim.AdamW([perturbation], lr=4e-3)
model.eval()
for step in range(1, 6):
    optimizer.zero_grad()
    loss, _ = get_loss_for_training(perturbation, waveforms, texts)
    (-loss).backward()
    print(f"Step {step}, loss: {loss.item():.4f}")
    optimizer.step()

    # torch.cuda.empty_cache()

# === 8. Final Predictions
print("\n=== Final Perturbed Audio ===")
final_input = torch.clamp(waveforms + perturbation, -1.0, 1.0).detach()
final_logits = get_logits(final_input)
# for pred, label in zip(decode(final_logits), texts):
#     print(f"→ Predicted: {pred}\n  Ground truth: {label}\n")

# === 10. Save first sample
def save_audio(filename, tensor, sample_rate=16000, amplify=1.0):
    tensor = tensor.detach().cpu() * amplify
    tensor = torch.clamp(tensor, -1.0, 1.0)
    int_tensor = (tensor * 32767).to(torch.int16)
    torchaudio.save(filename, int_tensor.unsqueeze(0), sample_rate, encoding="PCM_S", bits_per_sample=16)

os.makedirs("./trash", exist_ok=True)
save_audio("trash/clean.wav", waveforms[0])
save_audio("trash/perturbed.wav", final_input[0])
save_audio("trash/perturbation.wav", perturbation[0])
