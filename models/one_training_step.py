import os
import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ================================
# === 1. Download and save dataset
# ================================
print("Loading and caching dataset...")


ds = load_dataset(
    "librispeech_asr",
    "clean",  # This is the required config name
    split="train.100[:1%]",
    trust_remote_code=True
)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))  # force 16kHz
print(type(ds))
print(len(ds))
print(ds[0].shape)
sample = ds[0]

# ==========================
# === 2. Load pretrained model
# ==========================
print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()  # switch to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==========================
# === 3. Load waveform
# ==========================
waveform = torch.tensor(sample["audio"]["array"]).to(device)
if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)  # [1, T]
text = sample["text"]
print(f"Ground truth text: {text}")

# ================================
# === 4. Initialize perturbation
# ================================
waveform.requires_grad = False
perturbation = (torch.randn_like(waveform) / 25).detach().requires_grad_()
print(f'perturbation requires grad : {perturbation.requires_grad}')

# ================================
# === 5. Define helper functions
# ================================
def get_logits(wf):
    if isinstance(wf, torch.Tensor):
        wf = wf.squeeze().detach().cpu().numpy()
    inputs = processor(wf, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits
    return logits



def get_loss(wf, target_text):
    if isinstance(wf, torch.Tensor):
        wf = wf.squeeze().detach().cpu().numpy()
    inputs = processor(wf, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

    # Tokenize labels
    labels = processor(text=target_text, return_tensors="pt", padding=True).input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100  # for CTC loss

    # Forward pass with loss
    outputs = model(input_values=inputs.input_values, labels=labels)
    return outputs.loss, outputs.logits


def decode(logits):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def get_loss_for_training(perturbation, original_waveform, target_text):
    if original_waveform.ndim == 2:
        original_waveform = original_waveform.squeeze(0)

    # Process waveform to get input values (not differentiable, so use torch.no_grad)
    with torch.no_grad():
        inputs = processor(original_waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    input_values = inputs.input_values.to(device).float()
    input_values.requires_grad = True
    perturbed_input = input_values + perturbation.to(device).float()

    # Get labels
    labels = processor(text=[target_text], return_tensors="pt", padding=True).input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Compute loss
    outputs = model(input_values=perturbed_input, labels=labels)
    return outputs.loss, outputs.logits


# ====================================
# === 6. Evaluate on clean audio
# ====================================
print("\n=== Clean Audio ===")
loss_clean, logits_clean = get_loss(waveform, text)
print(f"Loss (clean): {loss_clean.item():.4f}")
print("Prediction:", decode(logits_clean))

# ====================================
# === 7. Evaluate on initial perturbation
# ====================================
print("\n=== Initial Perturbed Audio ===")
noisy_input = torch.clamp(waveform + perturbation, -1.0, 1.0)
loss_perturbed, logits_perturbed = get_loss(noisy_input, text)
print(f"Loss (initial perturbed): {loss_perturbed.item():.4f}")
print("Prediction:", decode(logits_perturbed))

# ====================================
# === 8. Backpropagate to perturbation
# ====================================
print("\n=== Optimizing Perturbation ===")
optimizer = torch.optim.SGD([perturbation], lr=7e-3)

for step in range(1, 6):
    optimizer.zero_grad()
    clamped_perturb = torch.clamp(perturbation, -0.95, 0.95)
    loss, _ = get_loss_for_training(clamped_perturb, waveform, text)
    (-loss).backward()  # maximize loss
    if perturbation.grad is not None:
        print(f"Step {step}, loss: {loss.item():.4f}, grad norm: {perturbation.grad.norm().item():.4f}")
    else:
        print(f"Step {step}, loss: {loss.item():.4f}, grad is None!")
    optimizer.step()

# ====================================
# === 9. Re-evaluate after optimization
# ====================================
print("\n=== Final Perturbed Audio ===")
final_input = torch.clamp(waveform + perturbation, -1.0, 1.0).detach()
final_logits = get_logits(final_input)
print("Prediction:", decode(final_logits))
print(f'final perturbation:\n'
      f'shape: {perturbation.shape}\n'
      f'mean: {perturbation.mean().item():.4f}\n'
      f'std: {perturbation.std().item():.4f}\n')


def save_audio(filename, tensor, sample_rate=16000, amplify=1.0):
    # Make sure it's CPU and float32
    tensor = tensor.detach().cpu()

    # Optional amplification (e.g. for perturbation)
    tensor = tensor * amplify

    # Clamp to [-1.0, 1.0]
    tensor = torch.clamp(tensor, -1.0, 1.0)

    # Convert to int16 PCM format
    int_tensor = (tensor * 32767).to(torch.int16)

    # Save as WAV
    torchaudio.save(filename, int_tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)

os.makedirs("./trash", exist_ok=True)
save_audio("trash/clean.wav", waveform)
save_audio("trash/perturbed.wav", final_input)
save_audio("trash/perturbation.wav", perturbation)  # Amplified so it's audible

