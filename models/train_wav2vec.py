from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from datasets import load_dataset
import torchaudio

# Load dataset
print('loading dataset')
# ds = load_dataset("librispeech_asr", "clean", split="train.100[:5]", trust_remote_code=True)
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy","clean", split="validation", trust_remote_code=True)
print(ds[0].keys())
print(f"Number of samples loaded: {len(ds)}")

# Pretrained model + processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.train()

# Resample audio to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

def prepare_sample(batch):
    audio = batch["audio"]
    waveform = torch.tensor(audio["array"])
    if audio["sampling_rate"] != 16000:
        waveform = resampler(waveform)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = processor(text=batch["text"], return_tensors="pt", padding=True).input_ids

    inputs["labels"] = labels
    return inputs

# Example training step
sample = ds[0]
inputs = prepare_sample(sample)
input_values = inputs["input_values"].squeeze(0)
labels = inputs["labels"].squeeze(0)

outputs = model(input_values=input_values.unsqueeze(0), labels=labels.unsqueeze(0))
loss = outputs.loss
logits = outputs.logits

# Decode prediction
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

# Decode ground truth labels
true_labels = labels[labels != -100]
true_transcription = processor.batch_decode(true_labels.unsqueeze(0))

# Print everything
print("=" * 60)
print(f"True transcription:\n{true_transcription[0]}")
print("-" * 60)
print(f"Predicted transcription:\n{transcription[0]}")
print("-" * 60)
print(f"Loss: {loss.item():.4f}")
print("=" * 60)

