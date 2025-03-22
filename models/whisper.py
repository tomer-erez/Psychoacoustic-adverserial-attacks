from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load the model & processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Load an audio file
audio, sr = librosa.load("./sample_wavs/hey_doc.wav", sr=16000)

# Process input
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cpu")

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription:", transcription)
