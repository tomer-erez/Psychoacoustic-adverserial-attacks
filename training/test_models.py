from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Prepare your audio
import torch
import librosa

audio_input, _ = librosa.load("wav_files\jackhammer.wav", sr=16000)
inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)

# Make prediction
with torch.no_grad():
    logits = model(input_values=inputs.input_values).logits

# Decode the logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print(transcription)
