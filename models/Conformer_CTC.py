from transformers import AutoModelForCTC, AutoProcessor
import torch
import librosa

model = AutoModelForCTC.from_pretrained("nvidia/stt_en_conformer_transducer_large").to("cuda")
processor = AutoProcessor.from_pretrained("nvidia/stt_en_conformer_transducer_large")

audio, sr = librosa.load("./sample_wavs/hey_doc.wav", sr=16000)
input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to("cpu").detach()

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print("Transcription:", transcription)
