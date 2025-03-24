print('hello world')
import time
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# Load the model
print("loading wav2vec model...")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
print("done loading")
# Load audio
audio, sr = librosa.load("./sample_wavs/hey_doc.wav", sr=16000)

# Process input
input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to("cpu").detach()

# Get predictions
with torch.no_grad():
    for i in range(10):
        start = time.time()
        print(type(input_values))
        print(input_values.shape)
        logits = model(input_values).logits
        print(type(logits))
        print(logits.shape)
        end=time.time()
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print("Transcription:", transcription)
        print("time:", end-start)
        print('\n\n\n')
# Decode output
