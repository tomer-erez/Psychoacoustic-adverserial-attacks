import torch
import torchaudio
def save_perturbation_as_wav(p,path):
    pass
def save_perturbation_as_pt(p,path):
    pass
def save_perturbation(p,args):
    save_perturbation_as_wav(p,args.save_path)
    save_perturbation_as_pt(p,args.save_path)

def pert_audio_and_save_as_wav():
    pass

def save_audio(filename, tensor, sample_rate=16000, amplify=1.0):
    tensor = tensor.detach().cpu()
    # Optional amplification (e.g. for perturbation)
    tensor = tensor * amplify
    # Clamp to [-1.0, 1.0]
    tensor = torch.clamp(tensor, -1.0, 1.0)
    # Convert to int16 PCM format
    int_tensor = (tensor * 32767).to(torch.int16)
    # Save as WAV
    torchaudio.save(filename, int_tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)
