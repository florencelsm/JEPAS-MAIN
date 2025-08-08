import torch
from transformers import ASTModel, Wav2Vec2Model

def load_audio_models(audio_mode: str, device: str = "cpu") -> torch.nn.Module:
    if audio_mode.lower() == "spectrogram":
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    elif audio_mode.lower() == "waveform":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", trust_remote_code=False)
    else:
        raise ValueError("Audio Model is non implemented.")
    return model.to(device)