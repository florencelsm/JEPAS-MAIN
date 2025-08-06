from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ASTModel, Wav2Vec2Model, AutoFeatureExtractor

def load_audio_models(mode: str, device: str = "cpu") -> torch.nn.Module:
    if mode.lower() == "ast":
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
    if mode.lower() == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", trust_remote_code=False).to(device)
    return model

def pad_waveform(waveform: torch.Tensor, min_waveform_len: int = 400) -> torch.Tensor:
    """Zero-pad ``waveform`` to at least ``min_len`` samples."""

    if waveform.numel() < min_waveform_len:
        waveform = F.pad(waveform, (0, int(min_waveform_len - waveform.numel())))
    return waveform

def load_waveform(path: Path, sample_rate: int = 16000, min_waveform_len: int = 400) -> torch.Tensor:
    """Load a ``.wav`` file as a mono waveform and pad if necessary.
    Parameters
    ----------
    path : Path
        Location of the ``.wav`` file.
    sample_rate : int, optional
        Desired sampling rate. If the file's rate differs it will be
        resampled to ``sample_rate``.
    """

    waveform, sr = torchaudio.load(str(path))
    # Convert to mono by averaging channels when needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform.squeeze(0)
    waveform = pad_waveform(waveform, min_waveform_len=min_waveform_len)
    return waveform
