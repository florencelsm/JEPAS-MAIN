"""Utilities for loading audio pretrained models and extracting waveforms."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers import ASTModel, Wav2Vec2Model, AutoFeatureExtractor


def load_audio_models(device: str = "cpu") -> Dict[str, Tuple[torch.nn.Module, AutoFeatureExtractor]]:
    """Load pretrained AST and Wav2Vec2 models along with their feature extractors."""
    ast_model = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(device)
    ast_model.eval()
    ast_extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    wav_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h", trust_remote_code=False
    ).to(device)
    wav_model.eval()
    wav_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    return {
        "ast": (ast_model, ast_extractor),
        "wav2vec2": (wav_model, wav_extractor),
    }

MIN_WAVEFORM_LEN = 400  # AST's fbank extractor expects at least 400 samples


def pad_waveform(waveform: torch.Tensor, *, min_len: int = MIN_WAVEFORM_LEN) -> torch.Tensor:
    """Zero‑pad ``waveform`` to at least ``min_len`` samples."""

    if waveform.numel() < min_len:
        waveform = F.pad(waveform, (0, int(min_len - waveform.numel())))
    return waveform


def load_waveform(path: Path, *, sample_rate: int = 16_000) -> torch.Tensor:
    """Load a mono waveform from ``path`` and ensure a minimum length."""

    waveform_np = np.load(path)
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    if waveform.numel() < MIN_WAVEFORM_LEN:
        pad = MIN_WAVEFORM_LEN - waveform.numel()
        waveform = F.pad(waveform, (0, pad))

    return waveform


if __name__ == "__main__":
    models = load_audio_models()
    dummy = load_waveform(Path("vgg_ss_processed_data/waveforms/16CvcIXIjzQ_000332.npy"))
    ast_model, ast_extractor = models["ast"]
    inputs = ast_extractor(dummy, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = ast_model(**inputs)
    print(outputs.last_hidden_state.shape)