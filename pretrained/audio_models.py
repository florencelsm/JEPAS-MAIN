"""Utilities for loading audio pretrained models and extracting waveforms."""
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import wave

from transformers import ASTModel, Wav2Vec2Model, AutoFeatureExtractor


def load_audio_models(
    device: str = "cpu", mode: str | None = None
) -> Dict[str, Tuple[torch.nn.Module, AutoFeatureExtractor]]:
    """Load pretrained audio models and their feature extractors.

    Parameters
    ----------
    device : str
        Device on which to load the models.
    mode : str | None
        If provided, only the specified model (``"ast"`` or
        ``"wav2vec2"``) is loaded. When ``None`` both models are
        returned.
    """

    models: Dict[str, Tuple[torch.nn.Module, AutoFeatureExtractor]] = {}

    if mode is None or mode.lower() == "ast":
        ast_model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(device)
        
        # ast_model.eval()  # ali
        ast_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        models["ast"] = (ast_model, ast_extractor)

    if mode is None or mode.lower() == "wav2vec2":
        wav_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h", trust_remote_code=False
        ).to(device)
        # wav_model.eval() # ali
        wav_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        models["wav2vec2"] = (wav_model, wav_extractor)

    return models


MIN_WAVEFORM_LEN = 400  # AST's fbank extractor expects at least 400 samples


def pad_waveform(waveform: torch.Tensor, *, min_len: int = MIN_WAVEFORM_LEN) -> torch.Tensor:
    """Zero‑pad ``waveform`` to at least ``min_len`` samples."""

    if waveform.numel() < min_len:
        waveform = F.pad(waveform, (0, int(min_len - waveform.numel())))
    return waveform


def load_waveform(path: Path, *, sample_rate: int = 16_000) -> torch.Tensor:
    """Load a ``.wav`` file as a mono waveform and pad if necessary.
    Parameters
    ----------
    path : Path
        Location of the ``.wav`` file.
    sample_rate : int, optional
        Desired sampling rate. If the file's rate differs it will be
        resampled to ``sample_rate``.
    """

    waveform, sr = torchaudio.load(path)
    # Convert to mono by averaging channels when needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform.squeeze(0)
    waveform = pad_waveform(waveform, min_len=MIN_WAVEFORM_LEN)
    return waveform
