"""Utilities for loading audio pretrained models and extracting waveforms."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

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


def load_waveform(path: Path) -> torch.Tensor:
    """Load a mono waveform from a ``.npy`` file.

    The dataset provides pre-extracted waveforms under ``waveforms`` so we simply
    load the array and ensure the result is a 1D float tensor.
    """

    waveform_np = np.load(path)
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    return waveform


if __name__ == "__main__":
    models = load_audio_models()
    dummy = load_waveform(Path("vgg_ss_processed_data/waveforms/16CvcIXIjzQ_000332.npy"))
    ast_model, ast_extractor = models["ast"]
    inputs = ast_extractor(dummy, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = ast_model(**inputs)
    print(outputs.last_hidden_state.shape)