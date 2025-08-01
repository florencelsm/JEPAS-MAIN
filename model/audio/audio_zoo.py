# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

import torch
import torchaudio

# To enable Hugging Face fallback, please pip install transformers>=4.45
try:
    from transformers import ASTModel, Wav2Vec2Model
    _HF_READY = True
except ImportError:
    _HF_READY = False

from model.vision.vit import VisionTransformer
from model.patch_embed import PatchEmbed1D


# Base model construction
def create_spec_vit(embed_dim: int = 768) -> VisionTransformer:
    """Default is a 12-layer / 768-dim ViT for 96-patch spectrogram input"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        enc_depth=12,
        num_heads=12,
        post_emb_norm=True,
        post_enc_norm=True,
        layer_dropout=0.1,
    )


def create_wave_1dt(embed_dim: int = 768) -> VisionTransformer:
    """Replace ViT's PatchEmbed with 1D patch-embedding to process waveform sequences"""
    vit = VisionTransformer(img_size=224, patch_size=16)
    vit.patch_embed = PatchEmbed1D()
    vit.num_patches = vit.patch_embed.patch_shape[0]
    vit.pos_embedding = vit.pos_embedding[:, : vit.num_patches, :]
    return vit



# Helper: Load torchaudio / Hugging Face pretrained weights
def _ta_try_ast() -> Optional[dict]:
    """Try to find any AST bundle in torchaudio.pipelines and return state_dict"""
    for name in ("AST_BASE", "AST", "AUDIO_SPECTROGRAM_TRANSFORMER_BASE"):
        bundle = getattr(torchaudio.pipelines, name, None)
        if bundle is None:
            continue
        try:
            return bundle.get_model().state_dict()
        except Exception as e:
            warnings.warn(f"[audio_zoo] torchaudio bundle {name} failed to load: {e}")
    return None


def _hf_try_ast() -> Optional[dict]:
    if not _HF_READY:
        return None
    try:
        model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            trust_remote_code=False,
        )
        return model.state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] Hugging Face AST download failed: {e}")
        return None


def _ta_try_wav2vec() -> Optional[dict]:
    bundle = getattr(torchaudio.pipelines, "WAV2VEC2_BASE", None)
    if bundle is None:
        return None
    try:
        return bundle.get_model().state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] torchaudio WAV2VEC2_BASE failed to load: {e}")
        return None


def _hf_try_wav2vec() -> Optional[dict]:
    if not _HF_READY:
        return None
    try:
        model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            trust_remote_code=False,
        )
        return model.state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] Hugging Face Wav2Vec2 download failed: {e}")
        return None


# Pretrained model construction
def create_spec_vit_pretrained(embed_dim: int = 768,
                               device: torch.device | str | None = None
                               ) -> VisionTransformer:
    """Spectrogram-ViT, try to load AST pretrained weights if possible; return randomly initialized model if failed"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_spec_vit(embed_dim).to("cpu")

    state_dict = _ta_try_ast() or _hf_try_ast()
    if state_dict is None:
        warnings.warn(
            "[audio_zoo] No AST pretrained weights found, Spec-ViT remains randomly initialized"
        )
        return model.to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"[audio_zoo] AST weights missing {len(missing)} parameters")
    if unexpected:
        warnings.warn(f"[audio_zoo] AST weights have {len(unexpected)} unexpected parameters ignored")

    return model.to(device)


def create_wave_1dt_pretrained(embed_dim: int = 768,
                               device: torch.device | str | None = None
                               ) -> VisionTransformer:
    """Wave-1DT, load Wav2Vec2 pretrained weights"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_wave_1dt(embed_dim).to("cpu")

    state_dict = _ta_try_wav2vec() or _hf_try_wav2vec()
    if state_dict is None:
        warnings.warn(
            "[audio_zoo] No Wav2Vec2 pretrained weights found, Wave-1DT remains randomly initialized"
        )
        return model.to(device)

    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"[audio_zoo] Wav2Vec2 weights missing {len(missing)} parameters")
    if unexpected:
        warnings.warn(f"[audio_zoo] Wav2Vec2 weights have {len(unexpected)} unexpected parameters ignored")

    return model.to(device)


# Register to factory dictionary
audio_model_builders: Dict[str, Callable[[], VisionTransformer]] = {
    "spec_vit": create_spec_vit,
    "wave_1dt": create_wave_1dt,
    "spec_vit_pretrain": create_spec_vit_pretrained,
    "wave_1dt_pretrain": create_wave_1dt_pretrained,
}

# Backward compatibility for old names
spec_vit_base = create_spec_vit
Wave1DT = create_wave_1dt