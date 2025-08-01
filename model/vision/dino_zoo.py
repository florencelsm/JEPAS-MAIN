# -*- coding: utf-8 -*-
"""Utilities to load Hugging Face DINO Vision Transformers."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from transformers import ViTModel
    _HF_READY = True
except Exception:  # pragma: no cover - optional dependency
    _HF_READY = False

__all__ = ["load_dino_model", "DINO_PATCH_MODELS"]

DINO_PATCH_MODELS: Dict[str, str] = {
    "small": "facebook/dino-vits16",
    "base": "facebook/dino-vitb16",
    "large": "facebook/dino-vitl16",
}

class HFViTPatchEmbed(nn.Module):
    """Patch embedding wrapper for Hugging Face ViT models."""

    def __init__(self, hf_model: ViTModel) -> None:
        super().__init__()
        self.proj = hf_model.embeddings.patch_embeddings.projection
        patch_size = hf_model.config.patch_size
        img_size = hf_model.config.image_size
        self.patch_shape = (
            img_size // patch_size,
            img_size // patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def load_dino_model(size: str = "base") -> Dict[str, nn.Module]:
    """Load a pretrained DINO ViT model from Hugging Face."""
    if not _HF_READY:
        raise ImportError(
            "transformers is required to load DINO models; please install it"
        )

    if size not in DINO_PATCH_MODELS:
        raise ValueError(f"Unknown DINO size '{size}'")

    model_name = DINO_PATCH_MODELS[size]
    vit = ViTModel.from_pretrained(model_name)
    vit.eval()

    patch_embed = HFViTPatchEmbed(vit)
    pos_embedding = vit.embeddings.position_embeddings
    encoder = vit.encoder

    for p in vit.parameters():
        p.requires_grad = False

    return {
        "patch_embed": patch_embed,
        "pos_embedding": pos_embedding,
        "encoder": encoder,
    }