"""Utilities for loading pretrained Dinov2 image model."""
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Model


def load_image_models(device: str = "cpu", mode: str | None = None) -> Dinov2Model:
    if mode is None or mode.lower() == "dinov2":
        model: Dinov2Model = AutoModel.from_pretrained("facebook/dinov2-base", 
                                                       trust_remote_code=False).to(device)
        model.eval()
    return model