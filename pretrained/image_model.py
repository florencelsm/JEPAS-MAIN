"""Utilities for loading pretrained Dinov2 image model."""
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Model


def load_image_models(
    device: str = "cpu", mode: str | None = None
) -> Dict[str, Tuple[Dinov2Model, AutoImageProcessor]]:
    """Load pretrained image model(s) and their processors.

    Parameters
    ----------
    device : str
        Device on which to load the model.
    mode : str | None
        If provided, only the specified model (``"dinov2"``) is loaded.
        When ``None`` all available models are returned.
    """

    models: Dict[str, Tuple[Dinov2Model, AutoImageProcessor]] = {}

    if mode is None or mode.lower() == "dinov2":
        model: Dinov2Model = AutoModel.from_pretrained(
            "facebook/dinov2-base", trust_remote_code=False
        ).to(device)
        model.eval()
        processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        models["dinov2"] = (model, processor)

    return models


def load_image(path: Path, processor: AutoImageProcessor) -> torch.Tensor:
    """Load an image from ``path`` and process it for Dinov2."""
    image = Image.open(path).convert("RGB")
    return processor(images=image, return_tensors="pt").pixel_values
