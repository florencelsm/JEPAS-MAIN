"""Utilities for loading pretrained Dinov2 image model."""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Model


def load_image_model(device: str = "cpu") -> Tuple[Dinov2Model, AutoImageProcessor]:
    """Return a pretrained Dinov2 model and its image processor."""
    model: Dinov2Model = AutoModel.from_pretrained(
        "facebook/dinov2-base", trust_remote_code=False
    ).to(device)
    model.eval()
    processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base"
    )
    return model, processor


def load_image(path: Path, processor: AutoImageProcessor) -> torch.Tensor:
    """Load an image from ``path`` and process it for Dinov2."""
    image = Image.open(path).convert("RGB")
    return processor(images=image, return_tensors="pt").pixel_values


if __name__ == "__main__":
    model, proc = load_image_model()
    tensor = load_image(Path("vgg_ss_processed_data/images"), proc)
    with torch.no_grad():
        out = model(tensor)
    print(out.last_hidden_state.shape)