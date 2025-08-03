"""Simple training pipeline combining audio and image models with JEPA_base."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from pretrained.audio_models import load_audio_models
from pretrained.image_model import load_image_model
from model.base_model import JEPA_base


class AudioImageDataset(Dataset):
    """Dataset pairing precomputed ``.npy`` waveforms with image paths.

    Returning image paths rather than loaded ``PIL.Image`` objects avoids
    issues with PyTorch's default collate function, which cannot stack PIL
    images. Images will be loaded in the training loop.
    """

    def __init__(self, root: Path, sample_rate: int = 16000) -> None:
        self.wave_dir = root / "waveforms"
        self.image_dir = root / "images"
        self.sample_rate = sample_rate
        self.items: List[str] = [
            p.stem
            for p in self.wave_dir.glob("*.npy")
            if (self.image_dir / f"{p.stem}.jpg").exists()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        stem = self.items[idx]
        wave_path = self.wave_dir / f"{stem}.npy"
        img_path = self.image_dir / f"{stem}.jpg"
        waveform_np = np.load(wave_path)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        return waveform, str(img_path)


def main() -> None:
    device = "cpu"
    root = Path("vgg_ss_processed_data")
    dataset = AudioImageDataset(root)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    audio_models = load_audio_models(device)
    ast_model, ast_extractor = audio_models["ast"]
    vision_model, vision_proc = load_image_model(device)

    jepa = JEPA_base(
        vision_model=vision_model,
        audio_model=ast_model,
        decoder_depth=6,
        num_heads=8,
        device=device,
    )

    criterion = torch.nn.MSELoss()

    for waveforms, img_paths in loader:
        # ``waveforms`` has shape (B, T); ``img_paths`` is a list of image paths.
        audio_inputs = ast_extractor(
            waveforms.squeeze(0), sampling_rate=16000, return_tensors="pt"
        ).input_values.to(device)
        images: List[Image.Image] = [
            Image.open(p).convert("RGB") for p in img_paths
        ]
        image_inputs = vision_proc(images=images, return_tensors="pt").pixel_values.to(
            device
        )
        preds, targets = jepa.forward_base(audio=audio_inputs, image=image_inputs)
        loss = criterion(preds, targets)
        print("loss", loss.item())


if __name__ == "__main__":
    main()