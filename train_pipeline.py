"""Simple training pipeline combining audio and image models with JEPA_base."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm.auto import tqdm


from configs.load_config import load_config
from pretrained.audio_models import (
    MIN_WAVEFORM_LEN,
    load_audio_models,
    load_waveform,
    pad_waveform,
)
from pretrained.image_model import load_image_models
from model.base_model import JEPA_base


class AudioImageDataset(Dataset):


    def __init__(self, root: Path, sample_rate: int = 16_000) -> None:
        self.wave_dir = root / "waveforms"
        self.image_dir = root / "images"
        self.sample_rate = sample_rate
        self.items: List[str] = []
        self.max_len: int = MIN_WAVEFORM_LEN

        for p in self.wave_dir.glob("*.wav"):
            if (self.image_dir / f"{p.stem}.jpg").exists():
                self.items.append(p.stem)
                wf = load_waveform(p, sample_rate=self.sample_rate)
                if wf.numel() > self.max_len:
                    self.max_len = int(wf.numel())

    def __len__(self) -> int: 
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        stem = self.items[idx]
        wave_path = self.wave_dir / f"{stem}.wav"
        waveform = load_waveform(wave_path, sample_rate=self.sample_rate)
        waveform = pad_waveform(waveform, min_len=self.max_len)
        img_path = self.image_dir / f"{stem}.jpg"
        return waveform, str(img_path)

def train(audio_mode: str = "spectrogram") -> None:
    """Train the JEPA model using the specified audio mode."""

    config = load_config()
    cfg = config["audio_image"]
    data_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    runtime_cfg = cfg["runtime"]
    track_cfg = cfg["tracking"]

    torch.manual_seed(exp_cfg.get("SEED", 0))
    torch.set_float32_matmul_precision(
        runtime_cfg.get("FLOAT32_MATMUL_PRECISION", "medium")
    )

    device = runtime_cfg.get("ACCELERATOR", "cpu")

    data_root = Path(data_cfg["DATASET_PATH"])
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parent / data_root).resolve()

    if not ((data_root / "waveforms").is_dir() and (data_root / "images").is_dir()):
        if data_root.name in {"images", "waveforms"}:
            data_root = data_root.parent

    dataset = AudioImageDataset(data_root)
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"No paired waveforms and images found in {data_root}. "
            "Verify DATASET_PATH points to a directory containing matching "
            "'waveforms' and 'images' subfolders."
        )
    
    loader = DataLoader(
        dataset,
        batch_size=exp_cfg["BATCH_SIZE"],
        shuffle=data_cfg.get("SHUFFLE_DATASET", True),
        num_workers=exp_cfg.get("NUM_WORKERS", 0),
        pin_memory=exp_cfg.get("PIN_MEMORY", False),
        persistent_workers=exp_cfg.get("PERSISTENT_WORKERS", False),
        prefetch_factor=exp_cfg.get("PREFETCH_FACTOR", 2),
    )

    which = "wav2vec2" if audio_mode.lower() == "waveform" else "ast"
    audio_models = load_audio_models(device, mode=which)
    audio_model, audio_extractor = audio_models[which]

    vision_models = load_image_models(device)
    vision_model, vision_proc = vision_models["dinov2"]

    jepa = JEPA_base(
        vision_model=vision_model,
        audio_model=audio_model,
        decoder_depth=6,
        num_heads=8,
        device=device,
    )

    optimizer = torch.optim.Adam(jepa.parameters(), lr=exp_cfg["LR"])
    criterion = torch.nn.MSELoss()


    writer = SummaryWriter(track_cfg["LOG_DIR"])
    ckpt_dir = Path(track_cfg["CHECKPOINT_DIR"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(exp_cfg["MAX_EPOCHS"]):
        best_loss = float("inf")
        progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{exp_cfg['MAX_EPOCHS']}", leave=False
        )
        for waveforms, img_paths in progress:
            # Convert batched tensors to a list of lists so the feature extractor
            # treats each waveform independently and pads as needed.
            wave_list = [w.tolist() for w in waveforms]

            audio_inputs = audio_extractor(
                wave_list, sampling_rate=dataset.sample_rate, return_tensors="pt"
            ).input_values.to(device)

            images: List[Image.Image] = [
                Image.open(p).convert("RGB") for p in img_paths
            ]
            image_inputs = vision_proc(
                images=images, return_tensors="pt"
            ).pixel_values.to(device)

            preds, targets = jepa.forward_base(audio=audio_inputs, image=image_inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
            progress.set_postfix(loss=loss.item(), best_loss=best_loss)

        writer.add_scalar("train/best_loss_epoch", best_loss, epoch)
        print(f"Epoch {epoch+1}: best loss {best_loss:.4f}")

        ckpt_path = ckpt_dir / f"jepa_{audio_mode}_epoch{epoch+1}.pt"
        torch.save(jepa.state_dict(), ckpt_path)

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA on audio-image pairs")
    parser.add_argument(
        "--audio-mode",
        choices=["spectrogram", "waveform"],
        default="spectrogram",
        help="Audio representation to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(audio_mode=args.audio_mode)






