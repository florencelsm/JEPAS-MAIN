from __future__ import annotations
import torch
import numpy as np
import argparse
import random
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from configs.load_config import load_config
from pretrained.audio_models import load_audio_models             
from pretrained.image_model import load_image_models
from model.base_model import JEPA_base
from dataset.VGGSS_dataset_clean import VGGSS_Dataset

def train(audio_mode: str = "spectrogram") -> None:
    config = load_config()
    cfg = config["audio_image"]
    data_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    runtime_cfg = cfg["runtime"]
    track_cfg = cfg["tracking"]

    torch.manual_seed(exp_cfg.get("SEED", 0))
    torch.cuda.manual_seed_all(exp_cfg.get("SEED", 0))
    random.seed(exp_cfg.get("SEED", 0))
    np.random.seed(exp_cfg.get("SEED", 0))
    torch.set_float32_matmul_precision(runtime_cfg.get("FLOAT32_MATMUL_PRECISION", "medium")) 

    device = runtime_cfg.get("ACCELERATOR", "cuda")

    data_root = Path(data_cfg["DATASET_PATH"])
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parent / data_root).resolve()

    if not ((data_root / "waveforms").is_dir() and (data_root / "images").is_dir()):
        if data_root.name in {"images", "waveforms"}:
            data_root = data_root.parent

    dataset = VGGSS_Dataset(audio_mode=audio_mode.lower(),
                            config=data_cfg)
    
    if len(dataset) == 0:
        raise RuntimeError(f"No paired waveforms and images found in {data_root}. "
                           "Verify DATASET_PATH points to a directory containing matching "
                           "'waveforms' and 'images' subfolders.")
    
    loader = DataLoader(dataset,
                        batch_size=exp_cfg["BATCH_SIZE"],
                        shuffle=data_cfg.get("SHUFFLE_DATASET", True),
                        num_workers=exp_cfg.get("NUM_WORKERS", 0),
                        pin_memory=exp_cfg.get("PIN_MEMORY", False),
                        persistent_workers=exp_cfg.get("PERSISTENT_WORKERS", False),
                        prefetch_factor=exp_cfg.get("PREFETCH_FACTOR", 2),
                        ) # ali
    
    audio_model = load_audio_models(audio_mode=audio_mode, device=device)
    vision_model = load_image_models(device=device)

    jepa = JEPA_base(vision_model=vision_model,
                     audio_model=audio_model,
                     decoder_depth=6,
                     num_heads=8,
                     mode = "train",
                     device=device,) # ali
    
    # ali
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=exp_cfg["LR"], weight_decay=exp_cfg["WEIGHT_DECAY"])
    criterion = torch.nn.MSELoss()

    writer = SummaryWriter(track_cfg["LOG_DIR"])
    ckpt_dir = Path(track_cfg["CHECKPOINT_DIR"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(exp_cfg["MAX_EPOCHS"]):
        best_loss = float("inf")
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{exp_cfg['MAX_EPOCHS']}", leave=False)
        for data in progress:
            audio, images = data["waveform"].to(device).squeeze(0), data["image"].to(device).squeeze(0)
            preds, targets = jepa.forward_base(audio=audio, image=images)
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

        if epoch % 10 == 0 and epoch != 0:
            ckpt_path = ckpt_dir / f"jepa_{audio_mode}_epoch{epoch+1}.pt"
            torch.save(jepa.state_dict(), ckpt_path)

    writer.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA on audio-image pairs")
    parser.add_argument("--audio-mode", 
                        choices=["spectrogram", "waveform"],
                        default="spectrogram",
                        help="Audio representation to use",)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(audio_mode=args.audio_mode)