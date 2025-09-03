from __future__ import annotations
import torch
import numpy as np
import argparse
import random
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from configs.load_config import load_config
from pretrained.audio_models import load_audio_models             
from pretrained.image_model import load_image_models
from model.vsl import VisualSoundLocalizer
from dataset.VGGSS_dataset_clean import VGGSS_Dataset
from losses.hungarian_matcher_loss import MatchingLoss
from utils.bbox_utils import plot_bbox, compute_metrics

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

    data_root = Path(data_cfg["TRAIN_DATASET_PATH"])
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parent / data_root).resolve()

    if not ((data_root / "waveforms").is_dir() and (data_root / "images").is_dir()):
        if data_root.name in {"images", "waveforms"}:
            data_root = data_root.parent

    train_dataset = VGGSS_Dataset(audio_mode=audio_mode.lower(),
                                  config=data_cfg,
                                  mode='train')
    test_dataset = VGGSS_Dataset(audio_mode=audio_mode.lower(),
                                 config=data_cfg,
                                 mode='test')
    
    if len(train_dataset) == 0:
        raise RuntimeError(f"No paired waveforms and images found in {data_root}. "
                           "Verify TRAIN_DATASET_PATH points to a directory containing matching "
                           "'waveforms' and 'images' subfolders.")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=exp_cfg["BATCH_SIZE"],
                              shuffle=data_cfg.get("SHUFFLE_DATASET", True),
                              num_workers=exp_cfg.get("NUM_WORKERS", 0),
                              pin_memory=exp_cfg.get("PIN_MEMORY", False),
                              persistent_workers=exp_cfg.get("PERSISTENT_WORKERS", False),
                              prefetch_factor=exp_cfg.get("PREFETCH_FACTOR", 2),)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=exp_cfg.get("NUM_WORKERS", 0),
                             pin_memory=exp_cfg.get("PIN_MEMORY", False),
                             persistent_workers=exp_cfg.get("PERSISTENT_WORKERS", False),
                             prefetch_factor=exp_cfg.get("PREFETCH_FACTOR", 2),)
    
    audio_model = load_audio_models(audio_mode=audio_mode, device=device)
    vision_model = load_image_models(device=device)

    state_dict = torch.load("/home/ec2-user/vggss/jepa_spectrogram_epoch121.pt", map_location="cpu", weights_only=True)
    audio_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("audio_encoder."):
            audio_state_dict[k.replace("audio_encoder.", "", 1)] = v
    audio_model.load_state_dict(audio_state_dict, strict=True)
    print("LOADED AUDIO ENCODER CHECKPOINT")

    ISL = VisualSoundLocalizer(image_encoder=vision_model,
                               audio_encoder=audio_model,
                               freeze_backbones=False).cuda()
    
    # optimizer = torch.optim.AdamW(ISL.parameters(), lr=exp_cfg["LR"], weight_decay=exp_cfg["WEIGHT_DECAY"])
    DINO_param = ISL.image_encoder.parameters()
    AUDIO_param = ISL.audio_encoder.parameters()
    ISL_param = [p for p in ISL.parameters() if p not in DINO_param and p not in AUDIO_param]
    optimizer = torch.optim.AdamW([{'params': DINO_param, 'lr': 5e-5},
                                   {'params': AUDIO_param, 'lr': 5e-5},
                                   {'params': ISL_param, 'lr': 1e-4},],
                                   betas=(0.9, 0.999),
                                   weight_decay=0.01)
    criterion = MatchingLoss()

    writer = SummaryWriter(track_cfg["LOG_DIR"])
    ckpt_dir = Path(track_cfg["CHECKPOINT_DIR"], "localisation")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    figures_path = '/home/ec2-user/vggss/JEPAS-MAIN/figures'
    os.makedirs(figures_path, exist_ok=True)

    global_step = 0
    for epoch in range(exp_cfg["MAX_EPOCHS"]):
        best_loss = float("inf")
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{exp_cfg['MAX_EPOCHS']}", leave=False)
        ISL.train()
        for i, data in enumerate(progress):
            audio = data["waveform"].to(device).squeeze(0)
            images = data["image"].to(device).squeeze(0)
            bbox = data["bbox"].to(device)
            
            output = ISL(audio=audio, image=images)
            loss = criterion(output, bbox)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                ISL.eval()
                with torch.no_grad():
                    output = ISL(audio=audio, image=images)
                    output = ISL.postprocess(output, images.shape[-2:])
                    plot_bbox(images, output, data["bbox"], figures_path, i, train=True)
                ISL.train()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
            progress.set_postfix(loss=loss.item(), best_loss=best_loss)

        writer.add_scalar("train/best_loss_epoch", best_loss, epoch)
        print(f"Epoch {epoch+1}: best loss {best_loss:.4f}")

        if epoch % 5 == 0 and epoch != 0:
            ckpt_path = ckpt_dir / f"ISL_{audio_mode}_epoch{epoch+1}.pt"
            torch.save({"epoch": epoch,
                        "model_state_dict": ISL.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,}, 
                        ckpt_path)
        
        test_progress = tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{exp_cfg['MAX_EPOCHS']}", leave=False)
        ISL.eval()
        total_val_loss = 0
        val_step = 0
        val_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'mean_iou': 0, 'num_predictions': 0, 'num_gt': 0}
        with torch.no_grad():
            for i, data in enumerate(test_progress):
                audio = data["waveform"].to(device)
                images = data["image"].to(device)
                bbox = data["bbox"].to(device)
                
                output = ISL(audio=audio, image=images)
                val_loss = criterion(output, bbox)
                
                total_val_loss += val_loss.item()
                output = ISL.postprocess(output, images.shape[-2:])
                if i % 10 == 0:
                    plot_bbox(images, output, bbox, figures_path, i)
                metrics = compute_metrics(output, bbox, image_shape=images.shape[-2:])
                val_metrics = {k: (val_metrics[k] + metrics[k]) for k in val_metrics}
                val_step += 1
            
            val_metrics = {k: v / len(test_loader) for k, v in val_metrics.items()}
            writer.add_scalar("test/loss", total_val_loss / len(test_loader), epoch)
            print(f"Epoch {epoch+1}: val loss {total_val_loss / len(test_loader):.4f}")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
                writer.add_scalar(f"test/{k}", v, epoch)

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