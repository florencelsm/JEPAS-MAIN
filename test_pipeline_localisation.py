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
    torch.set_float32_matmul_precision(runtime_cfg.get("FLOAT32_MATMUL_PRECISION", "medium")) # ali chenged to highest in config.json

    device = runtime_cfg.get("ACCELERATOR", "cuda")

    data_root = Path(data_cfg["TRAIN_DATASET_PATH"])
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parent / data_root).resolve()

    if not ((data_root / "waveforms").is_dir() and (data_root / "images").is_dir()):
        if data_root.name in {"images", "waveforms"}:
            data_root = data_root.parent

    test_dataset = VGGSS_Dataset(audio_mode=audio_mode.lower(),
                                 config=data_cfg,
                                 mode='test')
    
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=exp_cfg.get("NUM_WORKERS", 0),
                             pin_memory=exp_cfg.get("PIN_MEMORY", False),
                             persistent_workers=exp_cfg.get("PERSISTENT_WORKERS", False),
                             prefetch_factor=exp_cfg.get("PREFETCH_FACTOR", 2),)
    
    audio_model = load_audio_models(audio_mode=audio_mode, device=device)
    vision_model = load_image_models(device=device)

    ISL = VisualSoundLocalizer(image_encoder=vision_model,
                               audio_encoder=audio_model,
                               freeze_backbones=False).cuda()
    checkpoint = '/home/ec2-user/vggss/JEPAS-MAIN/figures_spectrogram_audioQ_imageKV/ISL_spectrogram_epoch11.pt'
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)['model_state_dict']
    ISL.load_state_dict(checkpoint, strict=True)


    figures_path = '/home/ec2-user/vggss/JEPAS-MAIN/figures_test'
    os.makedirs(figures_path, exist_ok=True)

    test_progress = tqdm(test_loader, leave=False)
    ISL.eval()
    val_step = 0
    val_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'mean_iou': 0, 'num_predictions': 0, 'num_gt': 0}
    with torch.no_grad():
        for i, data in enumerate(test_progress):
            audio = data["waveform"].to(device)
            images = data["image"].to(device)
            bbox = data["bbox"].to(device)
            
            output = ISL(audio=audio, image=images)
            
            output = ISL.postprocess(output, images.shape[-2:])
            plot_bbox(images, output, bbox, figures_path, i)
            metrics = compute_metrics(output, bbox, image_shape=images.shape[-2:])
            val_metrics = {k: (val_metrics[k] + metrics[k]) for k in val_metrics}
            val_step += 1
        
        val_metrics = {k: v / len(test_loader) for k, v in val_metrics.items()}
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

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