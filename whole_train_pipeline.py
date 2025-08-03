"""Training pipeline for JEPA image–audio alignment.

Example – run the two experiments on CPU as defined in the default
configuration:
    $ python train_pipeline.py --config ./config.json

If you only want to run one backbone or override a hyper‑parameter from the
command line, simply add it, e.g.:
    $ python train_pipeline.py --config ./config.json --audio_backbones ast \
                               --max_epochs 10 --accelerator gpu --devices 1
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.load_config import load_config  # local helper already provided
from pretrained.audio_models import load_audio_models  # utilities for AST / Wav2Vec2
from pretrained.image_model import load_image_model  # utility for Dinov2 + processor
from model.base_model import JEPA_base


# ---------------------------------------------------------------------------
#  Kaldi F‑bank extractor inside AST requires **len(waveform) >= window_size**
#  where window_size ≈ sample_rate * 0.025.  We therefore compute it once from
#  the (assumed) sample‑rate and pad any shorter clip so that the assertion in
#  ``torchaudio.compliance.kaldi.fbank`` is never triggered.
# ---------------------------------------------------------------------------
FRAME_MS = 25  # analysis window length used by AST (25 ms)
DEFAULT_SR = 16_000  # expected by pre‑trained AST
WINDOW_SIZE = int(DEFAULT_SR * FRAME_MS / 1_000 + 0.5)  # → 400
MIN_WAVEFORM_LEN = WINDOW_SIZE  # pad clips < 400 samples to exactly 400


# ────────────────────────────────  Dataset  ──────────────────────────────────

class AudioImageDataset(Dataset):
    """Pairs ``waveforms`` stored as ``.npy`` with corresponding ``.jpg`` files."""

    def __init__(self, root: Path, sample_rate: int = 16_000) -> None:  # noqa: D401
        self.root = Path(root).expanduser()
        self.wave_dir = self.root / "waveforms"
        self.image_dir = self.root / "images"
        self.sample_rate = sample_rate

        # Only keep items for which both modalities are available
        self.items: List[str] = [
            p.stem
            for p in self.wave_dir.glob("*.npy")
            if (self.image_dir / f"{p.stem}.jpg").exists()
        ]

        if not self.items:
            raise FileNotFoundError(
                "No paired <waveform, image> samples were found under "
                f"{self.root}. Ensure directories `waveforms/` and `images/` "
                "exist and contain matching filenames."
            )

    def __len__(self) -> int:  
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:  
        stem = self.items[idx]
        wave_path = self.wave_dir / f"{stem}.npy"
        img_path = self.image_dir / f"{stem}.jpg"

        # Waveform => 1‑D float32 tensor (mono)
        waveform_np = np.load(wave_path)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)  # convert stereo → mono

        if waveform.numel() < MIN_WAVEFORM_LEN:
            pad_len = MIN_WAVEFORM_LEN - waveform.numel()
            waveform = F.pad(waveform, (0, pad_len))

        return waveform.contiguous(), str(img_path)


# ──────────────────────────────  Utils  ─────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across numpy / torch / python."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Performance / determinism trade‑off – OK for research purposes.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch: Sequence[Tuple[torch.Tensor, str]]):  
    """Custom collate that keeps variable‑length waveforms & PIL images as lists."""

    waveforms, img_paths = zip(*batch)  # tuple of tensors / paths

    # PIL images are loaded *lazily* here so as not to keep all images in RAM.
    images = [Image.open(p).convert("RGB") for p in img_paths]

    # Waveforms remain a list of 1‑D tensors; the feature‑extractor will handle
    # padding/stacking internally.
    return {"audio": list(waveforms), "image": images}


# ────────────────────────────────  Train  ───────────────────────────────────=

def train_one_epoch(
    *,
    model: JEPA_base,
    dataloader: DataLoader,
    audio_extractor,
    vision_processor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    epoch_idx: int,
    log_every: int = 10,
):
    model.train()
    pbar = tqdm(dataloader, desc=f"[train] Epoch {epoch_idx}", leave=False)
    running_loss = 0.0

    for step, batch in enumerate(pbar):
        waveforms: List[torch.Tensor] = batch["audio"]
        images: List[Image.Image] = batch["image"]

        # ------------------------  Feature extraction  -----------------------
        audio_inputs = audio_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device)

        image_inputs = vision_processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        # ----------------------------  Forward  ------------------------------
        optimizer.zero_grad(set_to_none=True)
        preds, targets = model.forward_base(audio=audio_inputs, image=image_inputs)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % log_every == 0:
            global_step = epoch_idx * len(dataloader) + step
            writer.add_scalar("Loss/train", loss.item(), global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    writer.add_scalar("Loss/train_epoch", epoch_loss, epoch_idx)


@torch.inference_mode()
def validate(
    *,
    model: JEPA_base,
    dataloader: DataLoader,
    audio_extractor,
    vision_processor,
    device: torch.device,
    writer: SummaryWriter,
    epoch_idx: int,
):
    model.eval()
    losses = []

    pbar = tqdm(dataloader, desc=f"[val]   Epoch {epoch_idx}", leave=False)
    for batch in pbar:
        waveforms: List[torch.Tensor] = batch["audio"]
        images: List[Image.Image] = batch["image"]

        audio_inputs = audio_extractor(
            waveforms, sampling_rate=16_000, return_tensors="pt", padding=True
        ).input_values.to(device)
        image_inputs = vision_processor(images=images, return_tensors="pt").pixel_values.to(
            device
        )

        preds, targets = model.forward_base(audio=audio_inputs, image=image_inputs)
        loss = F.mse_loss(preds, targets)
        losses.append(loss.item())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    writer.add_scalar("Loss/val_epoch", mean_loss, epoch_idx)
    return mean_loss


# ────────────────────  Experiment (AST vs Wav2Vec2)  ────────────────────────

def run_experiment(
    *,
    backbone_key: str,  # "ast" | "wav2vec2"
    config: Dict,
    device: torch.device,
    global_seed: int,
    resume_path: str | None = None,
):
    """Runs one full training loop and returns validation losses per epoch."""

    seed_everything(global_seed)

    # -----------------------  Load models / processors  ---------------------
    vision_model, vision_processor = load_image_model(device)

    audio_models = load_audio_models(device)
    audio_model, audio_extractor = audio_models[backbone_key]

    # -----------------------  JEPA instantiation  ---------------------------
    model = JEPA_base(
        vision_model=vision_model,
        audio_model=audio_model,
        decoder_depth=6,
        num_heads=8,
        device=str(device),
    ).to(device)

    # ---------------------------  Optimiser  --------------------------------
    lr = float(config["experiment"]["LR"])
    weight_decay = 0.05  # hard‑coded – can also move to config
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -------------------------  Datasets & Loaders  -------------------------
    root_path = Path(config["dataset"]["DATASET_PATH"])

    # Train / val split = 90 / 10 based on filenames for simplicity.
    dataset_full = AudioImageDataset(root=root_path)
    indices = list(range(len(dataset_full)))
    split = int(0.9 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    train_ds = torch.utils.data.Subset(dataset_full, train_indices)
    val_ds = torch.utils.data.Subset(dataset_full, val_indices)

    dataloader_common = dict(
        batch_size=int(config["experiment"]["BATCH_SIZE"]),
        num_workers=int(config["experiment"]["NUM_WORKERS"]),
        pin_memory=bool(config["experiment"]["PIN_MEMORY"]),
        persistent_workers=bool(config["experiment"]["PERSISTENT_WORKERS"]),
        prefetch_factor=int(config["experiment"].get("PREFETCH_FACTOR", 2)),
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **dataloader_common)
    val_loader_kwargs = {**dataloader_common, "shuffle": False}
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    # -----------------------  Logging & Checkpoints  ------------------------
    log_dir_root = Path(config["tracking"]["LOG_DIR"]).expanduser()
    ckpt_root = Path(config["tracking"]["CHECKPOINT_DIR"]).expanduser()

    experiment_name = f"JEPA_{backbone_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    writer = SummaryWriter(log_dir=log_dir_root / experiment_name)

    start_epoch = 0

    if resume_path:
        print(f"[info] Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1

    max_epochs = int(config["experiment"]["MAX_EPOCHS"])
    val_losses: List[float] = []

    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            audio_extractor=audio_extractor,
            vision_processor=vision_processor,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch_idx=epoch,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            audio_extractor=audio_extractor,
            vision_processor=vision_processor,
            device=device,
            writer=writer,
            epoch_idx=epoch,
        )
        val_losses.append(val_loss)

        # -------------  Save checkpoint after each epoch  -------------------
        ckpt_dir = ckpt_root / experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch-{epoch}.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"[info] Saved checkpoint: {ckpt_path.relative_to(Path.cwd())}")

    writer.flush()
    writer.close()
    return val_losses


# ============================================================================
# ────────────────────────────────  Main  ────────────────────────────────────
# ============================================================================


def parse_args():  
    """Very small CLI – all heavy lifting is in the config file."""

    p = argparse.ArgumentParser(description="JEPA audio–image training")
    p.add_argument("--config", type=str, default="./config.json")
    p.add_argument("--audio_backbones", nargs="*", default=["ast", "wav2vec2"], help="Subset to train; default: both")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():  
    args = parse_args()
    cfg = load_config(args.config)

    # Choose runtime device(s) – here we keep it simple (single‑device).
    accelerator = cfg["audio"]["runtime"].get("ACCELERATOR", "cpu").lower()
    if accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    seed_everything(int(cfg["audio"]["experiment"].get("SEED", 0)))

    val_history: Dict[str, List[float]] = {}

    for backbone_key in args.audio_backbones:
        print(f"\n[experiment] Starting run for audio backbone: {backbone_key}")
        val_losses = run_experiment(
            backbone_key=backbone_key,
            config=cfg["audio"],  # use audio‑only section for hyper‑params
            device=device,
            global_seed=int(cfg["audio"]["experiment"].get("SEED", 0)),
            resume_path=args.resume,
        )
        val_history[backbone_key] = val_losses

    # Simple summary
    print("\n[summary] Validation losses per epoch:")
    for bk, losses in val_history.items():
        joined = ", ".join(f"{x:.4f}" for x in losses)
        print(f"  {bk}: {joined}")


if __name__ == "__main__":
    main()
