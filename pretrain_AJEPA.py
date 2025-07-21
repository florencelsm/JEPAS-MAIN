import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs import (
    load_config,
)
from jepa_datasets import create_audio_datamodule
from model import IJEPA
from model.audio import audio_model_builders


if __name__ == "__main__":
    import torch

    config = load_config()
    audio_cfg = config.get("audio", {})

    model_name = audio_cfg.get("experiment", {}).get("MODEL_NAME", "spec_vit")
    lr = audio_cfg.get("experiment", {}).get("LR", 1e-4)
    seed = audio_cfg.get("experiment", {}).get("SEED", 0)
    max_epochs = audio_cfg.get("experiment", {}).get("MAX_EPOCHS", 1)

    runtime = audio_cfg.get("runtime", {})
    accelerator = runtime.get("ACCELERATOR", "cpu")
    devices = runtime.get("DEVICES", 1)
    precision = runtime.get("PRECISION", "32-true")

    torch.set_float32_matmul_precision(runtime.get("FLOAT32_MATMUL_PRECISION", "medium"))

    vit_builder = audio_model_builders[model_name]
    vit = vit_builder()
    model = IJEPA()
    model.encoder = vit

    datamodule = create_audio_datamodule(audio_cfg)

    tracking = audio_cfg.get("tracking", {})
    checkpoint_callback = ModelCheckpoint(dirpath=tracking.get("CHECKPOINT_DIR", "checkpoints"), filename=model_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=tracking.get("LOG_DIR", "logs"), name="AJEPA")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)