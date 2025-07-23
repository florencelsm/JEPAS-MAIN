import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs import load_config
from jepa_datasets.audio import create_windows_audio_datamodule
from model.image.image_zoo import ijepa_model_builders
from model.audio.audio_zoo import audio_model_builders
from model.predictor import MoEPredictor
from model import IJEPA

if __name__ == "__main__":
    import torch

    cfg = load_config()
    ai_cfg = cfg.get("audio_image", {})

    exp_cfg = ai_cfg.get("experiment", {})
    use_spec = ai_cfg.get("dataset", {}).get("USE_SPEC", True)
    student_name = "spec_vit_pretrain" if use_spec else "wave_1dt_pretrain"
    teacher_size = exp_cfg.get("TEACHER_SIZE", "base")
    lr = exp_cfg.get("LR", 1e-4)
    seed = exp_cfg.get("SEED", 0)
    max_epochs = exp_cfg.get("MAX_EPOCHS", 1)

    runtime = ai_cfg.get("runtime", {})
    accelerator = runtime.get("ACCELERATOR", "cpu")
    devices = runtime.get("DEVICES", 1)
    precision = runtime.get("PRECISION", "32-true")

    torch.set_float32_matmul_precision(runtime.get("FLOAT32_MATMUL_PRECISION", "medium"))
    pl.seed_everything(seed)

    student_builder = audio_model_builders[student_name]
    teacher_builder = ijepa_model_builders[teacher_size]

    student_model = student_builder()
    teacher_model = teacher_builder()

    teacher_patch = teacher_model.patch_embed
    teacher_pos = teacher_model.pos_embedding
    teacher_enc = teacher_model.teacher_encoder

    model: IJEPA = teacher_model
    model.patch_embed = student_model.patch_embed
    model.pos_embedding = student_model.pos_embedding
    model.num_patches = student_model.num_patches
    model.encoder = student_model.encoder
    model.teacher_patch_embed = teacher_patch
    model.teacher_pos_embedding = teacher_pos
    model.teacher_encoder = teacher_enc

    model.teacher_encoder.requires_grad_(False)
    model.predictor = MoEPredictor(
        embed_dim=model.embed_dim,
        num_heads=model.num_heads,
        depth=model.decoder_depth,
    )

    datamodule = create_windows_audio_datamodule(ai_cfg)

    tracking = ai_cfg.get("tracking", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=tracking.get("CHECKPOINT_DIR", "checkpoints"), filename=student_name
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=tracking.get("LOG_DIR", "logs"), name="AIJEPA")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)