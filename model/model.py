import copy
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertEmbeddings

from model.base_model import JEPA_base
from utils.types_utils import Number

from transformers import (
    ASTFeatureExtractor,
    ASTModel,
    AutoImageProcessor,
    BitImageProcessor,
    Dinov2Model,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

# pylint: disable=pointless-string-statement

BERT_MODEL_NAME: str = "bert-base-uncased"
PRETRAINED_TEXT_ENCODER: bool = True


class ImageAudioJEPA(JEPA_base, pl.LightningModule):
    def __init__(
        self,
        # JEPA Base
        vision_model: Dinov2Model,
        audio_model: Union[ASTModel, Wav2Vec2Model],

        vision_feature_extractor: BitImageProcessor,
        audio_feature_extractor: Union[ASTFeatureExtractor, Wav2Vec2FeatureExtractor],

        decoder_depth: int = 6,
        num_heads: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        num_target_blocks: int = 4,  # number of distinct target blocks per image
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
        audio_backbone: str = "spec",
        teacher_mask_ratio: float = 0.0,
        testing_purposes_only: bool = False,
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            vision_model=vision_model,
            audio_model=audio_model,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            **kwargs,
        )

        self.vision_feature_extractor = vision_feature_extractor
        self.audio_feature_extractor = audio_feature_extractor

        self.decoder_depth = decoder_depth

        if not testing_purposes_only:
            self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m  # momentum
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.teacher_mask_ratio = teacher_mask_ratio

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()
      
   
    def forward(  # pylint: disable=arguments-differ
        self,
        *,
        waveform: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_tensor: torch.Tensor = self.audio_feature_extractor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values

        image_tensor = self.vision_feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values

        return self.forward_base(
            audio=audio_tensor,
            image=image_tensor,
        )

    def update_momentum(self, m: float) -> None:
        """
        Update the teacher model parameters using momentum.

        Args:
            m (float): Momentum coefficient for the exponential moving average update.
        """
        # Enable eval mode to disable layers like dropout and batch normalization
        student_model: nn.Module = self.encoder.eval()
        teacher_model: nn.Module = self.teacher_encoder.encoder.eval()

        """
        Manual parameter updates:
        Manually update the teacher's parameters using a momentum term, ensuring
        that the teacher's parameters are a smoothed version of the student's parameters,
        thus reducing the noise and fluctuations in the learning process.

        This smoothing provides more consistent and stable targets for the student to learn from,
        increasing training efficacy. Additionally, this decoupling permits more exploration in the
        student without directly affecting the teacher's parameters, preventing the student from
        overfitting to the techer's instantaneous updates.
        """
        # Disable gradient computation
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        audio_tensor = batch
        image_tensor = batch

        if isinstance(batch, dict):
            if "image" in batch:
                image_tensor = batch["image"]
            elif "img_rgb" in batch:
                image_tensor = batch["img_rgb"]
            elif "img" in batch:
                image_tensor = batch["img"]
            if "audio" in batch:
                audio_tensor = batch["audio"]
            else:
                audio_tensor = image_tensor


        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            audio=audio_tensor,
            image=image_tensor,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)


        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform a validation step for each image (tensor) in the batch of images (list of tensors).

        Parameters
        ----------
        batch : torch.Tensor
            A tensor representing the batch of data (images).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        audio_tensor = batch
        image_tensor = batch

        if isinstance(batch, dict):
            if "image" in batch:
                image_tensor = batch["image"]
            elif "img_rgb" in batch:
                image_tensor = batch["img_rgb"]
            elif "img" in batch:
                image_tensor = batch["img"]
            if "audio" in batch:
                audio_tensor = batch["audio"]
            else:
                audio_tensor = image_tensor

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            audio=audio_tensor,
            image=image_tensor,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_
        dataloader_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"

        audio_tensor = batch
        image_tensor = batch

        if isinstance(batch, dict):
            if "image" in batch:
                image_tensor = batch["image"]
            elif "img_rgb" in batch:
                image_tensor = batch["img_rgb"]
            elif "img" in batch:
                image_tensor = batch["img"]
            if "audio" in batch:
                audio_tensor = batch["audio"]
            else:
                audio_tensor = image_tensor

        return self(  # Return only student embedding using the student (ViT) encoder
            audio=audio_tensor,
            image=image_tensor,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        steps = max(1, getattr(self.trainer, "estimated_stepping_batches", 1))
        self.m += (
            self.momentum_limits[1] - self.momentum_limits[0]
        ) / steps

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Callable, Dict[str, Union[str, Callable]]]]:
        optimizer: Callable = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        steps = max(1, getattr(self.trainer, "estimated_stepping_batches", 1))
        scheduler: Callable = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
