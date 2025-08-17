import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.base_model import JEPA_base
import numpy as np
from typing import Callable, Dict, Tuple, Union
from utils.types_utils import Number
from transformers import ASTModel, Dinov2Model, Wav2Vec2Model

class ImageAudioJEPA(JEPA_base, pl.LightningModule):
    def __init__(self,
                 vision_model: Dinov2Model,
                 audio_model: Union[ASTModel, Wav2Vec2Model],
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
                 teacher_mask_ratio: float = 0.0,
                 testing_purposes_only: bool = False,
                 apply_ema: bool = False,
                 **kwargs,):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(self,
                           vision_model=vision_model,
                           audio_model=audio_model,
                           decoder_depth=decoder_depth,
                           num_heads=num_heads,
                           **kwargs,)
        
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
        self.apply_ema = apply_ema

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()
    
    def forward(self,
                waveform: torch.Tensor,
                image: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_base(audio=waveform, image=image,)

    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int,
                      dataloader_idx: int = 0) -> torch.Tensor:
                
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], 
                                                self.target_aspect_ratio[1])
        
        target_scale = np.random.uniform(low=self.target_scale_interval[0],
                                         high=self.target_scale_interval[1])

        context_scale = np.random.uniform(self.context_scale[0], 
                                          self.context_scale[1])
        
        y_student, y_teacher = self(audio=batch['waveform'],
                                    image=batch['image'],
                                    target_aspect_ratio=target_aspect_ratio,
                                    target_scale=target_scale,
                                    context_aspect_ratio=self.context_aspect_ratio,
                                    context_scale=context_scale,)

        loss = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int,
                        dataloader_idx: int = 0,) -> torch.Tensor:
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], 
                                                self.target_aspect_ratio[1])
        
        target_scale = np.random.uniform(low=self.target_scale_interval[0],
                                         high=self.target_scale_interval[1])

        context_scale = np.random.uniform(self.context_scale[0], 
                                          self.context_scale[1])

        y_student, y_teacher = self(audio=batch['waveform'],
                                    image=batch['image'],
                                    target_aspect_ratio=target_aspect_ratio,
                                    target_scale=target_scale,
                                    context_aspect_ratio=self.context_aspect_ratio,
                                    context_scale=context_scale,)

        loss = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        
        self.mode = "test"
        return self(audio=batch['waveform'], image=torch.empty([1]))

    def update_momentum(self, m: float) -> None:
        student_model: nn.Module = self.encoder.eval()
        teacher_model: nn.Module = self.teacher_encoder.encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), 
                                                    teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)
    
    def on_after_backward(self) -> None:
        if self.apply_ema:
            self.update_momentum(self.m)
            steps = max(1, getattr(self.trainer, "estimated_stepping_batches", 1))
            self.m += (self.momentum_limits[1] - self.momentum_limits[0]) / steps

    def configure_optimizers(self,) -> Dict[str, Union[Callable, Dict[str, Union[str, Callable]]]]:
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.lr,
                                                        total_steps=self.trainer.estimated_stepping_batches,)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step",}}
