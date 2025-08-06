from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ASTModel, Wav2Vec2Model, Dinov2Model

from .predictor import Predictor


class JEPA_base(nn.Module):

    def __init__(
        self,
        vision_model: Dinov2Model,
        audio_model: Union[ASTModel, Wav2Vec2Model],
        decoder_depth: int,
        num_heads: int,
        predictor_embed_dim: Optional[int] = None,
        post_enc_norm: bool = False,
        mode: str = "train",
        context_ratio_range: Tuple[float, float] = (0.85, 0.95),
        target_mask_range: Tuple[float, float] = (0.15, 0.25),
        device: str = "cpu",
        **_: Any,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode.lower()
        self.context_ratio_range = context_ratio_range
        self.target_mask_range = target_mask_range
        self.embed_dim = vision_model.config.hidden_size
        assert self.embed_dim == audio_model.config.hidden_size

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_jepa = (
            nn.LayerNorm(self.embed_dim) if self.post_enc_norm else nn.Identity()
        )

        # ali
        # student
        self.audio_encoder = audio_model
        if self.mode == "test":
            self.audio_encoder.eval()
        # teacher
        self.image_encoder = vision_model
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            depth=decoder_depth,
            predictor_embed_dim=predictor_embed_dim,
        )
        

        self.device = torch.device(device)
       
    def sample_num_blocks(
        self, T: int, min_ratio: float, max_ratio: float, exclude_cls: bool = True
    ) -> int:
        """
        Samples a single integer value based on a ratio range.

        Parameters
        ----------
        T : int
            Total number of tokens (including CLS).
        min_ratio : float
            Minimum proportion of tokens to sample.
        max_ratio : float
            Maximum proportion of tokens to sample.
        exclude_cls : bool
            Whether to exclude the CLS token (index 0) from sampleing.

        Returns
        -------
        int
            Number of tokens to sample.
        """
        num_candidates = T - 1 if exclude_cls else T
        min_num_samples = max(1, int(min_ratio * num_candidates))
        max_num_samples = max(
            min_num_samples + 1, int(max_ratio * num_candidates)
        )  # ensure > min

        return torch.randint(
            low=min_num_samples, high=max_num_samples + 1, size=(1,)
        ).item()

    def create_fixed_mask(
        self,
        batch_size: int,
        num_tokens: int,
        num_masked: int,
        device: torch.device,
        exclude_cls: bool = True,
    ) -> torch.BoolTensor:
        """
        Creates a per-sample boolean mask with exactly `num_masked` masked positions.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        num_tokens : int
            Number of total tokens per sample (e.g. 257).
        num_masked : int
            Number of tokens to mask per sample.
        device : torch.device
            Device to place the output mask on.
        exclude_cls : bool
            Whether to exclude the CLS token (index 0) from being masked.

        Returns
        -------
        torch.BoolTensor
            A tensor of shape (B, T) with exactly `num_masked` True values per row.
        """
        valid_indices: List[int] = (
            list(range(1, num_tokens)) if exclude_cls else list(range(num_tokens))
        )

        mask: torch.BoolTensor = torch.zeros(
            (batch_size, num_tokens), dtype=torch.bool, device=self.device
        )

        for b in range(batch_size):
            selected = torch.randperm(len(valid_indices), device=self.device)[
                :num_masked
            ]
            masked_indices = torch.tensor(valid_indices, device=self.device)[selected]
            mask[b, masked_indices] = True

        return mask

    def create_target_masks_and_blocks(
        self,
        last_hidden_state: torch.Tensor,  # (B, T, D)
        pos_embeddings: torch.Tensor,  # (B, T, D)
        mask_token: nn.Parameter,  # (1, 1, D)
        mask: torch.BoolTensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts target masks and target blocks from masked positions in ViT encoder output.

        For all masked positions (excluding [CLS]), returns:
        - target_masks: mask_token + pos_embedding
        - target_blocks: original hidden states at those positions

        Parameters
        ----------
        last_hidden_state : torch.Tensor
            ViT encoder output, shape (B, T, D)
        pos_embeddings : torch.Tensor
            Positional embeddings, shape (B, T, D)
        mask_token : nn.Parameter
            Learnable mask token, shape (1, 1, D)
        mask : torch.BoolTensor
            Boolean mask indicating which positions to mask, shape (B, T)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            target_masks : torch.Tensor of shape (B, N_masked, D)
                Mask token plus positional embedding at masked positions
            target_blocks : torch.Tensor of shape (B, N_masked, D)
                Ground-truth hidden states at the masked positions
        """
        B, T, D = last_hidden_state.shape

        if pos_embeddings.shape[0] == 1 and B > 1:
            pos_embeddings = pos_embeddings.expand(B, -1, -1)

        # Ensure CLS token is not masked (typically index 0)
        mask = mask.clone()
        mask[:, 0] = False

        # Expand mask token to (B, T, D)
        mask_token_expanded = mask_token.expand(B, T, D)  # (B, T, D)

        # Compute context masks
        target_masks: List[torch.Tensor] = []
        target_blocks: List[torch.Tensor] = []

        for b in range(B):
            masked_indices = mask[b]  # (T,)

            target_masks.append(
                mask_token_expanded[b][masked_indices]
                + pos_embeddings[b][masked_indices]
            )  # (N_masked, D)
            target_blocks.append(last_hidden_state[b][masked_indices])  # (N_masked, D)

        return torch.stack(target_masks), torch.stack(target_blocks)

    def sample_context_blocks(
        self,
        audio_embeddings: torch.Tensor,
        num_context_blocks: int,
        exclude_cls: bool = True,
    ) -> torch.Tensor:
        """
        Randomly samples `num_context_blocks` from audio_embeddings per sample.

        Parameters
        ----------
        audio_embeddings : torch.Tensor
            Input embeddings of shape (B, T, D)
        num_context_blocks : int
            Number of blocks to sample per sample
        exclude_cls : bool
            Whether to exclude the CLS token (index 0)

        Returns
        -------
        torch.Tensor
            Context blocks of shape (B, num_context_blocks, D)
        """
        B, T, D = audio_embeddings.shape
        context_blocks: List[torch.Tensor] = []

        # Range of valid indices (excluding CLS if needed)
        valid_indices = list(range(1, T)) if exclude_cls else list(range(T))

        for b in range(B):
            selected_indices = torch.randperm(len(valid_indices))[:num_context_blocks]
            selected_indices = torch.tensor(
                valid_indices, device=audio_embeddings.device
            )[selected_indices]

            blocks_b: torch.Tensor = audio_embeddings[b][
                selected_indices
            ]  # shape: (num_context_blocks, D)
            context_blocks.append(blocks_b)

        return torch.stack(context_blocks)  # shape: (B, num_context_blocks, D)

    def forward_base(
        self,
        *,
        audio: torch.Tensor,
        image: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        (Image/video invariant)

        Forward pass for generating predictions and targets within the JEPA architecture.

        Args:
            img_rgb (torch.Tensor): Image tensor for the teacher branch.
            aud_inp (torch.Tensor): Audio tensor for the student branch.
            target_patches (List[List[int]]): A list of lists containing indices of patches for each target block.
            context_patches (List[int]): A list of patch indices for the context block excluding target patches.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If self.mode is "test":
                    torch.Tensor: Full embedding tensor from the student encoder.
                - If self.mode is not "test":
                    Tuple[torch.Tensor, torch.Tensor]:
                        - prediction_blocks: Predicted blocks based on the context encoding.
                        - target_blocks: Actual target blocks.
        """
        test_mode: bool = self.mode == "test"

        # Encode the input tensor (already normalised)
        audio_embeddings: torch.Tensor = self.audio_encoder(
            input_values=audio
        ).last_hidden_state

        if test_mode:
            return audio_embeddings

        ### GET CONTEXT BLOCKS
        B, AT, AD = audio_embeddings.shape

        num_context_blocks: int = self.sample_num_blocks(
            T=AT,
            min_ratio=self.context_ratio_range[0],
            max_ratio=self.context_ratio_range[1],
            exclude_cls=True,
        )

        context_encoding: torch.Tensor = self.sample_context_blocks(
            audio_embeddings=audio_embeddings,
            num_context_blocks=num_context_blocks,
            exclude_cls=True,
        )

        ### GET TARGET BLOCKS AND MASKS
        # Encode the input tensor (already normalised)
        with torch.no_grad():
            image_embeddings: torch.Tensor = self.image_encoder(image).last_hidden_state

        B, IT, ID = image_embeddings.shape

        num_masks: int = self.sample_num_blocks(
            T=IT,
            min_ratio=self.target_mask_range[0],
            max_ratio=self.target_mask_range[1],
            exclude_cls=True,
        )

        target_mask: torch.Tensor = self.create_fixed_mask(
            batch_size=B,
            num_tokens=IT,
            num_masked=num_masks,
            device=self.device,
            exclude_cls=True,
        )

        B, C, H, W = image.shape
        pos_embeddings: torch.Tensor = (
            self.image_encoder.embeddings.interpolate_pos_encoding(
                image_embeddings, H, W
            )
        )

        (
            target_masks,  # (B, N_masked, ID)
            target_blocks,  # (B, N_masked, ID)
        ) = self.create_target_masks_and_blocks(
            last_hidden_state=image_embeddings,
            pos_embeddings=pos_embeddings,
            mask_token=self.mask_token,
            mask=target_mask,
        )
        batch_size, num_target_blocks, embed_dim = target_blocks.shape

        ### MAKE PREDICTIONS
        predictions: torch.Tensor = self.predictor(
            context_encoding=context_encoding,
            target_masks=target_masks,
        )  # (B, N_masked, ID)

        return (
            predictions,  # (B, N_masked, ID)
            target_blocks,  # (B, N_masked, ID)
        )
    
