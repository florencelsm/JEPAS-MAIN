from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from x_transformers import Encoder

from utils.types_utils import ensure_tuple

from ..patch_embed import PatchEmbed2D, PatchEmbed3D


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 64,
        enc_depth: int = 8,
        num_heads: int = 8,
        post_emb_norm: bool = True,
        post_enc_norm: bool = True,
        layer_dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = ensure_tuple(img_size)
        self.patch_size = ensure_tuple(patch_size)

        self.num_frames = num_frames
        self.is_video = num_frames > 1
        self.tubelet_size = tubelet_size

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        )
        self.num_patches: int = int(
            torch.prod(torch.Tensor(self.patch_embed.patch_shape)).item()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.post_emb_norm = post_emb_norm
        self.post_emb_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_emb_norm else nn.Identity()
        )

        self.layer_dropout = layer_dropout

        self.encoder = Encoder(  # student encoder
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_enc_norm else nn.Identity()
        )  # student encoder

    def forward_vit(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_embed_only: bool = False,
    ) -> torch.Tensor:
        # Obtain patch embeddings from the input tensor
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)

        # Add positional embeddings to the patch embeddings
        x = x + self.pos_embedding  # (batch, num_patches, embed_dim)

        # Normalize the patch embeddings (if `self.post_emb_norm`)
        x = self.post_emb_norm_vit(x)  # (batch, num_patches, embed_dim)

        if patch_embed_only:
            return x  # (batch, num_patches, embed_dim)

        # Encode the patch embeddings using the student encoder
        x = self.encoder(x, attn_mask=attention_mask)  # (batch, num_patches, embed_dim)

        # Normalize the encoded patches (if `self.post_enc_norm`)
        x = self.post_enc_norm_vit(x)  # (batch, num_patches, embed_dim)

        return x
    
    # ------------------------------------------------------------------ #
    # NEW : public forward – 兼容「原始图像/视频」和「已 patch-embed 的 token」
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_embed_only: bool = False,
    ) -> torch.Tensor:
        """
        统一入口：
        • (B, C, H, W) or (B, C, T, H, W) → 像以前一样走 `forward_vit`.
        • (B, N, E) 已是 patch-token（含位置编码）→ 直接送 Encoder.
        """
        # ── 情况 1：输入还是原图 / 视频 ──────────────────────────────────
        if x.ndim in (4, 5):
            return self.forward_vit(x, attention_mask, patch_embed_only)

        # ── 情况 2：输入已是 patch-tokens ───────────────────────────────
        if x.ndim == 3:                                          # (B, N, E)
            if patch_embed_only:                                 # 只想要 token 本身
                return x
            x = self.encoder(x, attn_mask=attention_mask)        # 编码
            x = self.post_enc_norm_vit(x)
            return x

        raise ValueError(f"[VisionTransformer] unexpected input shape: {x.shape}")


def vit_nano(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=64,
        enc_depth=8,
        num_heads=8,
        **kwargs,
    )


def vit_tiny(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=192,
        enc_depth=12,
        num_heads=8,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=384,
        enc_depth=18,
        num_heads=8,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=768,
        enc_depth=18,
        num_heads=12,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=1280,
        enc_depth=32,
        num_heads=16,
        **kwargs,
    )
