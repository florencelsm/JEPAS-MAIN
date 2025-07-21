from typing import Callable, Dict

from model.model import IJEPA
from model.vision.vit import VisionTransformer
from model.patch_embed import PatchEmbed1D


def create_spec_vit(embed_dim: int = 768) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        enc_depth=12,
        num_heads=12,
        post_emb_norm=True,
        post_enc_norm=True,
        layer_dropout=0.1,
    )


def create_wave_1dt(embed_dim: int = 768) -> VisionTransformer:
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
    )
    vit.patch_embed = PatchEmbed1D()
    vit.num_patches = vit.patch_embed.patch_shape[0]
    vit.pos_embedding = vit.pos_embedding[:, : vit.num_patches, :]
    return vit


audio_model_builders: Dict[str, Callable[[], VisionTransformer]] = {
    "spec_vit": lambda: create_spec_vit(),
    "wave_1dt": lambda: create_wave_1dt(),
}