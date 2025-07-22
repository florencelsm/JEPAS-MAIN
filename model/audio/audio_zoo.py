import torchaudio

from typing import Callable, Dict
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


def create_spec_vit_pretrained(embed_dim: int = 768) -> VisionTransformer:
    model = create_spec_vit(embed_dim)
    bundle = torchaudio.pipelines.AST_BASE
    ast = bundle.get_model()
    model.load_state_dict(ast.state_dict(), strict=False)
    return model


def create_wave_1dt_pretrained(embed_dim: int = 768) -> VisionTransformer:
    model = create_wave_1dt(embed_dim)
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav = bundle.get_model()
    state_dict = {
        k: v for k, v in wav.state_dict().items() if k in model.state_dict()
    }
    model.load_state_dict(state_dict, strict=False)
    return model


audio_model_builders: Dict[str, Callable[[], VisionTransformer]] = {
    "spec_vit": create_spec_vit,
    "wave_1dt": create_wave_1dt,
    "spec_vit_pretrain": create_spec_vit_pretrained,
    "wave_1dt_pretrain": create_wave_1dt_pretrained,
}
