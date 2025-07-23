# -*- coding: utf-8 -*-
"""
Audio model zoo
~~~~~~~~~~~~~~~
* 兼容 torchaudio >= 2.1 移除 AST_BASE 的变动
* 支持 Hugging Face AST/Wav2Vec2 回退
* 找不到预训练权重时仅 warnings.warn，不抛异常
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

import torch
import torchaudio

# ↓ 若想启用 Hugging Face 回退，需 pip install transformers>=4.45
try:
    from transformers import ASTModel, Wav2Vec2Model
    _HF_READY = True
except ImportError:
    _HF_READY = False

from model.vision.vit import VisionTransformer
from model.patch_embed import PatchEmbed1D


# -----------------------------------------------------------------------------#
# 基础模型构建
# -----------------------------------------------------------------------------#
def create_spec_vit(embed_dim: int = 768) -> VisionTransformer:
    """默认为 12 层 / 768 维的 ViT，用于 96-patch 频谱输入"""
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
    """将 ViT 的 PatchEmbed 替换为 1D patch-embedding，以处理波形序列"""
    vit = VisionTransformer(img_size=224, patch_size=16)
    vit.patch_embed = PatchEmbed1D()
    vit.num_patches = vit.patch_embed.patch_shape[0]
    vit.pos_embedding = vit.pos_embedding[:, : vit.num_patches, :]
    return vit


# -----------------------------------------------------------------------------#
# Helper：加载 torchaudio / Hugging Face 预训练权重
# -----------------------------------------------------------------------------#
def _ta_try_ast() -> Optional[dict]:
    """尝试在 torchaudio.pipelines 中找到任一 AST bundle 并返回 state_dict"""
    for name in ("AST_BASE", "AST", "AUDIO_SPECTROGRAM_TRANSFORMER_BASE"):
        bundle = getattr(torchaudio.pipelines, name, None)
        if bundle is None:
            continue
        try:
            return bundle.get_model().state_dict()
        except Exception as e:
            warnings.warn(f"[audio_zoo] torchaudio bundle {name} 加载失败：{e}")
    return None


def _hf_try_ast() -> Optional[dict]:
    if not _HF_READY:
        return None
    try:
        model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            trust_remote_code=False,
        )
        return model.state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] Hugging Face AST 下载失败：{e}")
        return None


def _ta_try_wav2vec() -> Optional[dict]:
    bundle = getattr(torchaudio.pipelines, "WAV2VEC2_BASE", None)
    if bundle is None:
        return None
    try:
        return bundle.get_model().state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] torchaudio WAV2VEC2_BASE 加载失败：{e}")
        return None


def _hf_try_wav2vec() -> Optional[dict]:
    if not _HF_READY:
        return None
    try:
        model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            trust_remote_code=False,
        )
        return model.state_dict()
    except Exception as e:
        warnings.warn(f"[audio_zoo] Hugging Face Wav2Vec2 下载失败：{e}")
        return None


# -----------------------------------------------------------------------------#
# 预训练模型构建
# -----------------------------------------------------------------------------#
def create_spec_vit_pretrained(embed_dim: int = 768,
                               device: torch.device | str | None = None
                               ) -> VisionTransformer:
    """频谱-ViT，尽可能加载 AST 预训练权重；失败则返回随机初始化模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_spec_vit(embed_dim).to("cpu")

    state_dict = _ta_try_ast() or _hf_try_ast()
    if state_dict is None:
        warnings.warn(
            "[audio_zoo] 未找到任何 AST 预训练权重，Spec-ViT 保持随机初始化"
        )
        return model.to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"[audio_zoo] AST 权重缺失 {len(missing)} 个参数")
    if unexpected:
        warnings.warn(f"[audio_zoo] AST 权重多余 {len(unexpected)} 个参数已忽略")

    return model.to(device)


def create_wave_1dt_pretrained(embed_dim: int = 768,
                               device: torch.device | str | None = None
                               ) -> VisionTransformer:
    """Wave-1DT，加载 Wav2Vec2 预训练"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_wave_1dt(embed_dim).to("cpu")

    state_dict = _ta_try_wav2vec() or _hf_try_wav2vec()
    if state_dict is None:
        warnings.warn(
            "[audio_zoo] 未找到任何 Wav2Vec2 预训练权重，Wave-1DT 保持随机初始化"
        )
        return model.to(device)

    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"[audio_zoo] Wav2Vec2 权重缺失 {len(missing)} 个参数")
    if unexpected:
        warnings.warn(f"[audio_zoo] Wav2Vec2 权重多余 {len(unexpected)} 个参数已忽略")

    return model.to(device)


# -----------------------------------------------------------------------------#
# 注册到工厂字典
# -----------------------------------------------------------------------------#
audio_model_builders: Dict[str, Callable[[], VisionTransformer]] = {
    "spec_vit": create_spec_vit,
    "wave_1dt": create_wave_1dt,
    "spec_vit_pretrain": create_spec_vit_pretrained,
    "wave_1dt_pretrain": create_wave_1dt_pretrained,
}

# 向后兼容旧名字
spec_vit_base = create_spec_vit
Wave1DT = create_wave_1dt
