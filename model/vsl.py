import torch
import torch.nn as nn
from utils.bbox_utils import box_cxcywh_to_xyxy

class ImageAudioEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_expand=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_expand),
                                 nn.GELU(),
                                 nn.Linear(dim * mlp_expand, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, src: torch.Tensor, src2: torch.Tensor):
        src_attended, _ = self.cross_attn(src, src2, src2)
        src = self.norm1(src + src_attended)
        src_mlp = self.mlp(src)
        src = self.norm2(src + src_mlp)
        return src

class ImageAudioEncoder(nn.Module):
    def __init__(self, dim, num_layers=6):
        super().__init__()
        self.encoder_layers = nn.ModuleList([ImageAudioEncoderLayer(dim) for _ in range(num_layers)])
    
    def forward(self, src: torch.Tensor, src2: torch.Tensor):
        for layer in self.encoder_layers:
            src = layer(src, src2)
        return src

class BoxAudioImageDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_expand=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_expand),
                                 nn.GELU(),
                                 nn.Linear(dim * mlp_expand, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, trgt: torch.Tensor, encoder_output: torch.Tensor, box_queries: torch.Tensor):
        trgt_attended, _ = self.self_attn(trgt + box_queries, trgt + box_queries, trgt)
        trgt = self.norm1(trgt + trgt_attended)
        trgt_attended, _ = self.cross_attn(trgt + box_queries, encoder_output, encoder_output)
        trgt = self.norm2(trgt + trgt_attended)
        trgt_mlp = self.mlp(trgt)
        trgt = self.norm3(trgt + trgt_mlp)
        return trgt

class BoxAudioImageDecoder(nn.Module):
    def __init__(self, dim, num_layers=6):
        super().__init__()
        self.decoder_layers = nn.ModuleList([BoxAudioImageDecoderLayer(dim) for _ in range(num_layers)])
    
    def forward(self, box_queries: torch.Tensor, encoder_output: torch.Tensor):
        bs = encoder_output.shape[0]
        box_queries = box_queries.unsqueeze(0).repeat(bs, 1, 1)
        trgt = torch.zeros_like(box_queries)
        for layer in self.decoder_layers:
            trgt = layer(trgt, encoder_output, box_queries)
        return trgt

class VisualSoundLocalizer(nn.Module):
    def __init__(self,
                 image_encoder,
                 audio_encoder,
                 num_object_queries=5,
                 freeze_backbones=False) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        
        if freeze_backbones:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            for p in self.audio_encoder.parameters():
                p.requires_grad_(False)
        
        dim = audio_encoder.config.hidden_size
        self.image_audio_encoder = ImageAudioEncoder(dim)
        self.box_audio_image_decoder = BoxAudioImageDecoder(dim)
        self.box_queries = nn.Embedding(num_object_queries, dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.bbox_head = nn.Sequential(nn.Linear(dim, dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(dim, dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(dim, 4))

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_encoder(pixel_values=image).last_hidden_state
        img_emb = img_emb[:, 1:, :]
        aud_emb = self.audio_encoder(audio).last_hidden_state
        encoder_output = self.image_audio_encoder(img_emb, aud_emb)
        decoder_output = self.avg_pool(encoder_output.transpose(-1, -2).contiguous()).squeeze(-1)
        boxes = self.bbox_head(decoder_output).sigmoid()
        return {'pred_boxes': boxes}
    
    @torch.no_grad()
    def postprocess(self, output, image_size) -> torch.Tensor:
        bboxes = output["pred_boxes"]
        bboxes = box_cxcywh_to_xyxy(bboxes)
        img_h, img_w = image_size
        scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=bboxes.device)
        bboxes = bboxes * scale_factor
        return {'pred_boxes': bboxes}