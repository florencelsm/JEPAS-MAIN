import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from transformers import AutoImageProcessor, AutoFeatureExtractor
from pretrained.audio_models import load_waveform, pad_waveform

class AudioImageDataset(Dataset):
    def __init__(self,
                 root: Path,
                 sample_rate: int = 16000,
                 min_waveform_len: int= 400) -> None:
        self.wave_dir = root / "waveforms"
        self.image_dir = root / "images"
        self.sample_rate = sample_rate
        self.items: List[str] = []
        self.min_waveform_len: int = min_waveform_len

        self.audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.img_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        for p in self.wave_dir.glob("*.wav"):
            if (self.image_dir / f"{p.stem}.jpg").exists():
                self.items.append(p.stem)
                wf = load_waveform(p, sample_rate=self.sample_rate)
                if wf.numel() > self.max_len:
                    self.max_len = int(wf.numel())

    def __len__(self) -> int: 
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        stem = self.items[idx]
        wave_path = self.wave_dir / f"{stem}.wav"
        waveform = load_waveform(wave_path,
                                 sample_rate=self.sample_rate,
                                 min_waveform_len=self.min_waveform_len)
        waveform = pad_waveform(waveform,
                                min_waveform_len=self.min_waveform_len)
        waveform = self.audio_processor(waveform, 
                                        sampling_rate=self.sample_rate, 
                                        return_tensors="pt").input_values
        img_path = self.image_dir / f"{stem}.jpg"
        image = self.img_processor(Image.open(img_path), return_tensors="pt").pixel_values
        #bounding_box = bouding_box (load them properly)
        return {'waveform': waveform,
                'image':image,}
                # 'bounding_box', bounding_box}