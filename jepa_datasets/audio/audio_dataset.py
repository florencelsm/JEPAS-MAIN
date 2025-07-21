from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AudioImageDataset(Dataset):
    """Dataset returning paired image, spectrogram and optional waveform."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "val", "test"],
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()

        self.shuffle = shuffle
        self.transform = transform or transforms.ToTensor()

        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)

        self.data_dir = dataset_path / stage
        self.image_paths: List[Path] = sorted(self.data_dir.glob("*_frame.jpg"))
        self.spec_paths: List[Path] = sorted(self.data_dir.glob("*_spec.npy"))
        self.wav_paths: List[Path] = sorted(self.data_dir.glob("*.wav"))

        if self.shuffle:
            rng = np.random.default_rng(0)
            indices = np.arange(len(self.image_paths))
            rng.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.spec_paths = [self.spec_paths[i] for i in indices]
            if len(self.wav_paths) == len(self.image_paths):
                self.wav_paths = [self.wav_paths[i] for i in indices]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.image_paths[index]).convert("RGB")
        image_tensor = self.transform(img)

        spec = np.load(self.spec_paths[index])
        spec_tensor = torch.from_numpy(spec)

        wav_tensor = None
        if len(self.wav_paths) == len(self.image_paths):
            wav = np.load(self.wav_paths[index]) if self.wav_paths[index].suffix == ".npy" else None
            if wav is not None:
                wav_tensor = torch.from_numpy(wav).float()

        return {
            "image": image_tensor,
            "spec": spec_tensor,
            "wave": wav_tensor,
        }