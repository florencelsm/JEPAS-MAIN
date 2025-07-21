from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from .audio_dataset import AudioImageDataset


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: Optional[int],
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle

        self.train_dataset: Optional[AudioImageDataset] = None
        self.val_dataset: Optional[AudioImageDataset] = None
        self.test_dataset: Optional[AudioImageDataset] = None

        self.transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = AudioImageDataset(
            self.dataset_path,
            stage="train",
            shuffle=self.shuffle,
            transform=self.transform,
        )
        self.val_dataset = AudioImageDataset(
            self.dataset_path,
            stage="val",
            shuffle=False,
            transform=self.transform,
        )
        self.test_dataset = AudioImageDataset(
            self.dataset_path,
            stage="test",
            shuffle=False,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )


def create_audio_datamodule(audio_config: Dict[str, Any]) -> AudioDataModule:
    dataset_cfg = audio_config["dataset"]
    exp_cfg = audio_config["experiment"]

    return AudioDataModule(
        dataset_path=dataset_cfg["DATASET_PATH"],
        batch_size=exp_cfg["BATCH_SIZE"],
        num_workers=exp_cfg["NUM_WORKERS"],
        pin_memory=exp_cfg["PIN_MEMORY"],
        persistent_workers=exp_cfg["PERSISTENT_WORKERS"],
        prefetch_factor=exp_cfg["PREFETCH_FACTOR"],
        shuffle=dataset_cfg.get("SHUFFLE_DATASET", True),
    )