from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .windows_audio_dataset import WindowsAudioImageDataset
from torch.nn.utils.rnn import pad_sequence
import torch


class WindowsAudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: Optional[int],
        *,
        use_spec: bool = True,
        shuffle: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.use_spec = use_spec
        self.shuffle = shuffle
        self.image_size = image_size
        self.train_dataset: Optional[WindowsAudioImageDataset] = None
        self.val_dataset: Optional[WindowsAudioImageDataset] = None
        self.test_dataset: Optional[WindowsAudioImageDataset] = None

    @staticmethod
    def collate_fn(batch):
        """Pads ``neg_pool`` entries to equal length and stacks tensors."""
        images = torch.stack([b["image"] for b in batch])
        audios = torch.stack([b["audio"] for b in batch])
        neg_pools = [b["neg_pool"] for b in batch]
        neg_pool = pad_sequence(neg_pools, batch_first=True, padding_value=-1)
        neg_intra = torch.stack([b["neg_intra"] for b in batch])
        return {
            "image": images,
            "audio": audios,
            "neg_pool": neg_pool,
            "neg_intra": neg_intra,
        }
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WindowsAudioImageDataset(
            self.dataset_root,
            "train",
            use_spec=self.use_spec,
            shuffle=self.shuffle,
            image_size=self.image_size,
        )
        self.val_dataset = WindowsAudioImageDataset(
            self.dataset_root,
            "val",
            use_spec=self.use_spec,
            shuffle=False,
            image_size=self.image_size,
        )
        self.test_dataset = WindowsAudioImageDataset(
            self.dataset_root,
            "test",
            use_spec=self.use_spec,
            shuffle=False,
            image_size=self.image_size,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError(
                f"No training samples found in {self.dataset_root}. "
                "Check DATASET_PATH or preprocessing steps."
            )

    def _loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)


def create_windows_audio_datamodule(cfg: Dict[str, Any]) -> WindowsAudioDataModule:
    dataset_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    return WindowsAudioDataModule(
        dataset_root=dataset_cfg["DATASET_PATH"],
        batch_size=exp_cfg["BATCH_SIZE"],
        num_workers=exp_cfg["NUM_WORKERS"],
        pin_memory=exp_cfg["PIN_MEMORY"],
        persistent_workers=exp_cfg["PERSISTENT_WORKERS"],
        prefetch_factor=exp_cfg["PREFETCH_FACTOR"],
        use_spec=dataset_cfg.get("USE_SPEC", True),
        shuffle=dataset_cfg.get("SHUFFLE_DATASET", True),
        image_size=dataset_cfg.get("IMAGE_SIZE", 224),
    )
