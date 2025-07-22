from pathlib import Path
from typing import List, Literal, Union
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf

from postprocess_for_jepa import safe_read_table


class WindowsAudioImageDataset(Dataset):
    """Dataset loading paired image and audio segments using the precomputed
    ``windows_jepa`` metadata table.
    Parameters
    ----------
    root : str or Path
        Path to the dataset root containing ``processed/`` and ``metadata/``.
    stage : {"train", "val", "test"}
        Dataset split to use. Splits are defined in ``metadata/split.yaml``.
    use_spec : bool, default True
        If ``True`` return log-Mel spectrograms. Otherwise return raw waveform
        segments using the ``wav_path`` and ``start_sample`` fields.
    shuffle : bool, default True
        Whether to shuffle the table on load for deterministic training.
    """

    def __init__(
        self,
        root: Union[str, Path],
        stage: Literal["train", "val", "test"],
        *,
        use_spec: bool = True,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        meta = safe_read_table(self.root / "metadata/windows_jepa")
        with open(self.root / "metadata/split.yaml", "r", encoding="utf-8") as f:
            split = yaml.safe_load(f)
        vids = set(split[stage])
        self.df = meta[meta["vid"].isin(vids)].reset_index(drop=True)
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)
        self.use_spec = use_spec

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frame = np.load(self.root / row["frame_stack"], mmap_mode="r")
        img = torch.from_numpy(frame)
        if self.use_spec:
            spec = np.load(self.root / row["mel224_path"], mmap_mode="r")
            audio = torch.from_numpy(spec)
        else:
            with sf.SoundFile(self.root / row["wav_path"]) as f:
                f.seek(int(row["start_sample"]))
                wav = f.read(int(row["num_samples"]))
            audio = torch.from_numpy(wav).unsqueeze(0)
        neg_pool: List[int] = json.loads(row["neg_xvid"])
        neg_intra = int(row["neg_intra"])
        return {
            "image": img,
            "audio": audio,
            "neg_pool": torch.tensor(neg_pool, dtype=torch.long),
            "neg_intra": torch.tensor(neg_intra, dtype=torch.long),
        }