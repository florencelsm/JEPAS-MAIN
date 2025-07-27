# windows_audio_dataset.py  (patched)
from pathlib import Path
from typing import List, Literal, Union
import json, yaml
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
from postprocess_for_jepa import safe_read_table


class WindowsAudioImageDataset(Dataset):
    """
    Paired image–audio windows for Audio-Image JEPA.
    """

    # ----------------------- init ----------------------------------------
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

        # -------- read metadata --------
        meta = safe_read_table(self.root / "metadata/windows_jepa")
        with open(self.root / "metadata/split.yaml", "r", encoding="utf-8") as f:
            vids_in_split = set(yaml.safe_load(f)[stage])

        self.df = meta[meta["vid"].isin(vids_in_split)].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(
                f"No samples found for stage '{stage}' in {self.root}. "
                "Verify that the dataset is prepared correctly."
            )

        if shuffle:
            self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

        self.use_spec = use_spec

    # ----------------------- helpers -------------------------------------
    @staticmethod
    def _to_chw(img_np: np.ndarray) -> torch.Tensor:
        """Ensure (C,H,W) layout."""
        assert img_np.ndim == 3, "expected 3-D array"
        if img_np.shape[-1] == 4:               # drop alpha
            img_np = img_np[..., :3]

        if img_np.shape[0] == 3:
            t = torch.from_numpy(img_np)
        elif img_np.shape[1] == 3:
            t = torch.from_numpy(img_np).permute(1, 0, 2)
        elif img_np.shape[2] == 3:
            t = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            raise ValueError(f"Cannot locate channel dim in shape {img_np.shape}")
        return t.contiguous()

    # ----------------------- mandatory API -------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # -------- load image --------
        if "frame_stack" in row:
            # legacy format：11-frame npy stack
            stack = np.load(self.root / row["frame_stack"], mmap_mode="r")
            frame = stack[stack.shape[0] // 2].copy()
        else:
            frame = np.load(self.root / row["img_path"], mmap_mode="r").copy()
        img = self._to_chw(frame).float()                       # (3,H,W)

        # -------- load audio --------
        if self.use_spec:
            spec = np.load(self.root / row["mel224_path"], mmap_mode="r").copy()
            audio = (
                torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1).float()
            )                                                   # (3,F,T)
        else:                                                   # raw wav
            with sf.SoundFile(self.root / row["wav_path"]) as f:
                f.seek(int(row["start_sample"]))
                wav = f.read(int(row["num_samples"]))
            audio = torch.from_numpy(wav).unsqueeze(0).float()  # (1,L)

        # -------- negatives --------
        neg_pool: List[int] = json.loads(row["neg_xvid"])
        neg_intra = int(row["neg_intra"])

        return {
            "image": img,
            "audio": audio,
            "neg_pool": torch.tensor(neg_pool, dtype=torch.long),
            "neg_intra": torch.tensor(neg_intra, dtype=torch.long),
        }
