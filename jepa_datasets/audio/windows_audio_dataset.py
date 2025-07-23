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
    Load paired image–audio windows for Audio-Image JEPA training.
    Parameters
    ----------
    root : str or Path
        Dataset root that contains ``processed/`` and ``metadata/``.
    stage : {"train", "val", "test"}
        Dataset split, defined in ``metadata/split.yaml``.
    use_spec : bool, default True
        If True return log-Mel spectrograms, else return raw waveform.
    shuffle : bool, default True
        Shuffle rows on load for deterministic training.
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

        # -------- read metadata --------
        meta = safe_read_table(self.root / "metadata/windows_jepa")
        with open(self.root / "metadata/split.yaml", "r", encoding="utf-8") as f:
            split = yaml.safe_load(f)
        vids = set(split[stage])
        self.df = meta[meta["vid"].isin(vids)].reset_index(drop=True)

        # optional reproducible shuffle
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

        self.use_spec = use_spec

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _to_chw(img_np: np.ndarray) -> torch.Tensor:
        """
        Guarantee (C, H, W) irrespective of how frame was stored.
        Accepts HWC, HCW, CHW or HW4 (with alpha).
        """
        assert img_np.ndim == 3, "expected 3-D frame"

        # drop alpha if present
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # find where channel-dim == 3
        if img_np.shape[0] == 3:          # already CHW
            tensor = torch.from_numpy(img_np)
        elif img_np.shape[1] == 3:        # H C W
            tensor = torch.from_numpy(img_np).permute(1, 0, 2)   # → C H W
        elif img_np.shape[2] == 3:        # H W C
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)   # → C H W
        else:
            raise ValueError("No 3-channel axis found in frame of shape "
                             f"{img_np.shape}")
        return tensor.contiguous()        # ensure dense memory

    # ------------------------------------------------------------
    # mandatory Dataset API
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # -------- load centre frame --------
        stack = np.load(self.root / row["frame_stack"], mmap_mode="r")
        frame = stack[stack.shape[0] // 2].copy()               # numpy array
        img = self._to_chw(frame).float()                       # (3,H,W)

        # -------- load audio --------
        if self.use_spec:
            spec_key = "mel244_path" if "mel244_path" in row else "mel224_path"
            spec = np.load(self.root / row[spec_key], mmap_mode="r").copy()
            # spec shape: (F, T)  →  fake RGB by repeating
            audio = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1).float()
        else:                                                   # raw waveform
            with sf.SoundFile(self.root / row["wav_path"]) as f:
                f.seek(int(row["start_sample"]))
                wav = f.read(int(row["num_samples"]))
            audio = torch.from_numpy(wav).unsqueeze(0).float()  # (1, L)

        # -------- negatives --------
        neg_pool: List[int] = json.loads(row["neg_xvid"])
        neg_intra = int(row["neg_intra"])

        return {
            "image": img,                                       # (3, H, W)
            "audio": audio,                                     # (C, …)
            "neg_pool": torch.tensor(neg_pool, dtype=torch.long),
            "neg_intra": torch.tensor(neg_intra, dtype=torch.long),
        }
