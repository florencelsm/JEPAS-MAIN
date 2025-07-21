#!/usr/bin/env python3
"""
postprocess_for_jepa.py
~~~~~~~~~~~~~~~~~~~~~~~
Second-pass processing to adapt the already-generated Flickr-SoundNet style
dataset to the *single-direction JEPA* ablation experiments:

  • Resize 128-Mel to 224×224 (bilinear)  ➜ processed/mel_224/
  • Append wave–slice offsets (2-s)      ➜ windows_jepa.parquet
  • Re-emit train/val/test split YAML    ➜ metadata/split.yaml   (unchanged ratio)

Run examples
------------
python postprocess_for_jepa.py resize
python postprocess_for_jepa.py index   --train_ratio 0.7 --val_ratio 0.15
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import cv2
import yaml

# ---------- CONFIG ----------
SR             = 16_000          # wav2vec family expects 16 kHz
WIN_LEN_SEC    = 2.0
WIN_SAMPLES    = int(SR * WIN_LEN_SEC)
MEL_TARGET     = 224
PARQUET_IN     = "metadata/windows_neg"
PARQUET_OUT    = "metadata/windows_jepa"
MEL224_DIRNAME = "processed/mel_224"
# ----------------------------

# -------- parquet helpers (same as original) ----------
def safe_to_parquet(df: pd.DataFrame, stem: Path):
    """
    Save DataFrame to stem.{parquet|csv}.
    如果系统缺少 arrow/fastparquet，则自动写 CSV。
    """
    parquet_path = stem.with_suffix(".parquet")
    csv_path     = stem.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path)
        if csv_path.exists():
            csv_path.unlink()
    except (ImportError, ValueError, OSError):
        df.to_csv(csv_path, index=False)
        print(f"[WARN] Parquet engine unavailable → wrote CSV to {csv_path}")


def safe_read_table(stem: Path) -> pd.DataFrame:
    """
    读取 stem.{parquet|csv}，优先用 parquet。
    """
    parquet_path = stem.with_suffix(".parquet")
    csv_path     = stem.with_suffix(".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[WARN] Failed to read parquet ({e}); fallback to CSV.")
    return pd.read_csv(csv_path)
# ------------------------------------------------------


def resize_mel_numpy(arr: np.ndarray, target: int = MEL_TARGET) -> np.ndarray:
    """arr: (128, T) float32  →  (target, target) float32"""
    h, w = arr.shape
    # pad / crop width to target first
    if w < target:
        pad = np.zeros((h, target - w), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    elif w > target:
        left = (w - target) // 2
        arr = arr[:, left:left + target]

    # resize height to target via bilinear
    arr = cv2.resize(arr, (target, target), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def _worker_resize(args_tuple):
    mel128_path, mel224_dir = args_tuple
    mel128 = np.load(mel128_path, mmap_mode="r")
    mel224 = resize_mel_numpy(mel128)        # (224,224)
    out_path = mel224_dir / mel128_path.name
    np.save(out_path, mel224)
    return out_path.name                   # store relative name for index


def stage_resize(root: Path, workers: int):
    mel128_dir = root / "processed/mel_128"
    mel224_dir = root / MEL224_DIRNAME
    mel224_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(mel128_dir.glob("*.npy"))
    todo = [p for p in paths if not (mel224_dir / p.name).exists()]
    print(f"[INFO] Need to resize {len(todo)} / {len(paths)} spectrograms → 224×224")

    with mp.Pool(processes=workers) as pool:
        list(pool.imap_unordered(_worker_resize,
                                 [(p, mel224_dir) for p in todo],
                                 chunksize=64))

    print(f"[OK] All Mel-224 saved to {mel224_dir}.")


def stage_index(root: Path, train_ratio: float, val_ratio: float):
    df = safe_read_table(root / PARQUET_IN)

    # append Mel-224 relative path
    df["mel224_path"] = df["mel_path"].apply(
        lambda p: str(Path(MEL224_DIRNAME) / Path(p).name))

    # append wav path + start_sample
    df["wav_path"] = df["vid"].apply(
        lambda vid: str(Path("processed/audio_wav16k") / f"{vid}.wav"))
    df["start_sample"] = (df["t0"] * SR).astype(int)
    df["num_samples"] = WIN_SAMPLES

    # reorder / keep useful cols
    keep = [
        "vid", "win_idx", "t0", "t1",
        "frame_stack", "mel224_path",
        "wav_path", "start_sample", "num_samples",
        "neg_xvid", "neg_intra"
    ]
    df_final = df[keep]
    out_stem = root / PARQUET_OUT          # ❹ 仍用 stem
    safe_to_parquet(df_final, out_stem)    # ❺
    print(f"[OK] Wrote JEPA index to {out_stem}.parquet / .csv  ({len(df_final)} samples)")

    # make new split (video-level, non-overlap)
    vids = sorted(df_final.vid.unique())
    rng = np.random.default_rng(2025)
    rng.shuffle(vids)
    n = len(vids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    split = {
        "train": vids[:n_train],
        "val":   vids[n_train:n_train + n_val],
        "test":  vids[n_train + n_val:]
    }
    with open(root / "metadata/split.yaml", "w") as f:
        yaml.safe_dump(split, f)
    print("[OK] split.yaml updated.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["resize", "index"], help="stage to run")
    p.add_argument("--out_root", default=".", type=str, help="project root")
    p.add_argument("--workers", default=4, type=int, help="CPU processes for resize")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    args = p.parse_args()
    root = Path(args.out_root)

    if args.cmd == "resize":
        stage_resize(root, args.workers)
    elif args.cmd == "index":
        stage_index(root, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
