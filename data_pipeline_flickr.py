#!/usr/bin/env python3
"""
End‑to‑end data‑prep pipeline for Flickr‑SoundNet style dataset.

Folder assumptions
------------------
raw_root/
├── dataset/
│   ├── audio/vid_00001.flac
│   └── video/vid_00001.mp4
└── processed/               # will be created automatically
    ├── video_h264/
    ├── audio_wav16k/
    ├── frames_224/
    ├── mel_128/
    └── features/            # other derived files

CLI entry points
----------------
python data_pipeline.py remux      # container standardisation
python data_pipeline.py window     # ±5‑frame & Mel extraction (needs remux stage done)
python data_pipeline.py negpool    # build offline hard‑negative table
python data_pipeline.py index      # parquet / csv indices

Each sub‑command is idempotent; already‑processed files会被跳过。
"""

import argparse
import subprocess
import shutil
import json
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict
import math
import csv
import multiprocessing as mp

import numpy as np
import soundfile as sf
import librosa
import cv2
import pandas as pd

##############################
# Parquet‑safe helpers       #
##############################

def safe_to_parquet(df: pd.DataFrame, path: Path):
    """Save DataFrame to Parquet when a supported engine is available, otherwise fall back to CSV.
    A warning is printed so that the user can decide later to install `pyarrow` or `fastparquet`."""
    try:
        df.to_parquet(path)
    except (ImportError, ValueError, OSError) as e:
        alt_path = path.with_suffix('.csv')
        df.to_csv(alt_path, index=False)
        print(f"[WARN] Parquet engine unavailable ({e}). Data written to CSV → {alt_path}")


def safe_read_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet, falling back to CSV when Parquet isn't supported."""
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, OSError, FileNotFoundError):
        alt_path = path.with_suffix('.csv')
        return pd.read_csv(alt_path)

##############################
# General helpers            #
##############################

def run(cmd: List[str]) -> None:
    """Run external command, raise on non‑zero."""
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def video_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out) if out else 0.0


def audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate if info.frames else 0.0

##############################
# Stage 1: container standardisation
##############################

BROKEN_LOG = "metadata/broken_videos.txt"


def is_valid_video(path: Path) -> bool:
    """Quick probe to detect corrupt mp4 (e.g., moov atom missing)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
        "stream=codec_type", "-of", "csv=p=0", str(path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def remux(args):
    raw_video_dir = Path(args.dataset)/"video"
    raw_audio_dir = Path(args.dataset)/"audio"
    out_video_dir = Path(args.out_root)/"processed/video_h264"
    out_audio_dir = Path(args.out_root)/"processed/audio_wav16k"
    ensure_dir(out_video_dir)
    ensure_dir(out_audio_dir)
    ensure_dir(Path(args.out_root)/"metadata")

    broken = []
    vid_paths = sorted(raw_video_dir.glob("*.mp4"))
    for vp in vid_paths:
        vid_id = vp.stem
        # ---------- video ---------- #
        if not is_valid_video(vp):
            print(f"[WARN] Corrupt video skipped: {vp.name}")
            broken.append(vid_id)
            continue
        out_vp = out_video_dir/f"{vid_id}.mp4"
        if not out_vp.exists():
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(vp),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-r", "25",
                "-an", str(out_vp)
            ]
            run(cmd)
        # ---------- audio ---------- #
        ap = raw_audio_dir/f"{vid_id}.flac"
        if not ap.exists():
            print(f"[WARN] Missing audio for {vid_id}, skipping")
            continue
        out_ap = out_audio_dir/f"{vid_id}.wav"
        if not out_ap.exists():
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(ap),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out_ap)
            ]
            run(cmd)

    # 记录坏文件
    if broken:
        with open(Path(args.out_root)/BROKEN_LOG, "w") as f:
            f.write("\n".join(broken))
        print(f"\n[INFO] {len(broken)} corrupt videos logged to {BROKEN_LOG}\n")
    else:
        print("[INFO] No corrupt video detected.")

##############################
# Stage 2: feature derivation (±5‑frame stack + Mel128)
##############################

WINDOW_LEN = 2.0   # seconds
STRIDE      = 1.0   # seconds
N_MELS      = 128
N_FFT       = 512
HOP         = 160  # 10 ms
WIN_LEN     = 400


def extract_frames(vid_path: Path, center_sec: float, out_dir: Path, vid_id: str, win_idx: int):
    """Return relative npy path; skip decoding if file exists."""
    out_path = out_dir / f"{vid_id}_w{win_idx:04d}_stack.npy"
    if out_path.exists():
        # already done earlier run
        return out_path.relative_to(out_dir.parent.parent)

    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    center_f = int(center_sec * fps)
    # read sequentially to reduce random seek cost
    start_f = max(center_f - 5, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(11):
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame.transpose(2, 0, 1))
    cap.release()
    stack = np.stack(frames, axis=0)
    ensure_dir(out_dir)
    np.save(out_path, stack)
    return out_path.relative_to(out_dir.parent.parent)


def extract_mel(audio_path: Path, start_sec: float, out_dir: Path, vid_id: str, win_idx: int):
    out_path = out_dir / f"{vid_id}_w{win_idx:04d}.npy"
    if out_path.exists():
        return out_path.relative_to(out_dir.parent.parent)

    y, sr = sf.read(str(audio_path), dtype='float32')
    start = int(start_sec * sr)
    end = start + int(WINDOW_LEN * sr)
    y_seg = y[start:end]
    if len(y_seg) < int(WINDOW_LEN * sr):
        y_seg = np.pad(y_seg, (0, int(WINDOW_LEN * sr) - len(y_seg)))
    S = librosa.feature.melspectrogram(
        y=y_seg,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LEN,
        n_mels=N_MELS,
        center=False)
    S = np.log(S + 1e-6).astype(np.float32)
    ensure_dir(out_dir)
    np.save(out_path, S)
    return out_path.relative_to(out_dir.parent.parent)


def process_video(args_tuple):
    """Helper for multiprocessing: (vp_path, skip_vids, frames_dir, mel_dir)"""
    vp, skip_vids, frames_dir, mel_dir = args_tuple
    records_local = []
    vid_id = vp.stem
    if vid_id in skip_vids:
        return records_local
    ap = vp.parent.parent/"audio_wav16k"/f"{vid_id}.wav"
    if not ap.exists():
        return records_local
    dur = video_duration_sec(vp)
    if dur < WINDOW_LEN:
        return records_local
    n_win = int(math.floor((dur - WINDOW_LEN) / STRIDE)) + 1
    for w_idx in range(n_win):
        t0 = w_idx * STRIDE
        t1 = t0 + WINDOW_LEN
        center = (t0 + t1) / 2
        frame_rel = extract_frames(vp, center, frames_dir, vid_id, w_idx)
        mel_rel = extract_mel(ap, t0, mel_dir, vid_id, w_idx)
        records_local.append({
            "vid": vid_id,
            "win_idx": w_idx,
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "frame_stack": str(frame_rel),
            "mel_path": str(mel_rel)
        })
    return records_local


def make_windows(args):
    video_dir = Path(args.out_root)/"processed/video_h264"
    frames_dir = Path(args.out_root)/"processed/frames_224"
    mel_dir = Path(args.out_root)/"processed/mel_128"
    ensure_dir(frames_dir)
    ensure_dir(mel_dir)

    skip_vids = set()
    broken_path = Path(args.out_root)/BROKEN_LOG
    if broken_path.exists():
        skip_vids = set(broken_path.read_text().splitlines())

    tasks = [(vp, skip_vids, frames_dir, mel_dir) for vp in sorted(video_dir.glob("*.mp4"))]

    workers = getattr(args, "workers", 4)
    with mp.Pool(processes=workers) as pool:
        all_records = pool.map(process_video, tasks)

    # flatten and drop empty lists
    records = [r for sub in all_records for r in sub]

    idx_path = Path(args.out_root)/"metadata/windows_raw.parquet"
    ensure_dir(idx_path.parent)
    safe_to_parquet(pd.DataFrame(records), idx_path)
    print(f"[INFO] Saved raw window index → {idx_path if idx_path.exists() else idx_path.with_suffix('.csv')}  (total {len(records)} windows)")

##############################
# Stage 3: offline hard‑negative pool
##############################


def hash_mel(path: Path) -> str:
    arr = np.load(path, mmap_mode='r')
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return h[:32]  # truncate to 128‑bit hex


def build_neg_pool(args):
    mel_dir = Path(args.out_root)/"processed/mel_128"
    index_path = Path(args.out_root)/"metadata/windows_raw.parquet"
    df = safe_read_parquet(index_path)

    # build hash buckets (LSH style)
    buckets: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        mel_path = mel_dir/Path(row["mel_path"]).name
        h = hash_mel(mel_path)
        bucket_key = h[:6]   # first 24‑bit as bucket
        buckets.setdefault(bucket_key, []).append(idx)

    # for each sample pick 3 cross‑video neg + 1 intra‑video neg (>=3s apart)
    neg_xvid = [[] for _ in range(len(df))]
    neg_intra = [None]*len(df)

    vid_to_indices: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        vid_to_indices.setdefault(row.vid, []).append(idx)

    rng = random.Random(42)

    for idx, row in df.iterrows():
        # cross‑video hard neg: same bucket different vid
        mel_path = mel_dir/Path(row["mel_path"]).name
        h = hash_mel(mel_path)
        bucket_key = h[:6]
        candidates = [i for i in buckets[bucket_key] if df.loc[i, 'vid'] != row.vid]
        if len(candidates) < 3:
            candidates += rng.sample(range(len(df)), 3)
        neg_xvid[idx] = rng.sample(candidates, 3)
        # intra‑video neg
        intra = [i for i in vid_to_indices[row.vid] if abs(df.loc[i,'t0'] - row.t0) > 3.0]
        if intra:
            neg_intra[idx] = rng.choice(intra)

    df["neg_xvid"] = [json.dumps(lst) for lst in neg_xvid]
    df["neg_intra"] = neg_intra

    out_path = Path(args.out_root)/"metadata/windows_neg.parquet"
    safe_to_parquet(df, out_path)
    print(f"[INFO] Saved negative‑augmented index → {out_path if out_path.exists() else out_path.with_suffix('.csv')}")

##############################
# Stage 4: final master & split
##############################


def make_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    vids = sorted(df.vid.unique())
    rng = random.Random(2025)
    rng.shuffle(vids)
    n = len(vids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": vids[:n_train],
        "val": vids[n_train:n_train + n_val],
        "test": vids[n_train + n_val:]
    }


def build_master(args):
    video_dir = Path(args.out_root)/"processed/video_h264"
    audio_dir = Path(args.out_root)/"processed/audio_wav16k"
    rows = []
    for vp in sorted(video_dir.glob("*.mp4")):
        vid = vp.stem
        dur_v = video_duration_sec(vp)
        ap = audio_dir/f"{vid}.wav"
        dur_a = audio_duration_sec(ap)
        offset_ms = int((dur_a - dur_v) * 1000)
        rows.append({"vid_id": vid, "duration": round(dur_v, 3), "offset_ms": offset_ms})
    out_csv = Path(args.out_root)/"metadata/master.csv"
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Master CSV written to {out_csv}")

    # generate split
    df_index = Path(args.out_root)/"metadata/windows_neg.parquet"
    df = safe_read_parquet(df_index)
    split = make_split(df)
    split_path = Path(args.out_root)/"metadata/split.yaml"
    with open(split_path, "w") as f:
        import yaml
        yaml.dump(split, f)
    print(f"[INFO] Dataset split saved to {split_path}")

##############################
# CLI glue
##############################

def main():
    parser = argparse.ArgumentParser(description="Data preparation pipeline")
    parser.add_argument("cmd", choices=["remux", "window", "negpool", "index"], help="Stage to run")
    parser.add_argument("--dataset", type=str, default="dataset", help="raw dataset root")
    parser.add_argument("--out_root", type=str, default=".", help="project root holding processed/")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of parallel worker processes for the window stage")

    args = parser.parse_args()

    if args.cmd == "remux":
        remux(args)
    elif args.cmd == "window":
        make_windows(args)
    elif args.cmd == "negpool":
        build_neg_pool(args)
    elif args.cmd == "index":
        build_master(args)


if __name__ == "__main__":
    main()
