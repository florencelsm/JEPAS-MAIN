#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_windows_jepa_vggss.py  (root-level version)

• 默认读取   ./vgg_ss_processed_data/  作为 --src
• 默认读取   ./vggss/vggss.json        作为元数据
• 默认输出到 ./vggss_jepa/             (可用 --dst 自定义)

生成：
    vggss_jepa/
      ├─ processed/
      │    ├─ img_npy/
      │    ├─ wav_npy/
      │    └─ mel224_npy/
      └─ metadata/
           ├─ windows_jepa.parquet
           └─ split.yaml
"""
import argparse, os, random, json, yaml, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------------------------- #
# Argument parsing                                                           #
# -------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("./vgg_ss_processed_data"),
                   help="Folder produced by previous download/convert scripts.")
    p.add_argument("--json", type=Path, default=Path("./vggss/vggss.json"),
                   help="Path to vggss.json metadata file.")
    p.add_argument("--dst", type=Path, default=Path("./vggss_jepa"),
                   help="Output root folder to create.")
    p.add_argument("--neg-pool-size", type=int, default=30,
                   help="#cross-video negatives per sample.")
    p.add_argument("--workers", type=int, default=8,
                   help="ThreadPool size for image/spec conversion.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# -------------------------------------------------------------------------- #
# Helpers                                                                    #
# -------------------------------------------------------------------------- #
def jpg_to_npy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    np.save(dst, arr)

def png_to_npy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    arr = np.asarray(Image.open(src).convert("L"), dtype=np.float32) / 255.0
    np.save(dst, arr)

def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)   # hard-link (same device, POSIX)
    except (AttributeError, OSError):
        shutil.copy2(src, dst)  # fallback


# -------------------------------------------------------------------------- #
# Main                                                                       #
# -------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    SRC: Path = args.src.resolve()
    DST: Path = args.dst.resolve()
    JSON: Path = args.json.resolve()

    if not SRC.exists():
        raise FileNotFoundError(f"--src folder not found: {SRC}")
    if not JSON.is_file():
        raise FileNotFoundError(f"--json file not found: {JSON}")

    (DST / "processed").mkdir(parents=True, exist_ok=True)
    (DST / "metadata").mkdir(exist_ok=True)

    # -------------------------- load json metadata ----------------------- #
    with open(JSON, "r") as f:
        raw_entries: List[Dict] = json.load(f)
    raw_map = {e["file"]: e for e in raw_entries}

    # -------------------------- discover files --------------------------- #
    img_dir   = SRC / "images"
    wav_dir   = SRC / "waveforms"
    spec_dir  = SRC / "spectrograms"

    jpgs  = {p.stem: p for p in img_dir.glob("*.jpg")}
    wavs  = {p.stem: p for p in wav_dir.glob("*.npy")}
    pngs  = {p.stem: p for p in spec_dir.glob("*.png")}

    print(f"Discovered {len(jpgs)} images  |  {len(wavs)} waveforms  |  {len(pngs)} specs")

    # --------------------- convert jpg/png to npy ------------------------ #
    img_npy_dir  = DST / "processed" / "img_npy"
    mel_npy_dir  = DST / "processed" / "mel224_npy"
    wav_npy_dir  = DST / "processed" / "wav_npy"   # hard-link/复制

    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for uid, jpg_p in jpgs.items():
            tasks.append(ex.submit(jpg_to_npy, jpg_p, img_npy_dir / f"{uid}.npy"))
        for uid, png_p in pngs.items():
            tasks.append(ex.submit(png_to_npy, png_p, mel_npy_dir / f"{uid}.npy"))
        for _ in tqdm(as_completed(tasks), total=len(tasks), ncols=80, desc="Converting"):
            pass

    # --------------------- build metadata rows --------------------------- #
    rows = []
    for uid, wav_src in wavs.items():
        if uid not in jpgs or uid not in pngs:
            print(f"[warn] trio missing for {uid}, skip")
            continue

        wav_dst = wav_npy_dir / f"{uid}.npy"
        hardlink_or_copy(wav_src, wav_dst)
        wav_np = np.load(wav_dst, mmap_mode="r")

        entry = raw_map.get(uid)
        if entry is None:
            print(f"[warn] {uid} not found in metadata json, skip")
            continue

        rows.append({
            "uid": uid,
            "vid": uid.split("_")[0],
            "img_path": str((img_npy_dir / f"{uid}.npy").relative_to(DST)),
            "wav_path": str((wav_npy_dir / f"{uid}.npy").relative_to(DST)),
            "mel224_path": str((mel_npy_dir / f"{uid}.npy").relative_to(DST)),
            "bbox": entry.get("bbox", []),
            "start_sample": 0,
            "num_samples": int(wav_np.shape[0]),
            "neg_xvid": [],     # later
            "neg_intra": -1,    # later
        })

    df = pd.DataFrame(rows)
    print(f"Metadata rows kept: {len(df)}")

    # --------------------- build negative pools -------------------------- #
    all_idx = df.index.tolist()
    vid_groups = df.groupby("vid").groups
    for idx, row in df.iterrows():
        # inter-vid negatives
        others = [i for i in all_idx if df.at[i, "vid"] != row["vid"]]
        df.at[idx, "neg_xvid"] = random.sample(
            others, min(args.neg_pool_size, len(others))
        )
        # intra-vid negative (if video有多clip)
        same = [i for i in vid_groups[row["vid"]] if i != idx]
        df.at[idx, "neg_intra"] = random.choice(same) if same else -1

    # --------------------- save parquet & split -------------------------- #
    meta_dir = DST / "metadata"
    parquet_path = meta_dir / "windows_jepa.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved table: {parquet_path}")

    # split.yaml
    split_dict = {"train": [], "val": [], "test": []}
    for uid, ent in raw_map.items():
        split_dict[ent.get("split", "train")].append(uid)
    with open(meta_dir / "split.yaml", "w") as f:
        yaml.safe_dump(split_dict, f)
    print(f"✓ Saved split file: {meta_dir / 'split.yaml'}")

    print("\nCompleted.  Directory structure:")
    for sub in ["img_npy", "wav_npy", "mel224_npy"]:
        print(f"  {DST/'processed'/sub}  (#={len(list((DST/'processed'/sub).glob('*.npy')))})")
    print("Ready for WindowsAudioImageDataset(root='vggss_jepa').")


if __name__ == "__main__":
    main()
