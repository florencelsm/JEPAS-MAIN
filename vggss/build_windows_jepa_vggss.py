#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_windows_jepa_vggss.py  (root-level version, self-contained)

默认目录
--------
./vgg_ss_processed_data/              --src   (images/ spectrograms/ waveforms/)
./vggss/vggss.json                    --json  (可缺；仅提供 bbox)
./vggss_jepa/                         --dst   (输出 processed/ 与 metadata/)

输出结构
--------
vggss_jepa/
 ├─ processed/
 │    ├─ img_npy/       <uid>.npy   # RGB 224×224 float32[0,1]
 │    ├─ mel224_npy/    <uid>.npy   # 224×224 float32[0,1]
 │    └─ wav_npy/       <uid>.npy   # 16-kHz waveform
 └─ metadata/
      ├─ windows_jepa.parquet  (缺 parquet 引擎时写 windows_jepa.csv)
      └─ split.yaml            # train/val/test 按 video-id 随机 70/15/15
"""
from __future__ import annotations
import argparse, os, random, json, yaml, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src",  type=Path, default=Path("vgg_ss_processed_data"),
                   help="Folder containing images/, spectrograms/, waveforms/")
    p.add_argument("--json", type=Path, default=Path("vggss/vggss.json"),
                   help="vggss.json for bbox (optional)")
    p.add_argument("--dst",  type=Path, default=Path("vggss_jepa"),
                   help="Output root; processed/ & metadata/ created inside")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--neg-pool", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.15)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def jpg_to_npy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    arr = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    np.save(dst, arr)


def png_to_npy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    arr = np.asarray(Image.open(src).convert("L"), np.float32) / 255.0
    np.save(dst, arr)


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)                  # same-disk hard-link
    except (AttributeError, OSError):
        shutil.copy2(src, dst)             # fallback copy


def safe_to_parquet(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
        print(f"[INFO] parquet saved → {path}")
    except (ImportError, ValueError, OSError) as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        print(f"[WARN] parquet engine unavailable ({e}). CSV saved → {alt}")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    src = args.src.resolve()
    dst = args.dst.resolve()
    meta_json = args.json.resolve()

    if not src.exists():
        raise FileNotFoundError(f"--src not found: {src}")

    # ------- src sub-folders ------- #
    src_imgs  = src / "images"
    src_specs = src / "spectrograms"
    src_wavs  = src / "waveforms"
    for p in (src_imgs, src_specs, src_wavs):
        if not p.is_dir():
            raise FileNotFoundError(f"Missing folder: {p}")

    # ------- dst processed folders ------- #
    p_img = dst / "processed" / "img_npy"
    p_mel = dst / "processed" / "mel224_npy"
    p_wav = dst / "processed" / "wav_npy"
    for p in (p_img, p_mel, p_wav):
        p.mkdir(parents=True, exist_ok=True)
    (dst / "metadata").mkdir(parents=True, exist_ok=True)

    # ------- step 1: convert / link -------------------------------------- #
    print("[STEP 1] Converting images & spectrograms → npy, linking wavs")
    tasks = []
    with ThreadPoolExecutor(args.workers) as ex:
        # images
        for jp in src_imgs.glob("*.jpg"):
            tasks.append(ex.submit(jpg_to_npy, jp, p_img / f"{jp.stem}.npy"))
        # spectrograms
        for pp in src_specs.glob("*.png"):
            tasks.append(ex.submit(png_to_npy, pp, p_mel / f"{pp.stem}.npy"))
        # wav_npy 直接硬链/复制
        for wp in src_wavs.glob("*.npy"):
            tasks.append(ex.submit(link_or_copy, wp, p_wav / wp.name))
        for _ in tqdm(as_completed(tasks), total=len(tasks), ncols=80):
            pass

    # ------- step 2: collect valid uids ---------------------------------- #
    imgs = {f.stem for f in p_img.glob("*.npy")}
    wavs = {f.stem for f in p_wav.glob("*.npy")}
    mels = {f.stem for f in p_mel.glob("*.npy")}
    uids = sorted(imgs & wavs & mels)
    print(f"[INFO] Triplets present = {len(uids)}")

    if not uids:
        raise RuntimeError("No complete img/wav/mel triplet found. Abort.")

    # ------- bbox map (optional) ----------------------------------------- #
    bbox_map: Dict[str, List] = {}
    if meta_json.is_file():
        with open(meta_json, "r", encoding="utf-8") as f:
            bbox_map = {e["file"]: e.get("bbox", []) for e in json.load(f)}
        print(f"[INFO] bbox entries loaded = {len(bbox_map)}")

    # ------- build DataFrame --------------------------------------------- #
    rows = []
    for uid in uids:
        wav_np = np.load(p_wav / f"{uid}.npy", mmap_mode="r")
        rows.append({
            "uid": uid,
            "vid": uid.split("_")[0],
            "img_path": str((p_img / f"{uid}.npy").relative_to(dst)),
            "wav_path": str((p_wav / f"{uid}.npy").relative_to(dst)),
            "mel224_path": str((p_mel / f"{uid}.npy").relative_to(dst)),
            "bbox": json.dumps(bbox_map.get(uid, [])),
            "start_sample": 0,
            "num_samples": int(wav_np.shape[0]),
            "neg_xvid": [],   # to be filled
            "neg_intra": -1,
        })

    df = pd.DataFrame(rows)

    # ------- negative pools ---------------------------------------------- #
    all_idx = df.index.tolist()
    vid_groups = df.groupby("vid").groups
    for i, row in df.iterrows():
        others = [j for j in all_idx if df.at[j, "vid"] != row["vid"]]
        df.at[i, "neg_xvid"] = rng.sample(others, min(args.neg_pool, len(others)))
        same = [j for j in vid_groups[row["vid"]] if j != i]
        df.at[i, "neg_intra"] = rng.choice(same) if same else -1
    df["neg_xvid"] = df["neg_xvid"].apply(json.dumps)

    # ------- save windows_jepa table ------------------------------------- #
    table_path = dst / "metadata" / "windows_jepa.parquet"
    safe_to_parquet(df, table_path)

    # ------- split.yaml --------------------------------------------------- #
    vids = sorted(df.vid.unique())
    rng.shuffle(vids)
    n = len(vids)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    split = {
        "train": vids[:n_train],
        "val":   vids[n_train:n_train + n_val],
        "test":  vids[n_train + n_val:],
    }
    with open(dst / "metadata" / "split.yaml", "w") as f:
        yaml.safe_dump(split, f)
    print(f"[INFO] split.yaml written ({len(split['train'])}/{len(split['val'])}/{len(split['test'])} vids)")

    print("\n[FINISHED] Dataset ready at", dst)


if __name__ == "__main__":
    main()
