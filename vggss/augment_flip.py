#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_flip_vggss_jepa.py  -  H‑flip augmentation for JEPA dataset.

• 读取原数据根 (--src, 默认 ./vggss_jepa)
• 水平翻转所有 img_npy，修正 bbox
• 音频 / 频谱文件硬链接复用
• 写入新根 (--dst, 默认 ./vggss_jepa_flip)：
      processed/{img_npy|wav_npy|mel224_npy}/
      metadata/{windows_jepa.parquet|csv, split.yaml}

运行:
    python augment_flip_vggss_jepa.py
    # 或自定义
    python augment_flip_vggss_jepa.py --src vggss_jepa --dst vggss_jepa_flip
"""
from __future__ import annotations
import argparse, os, json, shutil, random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("vggss_jepa"),
                   help="Existing JEPA dataset root (processed/ metadata/)")
    p.add_argument("--dst", type=Path, default=Path("vggss_jepa_flip"),
                   help="Destination root to write augmented dataset")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--neg-pool", type=int, default=30,
                   help="# cross-video negatives per sample")
    return p.parse_args()


# --------------------------------------------------------------------- #
# I/O helpers                                                           #
# --------------------------------------------------------------------- #
def hardlink_copy(src: Path, dst: Path):
    """Hard‑link if possible, otherwise copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:  # e.g. cross‑device link not permitted
        shutil.copy2(src, dst)


def flip_save(src: Path, dst: Path):
    """Load HWC float32 npy, flip horizontally, save."""
    if dst.exists():
        return
    arr = np.load(src, mmap_mode="r")
    flipped = arr[:, ::-1, :].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, flipped)


# -------------------- metadata: robust read / write ------------------- #
def load_metadata(path: Path) -> pd.DataFrame:
    """Auto‑detect .csv / .parquet and read accordingly."""
    if path.suffix == ".csv":
        print(f"[INFO] Reading CSV metadata → {path}")
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        try:
            print(f"[INFO] Reading Parquet metadata → {path}")
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Parquet ({e}). "
                "If this file is actually CSV, rename它 to .csv."
            )
    else:
        raise ValueError(f"Unsupported metadata format: {path}")


def save_metadata(df: pd.DataFrame, out_dir: Path, base_name: str = "windows_jepa"):
    """
    Write Parquet if possible; else CSV.
    Always ensure at least one usable file exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{base_name}.parquet"
    csv_path     = out_dir / f"{base_name}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] metadata saved → {parquet_path}")
        # 额外写一份 CSV 方便查看
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[WARN] Parquet write failed ({e}); falling back to CSV.")
        df.to_csv(csv_path, index=False)
        print(f"[OK] metadata saved → {csv_path}")


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    SRC = args.src.resolve()
    DST = args.dst.resolve()

    # ---------- check source dir structure ---------- #
    proc_src = SRC / "processed"
    meta_src = SRC / "metadata"

    # 自动匹配 windows_jepa.* （可能是 .csv 或 .parquet）
    table_candidates = list(meta_src.glob("windows_jepa.*"))
    if not table_candidates:
        raise FileNotFoundError(f"[table] windows_jepa.* not found in {meta_src}")
    table_file = table_candidates[0]

    paths_src = {
        "img": proc_src / "img_npy",
        "mel": proc_src / "mel224_npy",
        "wav": proc_src / "wav_npy",
        "table": table_file,            # 真正格式
        "split": meta_src / "split.yaml",
    }
    for k, p in paths_src.items():
        if not p.exists():
            raise FileNotFoundError(f"[{k}] not found: {p}")

    # ---------- prepare destination dirs ---------- #
    proc_dst = DST / "processed"
    meta_dst = DST / "metadata"
    for sub in ("img_npy", "mel224_npy", "wav_npy"):
        (proc_dst / sub).mkdir(parents=True, exist_ok=True)
    meta_dst.mkdir(parents=True, exist_ok=True)

    # ---------- 1. link wav & mel ---------- #
    print("[STEP 1] Linking wav & mel …")
    for sub in ("wav_npy", "mel224_npy"):
        for f in (proc_src / sub).glob("*.npy"):
            hardlink_copy(f, proc_dst / sub / f.name)

    # ---------- 2. copy orig img & create flips ---------- #
    print("[STEP 2] Copying orig imgs & creating flips …")
    img_files = list((proc_src / "img_npy").glob("*.npy"))
    for f in tqdm(img_files, ncols=80):
        hardlink_copy(f, proc_dst / "img_npy" / f.name)
        flip_save(f, proc_dst / "img_npy" / f"{f.stem}_fh.npy")

    # ---------- 3. load metadata ---------- #
    print("[STEP 3] Loading metadata …")
    df_orig = load_metadata(paths_src["table"])

    # ---------- 4. build augmented rows ---------- #
    rows: List[Dict] = []
    for _, row in df_orig.iterrows():
        uid = row.uid

        # original row (paths updated)
        rows.append({
            **row,
            "img_path": f"processed/img_npy/{uid}.npy",
            "wav_path": f"processed/wav_npy/{uid}.npy",
            "mel224_path": f"processed/mel224_npy/{uid}.npy",
        })

        # flipped row
        bboxes = json.loads(row.bbox) if isinstance(row.bbox, str) else row.bbox
        flipped_bbox = [
            [1 - x_max, y_min, 1 - x_min, y_max] for x_min, y_min, x_max, y_max in bboxes
        ]
        uid_fh = f"{uid}_fh"
        rows.append({
            **row,
            "uid": uid_fh,
            "img_path": f"processed/img_npy/{uid_fh}.npy",
            "wav_path": f"processed/wav_npy/{uid}.npy",        # same audio
            "mel224_path": f"processed/mel224_npy/{uid}.npy",  # same spec
            "bbox": json.dumps(flipped_bbox),
        })

    df_aug = pd.DataFrame(rows).reset_index(drop=True)

    # ---------- 5. rebuild negative pools ---------- #
    vid_groups = df_aug.groupby("vid").groups
    all_idx = list(df_aug.index)

    neg_xvid_col, neg_intra_col = [], []
    for idx, row in df_aug.iterrows():
        # inter‑video negatives
        others = [j for j in all_idx if df_aug.at[j, "vid"] != row["vid"]]
        neg_xvid_col.append(json.dumps(rng.sample(others, min(args.neg_pool, len(others)))))
        # intra‑video negative
        same = [j for j in vid_groups[row["vid"]] if j != idx]
        neg_intra_col.append(rng.choice(same) if same else -1)

    df_aug["neg_xvid"] = neg_xvid_col
    df_aug["neg_intra"] = neg_intra_col

    # ---------- 6. save metadata ---------- #
    save_metadata(df_aug, meta_dst)

    # ---------- 7. copy split.yaml ---------- #
    shutil.copy2(paths_src["split"], meta_dst / "split.yaml")
    print(f"[OK] split.yaml copied")

    print("\n[FINISHED] Augmented dataset located at:", DST)


if __name__ == "__main__":
    main()
