#!/usr/bin/env python3
# augment_dataset.py
"""
Augment a VGGSound-style dataset by generating additional images
while duplicating matching audio files, ensuring 1-to-1 correspondence.

Directory layout (input):
    dataset_root/
        images/   abcd_000123.jpg …
        audio/    abcd_000123.wav …

Output (default in-place under dataset_root):
    images_aug/
        abcd_000123.jpg         # original (optional copy)
        abcd_000123_aug0.jpg    # augmented variants
        …
    audio_aug/
        abcd_000123.wav
        abcd_000123_aug0.wav    # byte-wise copy of original

Example:
    python augment_dataset.py ./vgg_dataset \
           --target_size 20000 --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# --------------------------- augmentation ops --------------------------- #
def random_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def random_vflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def random_rotate(img: Image.Image, max_angle: int = 15) -> Image.Image:
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=True)

def random_scale_crop(
    img: Image.Image, scale_range: Tuple[float, float] = (0.8, 1.2)
) -> Image.Image:
    """Randomly rescale then center-crop / pad back to original size."""
    w, h = img.size
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

    # if scaled bigger -> center crop, else -> pad
    if new_w > w or new_h > h:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return img_scaled.crop((left, top, left + w, top + h))
    else:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        return ImageOps.expand(img_scaled, border=(pad_w, pad_h), fill=0)

def random_brightness(img: Image.Image, factor_range=(0.7, 1.3)) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

def random_contrast(img: Image.Image, factor_range=(0.7, 1.3)) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

AUG_POOL = [
    random_hflip,
    random_vflip,
    random_rotate,
    random_scale_crop,
    random_brightness,
    random_contrast,
]
# ----------------------------------------------------------------------- #

def augment_once(img: Image.Image) -> Image.Image:
    """
    Apply 1-3 random transformations from AUG_POOL in random order.
    """
    ops = random.sample(AUG_POOL, k=random.randint(1, 3))
    out = img
    for op in ops:
        out = op(out)
    return out

# ------------------------------ main logic ------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root",
                        help="Folder containing images/ and audio/")
    parser.add_argument("--target_size", type=int, default=20000,
                        help="Desired total number of items after augmentation")
    parser.add_argument("--out_suffix", default="_aug",
                        help="Suffix for augmented sub-folders")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--copy_original", action="store_true",
                        help="Also copy original files into output folders")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    root = Path(args.dataset_root).resolve()
    img_dir_in  = root / "images"
    aud_dir_in  = root / "audio"
    img_dir_out = root / f"images{args.out_suffix}"
    aud_dir_out = root / f"audio{args.out_suffix}"
    img_dir_out.mkdir(parents=True, exist_ok=True)
    aud_dir_out.mkdir(parents=True, exist_ok=True)

    # collect pairs
    img_paths = sorted(img_dir_in.glob("*.jpg"))
    n_ori = len(img_paths)
    if n_ori == 0:
        raise RuntimeError(f"No JPG found in {img_dir_in}")

    # option: copy originals first
    if args.copy_original:
        for img_p in img_paths:
            shutil.copy2(img_p, img_dir_out / img_p.name)
            wav_p = aud_dir_in / (img_p.stem + ".wav")
            if wav_p.exists():
                shutil.copy2(wav_p, aud_dir_out / wav_p.name)

    # compute augmentation factor
    need_total = args.target_size
    aug_per_img = max(1, math.ceil(need_total / n_ori) - 1)

    print(f"Original images : {n_ori}")
    print(f"Target total    : {need_total}")
    print(f"Aug per image   : {aug_per_img}")
    print(f"Output folders  : {img_dir_out.name} / {aud_dir_out.name}\n")

    for idx, img_p in enumerate(img_paths, 1):
        with Image.open(img_p).convert("RGB") as img:
            for k in range(aug_per_img):
                aug_img = augment_once(img)
                aug_name = f"{img_p.stem}_aug{k}.jpg"
                aug_img.save(img_dir_out / aug_name, quality=95)

                # duplicate audio file with matching name
                src_wav = aud_dir_in / f"{img_p.stem}.wav"
                dst_wav = aud_dir_out / f"{img_p.stem}_aug{k}.wav"
                if src_wav.exists():
                    shutil.copy2(src_wav, dst_wav)

        if idx % 200 == 0 or idx == n_ori:
            print(f"Processed {idx}/{n_ori} images")

    print("\nAugmentation completed.")

if __name__ == "__main__":
    main()
