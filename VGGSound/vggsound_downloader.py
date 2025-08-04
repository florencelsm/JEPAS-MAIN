#!/usr/bin/env python3
# vggsound_downloader.py
"""
Download a fixed number of VGGSound-test clips per class, saving
    • 10-s audio (16 kHz mono WAV)
    • one midpoint frame (JPG)

CSV (no header): YTID_startsec.mp4,<class label>

Usage:
    python vggsound_downloader.py test.csv \
           --output_dir vgg_dataset \
           --samples_per_class 2 \
           --workers 2
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

CLIP_SEC = 10
FRAME_T  = CLIP_SEC / 2
SAMPLE_RATE = 16_000
RETAIN_MP4  = False          # change to True to keep mp4 files

# ─────────── helpers ────────────
def parse_name(name: str) -> Tuple[str, int]:
    stem = Path(name).stem
    vid, sec = stem.split('_')
    return vid, int(sec)

def sh(cmd: List[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f'Cmd failed: {" ".join(cmd)}\n{res.stderr.strip()}')

def yt_download(vid: str, out: Path, start: int) -> None:
    tmp = out.with_suffix('.tmp.mp4')
    tmp.unlink(missing_ok=True)
    try:
        sh([
            'yt-dlp', '--quiet', '--no-warnings',
            '--download-sections', f'*{start}-{start+CLIP_SEC}',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            '-o', str(tmp),
            f'https://www.youtube.com/watch?v={vid}'
        ])
        tmp.rename(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def ffmpeg_audio(mp4: Path, wav: Path) -> None:
    sh([
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', str(mp4), '-vn',
        '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
        str(wav)
    ])

def ffmpeg_frame(mp4: Path, sec: float, jpg: Path) -> None:
    sh([
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{sec:.3f}', '-i', str(mp4),
        '-vframes', '1', '-q:v', '2', str(jpg)
    ])
# ────────────────────────────────

def handle(row_idx: int, fname: str, label: str,
           out_dir: Path, tmp_dir: Path) -> None:
    try:
        vid, start = parse_name(fname)
    except Exception as e:
        print(f'[{row_idx}] invalid filename "{fname}": {e}')
        return

    wav = out_dir / 'audio'  / f'{vid}_{start:06d}.wav'
    jpg = out_dir / 'images' / f'{vid}_{start:06d}.jpg'
    mp4 = tmp_dir / f'{vid}_{start:06d}.mp4'

    if wav.exists() and jpg.exists():
        print(f'[{row_idx}] already done, skip')
        return

    try:
        yt_download(vid, mp4, start)
        ffmpeg_audio(mp4, wav)
        ffmpeg_frame(mp4, FRAME_T, jpg)
        print(f'[{row_idx}] ok {fname}')
    except Exception as e:
        print(f'[{row_idx}] fail {fname}: {e}')
    finally:
        if not RETAIN_MP4:
            mp4.unlink(missing_ok=True)

def read_csv(csv_path: Path) -> List[Tuple[str, str]]:
    with csv_path.open(newline='') as f:
        return [(a.strip(), b.strip()) for a, b in csv.reader(f)]

def limit_per_class(rows, k: int):
    random.seed(0)
    buckets = {}
    for fn, lbl in rows:
        buckets.setdefault(lbl, []).append((fn, lbl))
    out = []
    for lbl, items in buckets.items():
        pick = items if k <= 0 else random.sample(items, min(k, len(items)))
        out.extend(pick)
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('csv_path')
    p.add_argument('--output_dir', default='vgg_dataset')
    p.add_argument('--samples_per_class', type=int, default=5)
    p.add_argument('--workers', type=int, default=5)
    args = p.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')

    out = Path(args.output_dir).resolve()
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    tmp = out / '__tmp__'
    tmp.mkdir(exist_ok=True)

    rows = read_csv(csv_path)
    todo = limit_per_class(rows, args.samples_per_class)

    print(f'total rows: {len(rows)}')
    print(f'to download: {len(todo)} '
          f'({args.samples_per_class} per class)')
    print(f'workers: {args.workers}\n')

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(handle, i, fn, lbl, out, tmp)
                for i, (fn, lbl) in enumerate(todo)]
        for f in as_completed(futs):
            f.result()

    if not RETAIN_MP4:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nall done.')

if __name__ == '__main__':
    main()
