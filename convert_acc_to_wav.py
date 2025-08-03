#!/usr/bin/env python
"""Convert all ACC/AAC audio files in <dataset_root>/audio to PCM WAV.

Usage
-----
$ python convert_acc_to_wav.py --root vgg_ss_processed_data \
                               --sr 16000  --channels 1

The script will create ``<dataset_root>/waveforms`` (if missing) and store
files as ``<waveforms>/<stem>.wav`` using the requested sample‑rate and number
of channels.  Requires *pydub* and an FFmpeg backend:
    pip install pydub
    # FFmpeg: https://ffmpeg.org/download.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover
    sys.exit("[Error] missing dependency: install via `pip install pydub` and make sure FFmpeg is in PATH.")

ACC_EXTS = ("*.acc", "*.aac")


# ────────────────────────────────  Helpers  ────────────────────────────────

def find_audio_dir(root: Path) -> Optional[Path]:
    """Return the directory that actually contains .acc/.aac files."""

    # 1) 典型结构: <root>/audio
    audio_dir = root / "audio"
    if audio_dir.is_dir() and any(audio_dir.glob(ext) for ext in ACC_EXTS):
        return audio_dir

    # 2) .acc 文件直接在 <root>
    if any(root.glob(ext) for ext in ACC_EXTS):
        return root

    # 3) 递归搜索第一个包含目标文件的子目录
    for sub in root.rglob("*"):
        if sub.is_dir() and any(sub.glob(ext) for ext in ACC_EXTS):
            return sub
    return None


def iter_acc_files(audio_dir: Path) -> Iterable[Path]:
    for ext in ACC_EXTS:
        yield from audio_dir.glob(ext)


def convert(src: Path, dst: Path, sr: int, ch: int) -> None:
    try:
        seg = AudioSegment.from_file(src)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Cannot load {src.name}: {e}")
        return
    seg = seg.set_frame_rate(sr).set_channels(ch)
    dst.parent.mkdir(parents=True, exist_ok=True)
    seg.export(dst, format="wav")
    print(f"[OK] {src.name} → {dst.relative_to(dst.parents[2])}")


# ─────────────────────────────────  Main  ──────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ACC/AAC batch converter → WAV")
    p.add_argument("--root", required=True, help="Dataset root path")
    p.add_argument("--sr", type=int, default=16_000, help="Target sample‑rate (Hz)")
    p.add_argument("--channels", type=int, choices=[1, 2], default=1, help="Output channels (1 mono | 2 stereo)")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"[Error] Root path {root} does not exist")

    audio_dir = find_audio_dir(root)
    if audio_dir is None:
        sys.exit(f"[Error] Could not find any .acc/.aac files under {root}")

    out_dir = root / "waveforms"
    files = list(iter_acc_files(audio_dir))
    print(f"Found {len(files)} files in {audio_dir.relative_to(root)}; converting …")

    for src in files:
        dst = out_dir / f"{src.stem}.wav"
        convert(src, dst, sr=args.sr, ch=args.channels)

    print(f"Done ✔ Converted files saved to {out_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
