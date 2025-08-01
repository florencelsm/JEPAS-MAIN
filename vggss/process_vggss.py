#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vggss_audio_to_wave_spec.py

Convert VGG-SS audio clips to:
1. 16 kHz mono waveform (.npy)
2. 224×224 Mel spectrogram (.png)

Directory layout
----------------
project_root/
 ├─ vggss/                   # this script lives here
 ├─ vgg_ss_processed_data/   # <-- dataset_root (contains audio/)
 │   ├─ audio/
 │   ├─ images/
 │   ├─ videos/
 │   └─ ...
 └─ ...

Example
-------
python vggss_audio_to_wave_spec.py \
       --dataset_root vgg_ss_processed_data \
       --sr 16000 \
       --n_mels 128 \
       --workers 6
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

import torchaudio

def safe_load(path, sr=16000):
    try:
        x, orig_sr = torchaudio.load(path)   # 支持 AAC
        if orig_sr != sr:
            x = torchaudio.functional.resample(x, orig_sr, sr)
        return x.squeeze(0).numpy()
    except Exception as e:
        print(f"[WARN] torchaudio failed to read {path}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VGG-SS audio clips to waveform (.npy) and 224×224 spectrogram (.png)."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Dataset root folder that contains the audio/ sub-folder."
             "It can be an absolute path or a path relative to either the current working directory "
             "or the repository root (parent of this script).",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (Hz).")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bins.")
    parser.add_argument("--workers", type=int, default=4, help="ThreadPool size.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Core processing                                                             #
# --------------------------------------------------------------------------- #
def process_one_audio(
    audio_path: Path,
    wave_dir: Path,
    spec_dir: Path,
    sr: int,
    n_mels: int,
) -> None:
    """Generate .npy waveform and 224×224 spectrogram for one audio clip."""
    stem = audio_path.stem                                           # e.g. zpWuikVorYg_000032
    wave_out = wave_dir / f"{stem}.npy"
    spec_out = spec_dir / f"{stem}.png"

    # Skip if outputs already exist (idempotent / resumable)
    if wave_out.exists() and spec_out.exists():
        return

    # --- Load audio ---
    try:
        # y, _ = librosa.load(audio_path, sr=sr, mono=True)
        y = safe_load(audio_path, sr)
        if y is None:
            return  # 或者记录错误再 continue

    except Exception as exc:
        print(f"[!] Failed to read {audio_path.name}: {exc}")
        return

    # --- Save waveform .npy ---
    try:
        if not wave_out.exists():
            np.save(wave_out, y.astype(np.float32))
    except Exception as exc:
        print(f"[!] Failed to save waveform {wave_out.name}: {exc}")

    # --- Compute Mel spectrogram ---
    try:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to 0-255 uint8
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        img_arr = (mel_norm * 255).astype(np.uint8)

        # PIL grayscale → resize to 224×224 → save
        img = Image.fromarray(img_arr, mode="L")
        img = img.resize((224, 224), Image.BICUBIC)
        img.save(spec_out)
    except Exception as exc:
        print(f"[!] Failed to save spectrogram {spec_out.name}: {exc}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def resolve_dataset_root(arg_root: Path) -> Path:
    """
    Resolve dataset_root in the following order:
    1. Absolute path as is.
    2. Relative to current working directory.
    3. Relative to repository root (parent folder of this script).
    """
    # 1. Absolute path: return immediately
    if arg_root.is_absolute():
        return arg_root

    # 2. Relative to CWD
    cwd_candidate = (Path.cwd() / arg_root).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # 3. Relative to repo root (parent of this script)
    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = (repo_root / arg_root).resolve()
    if repo_candidate.exists():
        return repo_candidate

    # If none matched, return original (will trigger FileNotFoundError later)
    return arg_root.resolve()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)

    audio_dir = dataset_root / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Audio folder not found: {audio_dir}\n"
            "Make sure the previous download script finished successfully."
        )

    wave_dir = dataset_root / "waveforms"
    spec_dir = dataset_root / "spectrograms"
    wave_dir.mkdir(exist_ok=True)
    spec_dir.mkdir(exist_ok=True)

    # Collect audio files
    audio_files = [
        p for p in audio_dir.glob("**/*")
        if p.suffix.lower() in {".aac", ".wav", ".mp3", ".flac"}
    ]

    print(f"Discovered {len(audio_files)} audio files. Start converting...")

    # ThreadPool + tqdm progress bar
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = [
            executor.submit(
                process_one_audio,
                audio_path=ap,
                wave_dir=wave_dir,
                spec_dir=spec_dir,
                sr=args.sr,
                n_mels=args.n_mels,
            )
            for ap in audio_files
        ]
        for _ in tqdm(as_completed(tasks), total=len(tasks), ncols=80):
            pass

    print("✔ Conversion finished.")
    print(f"Waveforms saved to:   {wave_dir}")
    print(f"Spectrograms saved to: {spec_dir}")


if __name__ == "__main__":
    main()

