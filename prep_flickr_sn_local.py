import argparse, json, shutil, subprocess, re
from pathlib import Path
from tqdm import tqdm
import ffmpeg, librosa, numpy as np, soundfile as sf
from PIL import Image

# -------------------- helpers --------------------#
def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_center_frame(mp4: Path, t: float, out_jpg: Path):
    (ffmpeg
        .input(str(mp4), ss=t)
        .filter("scale", 224, 224)
        .output(str(out_jpg), vframes=1, loglevel="quiet")
        .overwrite_output()
        .run())

def mp4_duration(mp4: Path) -> float:
    meta = ffmpeg.probe(str(mp4), select_streams="v")
    return float(meta["streams"][0]["duration"])

def wav_to_rgb_mel(wav: np.ndarray, sr: int, sec=1.28):
    """Cut centre segment & convert to 3-channel mel-spectrogram tensor."""
    half_len = int(sr * sec / 2)
    mid = wav.shape[0] // 2
    seg = wav[max(0, mid - half_len): mid + half_len]

    # pad/trim to fixed length
    seg = librosa.util.fix_length(seg, size=int(sr * sec))

    # ---- NEW: keyword-only librosa calls ----#
    # mel: (n_mels × time)  -> here 128 × 256
    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        power=2.0,
    )
    mel_db = librosa.power_to_db(S=mel, ref=1.0)

    # normalise to [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # three identical channels (C, H, W) → (3,128,256)
    return np.stack([mel_db, mel_db, mel_db], axis=0).astype(np.float32)



def first_file(p: Path, exts):
    for ext in exts:
        fs = sorted(p.glob(ext))
        if fs:
            return fs[0]
    return None

# -------------------- main --------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to Dataset/Data directory from HF repo")
    ap.add_argument("--dst", type=Path, default=Path("stageA_cache"))
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    ensure(args.dst / "train")
    ensure(args.dst / "val")

    subdirs = sorted([d for d in (args.src).iterdir() if d.is_dir()])
    val_num = int(len(subdirs) * args.val_frac)

    splits = {"val": subdirs[:val_num], "train": subdirs[val_num:]}

    for split, folders in splits.items():
        out_dir = args.dst / split
        ensure(out_dir)
        index = []

        for idx, folder in enumerate(tqdm(folders, desc=f"{split}", ncols=90)):
            # ---- get / generate image ----#
            img_file = first_file(folder, ["*.jpg", "*.png"])
            if not img_file:
                mp4 = first_file(folder, ["*.mp4", "*.webm"])
                if mp4:
                    dur = mp4_duration(mp4)
                    img_file = out_dir / f"{idx:06d}_frame.jpg"
                    extract_center_frame(mp4, dur/2, img_file)
                else:
                    print(f"skip {folder} (no image/video)")
                    continue
            else:
                # copy & resize
                dst_jpg = out_dir / f"{idx:06d}_frame.jpg"
                if not dst_jpg.exists():
                    img = Image.open(img_file).convert("RGB").resize((224,224))
                    img.save(dst_jpg, quality=95)
                img_file = dst_jpg

            # ---- get / generate wav -----#
            wav_file = first_file(folder, ["*.wav"])
            if not wav_file:
                mp4 = first_file(folder, ["*.mp4", "*.webm"])
                if not mp4:
                    print(f"skip {folder} (no audio)")
                    continue
                wav_file = out_dir / f"{idx:06d}.wav"
                (ffmpeg
                    .input(str(mp4))
                    .output(str(wav_file), ac=1, ar=16000, loglevel="quiet")
                    .overwrite_output()
                    .run())

            # ---- wav -> mel RGB npy ----#
            spec_path = out_dir / f"{idx:06d}_spec.npy"
            if not spec_path.exists():
                wav, sr = sf.read(wav_file)
                rgb_mel = wav_to_rgb_mel(wav, sr)
                np.save(spec_path, rgb_mel)

            index.append({
                "jpg": str(img_file.relative_to(args.dst)),
                "spec": str(spec_path.relative_to(args.dst))
            })

        with open(args.dst / f"{split}_index.json", "w") as f:
            json.dump(index, f, indent=2)

    print("✓ Done. Cached data under", args.dst)

if __name__ == "__main__":
    main()
