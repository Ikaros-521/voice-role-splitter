"""WAV loading, windowing, and file placement.

Amplitude convention: every backend is fed float32 in [-1, 1]. Feeding int16-magnitude
audio makes the frontends scale twice, every cosine similarity inflates, and hierarchical
clustering collapses into one giant cluster -- a real failure mode this module guards
against rather than trusts the caller to remember.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioError(RuntimeError):
    pass


class ScalingError(AudioError):
    pass


def read_wav(path, target_sr: int, trim_db: float | None = None) -> np.ndarray:
    """mono float32 @ target_sr, in [-1, 1]; optionally silence-trimmed"""
    try:
        y, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        raise AudioError(f"cannot read {path}: {exc}") from exc
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size and float(np.abs(y).max()) > 1.5:
        raise ScalingError(
            f"{Path(path).name}: samples exceed [-1, 1]. The file decodes as int-magnitude; "
            "vrs feeds backends normalized float audio and will not rescale it for you.")
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    if trim_db is not None and y.size:
        y, _ = librosa.effects.trim(y, top_db=trim_db)
    return np.clip(np.ascontiguousarray(y, dtype=np.float32), -1.0, 1.0)


def windows(y: np.ndarray, cfg) -> list:
    """Sliding windows; empty when there is no usable speech."""
    n = len(y)
    if n < int(cfg.min_speech * cfg.sample_rate):
        return []
    w = int(cfg.window * cfg.sample_rate)
    h = int(cfg.hop * cfg.sample_rate)
    if n <= w:
        return [y]
    out = [y[i:i + w] for i in range(0, n - w + 1, h)]
    if n - (len(out) - 1) * h < w:
        out.append(y[n - w:])
    return out


def file_info(path, target_sr: int) -> dict:
    """header-only probe: cheap enough to run over tens of thousands of files"""
    with sf.SoundFile(str(path)) as f:
        frames, sr, ch = f.frames, f.samplerate, f.channels
    return dict(file=Path(path).name, frames=int(frames), sample_rate=int(sr), channels=int(ch),
                duration=round(frames / sr, 3) if sr else np.nan,
                needs_resample=bool(sr != target_sr), bytes=int(os.path.getsize(path)))


def probe(audio_dir, target_sr: int = 16000) -> "pd.DataFrame":
    import pandas as pd
    rows = [file_info(p, target_sr) for p in sorted(Path(audio_dir).glob("*.wav"))]
    df = pd.DataFrame(rows)
    if not len(df):
        raise AudioError(f'no .wav files under "{audio_dir}"')
    return df


def rms_of(path, target_sr: int = 16000, max_sec: float = 5.0) -> float:
    y = read_wav(path, target_sr)
    y = y[: int(max_sec * target_sr)]
    return float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0


def place(src, dst, mode: str = "link") -> str:
    """link (default, zero extra disk) or copy. Falls back to copy across volumes."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "link":
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    shutil.copy2(src, dst)
    return "copy"
