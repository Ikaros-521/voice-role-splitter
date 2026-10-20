"""Per-file pitch statistics.

F0 is the model-independent signal in this pipeline. It cannot tell you *who* spoke, but it
reliably punishes merging different people (their medians differ) and it falsified the
"archive order groups lines by character" hypothesis that MFCC similarity had suggested --
which is why it is a first-class step here rather than an afterthought.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _one(args):
    path, sr, trim_db = args
    import librosa
    import soundfile as sf
    try:
        y, fs = sf.read(path, dtype="float32")
    except Exception:
        return (np.nan, np.nan, 0.0)
    if y.ndim > 1:
        y = y.mean(1)
    if fs != sr:
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    y = y[: sr * 5]
    frame, hop = 1024, 256
    if len(y) < frame:
        return (np.nan, np.nan, 0.0)
    f0 = np.stack([librosa.yin(y[i:i + frame], fmin=55, fmax=650, sr=sr)
                   for i in range(0, len(y) - frame, hop)], 0).ravel()
    ok = np.isfinite(f0) & (f0 > 55) & (f0 < 650)
    if ok.sum() < 8:
        return (np.nan, np.nan, 0.0)
    v = f0[ok]
    return (float(np.median(v)), float(np.percentile(v, 90) - np.percentile(v, 10)), float(ok.mean()))


def measure(cfg, workers: int = 0) -> Path:
    """write pitch.npz with median F0, range and voiced fraction per file"""
    from multiprocessing import Pool, cpu_count
    files = sorted(Path(cfg.audio_dir).glob("*.wav"))
    jobs = [(str(p), cfg.sample_rate, cfg.trim_db) for p in files]
    workers = workers or max(1, cpu_count() - 2)
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=24)
    arr = np.array(res, dtype=np.float32)
    out = cfg.path("pitch.npz")
    np.savez_compressed(out, files=np.array([p.name for p in files]),
                        f0=arr[:, 0], spread=arr[:, 1], voiced=arr[:, 2])
    ok = arr[:, 0][np.isfinite(arr[:, 0])]
    q = np.percentile(ok, [5, 25, 50, 75, 95]) if ok.size else [np.nan] * 5
    print(f"pitch: {ok.size}/{len(files)} files measured; "
          f"p5={q[0]:.0f} p25={q[1]:.0f} p50={q[2]:.0f} p75={q[3]:.0f} p95={q[4]:.0f} Hz")
    print(f"-> {out}")
    return out


def load(cfg):
    f = cfg.path("pitch.npz")
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {str(k): (float(a), float(b)) for k, a, b in zip(d["files"], d["f0"], d["spread"])}
