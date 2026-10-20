"""Embedding extraction with resume and a corruption guard.

The guard matters: mean pairwise cosine similarity of per-file embeddings is a cheap
health check that catches the single most damaging mistake in this pipeline (wrong input
scaling, or a frontend fed something it does not expect). Healthy speaker embeddings on
speech sit around 0.2-0.4; inflated ones sit above ~0.55 and clustering then collapses.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .audio import read_wav, windows
from .models import BACKENDS, BackendError, l2


class EmbedError(RuntimeError):
    pass


def collect_windows(cfg, files=None):
    """read + trim + window every file once; returns window counts, trimmed seconds, loudness"""
    from .audio import AudioError
    paths = sorted(Path(cfg.audio_dir).glob("*.wav")) if files is None else [Path(f) for f in files]
    counts, trimmed, wins, loud = [], [], [], []
    t0 = time.time()
    for p in paths:
        try:
            y = read_wav(p, cfg.sample_rate, trim_db=cfg.trim_db)
        except AudioError as exc:
            print(f"  skip unreadable: {exc}")
            y = np.zeros(0, np.float32)
        ws = windows(y, cfg)
        counts.append(len(ws))
        trimmed.append(len(y) / cfg.sample_rate)
        loud.append(float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0)
        wins.extend(ws)
    print(f"{len(paths)} files -> {len(wins)} windows in {time.time() - t0:.0f}s", flush=True)
    return [p.name for p in paths], counts, np.asarray(trimmed, np.float32), wins, np.asarray(loud, np.float32)


def embed(cfg, models=None, progress=500) -> dict:
    """extract one [N, D] matrix per backend into the run dir; existing files are reused"""
    models = models or cfg.models
    files, counts, trimmed, wins, loud = collect_windows(cfg)
    np.savez_compressed(cfg.path("meta.npz"), files=np.array(files),
                        nwindow=np.asarray(counts, np.int32), speech_sec=trimmed, rms=loud)
    out: dict = {}
    health = None
    for name in models:
        if name not in BACKENDS:
            raise EmbedError(f'unknown backend "{name}"; available: {", ".join(BACKENDS)}')
        target = cfg.path(f"embeddings/{name}.npy")
        if target.exists():
            F = np.load(target)
            if len(F) == len(files):
                health = check_health(F, name)
                print(f"{name}: reuse {target.name} {F.shape} | {health['note']}", flush=True)
                out[name] = F
                continue
            print(f"{name}: stale file ({len(F)} rows != {len(files)}), recomputing", flush=True)
        try:
            be = BACKENDS[name](cfg)
        except (Exception, ImportError) as exc:
            print(f"[skip] {name}: {exc}\n         install hint: {BACKENDS[name].hint}", flush=True)
            continue
        t0 = time.time()
        chunks = []
        for i in range(0, len(wins), progress):
            b = wins[i:i + progress]
            chunks.append(np.asarray(be(b), dtype=np.float32))
            print(f"  {name} {i + len(b)}/{len(wins)}  {time.time() - t0:.0f}s", flush=True)
        E = np.concatenate(chunks, 0)
        p, rows = 0, []
        for c in counts:
            rows.append(E[p:p + c].mean(0) if c else np.zeros(E.shape[1], np.float32))
            p += c
        F = np.stack(rows)
        np.save(target, F)
        health = check_health(F, name)
        out[name] = F
        print(f"{name}: {F.shape} -> {target} | {health['note']}", flush=True)
        del E, chunks, be
    if not out:
        raise EmbedError("no backend produced embeddings; run `vrs backends` to see what is installable")
    write_manifest(cfg, files, out, health)
    return out


def check_health(F, name, warn_above=0.55) -> dict:
    Fn = l2(F)
    S = Fn @ Fn.T
    off = S[~np.eye(len(S), dtype=bool)]
    mean, p99 = float(off.mean()), float(np.percentile(off, 99))
    bad = mean > warn_above
    return dict(backend=name, mean_pairwise_cos=round(mean, 4), p99=round(p99, 4),
                ok=not bad,
                note=("INFLATED: mean cosine %.3f > %.2f -- embeddings are not discriminative "
                      "(check that audio reaches backends as float [-1,1] and that the right "
                      "frontend path is used); clustering will collapse into one cluster"
                      % (mean, warn_above)) if bad else
                f"healthy (mean cosine {mean:.3f})")


def write_manifest(cfg, files, embs, last_health) -> Path:
    f = cfg.path("embeddings/manifest.json")
    f.write_text(json.dumps({
        "files": len(files),
        "backbones": {k: list(v.shape) for k, v in embs.items()},
        "sample_rate": cfg.sample_rate, "window": cfg.window, "hop": cfg.hop,
        "min_speech": cfg.min_speech, "trim_db": cfg.trim_db,
        "last_health": last_health,
    }, ensure_ascii=False, indent=1), "utf-8")
    return f


def load(cfg, names=None):
    """read previously produced embeddings; raises if none exist"""
    d = cfg.dir("embeddings")
    out = {}
    for p in sorted(d.glob("*.npy")):
        if names and p.stem not in names:
            continue
        out[p.stem] = np.load(p)
    if not out:
        raise EmbedError(f"no embeddings in {d}; run `vrs embed` first")
    return out
