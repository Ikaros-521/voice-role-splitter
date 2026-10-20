import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vrs.cli import main
from vrs.config import Config

SR = 16000


def tone(path, hz, sec=1.6, vib=0.0):
    t = np.linspace(0, sec, int(SR * sec), endpoint=False)
    f = hz * (1 + vib * np.sin(2 * np.pi * 5 * t))
    y = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32)
    sf.write(str(path), y, SR)


@pytest.fixture
def run(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for i, hz in enumerate((110, 170, 250, 330)):
        tone(audio / f"v{i}.wav", hz)
    main(["init", "--audio-dir", str(audio), "--run-dir", str(tmp_path / "run")])
    return Config.load(str(tmp_path / "run")), tmp_path / "run"


def test_pitch_measures_each_frequency(run):
    cfg, rd = run
    assert main(["pitch", "--run-dir", str(rd), "--workers", "2"]) == 0
    d = np.load(rd / "pitch.npz", allow_pickle=True)
    f0 = d["f0"]
    assert np.isfinite(f0).all(), f0
    # tones are the easy case: median F0 should track the synthesised fundamental
    for got, want in zip(np.sort(f0), (110, 170, 250, 330)):
        assert abs(got - want) / want < 0.2, (got, want)
    assert d["spread"].shape == f0.shape and (d["voiced"] > 0).all()


def test_pitch_records_files_in_order(run):
    cfg, rd = run
    main(["pitch", "--run-dir", str(rd), "--workers", "2"])
    d = np.load(rd / "pitch.npz", allow_pickle=True)
    assert list(d["files"]) == ["v0.wav", "v1.wav", "v2.wav", "v3.wav"]


def test_cluster_uses_pitch_when_available(run):
    """f0_disp only appears in the sweep if pitch.npz exists - and it must be the real column"""
    cfg, rd = run
    # synthesise embeddings so this stays fast and offline
    files = [f"w{i}.wav" for i in range(8)]           # 4 per cluster: enough for a spread
    rng = np.random.default_rng(0)
    (rd / "embeddings").mkdir(parents=True, exist_ok=True)
    for name in ("campplus", "ecapa"):
        E = rng.normal(size=(8, 16)).astype(np.float32)
        E[[0, 1, 4, 5]] += 3.0                        # two groups of four
        np.save(rd / "embeddings" / f"{name}.npy", E)
    np.savez_compressed(rd / "meta.npz", files=np.array(files),
                        nwindow=np.ones(8, np.int32),
                        speech_sec=np.array([1.6] * 8, np.float32),
                        rms=np.array([0.2] * 8, np.float32))
    main(["pitch", "--run-dir", str(rd), "--workers", "2"])
    # stretch pitch.npz to the 8 rows the embeddings now describe
    d = np.load(rd / "pitch.npz", allow_pickle=True)
    np.savez_compressed(rd / "pitch.npz", files=np.array(files),
                        f0=np.tile(d["f0"], 2), spread=np.tile(d["spread"], 2),
                        voiced=np.tile(d["voiced"], 2))
    main(["cluster", "--run-dir", str(rd), "--ks", "2,3"])
    sweep = pd.read_csv(rd / "cluster_sweep.csv")
    assert "f0_disp" in sweep.columns and sweep.f0_disp.notna().any()
    assert (sweep.f0_disp > 0).all()
