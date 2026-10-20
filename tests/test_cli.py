import json
import numpy as np
import pandas as pd
import pytest

from vrs.cli import build_parser, main
from vrs.config import Config


def make_audio(tmp_path, n=6, sr=16000, sec=1.5):
    import soundfile as sf
    d = tmp_path / "audio"
    d.mkdir(exist_ok=True)
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    for i in range(n):
        y = (0.2 + 0.05 * i) * np.sin(2 * np.pi * (120 + 60 * i) * t).astype(np.float32)
        sf.write(str(d / f"v{i}.wav"), y, sr)
    return d


def run(*argv):
    return main(list(argv))


def test_parser_exposes_the_documented_commands():
    p = build_parser()
    subs = set(p._subparsers._group_actions[0].choices)
    assert {"init", "status", "probe", "backends", "pitch", "embed", "cluster",
            "page", "apply", "refine", "export", "transcribe", "serve"} <= subs


def test_init_then_status_roundtrip(tmp_path, capsys):
    audio = make_audio(tmp_path)
    run("init", "--audio-dir", str(audio), "--run-dir", str(tmp_path / "run"),
        "--models", "campplus,ecapa", "--set", "window=2.5")
    cfg = Config.load(str(tmp_path / "run"))
    assert cfg.models == ["campplus", "ecapa"] and cfg.window == 2.5
    assert run("status", "--run-dir", str(tmp_path / "run")) == 0
    assert "labels.csv" in capsys.readouterr().out


def test_init_rejects_unknown_set_key(tmp_path):
    audio = make_audio(tmp_path)
    assert run("init", "--audio-dir", str(audio), "--run-dir", str(tmp_path / "r"),
               "--set", "nonsense=1") == 2          # exit code 2 = config error


def test_init_reports_missing_audio_dir(tmp_path):
    assert run("init", "--audio-dir", str(tmp_path / "nope"), "--run-dir", str(tmp_path / "r")) == 2


def test_probe_writes_csv(tmp_path):
    audio = make_audio(tmp_path)
    run("init", "--audio-dir", str(audio), "--run-dir", str(tmp_path / "r"))
    assert run("probe", "--run-dir", str(tmp_path / "r")) == 0
    df = pd.read_csv(tmp_path / "r" / "probe.csv")
    assert len(df) == 6 and abs(df.duration.iloc[0] - 1.5) < 0.01


def test_backends_reports_without_a_run(tmp_path, capsys):
    assert run("backends", "--run-dir", str(tmp_path / "missing")) == 0
    out = capsys.readouterr().out
    assert "campplus" in out and "install:" in out or "ready" in out


def test_page_clusters_after_a_synthetic_cluster_run(tmp_path, monkeypatch):
    """init -> embed -> cluster -> page, with the speaker backends replaced by a stub"""
    audio = make_audio(tmp_path, n=12, sec=2.5)
    run("init", "--audio-dir", str(audio), "--run-dir", str(tmp_path / "r"), "--models", "fake")

    class FakeBackend:
        """returns one 8-d vector per window: two loudness groups, so clustering has signal"""
        def __init__(self, cfg):
            pass

        def __call__(self, wins):
            # two genuinely separate groups by loudness, so clustering has real signal
            out = np.zeros((len(wins), 8), np.float32)
            for i, w in enumerate(wins):
                hot = float(np.sqrt(np.mean(w ** 2))) > 0.35
                out[i, 0 if hot else 1] = 1.0
                out[i, 2 + (i % 3)] += 0.01
            return out

    import vrs.embed as EM
    monkeypatch.setitem(EM.BACKENDS, "fake", FakeBackend)
    assert run("embed", "--run-dir", str(tmp_path / "r")) == 0
    assert (tmp_path / "r" / "embeddings" / "fake.npy").exists()
    assert run("cluster", "--run-dir", str(tmp_path / "r"), "--ks", "2,3") == 0
    labels = pd.read_csv(tmp_path / "r" / "labels.csv")
    assert len(labels) == 12 and labels.cluster.nunique() == 2
    assert run("page", "clusters", "--run-dir", str(tmp_path / "r")) == 0
    html = (tmp_path / "r" / "标注.html").read_text("utf-8")
    assert "const DATA=" in html and "labels_manual.csv" in html
    assert "v1.wav" in html or "v0.wav" in html               # clips referenced from the page
