import numpy as np
import pytest
import soundfile as sf

from vrs import audio
from vrs.config import Config


def make_wav(path, sec=2.0, sr=16000, amp=0.3, silence_head=0.0):
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    y = amp * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    if silence_head:
        y[: int(sr * silence_head)] = 0.0
    sf.write(str(path), y, sr)
    return path


class C:
    sample_rate = 16000
    window = 1.0
    hop = 0.5
    min_speech = 0.3
    trim_db = 35.0


def test_read_wav_is_normalised_float(tmp_path):
    y = audio.read_wav(make_wav(tmp_path / "x1.wav"), 16000)
    assert y.dtype == np.float32 and np.abs(y).max() <= 1.0


def test_read_wav_rejects_int_magnitude_audio(tmp_path):
    p = tmp_path / "loud.wav"
    sf.write(str(p), (30000 * np.sin(np.linspace(0, 100, 8000))).astype(np.float32), 16000,
             subtype="FLOAT")
    with pytest.raises(audio.ScalingError):
        audio.read_wav(p, 16000)


def test_read_wav_resamples(tmp_path):
    p = make_wav(tmp_path / "x2.wav", sec=1.0, sr=44100)
    y = audio.read_wav(p, 16000)
    assert abs(len(y) / 16000 - 1.0) < 0.05


def test_windows_count_and_overlap():
    cfg = C()
    y = np.zeros(3 * 16000, np.float32)
    ws = audio.windows(y, cfg)
    assert len(ws) == 5 and len(ws[0]) == 16000


def test_windows_empty_below_min_speech():
    assert audio.windows(np.zeros(int(0.2 * 16000), np.float32), C()) == []


def test_windows_single_when_shorter_than_window():
    ws = audio.windows(np.zeros(int(0.8 * 16000), np.float32), C())
    assert len(ws) == 1


def test_trim_removes_leading_silence(tmp_path):
    p = make_wav(tmp_path / "x3.wav", sec=3.0, silence_head=2.5)
    y = audio.read_wav(p, 16000, trim_db=35)
    assert len(y) / 16000 < 1.2


def test_probe_reports_duration(tmp_path):
    make_wav(tmp_path / "a.wav", sec=1.5)
    make_wav(tmp_path / "b.wav", sec=0.4)
    df = audio.probe(tmp_path, 16000)
    assert len(df) == 2 and abs(df.duration.sum() - 1.9) < 0.01


def test_probe_raises_on_empty_dir(tmp_path):
    with pytest.raises(audio.AudioError):
        audio.probe(tmp_path)


def test_place_link_then_copy_fallback(tmp_path):
    src = make_wav(tmp_path / "s.wav")
    dst = tmp_path / "out" / "s.wav"
    assert audio.place(src, dst, "link") == "link"
    assert audio.place(src, dst, "link") == "exists"
    dst2 = tmp_path / "out2" / "s.wav"
    assert audio.place(src, dst2, "copy") == "copy"


def test_config_validation(tmp_path):
    cfg = Config(audio_dir=str(tmp_path))
    with pytest.raises(Exception):
        cfg.validate()                       # no wav files yet
    make_wav(tmp_path / "a.wav")
    cfg.validate()
    cfg.window, cfg.hop = 0.5, 1.0
    with pytest.raises(ValueError, match="window must be greater"):
        cfg.validate()


def test_config_roundtrip(tmp_path):
    make_wav(tmp_path / "a.wav")
    cfg = Config(audio_dir=str(tmp_path), run_dir=str(tmp_path / "run"), ks=[6, 8], models=["campplus"])
    f = cfg.save()
    back = Config.load(str(tmp_path / "run"))
    assert back.ks == [6, 8] and back.models == ["campplus"] and f.exists()
