import json
import numpy as np
import pandas as pd
import pytest

from vrs import roles as RO
from vrs.config import Config


@pytest.fixture
def env(tmp_path):
    cfg = Config(audio_dir=str(tmp_path / "audio"), run_dir=str(tmp_path / "run"))
    (tmp_path / "audio").mkdir()
    for i in range(6):
        (tmp_path / "audio" / f"v{i}.wav").write_bytes(b"RIFF" + bytes(44))
    np.savez_compressed(cfg.path("meta.npz"), files=np.array([f"v{i}.wav" for i in range(6)]),
                        speech_sec=np.array([2.0] * 6, np.float32),
                        rms=np.array([0.01, 0.02, 0.3, 0.4, 0.05, 0.02], np.float32))
    cfg.save()
    labels = pd.DataFrame(dict(file=[f"v{i}.wav" for i in range(6)], cluster=[0, 0, 0, 1, 1, -1],
                               speech_sec=[2.0] * 6, embedded=[True] * 5 + [False],
                               has_vector=[True] * 5 + [False],
                               own_pct=[0.9, 0.8, 0.2, 0.7, 0.1, np.nan], f0=[200, 210, 190, 350, 500, np.nan]))
    return cfg, labels


def write_export(cfg, rows):
    pd.DataFrame(rows, columns=["type", "cluster", "file", "role", "verdict"]).to_csv(
        cfg.root / "labels_manual.csv", index=False, encoding="utf-8-sig")


def test_read_export_splits_three_label_kinds(env):
    cfg, _ = env
    write_export(cfg, [
        ["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""],
        ["file", 1, "v3.wav", "鲍勃", "yes"], ["file", 0, "v2.wav", "改主见", ""]])
    e = RO.read_export(cfg.root / "labels_manual.csv")
    assert e["cluster_names"] == {0: "爱丽丝", 1: "鲍勃"}
    assert e["file_roles"] == {"v3.wav": "鲍勃", "v2.wav": "改主见"}


def test_read_export_missing_file_is_not_fatal(env):
    e = RO.read_export(env[0].root / "does_not_exist.csv")
    assert e == dict(cluster_names={}, file_roles={}, spot_yes=set(), spot_no=set())


def test_propagate_applies_cluster_names_and_overrides(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""],
                       ["spot", 0, "v2.wav", "", "no"]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    assert list(df.role[:5]) == ["爱丽丝", "爱丽丝", "需复核", "鲍勃", "鲍勃"]
    assert df.role.iloc[5] == "未定"                 # never a silent guess
    assert df.confidence.iloc[5] == "low"


def test_confidence_uses_typicality_and_length(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    assert list(df.confidence) == ["high", "high", "mid", "high", "low", "low"]


def test_tone_names_reflect_loudness_and_length(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    tones, stats = RO.tone_names(cfg, df)
    assert tones[1] in ("高声强调", "激动喊叫", "高亢活泼")
    assert tones[-1] == "无有效人声"
    assert set(stats.columns) >= {"cluster", "files", "tone", "rms_rank"}


def test_export_folders_creates_tone_subdirs_with_all_files(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    res = RO.export_folders(cfg, df, mode="link", tone=True, out_dir=str(cfg.root / "roles"))
    assert sum(res["placed"].values()) == 6
    dirs = sorted(p.name for p in (cfg.root / "roles").iterdir())
    assert "_未确定" in dirs and "爱丽丝" in dirs
    inner = [p.name for p in (cfg.root / "roles" / "爱丽丝").iterdir()]
    assert len(inner) == 1 and inner[0].startswith("c00_")
    assert (cfg.root / "clusters.csv").exists()


def test_export_copy_mode_and_flat_layout(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    res = RO.export_folders(cfg, df, mode="copy", tone=False, out_dir=str(cfg.root / "r2"))
    assert len(list((cfg.root / "r2" / "爱丽丝").glob("*.wav"))) == 3
    assert res["modes"] == ["copy"]


def test_write_roles_emits_review_list_and_summary(env):
    cfg, labels = env
    write_export(cfg, [["cluster", 0, "", "爱丽丝", ""], ["cluster", 1, "", "鲍勃", ""]])
    df = RO.propagate(cfg, labels, RO.read_export(cfg.root / "labels_manual.csv"))
    RO.write_roles(cfg, df)
    out = pd.read_csv(cfg.path("roles.csv"), encoding="utf-8-sig").fillna("")
    rv = pd.read_csv(cfg.path("review_list.csv"), encoding="utf-8-sig")
    assert len(out) == 6 and out.role.nunique() == 3
    assert set(out.columns) >= {"file", "role", "cluster", "confidence"}
    # review list = the unresolved file plus the one atypical row (own_pct 0.10 < 0.15)
    assert len(rv) == 2 and set(rv.file) == {"v5.wav", "v4.wav"}
