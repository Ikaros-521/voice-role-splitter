import numpy as np
import pytest
from pathlib import Path

from vrs import cache
from vrs.config import Config

ECAPA_NEED = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
              "label_encoder.txt", "classifier.ckpt"]
CAMP_DIR = "iic/speech_campplus_sv_zh_en_16k-common_advanced"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """pretend the machine has no caches and no env overrides, and that $HOME is tmp_path"""
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for v in ("VRS_MODEL_DIR", "MODELSCOPE_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        monkeypatch.delenv(v, raising=False)
    return home


def touch(p: Path, size=2000):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    return p


def make_ecapa(root: Path) -> Path:
    d = root / "ecapa"
    for f in ECAPA_NEED:
        touch(d / f)
    return d


def test_download_root_is_project_local_not_the_system_drive(tmp_path):
    assert cache.download_root(None, cwd=tmp_path) == tmp_path / "models"


def test_explicit_model_dir_and_env_win(tmp_path, monkeypatch):
    cfg = Config(audio_dir=str(tmp_path), model_dir=str(tmp_path / "weights"))
    assert cache.download_root(cfg, cwd=tmp_path) == tmp_path / "weights"
    monkeypatch.setenv("VRS_MODEL_DIR", str(tmp_path / "envdir"))
    assert cache.download_root(None, cwd=tmp_path) == tmp_path / "envdir"
    # an explicit config still beats the environment
    assert cache.download_root(cfg, cwd=tmp_path) == tmp_path / "weights"


def test_candidate_order_lists_explicit_and_env_first(tmp_path, monkeypatch):
    monkeypatch.setenv("VRS_MODEL_DIR", str(tmp_path / "envdir"))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms"))
    cfg = Config(audio_dir=str(tmp_path), model_dir=str(tmp_path / "weights"))
    roots = [Path(r) for r in cache.candidate_roots(cfg, cwd=tmp_path)]
    assert roots[0] == tmp_path / "weights"
    assert roots[1] == tmp_path / "envdir"
    assert roots[2] == tmp_path / "ms"
    assert tmp_path / "models" in roots


def test_ecapa_found_in_project_models_dir(tmp_path):
    want = make_ecapa(tmp_path / "models")
    assert cache.find("ecapa", None, cwd=tmp_path) == want.resolve()


def test_ecapa_found_in_huggingface_hub_snapshot(tmp_path, isolated):
    snap = isolated / ".cache" / "huggingface" / "hub" / \
        "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots" / "abc123"
    for f in ECAPA_NEED:
        touch(snap / f)
    assert cache.find("ecapa", None, cwd=tmp_path) == snap.resolve()


def test_truncated_partial_download_is_not_treated_as_cached(tmp_path):
    d = tmp_path / "models" / "ecapa"
    for f in ECAPA_NEED:
        touch(d / f)
    (d / "embedding_model.ckpt").write_bytes(b"partial")        # a failed fetch stub
    assert cache.find("ecapa", None, cwd=tmp_path) is None


@pytest.mark.parametrize("layout", ["models/", ""])         # new and legacy modelscope layout
def test_campplus_found_in_modelscope_layouts(tmp_path, layout):
    d = tmp_path / "models" / layout / CAMP_DIR
    touch(d / "pretrained_model.pt")
    assert cache.find("campplus", None, cwd=tmp_path) == d.resolve()


def test_campplus_respects_MODELSCOPE_CACHE_env(tmp_path, monkeypatch):
    root = tmp_path / "elsewhere"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(root))
    d = root / "models" / CAMP_DIR
    touch(d / "example.pt")
    assert cache.find("campplus", None, cwd=tmp_path) == d.resolve()


def test_cache_root_for_download_points_at_the_write_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cache.locate("campplus", None, cwd=tmp_path) is None
    from vrs.models import CampPlus
    cfg = Config(audio_dir=str(tmp_path))
    assert CampPlus.cache_root(cfg) == tmp_path / "models"


def test_where_reports_cache_hit_and_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from vrs.models import CampPlus
    cfg = Config(audio_dir=str(tmp_path))
    assert "not cached" in CampPlus.where(cfg)
    d = tmp_path / "models" / CAMP_DIR
    touch(d / "pretrained_model.pt")
    w = CampPlus.where(cfg)
    assert w.startswith("cached ->") and "speech_campplus_sv_zh_en_16k-common_advanced" in w

def test_configured_modelscope_cache_is_also_the_write_target(tmp_path, monkeypatch):
    """$MODELSCOPE_CACHE is a promise about where weights go; honour it, don't split copies"""
    root = tmp_path / "msroot"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(root))
    assert cache.download_root(None, cwd=tmp_path) == root
    # and a project-local default is still what happens with no configuration at all
    monkeypatch.delenv("MODELSCOPE_CACHE")
    assert cache.download_root(None, cwd=tmp_path) == tmp_path / "models"
