"""Where model weights live: find first, download last, never silently onto the system drive.

Resolution order for reading:
  1. run.yaml ``model_dir``        - explicit wins
  2. $VRS_MODEL_DIR
  3. every candidate cache root the machine already has, in a sane order
     ($MODELSCOPE_CACHE, ~/.cache/modelscope/hub, $HF_HOME/hub, $HF_HUB_CACHE,
     ~/.cache/huggingface, <cwd>/models, ~/.cache/vrs/models, <run>/models)
  4. nothing found -> the download root

Download root is ``<cwd>/models`` by default, i.e. next to the project you are working
on, so weights are visible and deletable instead of rotting on the system drive. ``runs/``
stays copyable because weights are never inside a run directory: speechbrain symlinks into
its savedir, and a run dir holding them cannot be copied.
"""
from __future__ import annotations

import os
from pathlib import Path

# files that make a location usable as a model root, per backend flavour
MARKERS = {
    "campplus": ("iic/speech_campplus_sv_zh_en_16k-common_advanced",),
    "eres2net": ("iic/speech_eres2netv2_sv_zh-cn_16k-common",),
    "ecapa": ("ecapa",),
}


def candidate_roots(cfg=None, cwd: Path | None = None) -> list[Path]:
    cwd = Path(cwd or Path.cwd())
    out: list[Path] = []
    if cfg is not None and getattr(cfg, "model_dir", ""):
        out.append(Path(cfg.model_dir))
    if os.environ.get("VRS_MODEL_DIR"):
        out.append(Path(os.environ["VRS_MODEL_DIR"]))
    if os.environ.get("MODELSCOPE_CACHE"):
        out.append(Path(os.environ["MODELSCOPE_CACHE"]))
    home = Path.home()
    out += [home / ".cache" / "modelscope" / "hub", home / ".cache" / "modelscope"]
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]).parent)
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]))
    out += [home / ".cache" / "huggingface", cwd / "models", home / ".cache" / "vrs" / "models"]
    if cfg is not None:
        out.append(Path(cfg.root) / "models")
    seen, uniq = set(), []
    for p in out:
        q = _norm(p)
        if q and q not in seen:
            seen.add(q)
            uniq.append(p)
    return uniq


def _norm(p: Path):
    try:
        return p.expanduser().resolve()
    except OSError:
        return None


def modelscope_layout(root: Path) -> list[Path]:
    """modelscope writes either <root>/models/<org>/<name> (new) or <root>/<org>/<name> (legacy)"""
    return [root / "models", root, root / "hub"]


def has_model(root: Path, repo: str) -> Path | None:
    org, _, name = repo.partition("/")
    for base in modelscope_layout(root):
        for cand in (base / org / name, base / repo):
            try:
                if not cand.is_dir():
                    continue
                if any(cand.glob("*.ckpt")) or any(cand.glob("*.pt")):
                    return cand
            except OSError:            # unreadable/foreign symlinks: just skip this candidate
                continue
    return None


def ecapa_dir(root: Path) -> Path | None:
    """a speechbrain ECAPA snapshot: hyperparams.yaml + the embedding checkpoint

    Found either as <root>/ecapa or inside an HF hub cache for
    speechbrain/spkrec-ecapa-voxceleb (snapshot dirs are symlink farms, so resolve them).
    """
    need = ("hyperparams.yaml", "embedding_model.ckpt")
    direct = root / "ecapa"
    if all((direct / f).exists() and (direct / f).stat().st_size > 1000 for f in need):
        return direct
    hub = root / "hub" if (root / "hub").is_dir() else root
    for snap in sorted((hub / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots").glob("*")):
        try:
            real = snap.resolve()
        except OSError:
            continue
        if all((real / f).exists() and (real / f).stat().st_size > 1000 for f in need):
            return real
    return None


def find(kind: str, cfg=None, cwd: Path | None = None) -> Path | None:
    """first existing location holding this backend's weights, or None"""
    hit = locate(kind, cfg, cwd)
    return hit[1] if hit else None


def locate(kind: str, cfg=None, cwd: Path | None = None):
    """(cache_root, model_dir) of the first candidate root that already holds the weights

    cache_root is what MODELSCOPE_CACHE has to point at for funasr/modelscope to reuse the
    files instead of fetching a second copy; model_dir is where they actually live.
    """
    for root in candidate_roots(cfg, cwd):
        r = _norm(root)
        if not r or not r.exists():
            continue
        if kind == "ecapa":
            hit = ecapa_dir(r)
        elif kind in MARKERS:
            hit = has_model(r, MARKERS[kind][0])
        else:
            hit = None
        if hit:
            return r, hit
    return None


def download_root(cfg=None, cwd: Path | None = None) -> Path:
    """where to fetch weights when discovery came up empty

    A cache root the user already configured (``$MODELSCOPE_CACHE``, ``$VRS_MODEL_DIR``) is
    also the write target - they told a downloader where things go, and ignoring that would
    scatter weights across two places. Otherwise: project-local ``./models``.
    """
    if cfg is not None and getattr(cfg, "model_dir", ""):
        return Path(cfg.model_dir)
    for var in ("VRS_MODEL_DIR", "MODELSCOPE_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path(cwd or Path.cwd()) / "models"
