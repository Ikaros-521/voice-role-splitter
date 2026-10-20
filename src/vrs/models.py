"""Speaker-embedding backends.

Each backend is optional and independent: install what your environment can reach and
`vrs backends` tells you what is usable and why the rest is not. Embeddings from different
backends are fused later; having two agreeing is what makes the labels trustworthy, since
one alone was shown to be unable to separate roles on emotional anime dialogue.

Amplitudes handed to every backend are float32 in [-1, 1] (see audio.py).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import cache


class BackendError(RuntimeError):
    pass


def l2(F, eps=1e-8):
    F = np.asarray(F, dtype=np.float32)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return np.where(n > eps, F / np.maximum(n, eps), 0).astype(np.float32)


class Backend:
    """interface: __init__(cfg) -> build model; __call__(list[np.ndarray]) -> [N, D] float32"""
    name = ""
    dim = 0
    hint = ""

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def where(cls, cfg) -> str:
        """answer "where are my weights" for `vrs backends`, without downloading anything"""
        found = cache.locate(cls.name, cfg)
        if found:
            return f"cached -> {found[1]}"
        return f"not cached, would fetch into {cls.fetch_target(cfg)}"

    @classmethod
    def fetch_target(cls, cfg) -> Path:
        """where a download would land: project-local `models/`, not the system drive"""
        return cache.download_root(cfg)

    @classmethod
    def cache_root(cls, cfg) -> Path:
        """the root MODELSCOPE_CACHE must point at so existing weights get reused"""
        found = cache.locate(cls.name, cfg)
        return found[0] if found else cache.download_root(cfg)

    @classmethod
    def status(cls, cfg) -> str:
        try:
            cls._probe(cfg)
            return "ready"
        except Exception as exc:
            return f"unavailable: {exc}"

    @classmethod
    def _probe(cls, cfg):
        raise NotImplementedError

    def __call__(self, windows) -> np.ndarray:
        raise NotImplementedError


class CampPlus(Backend):
    """3D-Speaker CAM++ (zh+en, 192-d) through FunASR's own preprocessing path."""
    name = "campplus"
    dim = 192
    hint = 'pip install "funasr>=1.0"  (plus kaldiio, torch_complex, omegaconf, jaconv, jieba)'
    repo = "iic/speech_campplus_sv_zh_en_16k-common_advanced"

    @classmethod
    def _probe(cls, cfg):
        import funasr  # noqa: F401

    def __init__(self, cfg):
        super().__init__(cfg)
        os.environ["MODELSCOPE_CACHE"] = str(self.cache_root(cfg))
        from funasr import AutoModel
        self.m = AutoModel(model=self.repo, disable_update=True, device=cfg.resolved_device())

    def __call__(self, windows):
        import torch
        out = np.zeros((len(windows), self.dim), np.float32)
        for i, w in enumerate(windows):
            e = self.m.generate(input=w)[0]["spk_embedding"]
            e = e.detach().cpu().numpy().reshape(-1) if isinstance(e, torch.Tensor) else np.asarray(e).reshape(-1)
            out[i, :e.size] = e
        return out


class Ecapa(Backend):
    """SpeechBrain ECAPA-TDNN trained on VoxCeleb (English, 192-d): independent training
    data from the Chinese models, which is exactly why the cross-vote means something."""
    name = "ecapa"
    dim = 192
    hint = 'pip install "speechbrain>=0.5.16,<1.0"  (1.x needs torch>=2.4 for torch.amp)'
    files = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
             "label_encoder.txt", "classifier.ckpt"]
    repo = "speechbrain/spkrec-ecapa-voxceleb"

    @classmethod
    def _probe(cls, cfg):
        import speechbrain  # noqa: F401
        if not cache.find("ecapa", cfg):
            raise BackendError(f"weights not in any cache; run: vrs backends --fetch ecapa "
                               f"(downloads into {cls.fetch_target(cfg) / 'ecapa'})")

    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:                     # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        found = cache.find("ecapa", cfg)
        if not found:
            raise BackendError("ECAPA weights not found; run: vrs backends --fetch ecapa")
        d = str(found)
        missing = [f for f in self.files if not (Path(d) / f).exists()]
        if missing:
            raise BackendError(f"missing {missing} in {d}; run: vrs backends --fetch ecapa")
        # savedir == source dir is what makes speechbrain resolve the local .ckpt instead of
        # delegating to the Hub. A fetched dir accumulates self-referential symlinks, which
        # is another reason weights never go inside a run dir (those must stay copyable).
        self.m = EncoderClassifier.from_hparams(source=d, hparams_file="hyperparams.yaml",
                                                savedir=d, run_opts={"device": cfg.resolved_device()})
        self.device = cfg.resolved_device()

    def __call__(self, windows, batch=64):
        import torch
        outs = []
        for i in range(0, len(windows), batch):
            b = windows[i:i + batch]
            m = max(len(w) for w in b)
            x = torch.from_numpy(np.stack([w if len(w) == m else np.pad(w, (0, m - len(w))) for w in b]))
            with torch.no_grad():
                outs.append(self.m.encode_batch(x.to(self.device)).squeeze(1).cpu().float().numpy())
        return np.concatenate(outs, 0).astype(np.float32)


class ERes2Net(Backend):
    """3D-Speaker ERes2NetV2 (192-d) via the ModelScope pipeline (needs a newer torch)."""
    name = "eres2net"
    dim = 192
    hint = 'pip install modelscope (and addict, einops, simplejson, sortedcontainers, pillow, datasets)'
    repo = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

    @classmethod
    def _probe(cls, cfg):
        from modelscope.pipelines import pipeline  # noqa: F401

    def __init__(self, cfg):
        super().__init__(cfg)
        os.environ["MODELSCOPE_CACHE"] = str(self.cache_root(cfg))
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipe = pipeline(task=Tasks.speaker_verification, model=self.repo,
                             device=cfg.resolved_device())

    def __call__(self, windows, batch=16):
        outs = []
        for i in range(0, len(windows), batch):
            b = windows[i:i + batch]
            e = np.asarray(self.pipe(b, output_emb=True)["embs"], dtype=np.float32)
            if len(e) == 1 and len(b) > 1:
                e = np.repeat(e, len(b), 0)
            outs.append(e[:len(b)])
        return np.concatenate(outs, 0)


BACKENDS = {c.name: c for c in (CampPlus, Ecapa, ERes2Net)}


def fetch_ecapa(cfg=None, endpoint: str = "https://hf-mirror.com", retries: int = 4) -> Path:
    """Pull the ECAPA snapshot with retries: anonymous mirrors rate-limit (HTTP 429) and a
    plain retry loop is enough; huggingface.co itself is often unreachable in CN networks.

    Reuses an existing copy wherever it was found - discovery first, download last.
    """
    import subprocess
    existing = cache.find("ecapa", cfg)
    if existing:
        print(f"  already cached: {existing}")
        return existing
    d = cache.download_root(cfg) / "ecapa"
    d.mkdir(parents=True, exist_ok=True)
    for f in Ecapa.files:
        dst = d / f
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        url = f"{endpoint}/{Ecapa.repo}/resolve/main/{f}"
        for attempt in range(retries):
            code = subprocess.run(["curl", "-sL", "--retry", "3", "--retry-delay", "5",
                                   "--retry-all-errors", "-m", "600", "-o", str(dst),
                                   "-w", "%{http_code}", url], capture_output=True, text=True).stdout.strip()
            if code == "200" and dst.stat().st_size > 1000:
                print(f"  {f}: ok ({dst.stat().st_size} bytes)")
                break
            print(f"  {f}: http {code}, retry {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
        else:
            raise BackendError(
                f"could not fetch {f}. Try another --endpoint (default {endpoint}) or place the "
                f"files manually in {d}")
    return d
