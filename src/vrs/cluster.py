"""Fuse backbones, build one linkage tree, sweep k, and pick a cut.

Choices that took real experiments to reach, and are therefore encoded rather than left to
the caller:

* thresholds are not transferable between backbones (CAM++ and ECAPA distance scales differ
  by ~2x), so candidates are k values on one shared fused space, and per-backbone agreement
  is measured by centroid assignment, not by an independent cut.
* silhouette keeps rising as clusters merge, so it cannot choose k alone; it is ranked
  together with cross-backbone agreement (chance-corrected), intra-cluster ASV pass rate and
  intra-cluster pitch spread -- the last two are premise-free.
* clusters come from average linkage (connectivity-based), so "a backbone's nearest centroid
  is not my cluster" is expected and carries no information. Do not score that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .models import l2

# members needed before a cluster has a meaningful pitch spread at all
PITCH_MIN_MEMBERS = 3


def center(F):
    return l2(F - np.asarray(F, np.float32).mean(0, keepdims=True))


def asnorm(S, topk=20):
    """adaptive symmetric normalisation: removes the per-session/emotion level bias"""
    S = np.asarray(S, np.float32)
    n = len(S)
    idx = np.argsort(-S, axis=1)[:, 1:topk + 1]
    rows = np.arange(n)[:, None]
    mu = S[rows, idx].mean(1, keepdims=True)
    sd = S[rows, idx].std(1, keepdims=True) + 1e-6
    base = (S - mu) / sd + (S - mu.T) / (sd.T) - (S - S.mean()) / (S.std() + 1e-6)
    return base.astype(np.float32)


def centroid_map(labels, E):
    return {c: l2(E[labels == c].mean(0, keepdims=True))[0] for c in np.unique(labels)}


def nearest_labels(E, cent):
    keys = np.array(sorted(cent))
    C = np.stack([cent[k] for k in keys])
    return keys[(np.asarray(E, np.float32) @ C.T).argmax(1)]


def merge_small(labels, E, floor):
    """fold clusters below `floor` into their nearest surviving centroid.

    Emptied labels must be skipped: np.bincount reports size 0 for them, they stay under the
    floor forever, and the loop then never converges (this hung two full runs at 100% CPU).
    """
    labels = np.asarray(labels).copy()
    while True:
        cent = centroid_map(labels, E)
        if len(cent) <= 1:
            return labels
        sizes = np.bincount(labels)
        small = [c for c, s in enumerate(sizes) if 0 < s < floor]
        if not small:
            return labels
        victim = min(small, key=lambda c: sizes[c])
        others = {c: v for c, v in cent.items() if c != victim}
        idx = np.where(labels == victim)[0]
        labels[idx] = nearest_labels(E[idx], others)


def fuse(embs: dict, cfg, weights: dict | None = None):
    """per-model prepared space + the fused space used for clustering"""
    names = list(embs)
    weights = weights or {n: 1.0 / len(names) for n in names}
    total = sum(weights.values())
    weights = {n: weights[n] / total for n in names}
    prep = center if cfg.center else l2
    per = {n: prep(np.asarray(embs[n], np.float32)) for n in names}
    A = l2(sum(weights[n] * per[n] for n in names))
    return A, per, weights


def eligible(meta: pd.DataFrame, embs: dict, cfg):
    """files with usable speech for at least one backend and long enough to cluster"""
    stacked = np.stack([np.asarray(v, np.float32) for v in embs.values()])
    has_vec = (np.linalg.norm(stacked, axis=2) > 0).any(0)
    return has_vec & (meta.speech_sec.to_numpy() >= cfg.cluster_min_speech), has_vec


def sweep(A, per, has, cfg, pitch=None, cam=None, verbose=True):
    """build the tree once, then evaluate every candidate k"""
    E = A[has]
    S = E @ E.T
    M = asnorm(S) if cfg.score == "asnorm" else S
    D = (M.max() - M).astype(np.float64)
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=cfg.linkage)
    sub = np.random.default_rng(cfg.seed).choice(len(E), size=min(3000, len(E)), replace=False)
    rng = np.random.default_rng(cfg.seed + 3)
    ip, jp = rng.integers(0, len(E), 30000), rng.integers(0, len(E), 30000)
    camS = None
    if cam is not None:
        camS = cam[has] @ cam[has].T
    rows = []
    for k in cfg.ks:
        if k >= len(E):
            continue
        lab = fcluster(Z, t=k, criterion="maxclust") - 1
        sizes = np.bincount(lab)
        rec = {"k": int(k), "n_clusters": int(len(sizes)), "min_size": int(sizes.min()),
               "largest_frac": round(float(sizes.max() / len(lab)), 3),
               "median_size": int(np.median(sizes))}
        ls = lab[sub]
        if 1 < len(np.unique(ls)) < len(ls):
            from sklearn.metrics import silhouette_score
            rec["silhouette"] = round(float(silhouette_score(E[sub], ls, metric="cosine")), 4)
        cent = centroid_map(lab, E)
        frac = sizes / sizes.sum()
        pe = float((frac ** 2).sum())
        po = float(np.mean([np.mean(nearest_labels(v[has], cent) == lab) for v in per.values()]))
        rec["model_agree"] = round(po, 4)
        rec["agree_kappa"] = round((po - pe) / (1 - pe), 4) if pe < 1 else 0.0
        if camS is not None:
            same = (lab[ip] == lab[jp]) & (ip != jp)
            rec["verify_pass"] = round(float((camS[ip[same], jp[same]] >= 0.33).mean()), 4)
            diff = ip != jp
            rec["verify_pass_random"] = round(float((camS[ip[diff], jp[diff]] >= 0.33).mean()), 4)
        if pitch is not None:
            fv = pitch[has]
            disp = [(sizes[c], float(np.nanpercentile(fv[lab == c], 90) - np.nanpercentile(fv[lab == c], 10)))
                    for c in range(len(sizes)) if np.isfinite(fv[lab == c]).sum() >= PITCH_MIN_MEMBERS]
            if disp:
                rec["f0_disp"] = round(float(np.average([d for _, d in disp],
                                                        weights=[w for w, _ in disp])), 1)
        rows.append(rec)
        if verbose:
            print(rec, flush=True)
    return Z, pd.DataFrame(rows), E


def choose(sweep_df: pd.DataFrame, force_k: int = 0):
    ok = sweep_df.dropna(subset=[c for c in ("agree_kappa", "verify_pass", "f0_disp", "silhouette")
                                 if c in sweep_df.columns])
    if ok.empty:
        raise ValueError("sweep produced no scorable k; widen ks or check embeddings")
    score = sum(ok[c].rank(pct=True) * s for c, s in
                (("agree_kappa", 1.0), ("verify_pass", 1.0), ("f0_disp", -1.0), ("silhouette", 0.6))
                if c in ok.columns)
    if force_k:
        sel = ok[ok.k == force_k]
        if sel.empty:
            raise ValueError(f"k={force_k} is not in the sweep")
        return int(sel.iloc[0].k), sel.iloc[0]
    return int(ok.loc[score.idxmax()].k), ok.loc[score.idxmax()]


def run(cfg, embs=None, force_k=0):
    from . import embed as _embed
    mz = np.load(cfg.path("meta.npz"), allow_pickle=True)
    meta = pd.DataFrame({"file": mz["files"], "speech_sec": mz["speech_sec"]})
    embs = embs or _embed.load(cfg)
    A, per, weights = fuse(embs, cfg)
    has, has_vec = eligible(meta, embs, cfg)
    pitch = None
    pf = cfg.path("pitch.npz")
    if pf.exists():
        d = np.load(pf, allow_pickle=True)
        pitch = d["f0"]
        if len(pitch) != len(A):
            pitch = None
    cam = per.get("campplus")
    print(f"clustering {int(has.sum())}/{len(A)} files "
          f"(linkage={cfg.linkage} score={cfg.score} center={cfg.center})", flush=True)
    Z, sweep_df, E = sweep(A, per, has, cfg, pitch=pitch, cam=cam)
    sweep_df.to_csv(cfg.path("cluster_sweep.csv"), index=False)
    print("\n" + sweep_df.to_string(index=False), flush=True)
    k, best = choose(sweep_df, force_k)
    print(f"\nselected k={k} kappa={best.agree_kappa} verify_pass={best.get('verify_pass')} "
          f"f0_disp={best.get('f0_disp')} silhouette={best.get('silhouette')}", flush=True)

    lab = np.full(len(A), -1, dtype=int)
    # a fixed floor would swallow every cluster on a small corpus: cap it by dataset size
    floor = min(cfg.min_cluster, max(2, len(E) // (2 * max(k, 1))))
    if floor < cfg.min_cluster:
        print(f"min_cluster lowered {cfg.min_cluster} -> {floor} for {len(E)} files", flush=True)
    lab[has] = merge_small(fcluster(Z, t=k, criterion="maxclust") - 1, E, floor)
    cent = centroid_map(lab[has], A[has])
    order = {c: i for i, c in enumerate(pd.Series(lab[has]).value_counts().index)}
    lab = np.where(lab >= 0, [order.get(c, -1) for c in lab], -1)
    cent = {order[c]: v for c, v in cent.items()}
    keys = np.array(sorted(cent))
    C = np.stack([cent[c] for c in keys])
    col = {c: i for i, c in enumerate(keys)}
    own_c = np.array([col.get(c, -1) for c in lab], dtype=int)
    own_ok = own_c >= 0
    SC = A @ C.T
    own_s = np.where(own_ok, SC[np.arange(len(A)), np.maximum(own_c, 0)], np.nan)
    out = pd.DataFrame({"file": meta.file, "cluster": lab, "speech_sec": meta.speech_sec,
                        "embedded": has, "has_vector": has_vec,
                        "centroid_score": np.round(own_s, 4)})
    out["own_pct"] = np.where(own_ok, out.groupby("cluster").centroid_score.rank(pct=True), np.nan)
    if pitch is not None:
        out["f0"] = pitch
    for name, v in per.items():
        out[f"clu_{name}"] = [keys[j] for j in (v @ C.T).argmax(1)]
    out = out.sort_values(["cluster", "own_pct"])
    out.to_csv(cfg.path("labels.csv"), index=False)
    cfg.path("cluster_model.json").write_text(json.dumps(
        {"k": int(k), "linkage": cfg.linkage, "score": cfg.score, "center": cfg.center,
         "weights": weights, "selected": {a: (None if pd.isna(b) else float(b)) for a, b in best.items()},
         "clusters": [int(c) for c in keys]}, ensure_ascii=False, indent=1), "utf-8")
    vc = out.cluster.value_counts().sort_index()
    print("\ncluster sizes:\n" + vc.to_string())
    print(f"\nno-cluster files: {int((lab < 0).sum())}")
    print(f"-> {cfg.path('labels.csv')}")
    return out
