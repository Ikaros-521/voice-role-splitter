"""Human-label ingest, role propagation, tone naming and folder export.

The tool never overwrites a human decision with a model opinion. Model output is used for
three things only: naming unlabeled clusters is the human's job, filling files that had no
usable speech, and flagging risk. Anything the model merely "suggests" stays in the review
list with its suggestion attached.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import place
from .scoring import UNRES

UNRESOLVED_LABELS = (UNRES, "", "unresolved")


def read_export(*paths) -> dict:
    """labels_manual.csv from the annotation page: cluster names, per-file roles, spot verdicts"""
    out = dict(cluster_names={}, file_roles={}, spot_yes=set(), spot_no=set())
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        m = pd.read_csv(p, encoding="utf-8-sig").fillna("")
        for r in m[m.type == "cluster"].itertuples():
            if r.role:
                out["cluster_names"][int(r.cluster)] = r.role
        for r in m[m.type == "file"].itertuples():
            if r.role and r.file:
                out["file_roles"][r.file] = r.role
        for r in m[m.type == "spot"].itertuples():
            if r.verdict == "yes":
                out["spot_yes"].add(r.file)
            elif r.verdict == "no":
                out["spot_no"].add(r.file)
    return out


def confidence(df: pd.DataFrame) -> np.ndarray:
    role = df.role.to_numpy()
    own = pd.to_numeric(df.get("own_pct"), errors="coerce")
    sec = pd.to_numeric(df.get("speech_sec"), errors="coerce").fillna(0)
    embedded = df.get("embedded", pd.Series(True, index=df.index)).to_numpy()
    unres = np.isin(role, list(UNRESOLVED_LABELS))
    return np.select(
        [unres, ~embedded, own < 0.15, (own < 0.35) | (sec < 1.5)],
        ["low", "low", "low", "mid"], default="high")


def propagate(cfg, labels: pd.DataFrame, export: dict, final: bool = False) -> pd.DataFrame:
    """apply cluster -> role naming (and per-file overrides) to every row"""
    names = export["cluster_names"]
    df = labels.copy()
    role = np.array([names.get(int(c), UNRES) for c in df.cluster], dtype=object)
    # unresolved files keep whatever the refine pass assigned, if that column exists
    if "suggested" in df and "verdict" in df:
        assign = df.verdict.to_numpy() == "assign"
        role = np.where(assign, df.suggested.to_numpy(), role)
    for f, r in export["file_roles"].items():
        role[df.file.to_numpy() == f] = r
    role = np.where(np.isin(df.file, list(export["spot_no"])), "需复核", role)
    df["role"] = role
    df["confidence"] = confidence(df)
    return df


def tone_names(cfg, df: pd.DataFrame) -> tuple:
    """human-readable tone label per cluster, from loudness / pitch / line length"""
    z = np.load(cfg.path("meta.npz"))
    rms = z["rms"] if "rms" in z.files else np.zeros(len(df), np.float32)
    d = df.copy()
    d["rms"] = rms[:len(d)] if len(rms) >= len(d) else np.pad(rms, (0, len(d) - len(rms)))
    d["rms_rank"] = pd.to_numeric(d.rms, errors="coerce").rank(pct=True)
    out, stats = {}, []
    for c, g in d.groupby("cluster"):
        sec = float(pd.to_numeric(g.speech_sec, errors="coerce").median())
        rr = float(g.rms_rank.median())
        hzv = pd.to_numeric(g["f0"], errors="coerce").dropna() if "f0" in g else pd.Series(dtype=float)
        spv = pd.to_numeric(g["f0_spread"], errors="coerce").dropna() if "f0_spread" in g else pd.Series(dtype=float)
        hz = float(hzv.median()) if len(hzv) else np.nan
        spread = float(spv.median()) if len(spv) else np.nan
        if c < 0:
            t = "无有效人声"
        elif sec <= 1.2:
            t = "短句语气词"
        elif rr >= 0.70 and spread == spread and spread >= 90:
            t = "激动喊叫"
        elif rr >= 0.62:
            t = "高声强调"
        elif rr <= 0.30 and sec <= 3.0:
            t = "低语轻柔"
        elif sec >= 4.5 and rr <= 0.45:
            t = "平稳长句"
        elif hz == hz and hz >= 330 and rr >= 0.5:
            t = "高亢活泼"
        elif hz == hz and hz <= 150:
            t = "低沉平静"
        else:
            t = "常规对话"
        out[int(c)] = t
        stats.append(dict(cluster=int(c), files=int(len(g)), median_sec=round(sec, 2),
                          rms_rank=round(rr, 3), f0_median=None if hz != hz else round(hz, 1),
                          f0_spread=None if spread != spread else round(spread, 1), tone=t))
    return out, pd.DataFrame(stats)


def export_folders(cfg, df: pd.DataFrame, mode="link", tone=True, roles=None, out_dir="roles") -> dict:
    """roles/<角色>/c<簇号>_<语气>/<file>.wav ; unresolved files land in _未确定"""
    dest = Path(cfg.root) / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
    names = {}
    if tone:
        names, stats = tone_names(cfg, df)
        cfg.path("clusters.csv").parent.mkdir(exist_ok=True)
        stats.to_csv(cfg.path("clusters.csv"), index=False, encoding="utf-8-sig")
    want = roles or sorted(set(df.role))
    src = Path(cfg.audio_dir)
    placed, modes, skipped = {}, set(), 0
    for r in want:
        rows = df[df.role == r]
        base = dest / ("_未确定" if r in UNRESOLVED_LABELS or r == "需复核" else str(r))
        n = 0
        for _, g in rows.groupby("cluster"):
            folder = base if not tone else base / f"c{int(g.cluster.iloc[0]):02d}_{names.get(int(g.cluster.iloc[0]), '未分类')}"
            for f in g.file:
                s = src / f
                if not s.exists():
                    skipped += 1
                    continue
                modes.add(place(s, folder / f, mode))
                n += 1
        placed[str(r)] = n
    return dict(placed=placed, modes=sorted(modes), skipped_missing=skipped, dest=str(dest))


def write_roles(cfg, df: pd.DataFrame, name="roles.csv") -> Path:
    cols = [c for c in ("file", "role", "cluster", "centroid_score", "own_pct", "role_margin",
                        "cohesive_margin", "cross_delta", "verdict", "suggested", "basis",
                        "refine_note", "confidence", "embedded", "has_vector", "speech_sec",
                        "f0", "text") if c in df.columns]
    p = cfg.path(name)
    df[cols].to_csv(p, index=False, encoding="utf-8-sig")
    rv = df[df.role.isin(list(UNRESOLVED_LABELS)) | (df.get("confidence") == "low")
            | (pd.to_numeric(df.get("own_pct"), errors="coerce") < 0.15)]
    rv.to_csv(cfg.path("review_list.csv"), index=False, encoding="utf-8-sig")
    summary = df.groupby("role").agg(files=("file", "size"), seconds=("speech_sec", "sum"),
                                    mean_own_pct=("own_pct", "mean")).sort_values("files", ascending=False)
    summary.to_csv(cfg.path("roles_summary.csv"), encoding="utf-8-sig")
    print(summary.to_string())
    print(f"-> {p}\n-> {cfg.path('review_list.csv')} ({len(rv)} rows)")
    return p
