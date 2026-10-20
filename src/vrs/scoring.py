"""Re-score selected files against reference *lines* instead of centroids, and calibrate.

Centroid matching was measured at chance level on emotional dialogue (0.153 top-1 over 6
roles), so scoring is top-k mean similarity to that role's reference lines: "does it
resemble SOME lines of this role". Two variants, and the difference is the point:

  cohesive : references may include the file's own cluster. Legitimate for files that have
             no cluster at all; self-confirming for clustered files -- do not trust it there.
  cross    : every reference from the file's own cluster is removed, so a clustered file has
             to be recovered from other clusters of its role. Only this can catch a bad
             cluster, and it is only measurable for roles that span >=2 clusters.

Calibration runs leave-one-out on the gold tier (the lines a human actually listened to) and
reports accuracy per margin bucket, so thresholds are chosen from measured numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import l2

UNRES = "未定"      # appears in roles.csv and as the _未确定 folder name


def topk_mean(M, k):
    k = max(1, min(int(k), M.shape[1]))
    return np.partition(M, -k, axis=1)[:, -k:].mean(1)


def rank2(S):
    o = np.argsort(-S, axis=1)
    top = S[np.arange(len(S)), o[:, 0]]
    gap = top - S[np.arange(len(S)), o[:, 1]]
    return o[:, 0], top, gap


def build_references(labels, names, cfg, gold_files=()):
    """reference pool = typical members of named clusters (silver) + human-heard files (gold).

    Returns aligned (row indices, role names) plus the roles that span >=2 clusters, which is
    the only subset cross-cluster scoring can be validated on.
    """
    known = set(names.values())
    role = labels.role.to_numpy() if "role" in labels else np.array([names.get(c, UNRES) for c in labels.cluster])
    own_pct = pd.to_numeric(labels.own_pct, errors="coerce").fillna(-1).to_numpy()
    idx = {f: i for i, f in enumerate(labels.file)}
    silver = np.where((own_pct >= cfg.silver_pct) & np.isin(role, list(known)))[0]
    ref = set(silver.tolist()) | {idx[f] for f in gold_files if f in idx}
    ref = np.array(sorted(ref), dtype=int)
    counts = pd.Series([names[c] for c in set(labels.cluster) & set(names)]).value_counts()
    multi = {r for r, n in counts.items() if n >= 2}
    return ref, role[ref], sorted(multi), role


def score(A, ref_idx, ref_role, labels, cfg):
    """[N, R] score matrices for the cohesive and cross-cluster variants.

    ref_idx: row indices of reference files; ref_role: their role names (aligned arrays).
    """
    cluster = labels.cluster.to_numpy()
    role_names = sorted(set(ref_role))
    ref_role = np.asarray(ref_role)
    A = l2(np.asarray(A, np.float32))                 # cosine needs unit rows; don't trust callers
    ref_idx = np.asarray(ref_idx, dtype=int)
    Sl = np.full((len(A), len(role_names)), -1.0, np.float32)
    Sx = np.full_like(Sl, -1.0)
    for j, rn in enumerate(role_names):
        sel = np.asarray(ref_idx)[ref_role == rn]
        M = A @ A[sel].T
        pos = {s: k for k, s in enumerate(sel)}
        M[list(pos), list(pos.values())] = -1.0          # never let a file cite itself
        Sl[:, j] = topk_mean(M, cfg.k_neighbors)
        Mx = M.copy()
        for c in set(cluster[sel]):                       # leave-own-cluster-out
            rows = np.where(cluster == c)[0]
            cols = np.where(cluster[sel] == c)[0]
            if len(rows) and len(cols):
                Mx[np.ix_(rows, cols)] = -1.0
        Sx[:, j] = topk_mean(Mx, cfg.k_neighbors)
    return role_names, Sl, Sx


def calibrate(gold_df, labels, role_names, Sl, Sx, multi):
    """leave-one-out accuracy of the cross score, bucketed by margin"""
    idx = {f: i for i, f in enumerate(labels.file)}
    role_of = labels.role.to_numpy()
    rows = []
    jc, tc, mc = rank2(Sl)
    jx, tx, mx = rank2(Sx)
    for f, gr in gold_df[["file", "role"]].itertuples(index=False):
        if f not in idx:
            continue
        i = idx[f]
        rows.append(dict(file=f, gold=gr,
                         cohesive_role=role_names[jc[i]], cohesive_margin=round(float(mc[i]), 4),
                         cross_role=role_names[jx[i]], cross_margin=round(float(mx[i]), 4),
                         testable=gr in multi, hit_cohesive=role_names[jc[i]] == gr,
                         hit_cross=role_names[jx[i]] == gr))
    cal = pd.DataFrame(rows)
    t = cal[cal.testable]
    buckets = pd.cut(t.cross_margin, [-9, 0, 0.03, 0.06, 0.10, 0.15, 9],
                     labels=["<0", "0-.03", ".03-.06", ".06-.10", ".10-.15", ">.15"])
    curve = t.groupby(buckets, observed=True).agg(n=("hit_cross", "size"), acc=("hit_cross", "mean")).round(3)
    return cal, curve


def decide(labels, cfg, role_names, Sl, Sx, multi, scope: set | None = None):
    """one verdict per file; with cfg.only_unassigned clustered files are never touched"""
    role = labels.role.to_numpy()
    cluster = labels.cluster.to_numpy()
    ri = {r: j for j, r in enumerate(role_names)}
    has_vec = labels.has_vector.to_numpy() if "has_vector" in labels else np.ones(len(labels), bool)
    jc, tc, mc = rank2(Sl)
    jx, tx, mx = rank2(Sx)
    own_x = np.array([Sx[i, ri[role[i]]] if role[i] in ri else np.nan for i in range(len(labels))])
    other_x = np.full(len(labels), np.nan)
    for i in range(len(labels)):
        if role[i] in ri:
            others = [j for j in range(len(role_names)) if role_names[j] != role[i]]
            if others:
                other_x[i] = Sx[i, others].max()
    delta = own_x - other_x

    verdict, suggest, basis = [], [], []
    for i in range(len(labels)):
        cur = role[i] if role[i] in ri else UNRES
        scoped = scope is None or labels.file.iloc[i] in scope
        if not scoped:
            v, s, b = "skip", cur, "不在复判范围"
        elif cfg.only_unassigned and cur != UNRES:
            v, s, b = "keep", cur, "原簇保留"
        elif not has_vec[i]:
            v, s, b = "manual", cur, "无可用声纹向量"
        elif cur == UNRES:
            if mc[i] >= cfg.assign_margin and role_names[jc[i]] in multi:
                v, s, b = "assign", role_names[jc[i]], "cohesive"
            else:
                v, s, b = "manual", role_names[jc[i]], "cohesive边际不足"
        elif cur not in multi:
            v, s, b = "keep", cur, "单簇角色无法跨簇验证"
        elif np.isnan(delta[i]) or other_x[i] < 0:
            v, s, b = "manual", cur, "跨簇参照不足"
        elif delta[i] >= cfg.keep_margin:
            v, s, b = "confirm", cur, "cross"
        elif delta[i] <= -cfg.keep_margin:
            others = [j for j in range(len(role_names)) if role_names[j] != cur]
            v, s, b = "propose", role_names[others[int(np.argmax(Sx[i, others]))]], "cross改判"
        else:
            v, s, b = "manual", cur, "cross边际不足"
        verdict.append(v); suggest.append(s); basis.append(b)
    return pd.DataFrame(dict(file=labels.file, verdict=verdict, suggested=suggest, basis=basis,
                             cohesive_role=[role_names[j] for j in jc],
                             cross_role=[role_names[j] for j in jx],
                             cohesive_margin=np.round(mc, 4), cross_delta=np.round(delta, 4)))


def apply(d: pd.DataFrame, labels) -> tuple:
    """merge verdicts into the final role column; manual/propose never overwrite silently"""
    final = labels.role.to_numpy().astype(object)
    note = np.array([None] * len(labels), dtype=object)
    v = d.verdict.to_numpy()
    for i in range(len(labels)):
        cur, sug = labels.role.iloc[i], d.suggested.iloc[i]
        if v[i] == "skip":
            final[i], note[i] = cur, "未复判"
        elif v[i] in ("keep", "confirm"):
            final[i], note[i] = cur, d.basis.iloc[i]
        elif v[i] == "assign":
            tier = "auto93" if d.cohesive_margin.iloc[i] >= 0.06 else "auto87"
            final[i], note[i] = sug, f"自动归属({tier})"
        elif v[i] == "propose":
            final[i], note[i] = cur, f"建议改判→{sug}(不自动应用)"
        else:
            final[i], note[i] = cur, "需人工"
    return final, note
