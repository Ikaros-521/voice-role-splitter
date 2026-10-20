"""Build the annotation pages from run artifacts.

One template, two payload shapes:
  clusters -- name each cluster (this is the step that only a human can do)
  items    -- confirm the model's suggestion per leftover file, grouped so a whole group
              can be accepted at once and sampled afterwards

Audio is referenced by path relative to the generated HTML, so the page works over file://
in a normal browser; `vrs serve` covers browsers that refuse media from file://.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).parent / "templates" / "label_page.html"


def _href(page: Path, target: Path) -> str:
    """path relative to the generated HTML, so the page works over file:// without a server"""
    r = os.path.relpath(str(target.resolve()), str(page.parent.resolve()))
    return r.replace("\\", "/")


def _roles(cfg):
    """dropdown entries; run.yaml `roles` is [[value, label], ...]"""
    rs = [list(x) for x in (cfg.roles or []) if isinstance(x, (list, tuple)) and len(x) == 2]
    if not rs:
        rs = [["unresolved", "— 其他/待定 —"]]
    return rs


def _num(x):
    try:
        f = float(x)
        return None if not np.isfinite(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


def cluster_payload(cfg, page: Path, labels: pd.DataFrame, per_cluster=6, spot=3, seed=0):
    texts = {}
    tf = cfg.root / "transcripts.csv"
    if tf.exists():
        t = pd.read_csv(tf, keep_default_na=False, encoding="utf-8-sig").drop_duplicates("file").set_index("file")
        texts = t.text.to_dict()
    has_f0 = "f0" in labels.columns
    src = Path(cfg.audio_dir).resolve()
    groups = []
    for c, g in labels.groupby("cluster"):
        if c < 0:
            continue
        g = g[g.embedded].sort_values("own_pct", ascending=False)
        if not len(g):
            continue
        f0med = _num(np.nanmedian(pd.to_numeric(g.f0, errors="coerce"))) if has_f0 else None
        ex = list(g.head(per_cluster).itertuples())
        pool = list(g.iloc[per_cluster:].itertuples())
        if pool and spot:
            rng = np.random.default_rng(seed)
            ex += [pool[i] for i in sorted(rng.choice(len(pool), size=min(spot, len(pool)), replace=False))]
        groups.append(dict(id=int(c), title=f"簇 {int(c)}", suggested="",
                           tags=[f"全簇 {len(g)} 条", f"{round(float(g.speech_sec.sum()), 0)} 秒"]
                                + ([f"F0 {f0med}"] if f0med is not None else []),
                           items=[dict(file=r.file, src=_href(page, src / r.file),
                                       dur=_num(r.speech_sec),
                                       tags=([f"F0 {_num(getattr(r, 'f0', np.nan))}"] if has_f0 else [])
                                            + [f"典型度 {_num(getattr(r, 'own_pct', np.nan))}"],
                                       text=str(texts.get(r.file, ""))[:150]) for r in ex]))
    return groups


def item_payload(cfg, page: Path, queue: pd.DataFrame, group_key="suggested"):
    src = Path(cfg.audio_dir).resolve()
    texts = {}
    tf = cfg.root / "transcripts.csv"
    if tf.exists():
        t = pd.read_csv(tf, keep_default_na=False, encoding="utf-8-sig").drop_duplicates("file").set_index("file")
        texts = t.text.to_dict()
    groups = []
    for key, g in queue.groupby(group_key):
        groups.append(dict(id=str(key), title=f"建议：{key}", suggested=str(key), tags=[],
                           items=[dict(file=r.file, src=_href(page, src / r.file),
                                       dur=_num(r.speech_sec),
                                       tags=[f"F0 {_num(getattr(r, 'f0', np.nan))}",
                                             f"原簇:{getattr(r, 'cluster_role', getattr(r, 'role', ''))}",
                                             f"coh {_num(getattr(r, 'cohesive_margin', np.nan))}"],
                                       text=str(texts.get(r.file, ""))[:150]) for r in g.itertuples()]))
    groups.sort(key=lambda x: -len(x["items"]))
    return groups


def write_page(cfg, mode, groups, out=None, title=None, export_name=None, audio_rel=None):
    page = Path(out) if out else (cfg.root / ("标注.html" if mode == "clusters" else "待人工.html"))
    page.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(mode=mode, roles=_roles(cfg), groups=groups,
                   storage_key=f"vrs_{cfg.root.name}_{mode}",
                   export_name=export_name or ("labels_manual.csv" if mode == "clusters"
                                               else "labels_manual_review.csv"),
                   title=title or ("簇-角色命名" if mode == "clusters" else "剩余文件复核"))
    html = (TEMPLATE.read_text("utf-8")
            .replace("__TITLE__", payload["title"])
            .replace("__EXPORT_NAME__", payload["export_name"])
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    page.write_text(html, "utf-8")
    n = sum(len(g["items"]) for g in groups)
    print(f'{mode} page: {len(groups)} groups, {n} clips -> {page}')
    print(f'  打开方式：双击该文件，或 vrs serve --run {cfg.run_dir}（浏览器禁止 file:// 播放时用后者）')
    return page
