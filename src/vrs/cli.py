"""`vrs` command line.

Workflow (each step is idempotent and writes into the run dir, so you can stop anywhere):

  vrs init      --audio-dir ./wav/voice --run-dir runs/show1 --roles-file my_cast.yaml
  vrs status
  vrs embed             # speaker embeddings per backend (+ a health check)
  vrs pitch             # F0 stats, used as a model-independent QC signal
  vrs cluster           # fuse, sweep k, cut -> labels.csv
  vrs page clusters     # 标注.html : name each cluster by ear
  vrs apply             # ingest the export -> roles.csv (cluster naming propagated)
  vrs refine            # re-score leftovers with calibrated top-k scoring -> roles_final.csv
  vrs page review       # 待人工.html : the leftovers that are still not trustworthy
  vrs apply --export labels_manual_review.csv
  vrs export            # roles/<角色>/c<簇>_<语气>/*.wav   (hard links by default)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio as A
from . import cluster as CL
from . import embed as EM
from . import labeling as LB
from . import pitch as PT
from . import roles as RO
from . import scoring as SC
from .config import Config, ConfigError
from .models import BACKENDS, fetch_ecapa, l2


def _load_cfg(args) -> Config:
    return Config.load(getattr(args, "run_dir", None) or "runs/default")


# ---------------------------------------------------------------- commands
def cmd_init(args):
    cfg = Config(audio_dir=str(Path(args.audio_dir).resolve()))
    if args.run_dir:
        cfg.run_dir = args.run_dir
    for m in args.models or []:
        cfg.models = m.split(",")
    if args.roles_file:
        data = json.loads(Path(args.roles_file).read_text("utf-8")) \
            if args.roles_file.endswith(".json") else __import__("yaml").safe_load(
            Path(args.roles_file).read_text("utf-8"))
        cfg.roles = [[str(k), str(v)] for k, v in data["roles"].items()] if isinstance(data.get("roles"), dict) \
            else [list(x) for x in data["roles"]]
    for s in args.set or []:
        k, _, v = s.partition("=")
        if not hasattr(cfg, k):
            raise ConfigError(f"unknown config key: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, list):
            import yaml
            v = yaml.safe_load(v)
        else:
            v = type(cur)(v) if cur is not None else v
        setattr(cfg, k, v)
    cfg.validate()
    f = cfg.save()
    print(f"run created: {f}\n  audio_dir : {cfg.audio_dir}\n  models    : {cfg.models}\n"
          f"  device    : {cfg.resolved_device()}")


def cmd_status(args):
    cfg = _load_cfg(args)
    r = cfg.root
    steps = [("meta/embeddings", any((r / "embeddings").glob("*.npy")) if (r / "embeddings").exists() else False),
             ("pitch.npz", (r / "pitch.npz").exists()),
             ("cluster_sweep.csv", (r / "cluster_sweep.csv").exists()),
             ("labels.csv", (r / "labels.csv").exists()),
             ("roles.csv (needs labels_manual.csv)", (r / "roles.csv").exists()),
             ("roles_final.csv", (r / "roles_final.csv").exists()),
             ("review_list.csv", (r / "review_list.csv").exists())]
    for k, v in steps:
        print(f"  [{'x' if v else ' '}] {k}")
    for name in ("labels_manual.csv", "labels_manual_review.csv"):
        p = r / name
        print(f"  [{'x' if p.exists() else ' '}] {name}"
              + (f" ({len(pd.read_csv(p, encoding='utf-8-sig'))} rows)" if p.exists() else " (未导出)"))
    if (r / "embeddings" / "manifest.json").exists():
        h = json.loads((r / "embeddings" / "manifest.json").read_text("utf-8")).get("last_health", {})
        print(f"  embedding health: {h.get('note', 'unknown')}")
    if (r / "roles_final.csv").exists():
        df = pd.read_csv(r / "roles_final.csv").fillna("")
        print("\n  role totals:")
        print(df.role.value_counts().to_string())
        print(f"\n  待人工 (review_list.csv): {len(pd.read_csv(r / 'review_list.csv')) if (r / 'review_list.csv').exists() else 0} 条")


def cmd_probe(args):
    cfg = _load_cfg(args)
    df = A.probe(cfg.audio_dir, cfg.sample_rate)
    dur = df.duration.sum()
    print(f"{len(df)} files | {dur/3600:.2f} h | median {df.duration.median():.2f}s | "
          f"<1s {int((df.duration < 1).sum())} | <0.6s {int((df.duration < 0.6).sum())}")
    odd = df[df.needs_resample]
    if len(odd):
        print(f"  note: {len(odd)} files at a different sample rate; they will be resampled to {cfg.sample_rate} Hz")
    df.to_csv(cfg.path("probe.csv"), index=False)
    print(f"-> {cfg.path('probe.csv')}")


def cmd_backends(args):
    cfg = Config.load(args.run_dir) if (Path(args.run_dir) / "run.yaml").exists() else Config(audio_dir=".")
    if args.fetch:
        print(f"ecapa weights -> {fetch_ecapa(cfg, endpoint=args.endpoint)}")
    print(f"device: {cfg.resolved_device()}   "
          f"download root when nothing is cached: {cfg.resolved_model_dir()}")
    print("search order: run.yaml model_dir > $VRS_MODEL_DIR > $MODELSCOPE_CACHE > "
          "~/.cache/modelscope > $HF_HOME / $HF_HUB_CACHE > ~/.cache/huggingface > ./models")
    for name, cls in BACKENDS.items():
        st = cls.status(cfg)
        print(f"  {name:<10} {st}")
        print(f"  {'':<10} {cls.where(cfg)}")
        if st != "ready":
            print(f"  {'':<10} install: {cls.hint}")
    print("\nrun.yaml `models` should list the ready ones.")


def cmd_pitch(args):
    PT.measure(_load_cfg(args), workers=args.workers)


def cmd_embed(args):
    cfg = _load_cfg(args)
    EM.embed(cfg, models=args.models.split(",") if args.models else None, progress=args.progress)


def cmd_cluster(args):
    cfg = _load_cfg(args)
    if args.linkage:
        cfg.linkage = args.linkage
    if args.score:
        cfg.score = args.score
    if args.ks:
        cfg.ks = [int(x) for x in args.ks.split(",")]
    if args.min_cluster is not None:
        cfg.min_cluster = args.min_cluster
    CL.run(cfg, force_k=args.k)


def cmd_page(args):
    cfg = _load_cfg(args)
    if args.kind == "clusters":
        labels = pd.read_csv(cfg.path("labels.csv"))
        page = cfg.root / (args.out or "标注.html")
        groups = LB.cluster_payload(cfg, page, labels, args.per_cluster, args.spot, cfg.seed)
        LB.write_page(cfg, "clusters", groups, out=page)
    else:
        q = pd.read_csv(cfg.path("queue.csv"))
        page = cfg.root / (args.out or "待人工.html")
        groups = LB.item_payload(cfg, page, q)
        LB.write_page(cfg, "items", groups, out=page)


def cmd_apply(args):
    cfg = _load_cfg(args)
    labels = pd.read_csv(cfg.path("labels.csv"))
    exports = list(args.export or []) or ["labels_manual.csv", "labels_manual_review.csv"]
    exp = RO.read_export(*[cfg.root / e for e in exports],
                         *[Path(e) for e in exports if Path(e).exists()])
    if not exp["cluster_names"] and not exp["file_roles"]:
        raise SystemExit("no names found in the exports -- open the page, name the clusters, "
                         "then export labels_manual.csv into the run dir")
    df = RO.propagate(cfg, labels, exp)
    if exp["spot_no"]:
        print(f"spot-check rejected {len(exp['spot_no'])} files -> 需复核")
    print(f"named clusters: {len(exp['cluster_names'])} | per-file labels: {len(exp['file_roles'])}")
    RO.write_roles(cfg, df, name=args.name or "roles.csv")


def cmd_refine(args):
    cfg = _load_cfg(args)
    if args.only_unassigned is not None:
        cfg.only_unassigned = args.only_unassigned
    for k in ("k_neighbors", "silver_pct", "keep_margin", "assign_margin"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    embs = EM.load(cfg)
    A_fused, per, weights = CL.fuse(embs, cfg)
    labels = pd.read_csv(cfg.path(args.base or "roles.csv"))
    names = {int(c): r for c, r in
             labels[labels.cluster >= 0].groupby("cluster").role.agg(lambda s: s.mode().iloc[0]).items()}
    gold_rows = []
    for f in (cfg.root / "labels_manual.csv", cfg.root / "labels_manual_review.csv"):
        if f.exists():
            m = pd.read_csv(f, encoding="utf-8-sig").fillna("")
            gold_rows += list(m[m.type == "spot"].file) + [r.file for r in m[m.type == "file"].itertuples() if r.role]
    ref_idx, ref_role, multi, _ = SC.build_references(labels, names, cfg, gold_files=set(gold_rows))
    if not len(ref_idx):
        raise SystemExit("no reference files: run `vrs apply` first (references come from named clusters)")
    print(f"references={len(ref_idx)} roles={sorted(set(ref_role))} cross-testable={multi}", flush=True)
    role_names, Sl, Sx = SC.score(A_fused, ref_idx, ref_role, labels, cfg)
    d = SC.decide(labels, cfg, role_names, Sl, Sx, multi,
                  scope=None if not args.scope_review else set(pd.read_csv(cfg.path("review_list.csv")).file))
    cal = None
    if gold_rows:
        gdf = pd.DataFrame({"file": sorted(set(gold_rows))})
        gdf["role"] = gdf.file.map(dict(zip(labels.file, labels.role)))
        gdf = gdf.dropna()
        cal, curve = SC.calibrate(gdf, labels, role_names, Sl, Sx, set(multi))
        cal.to_csv(cfg.path("gold_calibration.csv"), index=False)
        print(f"\ngold LOO accuracy: cohesive={cal.hit_cohesive.mean():.3f} "
              f"cross(testable)={cal[cal.testable].hit_cross.mean():.3f}")
        print("accuracy by cross margin:")
        print(curve.to_string())
    final, note = SC.apply(d, labels)
    out = labels.copy()
    out["role"] = final
    out["refine_note"] = note
    out = out.merge(d[["file", "verdict", "suggested", "basis", "cohesive_role", "cross_role",
                       "cohesive_margin", "cross_delta"]], on="file", how="left")
    RO.write_roles(cfg, out, name=args.name or "roles_final.csv")
    q = out[out.verdict.isin(["manual", "propose"])].copy()
    q["cluster_role"] = q.file.map(dict(zip(labels.file, labels.role)))
    q.to_csv(cfg.path("queue.csv"), index=False)
    print(f"queue for review: {len(q)} -> {cfg.path('queue.csv')}  (然后 vrs page review)")
    if note is not None:
        print("decisions:", pd.Series(note).value_counts().to_dict())


def cmd_export(args):
    cfg = _load_cfg(args)
    df = pd.read_csv(cfg.path(args.roles_csv)).fillna("")
    res = RO.export_folders(cfg, df, mode=args.mode, tone=not args.flat,
                            roles=args.roles or None, out_dir=args.out)
    for r, n in sorted(res["placed"].items(), key=lambda kv: -kv[1]):
        print(f"  {r:<24} {n:>6}")
    print(f"\ntotal {sum(res['placed'].values())} files | mode {res['modes']} | "
          f"missing {res['skipped_missing']} | -> {res['dest']}")
    if args.mode == "link":
        print("note: hard links share bytes with the source; edits propagate. Use --copy for independent files.")


def cmd_transcribe(args):
    cfg = _load_cfg(args)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit('transcribe needs faster-whisper:  pip install "faster-whisper"  (optional extra: --extra asr)')
    out = cfg.path("transcripts.csv")
    done = set(pd.read_csv(out, keep_default_na=False, encoding="utf-8-sig").file) if out.exists() else set()
    files = [p for p in sorted(Path(cfg.audio_dir).glob("*.wav")) if p.name not in done]
    model = WhisperModel(args.model, device=cfg.resolved_device().split(":")[0] or "cuda",
                         compute_type="float16")
    print(f"{len(files)} files to transcribe ({len(done)} already done)", flush=True)
    rows = pd.read_csv(out, keep_default_na=False, encoding="utf-8-sig").to_dict("records") if done else []
    for i, p in enumerate(files):
        segs, info = model.transcribe(str(p), language=args.language or None, beam_size=5,
                                      condition_on_previous_text=False)
        rows.append(dict(file=p.name, text=" ".join(s.text.strip() for s in segs).strip(),
                         language=info.language, no_speech=round(float(np.mean([s.no_speech_prob for s in []] or [0])), 3)))
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  {len(rows)} saved", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"-> {out}")


def cmd_serve(args):
    cfg = _load_cfg(args)
    import http.server, functools, os
    root = str(cfg.root)
    print(f"serving {root}  ->  open http://127.0.0.1:{args.port}/标注.html")
    os.chdir(root)
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                         functools.partial(http.server.SimpleHTTPRequestHandler,
                                                           directory=root)) as h:
        h.serve_forever()


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrs", description="Split voice lines into roles: cluster, confirm, calibrate, export.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a run")
    s.add_argument("--audio-dir", required=True, help="folder of .wav lines")
    s.add_argument("--run-dir", default="runs/default")
    s.add_argument("--models", action="append", help='comma list, e.g. campplus,ecapa,eres2net')
    s.add_argument("--roles-file", help="yaml/json with a roles list for the annotation dropdown")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any run.yaml field")
    s.set_defaults(fn=cmd_init)

    for name, fn in [("status", cmd_status), ("probe", cmd_probe), ("pitch", cmd_pitch),
                     ("embed", cmd_embed), ("cluster", cmd_cluster), ("page", cmd_page),
                     ("apply", cmd_apply), ("refine", cmd_refine), ("export", cmd_export),
                     ("transcribe", cmd_transcribe), ("serve", cmd_serve), ("backends", cmd_backends)]:
        sp = sub.add_parser(name, help=fn.__doc__ or "")
        sp.add_argument("--run-dir", "-r", default="runs/default")
        sp.set_defaults(fn=fn)

    sub.choices["pitch"].add_argument("--workers", type=int, default=0)
    sub.choices["embed"].add_argument("--models", help="override run.yaml models (comma list)")
    sub.choices["embed"].add_argument("--progress", type=int, default=500, help="windows per status print")
    c = sub.choices["cluster"]
    c.add_argument("--k", type=int, default=0, help="force this cluster count instead of the selection rule")
    c.add_argument("--ks", help="comma list, e.g. 6,8,10,12")
    c.add_argument("--linkage", choices=["average", "complete", "single"])
    c.add_argument("--score", choices=["asnorm", "cosine"])
    c.add_argument("--min-cluster", type=int, default=None)

    pg = sub.choices["page"]
    pg.add_argument("kind", choices=["clusters", "review"], nargs="?", default="clusters")
    pg.add_argument("--per-cluster", type=int, default=6)
    pg.add_argument("--spot", type=int, default=3)
    pg.add_argument("--out")

    ap = sub.choices["apply"]
    ap.add_argument("--export", action="append", help="label export csv(s) to ingest (default: both known names)")
    ap.add_argument("--name", default=None, help="output name, default roles.csv")

    rf = sub.choices["refine"]
    rf.add_argument("--base", help="input roles csv, default roles.csv")
    rf.add_argument("--name", help="output csv name, default roles_final.csv")
    rf.add_argument("--scope-review", action="store_true", help="only re-score files currently in review_list.csv")
    rf.add_argument("--only-unassigned", dest="only_unassigned", action="store_true", default=None,
                    help="never touch clustered files (default from run.yaml)")
    rf.add_argument("--allow-reassign", dest="only_unassigned", action="store_false",
                    help="also let cross-cluster scoring propose re-assignments")
    for k, t in (("k_neighbors", int), ("silver_pct", float), ("keep_margin", float), ("assign_margin", float)):
        rf.add_argument("--" + k.replace("_", "-"), dest=k, type=t, default=None)

    ex = sub.choices["export"]
    ex.add_argument("--roles-csv", default="roles_final.csv", help="which label file to materialise")
    ex.add_argument("--mode", choices=["link", "copy"], default="link")
    ex.add_argument("--out", default="roles")
    ex.add_argument("--flat", action="store_true", help="no tone subfolders")
    ex.add_argument("--roles", nargs="*", default=None)

    tr = sub.choices["transcribe"]
    tr.add_argument("--model", default="large-v3", help="faster-whisper model name or local CTranslate2 dir")
    tr.add_argument("--language", default=None)

    bk = sub.choices["backends"]
    bk.add_argument("--fetch", choices=["ecapa"], help="download weights for the named backend")
    bk.add_argument("--endpoint", default="https://hf-mirror.com")

    sv = sub.choices["serve"]
    sv.add_argument("--port", type=int, default=8010)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except (A.AudioError, EM.EmbedError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
