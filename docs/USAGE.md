# Usage

## Command reference

All commands take `--run-dir / -r` (default `runs/default`) and read only that directory.

| command | what it does | main flags |
|---|---|---|
| `vrs init` | create `run.yaml` | `--audio-dir`, `--roles-file`, `--models a,b`, `--set key=value` |
| `vrs status` | which artefacts exist, embedding health, current role totals | |
| `vrs probe` | file count, durations, sample rates, sub-1 s census | |
| `vrs backends` | which backbones are importable + weights present, with install hints | `--fetch ecapa`, `--endpoint` |
| `vrs pitch` | median F0 / range / voiced fraction per file (multiprocess) | `--workers` |
| `vrs embed` | per-backend embeddings + meta (windows, seconds, loudness) + health check | `--models`, `--progress` |
| `vrs cluster` | fuse, sweep k with quality metrics, cut, fold tiny clusters | `--ks`, `--k`, `--linkage`, `--score`, `--min-cluster` |
| `vrs page clusters` | `标注.html` - name each cluster by ear | `--per-cluster`, `--spot`, `--out` |
| `vrs apply` | ingest the page export → `roles.csv` + `review_list.csv` | `--export` (repeatable), `--name` |
| `vrs refine` | calibrate on the heard files, re-score files without a role | `--only-unassigned` / `--allow-reassign`, `--k-neighbors`, `--silver-pct`, `--keep-margin`, `--assign-margin`, `--scope-review` |
| `vrs page review` | `待人工.html` - leftovers grouped by suggestion | |
| `vrs export` | `roles/<role>/c<cluster>_<tone>/*.wav` + `clusters.csv` | `--roles-csv`, `--mode link\|copy`, `--flat`, `--roles ...` |
| `vrs transcribe` | optional local ASR → `transcripts.csv` (feeds page text + export) | `--model`, `--language` |
| `vrs serve` | serve the run dir over http for browsers that block `file://` media | `--port` |

## run.yaml

Written by `init`, re-read by every step, safe to hand-edit.

```yaml
audio_dir: ./audio/lines              # required, must contain *.wav
run_dir: runs/show1
sample_rate: 16000                    # everything is resampled to this
window: 3.0                           # embedding window length (s)
hop: 1.5                              # window step (s); must be < window
min_speech: 0.6                       # shorter than this -> no embedding at all
cluster_min_speech: 1.0               # shorter than this -> excluded from clustering
trim_db: 35.0                         # leading/trailing silence cut
models: [campplus, ecapa]             # backbones to fuse
model_dir: ""                         # "" -> ~/.cache/vrs/models (VRS_MODEL_DIR honoured)
device: auto                          # auto | cpu | cuda:0
linkage: average                      # average | complete | single
score: asnorm                         # asnorm | cosine
center: true                          # remove the common direction before cosine
ks: [5, 6, ..., 30]                   # cluster counts to evaluate
min_cluster: 25                       # smaller clusters fold into the nearest (auto-capped by dataset size)
k_neighbors: 12                       # reference lines averaged in refine
silver_pct: 0.55                      # typicality cut for reference files
keep_margin: 0.03                     # cross-cluster margin required to confirm/reject
assign_margin: 0.03                   # cohesive margin required to name a roleless file
only_unassigned: true                 # never touch files a human already accepted
per_cluster: 6                        # exemplars shown per cluster on the page
spot: 3                               # extra random clips per cluster for spot-checking
seed: 0
roles:                                # dropdown of the annotation page: value -> label
  伊卡洛斯: "イカロス（早見沙織）"
  樱井智树: "桜井智樹（保志総一朗）"
```

`--set key=value` overrides any of these; values are parsed as YAML, so
`--set ks=[6,8,10]` and `--set center=false` work. Unknown keys and empty lists are
rejected at `init` rather than discovered three steps later.

## Cast file for the dropdown

```yaml
roles:
  <value written into roles.csv and used as the folder name>: "<label shown in the page>"
```

Keep values filesystem-safe and stable - they become directory names. The page also allows
free text, and exports a UTF-8-BOM CSV (`labels_manual.csv`) which is what `apply` reads.

## The annotation pages

Both pages are single self-contained HTML files that reference audio by relative path, so
they work by double-clicking. Progress persists in `localStorage` (keyed by run + mode), so
closing the tab loses nothing.

**Cluster page** (`vrs page clusters`): one card per cluster with `--per-cluster` exemplars
plus `--spot` random ones. Tick several cards and apply one role to name/merge them in a
single action - merging is the normal case, because one character is usually split into
several clusters by emotion.

**Review page** (`vrs page review`): leftovers grouped by the model's suggestion. Buttons:

- 整组接受 - accept the suggestion for the whole group
- 抽检25% - accept everything, then mark one in four for a listening pass (the honest way
  to buy speed: you verify a sample instead of trusting a threshold)
- per-file dropdown + 是 / 不是 / 听不出 - override individual lines

"不是" is recorded as a rejection and lands in `需复核` rather than being reassigned silently.

## Typical session budget

400-line pilot: `embed` ~1 min, `pitch` ~1 min, `cluster` ~1 min on an RTX 4070 with two
backbones. 7,000 lines: `embed` ~15 min, `pitch` ~4 min, `cluster` ~3 min, plus ~15 min of
human listening to name ~17 clusters (2 s per clip + typing). Transcription is the long
tail: 6.7 h of audio ≈ 50-70 min with `faster-whisper large-v2` on the same GPU.

## Re-running with different parameters

`embed` and `pitch` reuse existing artefacts; delete `embeddings/<backend>.npy` to force a
redo. `cluster`, `apply`, `refine`, `export` are cheap and overwrite their own outputs. If
you change `window`/`hop`/`min_speech`, re-run `embed` (the file-count check catches a stale
matrix and recomputes automatically).
