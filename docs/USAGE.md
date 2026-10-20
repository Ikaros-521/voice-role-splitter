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
model_dir: ""                         # "" -> see "Model weights" below; else this path
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

## Cast file (role vocabulary of the annotation page)

`vrs init --roles-file cast.yaml` reads a role table and stores it as `roles:` in `run.yaml`.
It is the list of names the human gets to pick from, and it is worth 5 minutes of care because
the names it contains are what ends up on disk.

| field | what it is for |
|---|---|
| `value` (left) | the role id. Written verbatim into the `role` column of `roles.csv` / `roles_final.csv`, and used verbatim as the export folder `roles/<value>/…`. Two clusters given the same value **are** one character - that merging is the normal case, since one character usually splits into several clusters by emotion. |
| `label` (right) | display only: what the dropdown shows. Make it recognisable by ear - original-language name plus voice actor is the combination that works. |

`value` is the one to decide first: renaming a role afterwards means re-exporting, and the
value is what other tools and scripts will key on.

### The dropdown is a closed list

The page offers `— 未定 —` plus exactly your values; there is no free-text box. That is
deliberate: a typed name is a brand new role, and `伊卡洛斯` / `イカロス` / `伊卡洛斯（天使）`
would become three folders for one character, with no way to notice.

So when you hear a character that is missing from the table:

1. add the row to the cast file (or to `roles:` in `run.yaml` - same thing),
2. re-run `vrs page clusters`,
3. open the page again: names you already assigned are kept in browser `localStorage`, keyed
   by run directory + mode, not by the cast file, so nothing you typed is lost,
4. name that cluster and export the CSV again.

Step 2 is seconds. The one thing that does break old names is re-running `vrs cluster` with
different parameters, because clusters get renumbered - run `vrs apply` before you re-cluster,
or expect to re-name.

### Building one from scratch

1. **Get the names, not the audio.** The credited cast from the game's official site, the
   visual-novel/wiki page (Bangumi / VNDB / MAL), the packaging, or the ISO's own text files
   if the game ships them. You need every voice actor's name too - that is what disambiguates
   two similar-sounding girls when you are listening at 2 s per clip.
2. **Decide the value convention** and hold to it: one language, no decorations. Most people
   use the name they want to see in the folder tree (e.g. `伊卡洛斯`), and put everything else
   in the label (`イカロス（早見沙織）`).
3. **One row per voice, not per character.** Narrator / menu-and-system lines / SFX-only clips
   each need their own value, or they will pollute a character folder. Two catch-alls at the
   end of the list are cheap insurance - see `examples/cast_sora_no_otoshimono.yaml`.
4. **List the secondary characters too.** Better an unused row than a gap that stops a naming
   session.
5. **Let `init` check it**: `vrs init --roles-file cast.yaml` prints
   `roles : N in the page dropdown (…)`, and a bad row fails there with the entry number and
   the reason instead of surfacing as a broken path three steps later.

Rules enforced on every `value`, because it becomes a directory name:

| rule | why |
|---|---|
| no `\ / : * ? " < > \|` or control characters | illegal in a path (keep such characters in the label) |
| may not start or end with `.` | Windows refuses to create such a folder |
| non-empty after trimming, ≤ 60 characters | the full value is written to CSV and to the tree |
| unique ignoring case | `Alice` and `alice` would be one character in two folders |
| not `未定`, `需复核` or `unresolved` | those are how the pipeline marks "not decided yet"; using one as a role would route the files to `_未确定` |

`未定` itself never appears in your table - the page adds it.

### Accepted shapes

All three read the same way, so a hand-written file and a generated one both work:

```yaml
roles:                       # mapping: shortest to write
  爱丽丝: "アリス（CV）"
  鲍勃: Bob
```

```yaml
roles:                       # pairs: what a script emits
  - [爱丽丝, "アリス（CV）"]
  - [鲍勃, Bob]
```

```yaml
roles:                       # bare strings: value doubles as the label
  - 爱丽丝
  - 鲍勃
```

A `.json` file with `{"roles": …}` works too, and `vrs init --set 'roles=[[爱丽丝, アリス]]'`
overrides it in one go. An empty label falls back to the value. Top level may also be a bare
list, without the `roles:` key.

### Growing the table from a run you already named

The `role` column of `roles.csv` is a cast file waiting to be reused for the next dataset from
the same series:

```bash
python -c "import pandas as pd,yaml;d=pd.read_csv('runs/show1/roles.csv',encoding='utf-8-sig');print(yaml.safe_dump({'roles':sorted(set(d.role)-{'未定','需复核'})},allow_unicode=True,sort_keys=False))" > cast.yaml
```

Fill in the labels (voice actors) afterwards by hand - they are the part that helps the ear.

## Model weights

Discovery runs before download, in this order:

1. `model_dir` in `run.yaml`
2. `$VRS_MODEL_DIR`
3. `$MODELSCOPE_CACHE`
4. `~/.cache/modelscope/hub`, then `~/.cache/modelscope`
5. `$HF_HUB_CACHE` / `$HF_HOME`, then `~/.cache/huggingface` (an ECAPA snapshot there is used directly)
6. `./models` in the current project
7. `~/.cache/vrs/models`, and `<run>/models` as a last look

A location only counts if the weights are really there (for ECAPA that means
`hyperparams.yaml` + `embedding_model.ckpt` above 1 KB - a truncated stub from a failed
download is ignored so the fetch can resume instead of loading garbage). If nothing is
found, the download target is the first configured cache root, else `./models` next to
where you run the command. `vrs backends` prints both the search order and, per backend,
whether it is `cached -> <path>` or `not cached, would fetch into <path>`.

Weights are never placed inside a run directory: speechbrain symlinks into its own savedir,
which would make the run un-copyable.

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
