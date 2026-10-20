# voice-role-splitter

[中文](README.md) | English

![tests](https://github.com/Ikaros-521/voice-role-splitter/actions/workflows/tests.yml/badge.svg)

Split a folder of character voice lines into roles: speaker-embedding clustering, **human
confirmation in the browser**, calibrated re-scoring of the leftovers, and export into
`roles/<role>/<tone>/<file>.wav`.

Built for the case where naive speaker clustering quietly fails: expressive dialogue
(shouting, whispering, crying, 0.5 s interjections) where the embedding space is dominated
by *speaking style* rather than *identity*. Measured on a 7,055-line / 6.7 h anime game
corpus, a single centroid per role recovers the held-out role for only **15.3 %** of files
- chance level for 6 roles. This tool therefore never asks a model to name roles. It asks
the model to group, the human to name, and only then re-uses the model for the leftovers -
with the accuracy of each operating point measured on the human's own confirmations.

```
 vrs embed ──► vrs cluster ──► vrs page clusters ──► human names 17 clusters (~15 min)
                                                              │
 vrs export ◄── vrs refine ◄── vrs apply ◄──── labels_manual.csv
     │            │  calibrates on the heard files, re-scores only what has no role
     ▼            ▼
 roles/<角色>/c04_激动喊叫/*.wav      roles_final.csv + review_list.csv
```

## Contents

- [Install](#install) · [Quick start](#quick-start) · [The workflow](#workflow)
- [Outputs](#outputs) · [What the model is good at (measured)](#honest-limits)
- [Design rules](#design-rules) · [Docs](#docs) · [Data & licensing](#data--licensing)

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[campplus]"                            # core + one backend
pip install -e ".[dev]"                                 # tests
```

Backends are independent and each is optional - install what your environment can reach:

| backend | training data | install extra | notes |
|---|---|---|---|
| `campplus` | Chinese + English (3D-Speaker) | `[campplus]` (funasr) | needs `kaldiio torch_complex omegaconf jaconv jieba` |
| `ecapa` | English (VoxCeleb) | `[ecapa]` (`speechbrain<1.0`) | 1.x requires torch>=2.4 (`torch.amp`) |
| `eres2net` | Chinese (3D-Speaker) | `[eres2net]` (modelscope) | heaviest; strongest single model |

Two independent backbones are the point: cross-backbone agreement is one of the few
*premise-free* quality signals available, so a second backbone is not a luxury.

Fetch weights (`ecapa` is not auto-downloaded; the mirror rate-limits, so it retries):

```bash
vrs backends --run-dir runs/show1              # what is usable, and why not
vrs backends --run-dir runs/show1 --fetch ecapa
```

GPU is optional (CPU works, ~10x slower). Everything is local - no uploads, no API keys.

## Quick start

```bash
# 1. one run per dataset; run.yaml is the only hand-edited file
vrs init --audio-dir ./wav/voice --run-dir runs/show1 \
         --roles-file examples/cast_sora_no_otoshimono.yaml

vrs status --run-dir runs/show1        # what has been done, what is missing
vrs probe  --run-dir runs/show1        # count, durations, sample rates
vrs embed  --run-dir runs/show1        # per-backend embeddings + a health check
vrs pitch  --run-dir runs/show1        # F0 stats (model-independent QC)
vrs cluster --run-dir runs/show1 --ks 8,12,16,20,24
vrs page clusters --run-dir runs/show1 # 标注.html : name each cluster by ear

# ... open the page, listen, name the clusters, export labels_manual.csv into the run dir

vrs apply  --run-dir runs/show1        # propagate names -> roles.csv
vrs refine --run-dir runs/show1        # re-score files with no role, calibrated
vrs page review --run-dir runs/show1   # 待人工.html : leftovers, grouped by suggestion
vrs export --run-dir runs/show1        # roles/<角色>/c<簇>_<语气>/*.wav (hard links)
```

## Workflow

| step | artefact | who decides |
|---|---|---|
| `embed` | `embeddings/*.npy`, `meta.npz`, health note | model |
| `pitch` | `pitch.npz` (median F0, range, voiced fraction) | measurement |
| `cluster` | `cluster_sweep.csv`, `labels.csv` | model, by `agree_kappa` / `verify_pass` / `f0_disp` / silhouette |
| `page clusters` | `标注.html` | **human**: one name per cluster |
| `apply` | `roles.csv`, `review_list.csv` | human naming propagated verbatim |
| `refine` | `roles_final.csv`, `queue.csv`, `gold_calibration.csv` | model, only inside calibrated margins |
| `page review` | `待人工.html` | **human**: accept per group, spot-check 25 % |
| `export` | `roles/…`, `clusters.csv` | mechanical |

Every step is idempotent and reads only the run dir, so you can stop after any of them,
re-run with different parameters, or hand-edit `run.yaml`.

## Outputs

- **`roles.csv` / `roles_final.csv`** - one row per file: `role`, `cluster`, `own_pct`
  (typicality inside its cluster), `centroid_score`, `cohesive_margin`, `cross_delta`,
  `verdict`, `confidence`, `speech_sec`, `f0`, and `text` when transcripts exist.
- **`review_list.csv`** - the files a careful human should hear, sorted by risk.
- **`clusters.csv`** - per cluster: file count, median length, loudness percentile, F0
  median/range, and the generated tone name.
- **`roles/<角色>/c<簇号>_<语气>/*.wav`** - hard links by default (zero extra disk).
  `--copy` for independent files; `--flat` to skip the tone sub-folders.

## Honest limits

Numbers from the corpus above, 6 named roles, 7,055 files (see
[docs/METHOD.md](docs/METHOD.md) for how each was obtained):

| claim | measured |
|---|---|
| single centroid per role ≈ guessing | LOO top-1 **0.153** (chance = 0.167) |
| cluster ≠ role; emotion drives clusters | 1 role split into 7 clusters, another into 5 |
| top-k scoring, cross-cluster (needs >= 2 clusters/role) | **0.571** overall; 0.84 when margin >= 0.03 |
| top-k scoring, cohesive (files with no cluster) | 0.87 at margin 0.03-0.06; 0.93 at >= 0.06 |
| quality of an unsupervised cut | intra-cluster ASV pass 0.71 vs 0.12 for random pairs |

Consequences built into the tool: `refine` defaults to `only_unassigned` (it will not
re-shuffle files a human already accepted), `propose` verdicts are *never* applied
automatically, and calibration is printed on the human's own confirmed files so thresholds
are chosen from numbers rather than vibes. Roles that live in a single cluster cannot be
cross-validated at all - the tool says so instead of scoring them.

Short interjections are the honest hard case: on that corpus 466 of the leftovers had a
median length of **0.69 s** - some have no measurable pitch at all. Ship them as
`_未确定/` rather than pretending.

## Design rules

1. **WAV in [-1, 1], always.** Both frontends scale internally; feeding int16-magnitude
   audio inflates every cosine until clustering collapses into one 94 %-of-everything
   cluster. `audio.read_wav` raises `ScalingError` instead of letting that happen again.
2. **No positional priors without evidence.** Archive order looked like a speaker prior
   under MFCC similarity (adjacent 0.55 vs random 0.00) and was falsified by pitch
   (adjacent |ΔF0| 78 Hz vs random 88 Hz). Nothing in the pipeline uses file order.
3. **Centroid agreement is meaningless for connectivity clusters** (average linkage), so it
   is recorded as a column and never used as a risk score.
4. **Every quality metric must be premise-free or labelled**: cross-backbone kappa,
   intra-cluster ASV pass rate and pitch spread are; silhouette is reported but demoted.
5. **Human decisions are terminal.** Models annotate, propose and rank - they do not
   overwrite.

## Docs

- [docs/USAGE.md](docs/USAGE.md) - every command, every `run.yaml` key
- [docs/METHOD.md](docs/METHOD.md) - the pipeline, the measurements, why each choice
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - env/version matrix, downloads, encoding, hangs
- [CONTRIBUTING.md](CONTRIBUTING.md) - dev setup, the invariants a PR must not break, adding a backbone
- [examples/](examples) - a cast file and a `labels_manual.csv` in the shape `apply` expects

The docs under `docs/` are English-only for now; translation PRs welcome.
- [CONTRIBUTING.md](CONTRIBUTING.md) - dev setup, the invariants a PR must not break, adding a backbone
- [examples/](examples) - a cast file and a `labels_manual.csv` in the shape `apply` expects

```bash
pytest -q            # 44 tests, offline + CPU-only, including a regression for every bug above
```

## Data & licensing

The tool ships **no audio**. Datasets (`.wav`, embeddings, run outputs, exported role
folders, generated pages) are git-ignored - and voice lines from a game or anime are
copyrighted recordings, so publishing labelled role folders of one is a redistribution
problem even when the tool is MIT. Commit `run.yaml` and a cast YAML if you want to share
configuration; keep the media local.

MIT licensed; see [LICENSE](LICENSE). CI runs the suite on Linux and Windows, Python 3.10-3.12
([.github/workflows/tests.yml](.github/workflows/tests.yml)). CI runs the suite on Linux and Windows, Python 3.10-3.12
([.github/workflows/tests.yml](.github/workflows/tests.yml)).
