# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), versioning follows SemVer.

## [0.2.0] - 2026-10-20

### Added

- `vrs.cast`: the role table (`--roles-file`, `run.yaml: roles`) now has one parser and one
  contract. `value` is an identifier - it is written to `roles.csv` and becomes the export
  folder name - so it is checked at `init`: no path-hostile characters, no leading/trailing
  `.`, non-empty, ≤ 60 characters, unique ignoring case, and not one of the reserved
  "not decided yet" markers (`未定` / `需复核` / `unresolved`), which would otherwise have sent
  every file of that role to `_未确定`. A bad row fails with its entry number and the reason.
- Three input shapes parse to the same table - `roles: {value: label}`,
  `roles: [[value, label], ...]`, and a bare list of strings - plus `.json` and `{"roles": ...}`.
  Previously a bare list produced single-element entries, the page filtered them out, and the
  dropdown came up with only its fallback entry: a naming session with no names.
- `docs/USAGE.md` documents how to build the table (where to get the names, why the dropdown is
  a closed list, what happens when you hear a character that is missing, how to grow the table
  out of a finished `roles.csv`); the same guide is summarised in both READMEs and in the
  comments of `examples/cast_sora_no_otoshimono.yaml`.

### Changed

- `run.yaml` writes `roles:` as `value: label` instead of nested pair lists, because that field
  is meant to be hand-edited. Both forms still load.
- `vrs init` prints how many roles the page will offer.

### Fixed

- The example cast file and the old docs told you that anything not in the table stays
  reachable through an "其他/自定义" entry in the page. It does not: the dropdown is a closed
  list (by design, so a typo cannot create a second folder for one character), and the docs now
  say so along with the add-a-row-and-regenerate-the-page workflow.

## [0.1.1] - 2026-10-20

### Changed

- **Model weights no longer default to the system drive.** Backends now search existing
  caches first (`model_dir` > `$VRS_MODEL_DIR` > `$MODELSCOPE_CACHE` > `~/.cache/modelscope` >
  `$HF_HOME`/`$HF_HUB_CACHE` > `~/.cache/huggingface` > project `./models`) and reuse whatever
  is already there, so nothing gets downloaded twice; when nothing exists, the download target
  is the configured cache root or the project's own `models/`. `vrs backends` prints the search
  order and, per backend, `cached -> <path>` or `not cached, would fetch into <path>`.
- A configured `$MODELSCOPE_CACHE` is honoured as the write target too, so weights do not end
  up split across two roots.
- Re-running `vrs embed` against cached `.npy` files now still runs the embedding health check
  (it used to skip straight past it, so a corpus with a broken ingest path stopped being flagged
  after the first run).

### Fixed

- `vrs embed` crashed with `UnboundLocalError: health` whenever a backend came from cache or
  every backend was skipped, instead of reporting a health note or a clean error.
- A truncated ECAPA stub (partial download) was treated as a cached model; locations now need
  the real checkpoints before being used.

## [0.1.0] - 2026-10-19

First public release, extracted from a production run over 7,055 character voice lines
(6.7 h, 6 named roles, Japanese anime game dialogue).

### Added

- `vrs` CLI over a per-dataset run directory: `init`, `status`, `probe`, `backends`,
  `pitch`, `embed`, `cluster`, `page clusters`, `apply`, `refine`, `page review`,
  `export`, `transcribe`, `serve`.
- Three independent speaker backbones (`campplus`, `ecapa`, `eres2net`), each optional,
  with availability probing and per-backend install hints.
- Fusion + AS-Norm + one scipy linkage tree, swept over k with four premise-free quality
  measurements: cross-backbone `agree_kappa`, intra-cluster `verify_pass` against the
  model's own verification threshold, `f0_disp` (pitch spread inside a cluster), silhouette.
- Browser annotation pages generated from run artefacts: cluster naming with batch
  multi-select merge, and a review page grouped by suggestion with accept-group and
  25 % spot-check modes. Human state persists in `localStorage`; export is a UTF-8-BOM CSV.
- `refine`: top-k reference-line scoring (cohesive and cross-cluster variants) calibrated
  by leave-one-out on the files a human actually listened to; per-margin accuracy is
  printed so thresholds are chosen from measurements.
- Tone sub-folders in the export (`roles/<role>/c<cluster>_<tone>/`), named from measured
  loudness percentile, pitch level/range and line length; `clusters.csv` carries the
  numbers behind each name. Hard links by default, `--copy` for independent files.
- Regression suite of 44 tests covering the ingest, clustering, scoring, labelling and CLI
  paths, including explicit regressions for the bugs listed below.

### Fixed (bugs from the original ad-hoc pipeline, now guarded in code)

- int16-magnitude audio reaching the backends inflated every cosine until clustering
  collapsed into one cluster holding 94 % of the corpus; `read_wav` now raises
  `ScalingError` and `embed` reports a health note.
- `merge_small` looped forever once a cluster was emptied (its `np.bincount` entry stays
  below the fold threshold), hanging two full runs at 100 % CPU.
- Fixed-width numpy unicode arrays silently truncated Chinese label notes to three characters.
- Speechbrain `savedir` equal to the weights directory made runs un-copyable; the model
  cache now lives outside the run directory by default.
- Annotation page templates referencing a Python-only constant, which rendered a blank page.
- `--set key=value` accepted values that broke invariants (`models: None`); `validate()`
  now rejects them at `init`.
- Fixed-cluster thresholds that swallowed small datasets; `min_cluster` is now capped by
  dataset size, and the pitch metric needs 3 members instead of 7.

### Known limits

- Roles that live in a single cluster cannot be cross-cluster validated; the tool reports
  that instead of emitting a confidence number.
- Files shorter than ~0.6 s get no embedding at all and stay in `未确定/`.
- On expressive dialogue, a single centroid per role is at chance level (measured 0.153 for
  6 roles); any workflow that assumes otherwise will silently mislabel. See docs/METHOD.md.
