# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                            # 85 tests, offline, CPU-only, ~1 min
```

The test suite deliberately needs **no** speaker models, no GPU and no network: backbones
are stubbed (`tests/test_cli.py`) and synthetic tone files stand in for speech
(`tests/test_pitch.py`). Keep it that way - CI runs on a clean runner.

## Invariants worth defending

These came from measured failures; a PR that breaks one needs a good argument.

1. **Audio reaches a backend as float in [-1, 1].** `audio.read_wav` enforces it. Inflated
   input makes every cosine look similar and clustering collapses silently.
2. **No file-order priors.** Archive position is scene/recording similarity, not speaker
   identity (proven with pitch in docs/METHOD.md). Nothing may use it as evidence.
3. **Centroid agreement is not a risk score** for average-linkage clusters. It is recorded
   as a column and never used to decide confidence.
4. **Human decisions are terminal.** `apply` propagates names verbatim; `refine` only fills
   files without a role unless the user explicitly passes `--allow-reassign`, and a
   `propose` verdict is never applied automatically.
5. **Every quality number must be measured, not assumed.** New scoring paths ship with a
   calibration print (leave-one-out on the human-confirmed files, bucketed by margin).
6. **Everything a human opens is UTF-8 with BOM**; everything with Chinese labels is built
   with `dtype=object` (fixed-width numpy unicode arrays truncate silently).
7. **The run directory must stay copyable** - keep weights and caches out of it.

## Adding a backbone

Subclass `Backend` in `src/vrs/models.py`: set `name`, `dim`, `hint` (the exact pip line),
implement `_probe(cfg)` (raise with a message a user can act on) and `__call__(windows)`.
Register it in `BACKENDS`. It then appears in `vrs backends`, in `models:` in `run.yaml`,
and in fusion automatically. Do not add required dependencies for it - keep it an extra.

## Adding a step

Steps read the run dir and write new artefacts; never mutate an earlier step's output in
place, so re-running with different parameters stays safe. Add the subcommand in
`src/vrs/cli.py`, document it in `docs/USAGE.md`, and cover it in `tests/`.

## Before opening a PR

```bash
pytest -q
python -m vrs.cli init --audio-dir /path/to/any/wavs --run-dir /tmp/prcheck && \
python -m vrs.cli embed --run-dir /tmp/prcheck && \
python -m vrs.cli cluster --run-dir /tmp/prcheck && \
python -m vrs.cli page clusters --run-dir /tmp/prcheck
```

Open the generated page and confirm clips play - template regressions show up as a blank
page, and unit tests do not cover the browser side.
