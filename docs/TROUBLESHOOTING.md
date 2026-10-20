# Troubleshooting

## Clustering collapses into one giant cluster

`cluster_sweep.csv` shows `largest_frac` ≈ 0.9-1.0 for every k, and `embed` printed
`INFLATED: mean cosine ...`. Cause is almost always the ingest path:

- audio reaching a backend as int16-magnitude instead of float `[-1, 1]` (each frontend
  scales again, every cosine inflates). `audio.read_wav` now raises `ScalingError`; if you
  bypassed it, don't.
- skipping `center: true` on a corpus with a strong common direction.
- using a model's raw output per window without averaging per file.

Check with a 200-file run: `vrs embed --models campplus` then read the printed health note.
Healthy mean pairwise cosine on speech is ~0.2-0.4.

## `vrs cluster` runs forever at 100 % CPU on one core

Known regression class, fixed in `merge_small`: an emptied cluster label keeps size 0,
stays under the fold threshold, and the loop never converges. If you patched that function,
`pytest tests/test_cluster.py -k merge_small` must pass. Before assuming this, check for
several runs fighting for RAM - two full-corpus clustering processes on a 32 GB box swap,
which looks identical to a hang:

```bash
py-spy dump --pid <pid>          # tells you instantly whether it is looping
```

## A backend reports `unavailable`

`vrs backends` prints the reason and the install line. Recurring cases:

| symptom | cause | fix |
|---|---|---|
| `speechbrain` import fails with `torch.amp has no attribute custom_fwd` | speechbrain ≥1.0 needs torch ≥2.4 | `pip install "speechbrain>=0.5.16,<1.0"`, or upgrade torch |
| funasr: `No module named 'kaldiio'` / `'torch_complex'` | installed with `--no-deps` | `pip install kaldiio torch_complex omegaconf rapidfuzz jaconv jieba` |
| installing any ML extra silently replaces GPU torch | pip resolves `torch` from PyPI (CPU build) | reinstall `torch` from the PyTorch index afterwards; check `python -c "import torch;print(torch.cuda.is_available())"` |
| `Failed to initialize NumPy: _ARRAY_API not found` | torch built against NumPy 1.x running on NumPy 2 | `pip install "numpy<2"` |
| `tokenizers>=...<0.14 is required` (modelscope pipelines) | user-site install shadowing the env | run with `PYTHONNOUSERSITE=1`, or `pip uninstall modelscope` in user site |
| torch 2.1.2 + transformers 5.x: `Disabling PyTorch because PyTorch >= 2.6 is required` | a dependency pulled a newer transformers | pin transformers, or use the `campplus`/`ecapa` path instead of `eres2net` |

## Model downloads

- `huggingface.co` is unreachable from many networks; the default endpoint is
  `https://hf-mirror.com`, and `--fetch` retries because anonymous mirror requests get
  **HTTP 429** quite often. Re-running `vrs backends --fetch ecapa` finishes what a previous
  attempt left short (files under ~1 KB are treated as failures).
- Weights live outside the run dir (`~/.cache/vrs/models`, or `VRS_MODEL_DIR` / `model_dir`).
  That is deliberate: speechbrain symlinks into its own `savedir`, so a weights folder
  inside a run makes the run un-copyable (`cp` fails with "Too many levels of symbolic
  links"). Copying or archiving a run dir should never touch weights.
- ModelScope downloads need no token. Gated models (`pyannote/*`) do - and their pipelines
  pull a heavy dependency stack, which is why this tool avoids them: the backbones here are
  enough and run offline once fetched.

## Chinese / Japanese text and CSV

Windows consoles are usually GBK; printing Japanese transcripts raises `UnicodeEncodeError`
and - worse - CSVs written with the platform default encoding are unreadable later. The
tool always writes UTF-8 with BOM (`utf-8-sig`) for anything a human opens in Excel, and
reads exports the same way. If you script around it:

```python
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

Beware the numpy counterpart: `np.where(mask, "中文", None)` yields a *fixed-width* unicode
array and silently truncates every later assignment to that width. Build label arrays with
`dtype=object` (see `refine`'s `note` array).

## Annotation page

- **Blank page**: open the browser console; historically a template variable referenced
  from JS but defined only in Python produced exactly this. `vrs page clusters` regenerates.
- **Audio won't play**: some browsers block media from `file://`. Run `vrs serve --port 8010`
  and open `http://127.0.0.1:8010/标注.html`.
- **Progress lost**: state is in `localStorage`, keyed per run dir + mode. It survives
  reloads but not a different machine or a browser profile reset - export before closing
  for the day.
- **Page shows no transcripts**: run `vrs transcribe` (optional step); pages work without it,
  just without the text line.

## Recovering / restarting

`embed` and `pitch` skip work already present; `cluster`/`apply`/`refine`/`export` rewrite
their own outputs. To redo one backend only, delete `embeddings/<backend>.npy`. If you
changed `window`/`hop`/`min_speech`/audio set, the row-count check in `embed` and the length
checks in `apply`/`refine` detect the mismatch and recompute - they will not silently mix
stale matrices with new ones.

## Long-running jobs

Prefer explicit logging to a file over relying on a supervisor: `vrs ... > run.log 2>&1`.
Killing a parent shell can take down a chained job mid-run without an error in the log - the
signature is a log that simply stops. Every step resumes, so just re-run it.
