# Method

Why the pipeline looks the way it does. The numbers quoted here were measured on one
corpus - 7,055 character voice lines (6.7 h, median 3.26 s) from a Japanese anime game,
16-bit 44.1 kHz mono WAV, with 6 named roles confirmed by ear - so treat them as a
direction, not a spec sheet.

## 1. No labels exist, and the container will not give you any

The source was a disc image whose audio packs (`*.ptd`, magic `PETA`) store an index region
plus concatenated ADX blobs. That index looked like the obvious ground truth - a per-entry
table would have given `voice_id -> character` directly. It is encrypted: 7.64 bits/byte of
entropy, a near-uniform byte histogram, and a known-plaintext attack using the 7,055
(offset, size) pairs recovered by signature scanning found no transform that revealed them
(byte-order, 4-in-16 chunk reversal, stride de-interleaving, positional XOR all failed).
Between entries there is only zero padding. Extracting the mapping would mean reversing the
game's loader in `EBOOT.BIN`.

Lesson encoded in the tool: **assume no labels, plan a human step early.**

## 2. Embeddings, with a health check that pays for itself

Per file: resample to 16 kHz, silence-trim (`top_db=35`), slide 3.0 s windows with 1.5 s
hop, embed each window, average, L2-normalise. Backends: CAM++ (zh+en), ECAPA-TDNN
(VoxCeleb, en), ERes2NetV2 (zh) - different training data on purpose.

Input amplitude must be float in [-1, 1]. Both frontends scale internally; feeding
int16-magnitude audio inflated mean pairwise cosine from 0.31 to 0.62 and made hierarchical
clustering produce one cluster holding 94 % of the corpus, with every metric still looking
"fine". That is why `audio.read_wav` raises `ScalingError` and `embed` prints:

```
campplus: (7055, 192) saved; mean pairwise cos=0.311 p99=0.750   -> healthy
ecapa:    (7055, 192) saved; mean pairwise cos=0.187 p99=0.682   -> healthy
```

Anything above ~0.55 is a broken ingest path, not a hard dataset.

## 3. F0 kills the ordering prior

The archive order *looked* like a speaker prior: MFCC similarity between consecutive files
averaged 0.552 against 0.002 for random pairs, and 24.5 % of files had their nearest
neighbour within ±5 positions (2.5 % expected). Building any metric on that would have
"validated" the clustering circularly.

Median F0 says otherwise: `|ΔF0|` between consecutive files has median **78 Hz**, versus
**88 Hz** for random pairs - and only 23.2 % of adjacent pairs are within 30 Hz versus
19.1 % of random pairs. Consecutive lines are *not* the same speaker; the MFCC signal was
scene and recording-chain similarity (same loudness envelope, same processing), which is
exactly what MFCC mean/std encodes.

Consequence: pitch is kept as a **model-independent QC signal** (`f0_disp` in the sweep,
`f0`/`own_pct` in outputs) and file order is used nowhere.

## 4. Clustering

Fuse backbones (equal weights) after optional centering; cosine distances are *not*
comparable across backbones, so no per-backend threshold is ever shared. Score matrix goes
through AS-Norm (adaptive symmetric normalisation, top-20) to suppress per-file level bias,
is converted to a distance (`max - score`) and built into one scipy linkage tree; every
candidate k is then a cheap `fcluster`.

Per k the tool reports four premise-free measurements:

| metric | what it catches | value at the chosen cut (k=22 → 17 after folding) |
|---|---|---|
| `agree_kappa` | cross-backbone agreement on cluster centroids, chance-corrected | **0.70** |
| `verify_pass` | fraction of random same-cluster pairs passing CAM++'s own verification threshold | **0.71** (random pairs: 0.12) |
| `f0_disp` | size-weighted within-cluster spread of median F0 - penalises merging people | **107 Hz**, flat from k≈18 |
| `silhouette` | shape/compactness | 0.10 |

Selection is a rank sum of all four, with `verify_pass`, `agree_kappa` and `f0_disp`
dominating because silhouette rises monotonically as clusters merge (it scores 0.31 at k=5
and prefers coarse cuts). Plateaus are the useful signal: both `verify_pass` and `f0_disp`
stop improving past k≈18-22, so the corpus has roughly that many distinguishable voice
styles - more than the 6 named roles, which is the whole difficulty.

Clusters smaller than `min_cluster` fold into their nearest centroid. (The first
implementation of that loop hung at 100 % CPU for 1.5 h: `np.bincount` reports size 0 for a
label that has been emptied, it stays below the floor forever, so the loop never
converges. `tests/test_cluster.py` pins this.)

## 5. Why clusters are not roles - and what was done about it

Attempting supervised-style assignment on top of the clustering, with one centroid per role
built from the human-named clusters, gives **LOO top-1 accuracy 0.153** over 6 roles
(chance = 0.167). One role split into 7 clusters, another into 5. The embedding space's
dominant axis here is speaking style, not identity - so a centroid is the wrong model.

Replacing it with top-k mean similarity to the role's reference lines ("does it resemble
SOME lines of this role") is much better but still not free, and the two variants behave
very differently:

| variant | gold leave-one-out accuracy | margin 0.03-0.06 | margin ≥0.06 |
|---|---|---|---|
| cohesive (may cite the file's own cluster) | 0.641 | **0.87** (n=38) | **0.93** (n=15) |
| cross-cluster (own cluster removed from every reference set) | 0.571 on the testable subset (n=126) | **0.84** (n=37) | 0.83 (n=12) |

Two structural notes that matter more than the numbers:

- **Cross-cluster scoring is only measurable for roles spanning ≥2 clusters.** On this
  corpus that was 3 of 6 roles; for single-cluster roles the cross score says nothing and
  the tool reports `单簇角色无法跨簇验证` instead of inventing a confidence.
- **Cohesive scoring is self-confirming for clustered files** (its references include the
  cluster the label came from). It is legitimate only for files with no cluster at all -
  which is exactly where `refine` uses it.

Hence the defaults: `only_unassigned: true`, `propose` verdicts never applied, and
thresholds taken from the buckets above rather than a hunch. On the real run this resolved
329 of 730 role-less files (209 in the 87 % tier, 55 in the 93 % tier, reported as
`自动归属(auto87)` / `自动归属(auto93)`) and left 466 for the human.

Those 466 are the genuinely hard remainder: median **0.69 s** of trimmed speech, 62 with no
measurable pitch, 23 with no usable embedding at all. No scoring scheme fixes that; the
review page groups them by suggestion so a human can accept a group and spot-check 25 %.

## 6. Tone sub-folders

Since clusters inside a role *are* emotion/style groups, the export keeps that structure
instead of flattening it: `roles/<角色>/c<簇号>_<语气>/*.wav`, where the tone name comes from
measured loudness percentile, F0 level/range and line length (see `tone_names()` in
`roles.py`). `clusters.csv` ships the numbers behind each name, so a label can be corrected
without re-running anything.

## 7. Reproducing these measurements

```bash
vrs cluster --run-dir runs/show1                 # prints the whole k sweep table
vrs refine  --run-dir runs/show1                 # prints gold LOO accuracy per margin bucket
cat runs/show1/gold_calibration.csv              # per-file calibration rows
```

Everything above is computed by the shipped code on your own data - the numbers in this
document are what this corpus produced, not constants the code assumes.
