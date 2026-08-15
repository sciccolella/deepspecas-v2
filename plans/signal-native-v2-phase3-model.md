# signal-native-v2 Phase 3 — the model

## Context

Phase 1 (`dsignal/extract.py`) and Phase 2 (`sim/`, all five noise layers + window
randomization + `nn_novel` negatives + gene-level splitting, scaled to 1260 events) are
done and verified end to end — see `plans/signal-native-v2-checkpoint.md`. What has never
been built is the thing the whole rebuild exists for: **a model that consumes junction
identity**.

Everything scored so far came from `sim/baseline_model.py`, which was always explicitly a
sanity check — 24 crude summary features (junction *count*, coverage moments, ratios) with
no topology at all. Its persistent failure mode is exactly the one the v1 image pipeline
has: A3 and A5 are confused with each other about as often as called correctly (F1 0.108 /
0.160 at 1260 events), because nothing in those features encodes *shares-donor* vs
*shares-acceptor*. That is the discriminative object, and it is sitting unused in the
`.npz` junction tables.

Phase 3 delivers `dsignal/`'s modelling half: a shared junction-feature builder, a LightGBM
topology baseline (the honest-reporting gate), and the junction-set Transformer, plus the
first real evaluation against the untouched 140-event curated benchmark.

**Decisions taken with the user for this phase:**

| Question | Decision |
|---|---|
| Scope | Plan items 3.1 + 3.2 + 3.3. Item 3.4 (dilated coverage CNN) stays conditional. |
| Data scale | Develop against the existing 1260-event set; run a larger pasteur simulation in parallel, batched. |
| Label set | 5-class primary (`A3`,`A5`,`CSE`,`RI`,`NN`); also report the `A3`+`A5`→`A` collapse for comparability with v1's 123/140. 6-class is impossible — SUPPA emits `SE` only, no CE/SE distinction. |
| Eval | 5-fold `GroupKFold` on sim (identical protocol to `sim/baseline_model.py`, so numbers stay comparable to the whole layer-by-layer history) + the 140-event real benchmark as true held-out. |

## Step 0 — Local torch environment (blocking, mechanical)

There is currently **no torch anywhere on this machine**: the only v2 env is
`/datasets/deepSpecas-v2/envs/DSpoly2` (numpy/pandas/pysam/sklearn, no torch, no lightgbm),
and `/home/ciccolella/mambaforge/envs/` is empty. GPU is an RTX 4080 (16GB).

Create `/datasets/deepSpecas-v2/envs/DSnet` via `/home/ciccolella/mambaforge/bin/mamba`
with pytorch (CUDA 12.x build), lightgbm, scikit-learn, numpy, pandas, pysam, matplotlib,
seaborn. Verify `torch.cuda.is_available()` and that
`python -m sim.baseline_model --npz-dir sim/experiment/npz_scale1260 --labels
sim/experiment/labels_scale1260.tsv` still reproduces 48.8% acc / 0.325 macro-F1 in the new
env before anything else — that reproduction is the environment's acceptance test.

*Tier: cheap model.*

## Step 1 — `dsignal/features.py`, the shared feature builder

One module, consumed by **both** the LightGBM baseline and the Transformer, so the two are
compared on identical information rather than on incidentally different feature sets.

Input: an extractor record (`dsignal.extract.load_record`). Output two things:

**(a) Per-junction token matrix** — rows = distinct `(start, end)` in the *union* across
both conditions (a junction present in only one condition gets zeros for the other). Per
row:

- `log10(intron_length)`, `log1p(count_c)` and `log1p(nh_weighted_count_c)` per condition,
  the log-ratio between conditions, `min_anchor`, motif one-hot (7 levels incl. unknown).
- **Topology, the point of the whole exercise**: `n_sharing_donor`, `n_sharing_acceptor`
  (how many other junctions in the set share this `start` / this `end`), and the
  *donor-group* and *acceptor-group* count fractions per condition — this junction's count
  over the summed count of all junctions sharing its donor (resp. acceptor). A3 is
  "shares donor, group fraction shifts between conditions"; A5 is the mirror.
- **Coverage context, the RI signal**: mean coverage over a flank window (default 50bp)
  outside the donor and outside the acceptor; mean coverage over the intron interior; the
  interior:flank ratio; the fraction of intron-interior bases with nonzero coverage. All
  per condition, read from `cov_1`/`cov_2` at base resolution. Attaching coverage to
  junction tokens is deliberate — it means RI does not need a separate coverage encoder.
- Nesting/containment flags relative to the highest-count junction in the set.
- Relative position within the extraction window (kept, but see the geometry control in
  Verification — the model must not lean on it).

**(b) Event-level vector** — `log10(window_width)`, per-condition log mean/median/max
coverage, zero-coverage fraction, `n_junctions`, strand vote, plus pooled summaries of the
token matrix (max/mean of the shares-donor and shares-acceptor counts, max
interior:flank ratio, etc.). This is the CLS token for the Transformer and, concatenated
with a few more aggregates, the LightGBM feature row.

**Strand canonicalization**: if the strand vote (`extract.strand_vote`, already implemented
and GTF-free) is `-`, mirror all coordinates within the window before building features, so
"donor" always means transcription-direction donor. This makes A3/A5 well-defined
independent of strand, and makes the left-right-flip augmentation an exact A3↔A5 map.
Strand `?` events are left unflipped and flagged in the event vector.

*Tier: strong model — the topology and coverage-context definitions are the substance of
the contribution, and off-by-one errors on intron interiors (`jI` is 1-based inclusive of
the first/last intronic base, `cov` is 0-based relative to `start`) are silently wrong
rather than loud.*

## Step 2 — `dsignal/baseline_gbm.py` (plan item 3.1)

LightGBM 5-class over the event-level vector from Step 1, `GroupKFold(5)` grouped by
`gene_id_of(event_id)` with an `assert_no_gene_leak` call — reuse `sim.gene_split`'s
`gene_id_of` / `assert_no_gene_leak` and mirror `sim/baseline_model.py`'s CLI
(`--npz-dir` / `--labels` / `--folds` / `--seed`) so results drop straight into the
existing comparison table. Report per-class F1, macro-F1, accuracy, confusion matrix, and
LightGBM gain importances.

**This is the honest-reporting gate.** Two outcomes both matter:
- If topology features alone lift A3/A5 F1 materially above `baseline_model.py`'s
  0.108/0.160, that confirms junction identity — not model capacity — is the missing
  ingredient, which is the paper's central claim.
- If this baseline approaches the eventual Transformer (or the real 123/140), that has to
  be reported as such, per the top-level plan.

*Tier: cheap model.*

## Step 3 — `dsignal/dataset.py` + `dsignal/model.py` (plan item 3.2)

`dataset.py`: a torch `Dataset` over an npz dir + labels TSV, emitting `(tokens [J, F],
cls_vector [E], label)`, with length-bucketed batching and padding + key-padding mask.
Junction counts are heavy-tailed (median 22 across both conditions, p99 252, max 925) so
cap at `--max-junctions` (default 256, covering ~99% of events) keeping the highest-count
junctions; the discarded tail is summarized into the CLS vector rather than dropped
silently. `n_junctions == 0` events exist and must work (CLS-only forward pass).

`model.py` — `JunctionSetTransformer`:
- Linear projection of token features → `d_model=192`; learned CLS token from the
  event vector; 4 encoder layers, 4 heads, GELU, pre-norm, dropout 0.2 → ~1-2M params.
- **Relative-position bias**, per the top-level plan: for each junction pair, bucketize
  signed `log10(|midpoint distance| + 1)` into ~32 buckets → learned per-head additive
  attention bias. Raw bp cannot work across four orders of magnitude.
- **Relation bias on top**: learned per-head scalars for the discrete relations
  `shares_donor`, `shares_acceptor`, `nested`, `overlapping`, `disjoint`. This is the
  inductive bias that makes A3/A5/CSE topology directly reachable by attention rather than
  something the model must rediscover from coordinates.
- CLS output → 5-way head. Class-balanced cross-entropy.
- Attention weights over junctions are returned and saved — this is the `--explain` story
  that `deepSpecas.py predict` still raises `NotImplementedError` for.

`dsignal/train.py`: training CLI with `GroupKFold(5)`, AdamW, cosine schedule, early
stopping on held-out-fold macro-F1, writes per-fold checkpoints + an aggregate
classification report and confusion matrix in the same format as Step 2.

*Tier: strong model.*

## Step 4 — Augmentations (plan item 3.3)

In `dataset.py`, each on/off by CLI flag, all off in a first control run so their
contribution is measurable:

- **Left-right flip** — mirror coordinates within the window and swap the `A3`↔`A5` label.
  Exact under strand canonicalization (Step 1); the single most valuable augmentation here.
- **Condition swap** — exchange condition-1 and condition-2 features; label-invariant,
  since the label is event *type*, not direction.
- **Read subsampling** — binomially thin junction counts and scale coverage-derived
  features by the same factor, for depth robustness.
- **Junction dropout** — randomly drop low-count junctions.
- **Coordinate jitter** — re-crop to a random sub-window, dropping junctions that fall
  outside; complements the Phase-2 window randomization at the record level.

Report an ablation: no-aug vs each-alone vs all.

*Tier: cheap model, except the flip's label-swap correctness (strong).*

## Step 5 — `dsignal/score_benchmark.py` and the v2 CLI

The top-level plan's Verification section says to reuse `tools/score_benchmark.py` — **that
file does not exist in this repo**; it must be written here.

- Takes `benchmark/{A3,A5,ES,IR}.csv` (140 events total, local `benchmark/c1.bam`/`c2.bam`),
  runs `dsignal.extract.extract_event` on the fly, applies a trained model, and reports
  per-CSV and aggregate accuracy plus a confusion matrix — with **per-event region width**
  added to the output so accuracy-vs-width is plottable against v1's.
- The benchmark's CSV→label map is `A3.csv`→`A3`, `A5.csv`→`A5`, `ES.csv`→`CSE`,
  `IR.csv`→`RI`. There are no `NN` events in the benchmark, so it scores the positive
  classes only. Report both the 5-class number and the `A3`+`A5`→`A` collapse (the latter
  is what is comparable to v1's 123/140).
- A `dsignal/predict.py` accepting the same `region,bam1,bam2` CSV as `dsEnsemble.py`, so
  the end-to-end path is a drop-in comparison. Record wall-clock per event for the Phase-5
  cost comparison against the ensemble's ~17s/event of rendering.

Expect the sim-only model to score *poorly* here — this is zero-shot sim→real transfer with
no fine-tuning; Phase 4 is what closes that gap. The number's value is as a baseline for
the Phase-5 "noise realism transfers" ablation, not as a headline.

*Tier: cheap model for the driver, strong for the label-mapping/scoring semantics.*

## Step 6 — Sim scale-up on pasteur, in parallel

Independent of Steps 1-5; start it early and let it run while model work proceeds. See
`memory/reference_pasteur_signal_native.md` for host/env/paths.

**Run it batched, not as one giant design.** At 1260 events the pipeline produced 27.4M
fragments; a single 10x design would be ~270M fragments with matching FASTA/BAM footprint
and STAR wall-clock. Instead run ~5 additional batches of 1260 (seeds 1-5), each its own
polyester + STAR + extract cycle, accumulating `.npz` into one pooled directory →
~7,560 events (~1,080/class).

This needs **one small code change**: `sim/select_events.py` must accept
`--exclude-genes <file>` that pre-seeds its existing `used_genes` set, so batches select
disjoint genes. Without it, different seeds will re-pick the same genes and create
cross-batch gene leakage that `assert_no_gene_leak` would only catch after the fact. The
`used_genes` set is already threaded through `select_positives` /
`select_hard_negatives` / `select_constitutive_negatives`, so this is a ~10-line change.

Everything else is a rerun of the existing pipeline at the same parameters (seed per batch,
fraglen=250 fragsd=25 readlen=100, `--neighbors --novel-junctions`, layer-5
`pcr_rate=0.1 pcr_lambda=1 adapter_contamination=TRUE`). Pull back only `.npz` + labels +
psi-recovery, never BAMs/FASTA.

*Tier: cheap model — it is pipeline driving, with the one `--exclude-genes` change.*

## Files

New, all under `dsignal/`: `features.py`, `dataset.py`, `model.py`, `train.py`,
`baseline_gbm.py`, `score_benchmark.py`, `predict.py`.

Modified: `sim/select_events.py` (add `--exclude-genes`). Nothing else in `sim/` changes.

Reused rather than rewritten: `dsignal.extract.load_record` / `extract_event` /
`strand_vote` / `ReadFilters`, `sim.gene_split.gene_id_of` / `assert_no_gene_leak`, and
`sim/baseline_model.py`'s CLI shape and reporting format.

## Verification

Per the user's stated preference, each step is **built, verified, smoke-tested and
committed before the next one starts** — the same discipline used through all of Phase 2.

1. **Feature builder (Step 1)** — hand-construct a synthetic record with a known A3
   topology (two junctions sharing a donor) and assert `n_sharing_donor == 2`,
   `n_sharing_acceptor == 1`, and that the donor-group fractions sum to 1. Assert the
   minus-strand canonicalization is an involution (flip twice = identity) and that
   flipping an A3 record produces the same feature matrix as a natural A5 record with
   mirrored coordinates. Spot-check intron-interior coverage against a manual
   `cov[start-window_start : end-window_start]` slice on a real 1260-set record.
2. **Leakage audit** — `grep` `dsignal/features.py` and `dsignal/dataset.py` for any GTF /
   annotation input (there must be none); confirm the `jM % 20` mask is still in force in
   `extract.py`. Then the **geometry-only control**: train the Transformer with all
   junction and coverage features zeroed, leaving only window geometry — it must score at
   chance. If it does not, the window randomization is not doing its job and that is a
   Phase-2 bug, not a Phase-3 one.
3. **Split integrity** — `assert_no_gene_leak` on every fold assignment, in both
   `baseline_gbm.py` and `train.py`, as `sim/baseline_model.py` already does.
4. **Comparability** — run every new model with the same `GroupKFold(5)`, seed 0, on
   `sim/experiment/npz_scale1260` + `labels_scale1260.tsv`, and report against the standing
   reference (RF: 48.8% acc, 0.325 macro-F1; per-class F1 A3 0.108, A5 0.160, CSE 0.132,
   NN 0.657, RI 0.568). **A3/A5 F1 is the metric that matters** — that is the class pair
   the topology features exist to fix.
5. **Augmentation ablation** — no-aug run first as the control, then each augmentation
   alone, then all; report a table.
6. **End to end** — `dsignal/score_benchmark.py` over all four `benchmark/*.csv` against a
   trained checkpoint, reporting 5-class and 4-class-collapsed accuracy, confusion matrix,
   per-event region width, and wall-clock per event.
7. **Checkpoint** — append a Phase 3 section to `plans/signal-native-v2-checkpoint.md`
   recording results and any bug found, matching the existing per-layer update format.
