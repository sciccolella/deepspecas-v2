# signal-native-v2 — scale-up to ~7,500 events, class prior as a design variable

Next step after the window-geometry fix and the prior-correction probe (see the
last two update sections of `plans/signal-native-v2-checkpoint.md`). Branch:
`signal-native-v2`.

## Why this run, and why it is not just "rerun the recipe at a bigger n"

Three things converge on the same batch of compute:

1. **Sim-CV estimates are still noisy** at 180/class — the 420-event run's
   apparent `nn_novel` regression turned out to be exactly this, and was only
   resolved by more data.
2. **The current data carries an accepted mismatch**: the neighbour background
   gene set baked into the scale1260 BAMs was chosen under the *old* windows,
   because the window fix was re-extraction-only by agreed scope. That is the
   confound in every window-fix-vs-old comparison made so far. A fresh
   simulation removes it; nothing else does.
3. **The NN fraction is the single biggest lever on transfer found so far**
   (uniform prior correction: 50.0% -> 55.0% old windows, 42.9% -> 51.4% new),
   and it is set at *design* time, in this run.

## Decisions taken with the user before writing this plan

- **NN fraction: 25%**, down from 43%. Per batch: 180 per positive label
  (A3/A5/CSE/RI = 720) + 80 each `nn_plain`/`nn_hard`/`nn_novel` (240), i.e.
  960 rows/batch. All three negative flavors are kept — they probe different
  shortcuts and `nn_hard` is the informative one. Step 4 still measures whether
  this actually beat inference-time calibration rather than assuming it.
- **Batches run sequentially and gated**, not fired back-to-back: a bad batch
  is caught after ~2h instead of after ~10h, consistent with the
  build-verify-commit discipline used throughout Phase 2.

## Steps

### Step 0 — diagnose CSE->A3 before spending the compute
*Model tier: strong (interpretation, not mechanics). Local, ~1h.*

The one error calibration does not touch: 23/65 ES events called A3
uncorrected, essentially unmoved by any prior setting. Take those events from
`runs/benchmark_events_windowfix.tsv`, re-extract their junction sets, and test
the concrete hypothesis: **a curated ES window often also contains an
alternative-3' junction pair sharing a donor, so the model sees genuine A3
evidence and the "right" answer depends on which event the curator annotated,
not on which junctions exist.**

Report: how many of the 23 contain a shared-donor/shared-acceptor pair
consistent with A3; whether the model's A3 probability tracks that; and what
the SUPPA catalogue says about those same genomic regions (do A3 and SE events
genuinely overlap there?).

**This gates the rest.** If it is a label-definition problem, more data does not
fix it, and the scale-up's purpose narrows to (2) and (3) above rather than
"more data will fix CSE". Say so explicitly in that case rather than proceeding
as if the run will help.

### Step 1 — batch composition
*Model tier: cheap.*

`sim/select_events.py --n-per-label 180 --n-nn-plain 80 --n-nn-hard 80
--n-nn-novel 80 --seed <k>` per batch, `--exclude-genes` repeated for every
prior batch's `design.tsv`. Seeds 1-5 (seed 0 is the existing batch, which is
*not* reused: its composition is the old 43% NN and its BAMs carry the old
neighbour set). Target ~4,800-7,500 events depending on how many batches
survive their gates.

Note the catalogue ceiling: RI is the limiting label at ~7,300 catalogued
events, so 180/batch x 5 batches = 900 RI is comfortable.

### Step 2 — run the batches on pasteur, one at a time
*Model tier: cheap to drive; no new code.*

Per batch, the existing recipe end to end: `select_events` -> `gene_neighbors`
-> `gtf_exon_structure` -> `novel_junctions` -> `build_experiment` (with
`--neighbors --novel-junctions`) -> FASTA assembly (gffread on the filtered
GTF + `build_premrna_fasta.py` + `novel.fa`, reordered to match
`transcript_ids.txt` positionally) -> `run_polyester.R` (fraglen=250 fragsd=25
readlen=100, `pcr_rate=0.1 pcr_lambda=1 adapter_contamination=TRUE`) ->
`align_reads.sh` -> `dsignal.extract`.

**Gates before starting the next batch** (all four, every batch):
- STAR uniquely-mapped 95-98%, GT/AG-dominant splicing;
- PSI-recovery mean abs error per label in the 0.13-0.20 band;
- zero gene overlap with every prior batch (`select_events.py` already asserts
  this, but check the printed count);
- `select_events.py`'s own window report: clamp rate and pooled log10 width
  near the benchmark's 3.445 / 0.702.

Pull back only the `.npz` + labels per batch (never BAMs/FASTA), into
`sim/experiment/npz_scaleup/` with a batch-tagged labels TSV, so a partial run
is still usable if a later batch fails.

### Step 3 — re-measure everything on the combined set
*Model tier: cheap to run, strong to read.*

`sim/width_leak_check.py`, both geometry controls (LightGBM and Transformer),
LightGBM and Transformer at `no-geometry`, `score_benchmark.py`, and
`prior_sweep.py`. **The controls are cheap and must be re-confirmed at the new
scale, not assumed** — a bigger design changes the width distribution through
the label mix alone.

### Step 4 — the experiment that settles design-vs-calibration
*Model tier: strong.*

Train on the combined scaled set **twice**: once at the new 25% NN, once
subsampled back to 43% NN (same events, NN rows downsampled — same data, only
the mix changes, so this is a clean ablation). Score both with and without the
uniform prior correction. Four numbers, one answer: is design-time rebalancing
better than, worse than, or redundant with inference-time calibration? Report
all four even if the answer is "redundant", which is itself a useful result
(it would mean the cheap fix suffices and future designs need not be
rebalanced).

### Step 5 — write up and commit
*Model tier: cheap.*

Append to `plans/signal-native-v2-checkpoint.md`: the CSE->A3 diagnosis, the
per-batch gate results, the re-measured controls, the combined-set model
numbers vs the 1260-event ones, and the step-4 answer. Commit, no push.

## Verification checklist

- [ ] CSE->A3 diagnosis reported before any pasteur compute is spent
- [ ] no cross-batch gene overlap (asserted per batch)
- [ ] per-batch STAR / PSI-recovery / window-report gates all pass
- [ ] width-only leak test still at chance at the new scale
- [ ] both geometry controls still at chance at the new scale
- [ ] benchmark reported raw *and* uniform-corrected, carrying forward the
      caveats from the prior-correction update (nn_weight is benchmark-selected;
      correction reintroduces 3-4 A3<->A5 errors)
- [ ] step-4's four numbers reported together, whatever they say

## Known risks

- **A batch fails its gates.** Then stop, report, and keep the batches that
  passed — the combined set does not need to be exactly 5 batches.
- **The 25% NN design makes NN recall worse in a way the benchmark cannot
  see** (it has no NN events, so NN recall is unmeasurable there). Watch the
  sim-CV NN F1 as the only available signal, and say plainly that the
  benchmark cannot adjudicate this.
- **Compute**: ~1.5-2h per batch, ~10h total, plus ~1h of local training per
  model variant at 6x data.
