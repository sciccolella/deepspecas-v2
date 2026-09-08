# Stage 15/16/17 — attacking the real-vs-simulated shortcut

## Context

Stage 14 (`plans/signal-native-v2-arena-nn-attack.md`) mixed 474 real NN
events (from `aqr_arena`, AQR knockdown) unweighted into the 12,480-event
sim pool and trained 3 seeds. NN recall on `benchmark_new` dev jumped
13.9% → 57.0% (passes), but pooled collapsed accuracy on both gating
benchmarks dropped well beyond the branch's measured seed-noise ceiling, in
all 3 seeds, with zero compensating wins on A3/A5/CSE/RI. **Gate failed on
condition 3.**

The doc records an unverified hypothesis for why: the 474 real rows are the
*only* real BAM data in the whole pool and are labelled NN uniformly, so the
model may have learned "looks like real, not simulated, data" → NN, rather
than the intended splicing distinction. It also names two untried follow-ups
and one required diagnostic. The user asked to pursue all of it, as far as
possible: (1) verify the hypothesis, (2) attack it with real data spanning
multiple labels, (3) attack it architecturally with a domain-adversarial
term. This plan sequences those three as Stage 15/16/17, each pre-registered
and gated per branch convention, each building only on data/checkpoints/code
already confirmed to exist (per exploration below).

**Confirmed infrastructure (no new scoring/extraction machinery needed for
15 and most of 16):**
- `ptbp1_arena/candidates/{A3,A5,ES,IR}.csv` (152/113/1073/284 = 1,622 real
  positives, ground truth, local BAMs) and `aqr_arena/candidates/{A3,A5,ES,IR}.csv`
  (128/88/716/284 = 1,216 real positives, ground truth, **BAMs on pasteur
  only** — `aqr_work/align/{c1,c2}_aqr.bam`, never extracted) are both
  already in `region,bam1,bam2` no-header format.
- `dsignal/score_benchmark.py`'s `score()` (`CSV_LABEL` maps basename →
  truth) is the ready-made harness — point `--csv` at these candidate files
  directly, no new script needed.
- Checkpoints: `runs/nn_attack_realnn_seed{0,1,2}/fold{0..4}.pt` (Stage 14),
  `runs/ensembles/e4_union2.json` (baseline, 6-member ensemble).
- No real-vs-simulated domain tag exists anywhere in the `.npz` records —
  Stage 17 must add one.
- `JunctionSetTransformer` (`dsignal/model.py:270`) pools to a single CLS
  embedding (`pooled = self.norm(x[:, 0])`, line 366) before `head`/
  `type_head`; the two-head machinery (`compose_two_head`, `--diff-lambda`
  in `train.py`) is the existing pattern for adding a second loss term off
  that same pooled embedding.

---

## Stage 15 — verify the real-vs-sim shortcut hypothesis (diagnostic only, no training)

**Goal:** determine whether the Stage 14 model predicts NN on real *positive*
(non-NN) events at a higher rate than the baseline does, independent of
whether either model gets the true label right. If real-ness alone is
driving NN predictions, the Stage 14 models should over-predict NN on real
A3/A5/CSE/RI events specifically, not just fail to get them right.

1. **Pre-register the metric and pass/fail criterion** before scoring
   anything (judgment — do this carefully, it's the whole point of a
   diagnostic). Proposed metric: for each of the 3 Stage-14 seed models and
   the `e4_union2` baseline, compute the NN-prediction rate over
   `ptbp1_arena/candidates/{A3,A5,ES,IR}.csv` (1,622 real, non-NN-truth
   events — held out from Stage 14 training, since only `aqr_arena` supplied
   training rows). Paired McNemar (or exact binomial) on
   "predicted NN vs not" between each seed model and the baseline, same
   events. Hypothesis is **supported** if all 3 seeds show a significant,
   consistent increase in NN-prediction rate over baseline on this set,
   consistent in direction and magnitude with the recall/regression pattern
   already measured. *(model: Opus/Sonnet — this is the interpretive core)*
2. **Score.** Run `dsignal.score_benchmark` (or a thin wrapper around
   `Predictor.predict_region`, whichever fits `score()`'s existing CLI more
   directly) with `--csv ptbp1_arena/candidates/A3.csv ... IR.csv` against
   each of `runs/nn_attack_realnn_seed{0,1,2}/fold*.pt` and against
   `runs/ensembles/e4_union2.json`. This is read-only scoring of an
   already-extracted, already-local arena — no pasteur work, no retraining.
   *(model: cheap/mechanical — reuses an existing CLI unmodified)*
3. **Optional confirmatory extension:** if time allows, extract
   `aqr_arena`'s positive candidates from pasteur (extraction only, wrapped
   in `limitram`, four thread-pool env vars set, `nice -n 19` — same
   discipline as every other pasteur job) and repeat the same comparison
   there. Not required for the Stage 15 gate; strengthens it if the pattern
   replicates on a second real cohort. *(model: cheap/mechanical for the
   extraction command; Opus/Sonnet if `dsignal.verify` flags anything)*
4. **Write up and gate.** New doc `plans/signal-native-v2-stage15-shortcut-diagnostic.md`:
   the pre-registered criterion, the per-seed table, McNemar results,
   confirmed/not-confirmed verdict. Update `plans/signal-native-v2-resume-9.md`'s
   top section (Stage 15 supersedes the "unverified hypothesis" caveat #5 in
   the open-caveats list — resolve it either way). *(model: Opus/Sonnet —
   this is exactly the kind of write-up the branch has gotten wrong before
   by skipping the "why" section)*

**Decision rule into Stage 16/17:** proceed regardless of Stage 15's
verdict — Stage 16 and 17 are informative either way (they test whether a
*different* mechanism for injecting real data avoids the regression, not
whether the diagnosed mechanism is correct) — but the write-up shapes how
Stage 16/17's results get interpreted.

---

## Stage 16 — real data spanning multiple labels (the "homogeneous real data" attack)

**Goal:** per the Stage 14 doc's own follow-up note, test whether the
regression was caused by *homogeneous* real data (all real rows sharing one
label, NN) rather than by real data itself. Add real positives across all
four non-NN classes so the pool's real/simulated split is no longer
confounded with the label.

1. **Pre-register** the mixing design and the exact same 3-condition gate
   Stage 14 used (NN recall beats 13.9% baseline; ≥2/3 seeds; no
   more-than-noise regression on `benchmark_new` dev pooled collapsed and on
   `benchmark/` dev), plus a fourth comparison against Stage 14's own result
   (does this mix regress *less* than Stage 14 did, even if it still doesn't
   pass condition 3 outright). Keep `ptbp1_arena` (pair 1) entirely
   eval-only, unchanged from Stage 13/14 — only `aqr_arena` (pair 2) supplies
   training rows, preserving the "held-out confirmatory arena" structure.
   *(model: Opus/Sonnet)*
2. **Extract `aqr_arena`'s positive candidates** (`A3.csv`/`A5.csv`/`ES.csv`/`IR.csv`,
   1,216 rows) on pasteur via the same `dsignal.extract` / `build_extract_csv.py`
   path already used for its NN half — same `limitram`/thread-cap/`nice`
   discipline. Verify with `dsignal.verify`. Only `.npz`/labels/TSVs return
   from pasteur, never BAMs. *(model: cheap/mechanical — repeats a known-good
   procedure from Stage 14/B2 on a new candidate set, no new script logic)*
3. **Merge the pool:** `npz_union2` (12,480 sim) + existing 474 real NN rows
   + new ~1,216 real positive rows (real-data share rises from 3.7% to
   ~13.6%, no longer single-label). Build the merged labels TSV the same way
   `sim.merge_batches` / Stage 14's Step 5 did. *(model: cheap/mechanical)*
4. **Train 3 seeds**, identical hyperparameters/augmentation to Stage 14's
   run for comparability (`--augment all --feature-subset no-geometry --refit
   --inner-val-frac 0.30`, same seeds 0/1/2). *(model: cheap/mechanical to
   launch; Opus/Sonnet if training instability needs debugging)*
5. **Score** on `benchmark_new` dev, `benchmark/` dev (regression guard), and
   `ptbp1_arena` pair 1 (confirmatory, same as Stage 14) — reuse
   `score_benchmark.py` exactly as Stage 14 did.
6. **Apply the gate, write up** (`plans/signal-native-v2-stage16-multilabel-real.md`),
   update resume doc. If this passes where Stage 14 failed, it is the
   headline result of this whole line of work and needs the same "fold into
   an ensemble?" decision Stage 13's B3 result got. *(model: Opus/Sonnet)*

---

## Stage 17 — domain-adversarial term against the real-vs-sim shortcut

**Goal:** attack the same failure mode architecturally — an explicit penalty
for the model learning to distinguish real from simulated data — as a
second, independent line of defense regardless of whether Stage 16 already
fixes it (a positive Stage 16 result is a workaround; this tests whether the
underlying shortcut can be suppressed directly).

1. **Design** (architecture judgment, do this on the strongest available
   model): a gradient-reversal layer (GRL) wrapping a new
   `domain_head = nn.Linear(d_model, 1)` in `JunctionSetTransformer.__init__`
   (`dsignal/model.py`), fed from the same pooled CLS embedding
   (`pooled = self.norm(x[:, 0])`, line 366) that already feeds `head`/
   `type_head`. Binary cross-entropy domain loss, weighted by a new
   `--domain-lambda` CLI flag in `train.py` (mirrors the existing
   `--diff-lambda` two-head pattern), added into `train_fold`'s loss
   composition (`two_head_loss`, line 112, or alongside it).
2. **Thread domain labels through the data pipeline.** No `.npz` record
   currently carries a real/simulated tag — add one at the labels-TSV level
   (an extra column, `is_real`, set when merging in `aqr_arena` rows, 0
   everywhere else) and carry it through `dataset.py`'s batch construction
   (`batch_inputs`) to the training loop. This is the step most likely to
   need debugging — get it right before training anything on top of it.
3. **Verify** the change doesn't alter behavior when `--domain-lambda 0`
   (should reduce exactly to the current model) before trusting any run with
   it nonzero — a cheap regression check against an existing checkpoint's
   held-out fold metrics.
4. **Train 3 seeds** on the Stage 14 mix (474 real NN, single-label) first —
   this isolates whether the adversarial term alone fixes what Stage 14's
   architecture couldn't, without also changing the data mix. If Stage 16 is
   already done and passed, optionally repeat on Stage 16's mix too as a
   combined attempt, but only after the single-label test is scored.
5. **Score and gate**, same 3-condition criterion as Stage 14/16. Write up
   (`plans/signal-native-v2-stage17-domain-adversarial.md`), update resume
   doc, and — since this is now the fifth+ attempt on the NN-recall line —
   make sure the write-up explicitly compares against all prior attempts
   (Track A's 0.804 cap, Track D's reallocation, B3's seed-variance finding,
   Stage 14, Stage 16) rather than in isolation.

---

## Verification (applies to every stage)

- `dsignal.verify` after any new extraction (Stages 16/17's `aqr_arena`
  positives extraction).
- McNemar (or exact binomial for single-model-vs-fixed-baseline
  comparisons) on every benchmark comparison; 3-seed replication before
  believing any result, per the branch's own measured seed-noise ceiling
  (5 events at n=70, 10 at n=242).
- Never spend `benchmark_new`'s reserved test half (243 events) — everything
  above stays on dev.
- Build → verify → commit each step before starting the next, per repo
  convention. No `git push`; no new branches without asking.
- Update `plans/signal-native-v2-resume-9.md` at the end of each stage, not
  just at the end of all three — it is the entry point for whoever resumes
  next, including a future session picking this plan back up mid-way.
