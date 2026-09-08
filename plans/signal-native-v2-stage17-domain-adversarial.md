# Stage 17 — domain-adversarial term against the real-vs-simulated shortcut

Design + implementation record, per
`plans/signal-native-v2-stage15-17-shortcut-attack-plan.md`. Pre-registration
of the gate (Step 5) is below, before any scoring.

## Design (Steps 1-3, implemented)

**Gradient-reversal layer** (`dsignal/model.py`): `_GradientReversal`, a
`torch.autograd.Function` — identity on the forward pass, negated-and-scaled
on the backward pass (Ganin & Lempitsky 2015). `ModelConfig.domain_adversarial`
(default `False`) gates a new `domain_head = nn.Linear(d_model, 1)` in
`JunctionSetTransformer.__init__`.

**`domain_logits` method**, not a third `forward()` return value. `forward()`
already has call sites across `predict.py`, `score_benchmark.py`,
`train.evaluate`, `ensemble.py` that unpack a fixed `(logits, attn)` 2-tuple;
changing that contract for domain-adversarial checkpoints would touch every
one of them. Instead `domain_logits` mirrors the existing `forward_parts`
pattern (already established by the two-head factorization): it reruns the
encoder pass from the same `_cls_embedding`/`token_proj`/layer stack, ending
in `domain_head(gradient_reversal(pooled, grl_lambda))` instead of the type/
diff heads. Cost: an extra encoder pass per training step when the domain
term is enabled, paid only during training — `forward()`, and therefore every
inference path, is untouched byte-for-byte.

**Domain labels threaded through the data pipeline** (`dsignal/dataset.py`):
`read_labels` now returns 6-tuples (`ix, event_id, label, gene_id, npz_path,
kind`) instead of 5 — `kind` is the labels TSV's own existing provenance
column (`positive`/`nn_hard`/`nn_novel`/`nn_plain` for sim batches,
`real_nn_aqr`/`real_pos_aqr` for Stage 14/16's real batches), so no labels-
file schema change was needed. Every existing index-based read of an entry
(`e[1]` through `e[4]`) is unaffected by the new 6th element; only
`JunctionSetDataset.__getitem__`'s unpacking line changed. `is_real =
kind.startswith("real_")` becomes an item field, batched by `collate` into
`batch["is_real"]`.

**Training** (`dsignal/train.py`): `--domain-lambda` (default `0.0`, mirrors
the existing `--diff-lambda` pattern). `ModelConfig(..., domain_adversarial=
args.domain_lambda > 0)` — at the default, `domain_head` is never
constructed, consuming zero extra RNG draws from model init, so weight
initialization for every other layer is unaffected. `domain_loss()` (BCE
against `batch["is_real"]`, unweighted by domain-class-imbalance by design —
down-weighting the ~7% real fraction would blunt exactly the signal the
domain head is supposed to find hard) is added into the composed loss,
scaled by `--domain-lambda`, in both the epoch-selection loop and the
`--refit` loop.

**Verification (Step 3, done before trusting anything with `--domain-lambda`
nonzero):** `--domain-lambda 0` (the default) reduces to the pre-Stage-17
model exactly, not just approximately. Verified by `git stash` back to the
pre-Stage-17 commit, running `dsignal.train --folds 2 --epochs 2 --device cpu
--seed 0` on an 80-event smoke subset of `npz_union2_realmix`, then `git
stash pop` and rerunning the identical command with the new code (implicit
`--domain-lambda 0`): **identical held-out-fold metrics** (CSE
0.983/0.950/0.966, NN 0.864/0.950/0.905, confusion `[[57 3][1 19]]`, both
runs). Also smoke-tested the `--domain-lambda 1.0` path end-to-end (same
subset) to confirm it runs without error before committing to any real
training.

## Pre-registration (Step 4-5, before scoring)

**Design decision (Step 4):** train on **Stage 14's mix**
(`sim/experiment/npz_union2_realnn` / `labels_union2_realnn.tsv`, 474 real
NN rows only, single-label) first — this isolates whether the adversarial
term alone fixes what Stage 14's architecture couldn't, without also
changing the data mix. This is the harder, more informative test: Stage 14's
mix is exactly the setting where Stage 15 found the shortcut. If Stage 16
has already passed by the time this runs, repeating on Stage 16's mix too is
optional and secondary, not required for this stage's own gate.

**Hyperparameters:** identical to Stage 14/16 otherwise —
`--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`,
3 seeds (0, 1, 2), plus `--domain-lambda` at a value fixed **before training
starts** (not tuned post hoc): **1.0**, matching `--diff-lambda`'s existing
default scale for a second auxiliary loss term (`two_head_loss`'s own
`diff_lambda=1.0` default is the closest precedent in this codebase for
"weight an auxiliary term the same order of magnitude as the main task
loss"). `grl_lambda` (the backward-pass gradient scale inside
`domain_logits`, independent of `--domain-lambda`'s loss weight) stays fixed
at its default 1.0 rather than the Ganin-Lempitsky ramp schedule — a
deliberate simplification, noted here as a limitation: if this attack fails,
a ramped `grl_lambda` is an untried follow-up before concluding the
architecture doesn't work at all.

**Gate:** identical 3-condition gate to Stage 14/16, scored in the same
fixed order (`benchmark_new` dev primary, `benchmark/` dev regression guard,
`ptbp1_arena` confirmatory), against the same baselines
(`runs/bnew_dev_e4_union2.tsv` 142/242, `runs/benchmark_events_stack_dev.tsv`
53/70). Additionally compared against Stage 14's own 3 seeds (same mix,
`--domain-lambda 0`) — the direct ablation of this attack's one changed
variable.

**Standing constraints:** unchanged from Stage 14/16 — `benchmark_new` test
half reserved; never delete a measured number; build → verify → commit each
step; no `git push`; no branch changes; pasteur discipline (not needed here,
this stage is training-only, no new extraction).

## Results

3 seeds trained on Stage 14's mix (`npz_union2_realnn`, 474 real NN rows,
single-label, 12,954 events), `--domain-lambda 1.0`, otherwise identical
hyperparameters to Stage 14/16. Model grew by exactly 193 parameters per
seed (`domain_head = nn.Linear(192, 1)`, 192 weights + 1 bias), confirming
the term was actually active. In-sim macro-F1 0.916-0.921 (5-class
uncollapsed, A3+A5 collapsed), close to Stage 14's own 0.905-0.911 -- the
domain term doesn't cost simulation-side fit.

### benchmark_new dev (primary gate)

| seed | pooled collapsed | vs baseline (142/242=58.7%) | McNemar p | NN recall | vs baseline (11/79=13.9%) |
|---|---:|---:|---:|---:|---:|
| 0 | 128/242 = 52.9% | −14 events | 0.093 | 30/79 = 38.0% | +19 events |
| 1 | 130/242 = 53.7% | −12 events | 0.134 | 30/79 = 38.0% | +19 events |
| 2 | 130/242 = 53.7% | −12 events | 0.148 | 32/79 = 40.5% | +21 events |
| **median** | **53.7%** | **−12** | | **38.0%** | **+19** |

Conditions 1-2 **pass** (median 38.0% beats 13.9%, 3/3 seeds individually
beat it, each McNemar p<0.0001 on NN-prediction alone). Condition 3
**fails**: −12 to −14 events, all beyond the branch's 10-event/n=242 noise
ceiling, in all 3 seeds.

### benchmark/ dev (regression guard)

| seed | pooled collapsed | vs baseline (53/70=75.7%) | McNemar p |
|---|---:|---:|---:|
| 0 | 47/70 = 67.1% | −6 events | 0.180 |
| 1 | 52/70 = 74.3% | −1 event | 1.000 |
| 2 | 47/70 = 67.1% | −6 events | 0.210 |

2/3 seeds regress beyond the 5-event/n=70 noise ceiling; seed 1 is within
noise. **Condition 3 fails here too** (majority of seeds).

**Gate verdict: FAIL** (condition 3, both benchmarks, most seeds) — the
same failure mode as Stage 14, on the same data.

### But: measurably, consistently attenuated versus Stage 14 on identical data

This is the point of running it on Stage 14's mix specifically -- same
data, only the loss changed:

| | Stage 14 (no domain term) | Stage 17 (`--domain-lambda 1.0`) |
|---|---:|---:|
| `benchmark_new` dev NN recall (median) | 57.0% | 38.0% |
| `benchmark_new` dev pooled collapsed (median) | 52.1% (−16 events) | 53.7% (−12 events) |
| `benchmark/` dev pooled collapsed (median) | 53/70 → 45/70 (−8, all 3 seeds) | 53/70 → 47-52/70 (−1 to −6) |
| `ptbp1_arena` NN recall (confirmatory) | 641-675/800 (80-84%) | 566-577/800 (71-72%) |
| `ptbp1_arena` pooled collapsed (confirmatory) | 1175-1230/2422 | 1263-1277/2422 (higher) |

Direct per-seed McNemar, `benchmark_new` dev, Stage 14 vs Stage 17 (same
seed index, same data, architecture is the only variable):

| seed | pooled collapsed p | NN-recall discordant (b/c) | NN-recall p |
|---|---:|---:|---:|
| 0 | 1.000 (128 vs 128) | 16/0 | <0.0001 |
| 1 | 0.280 (122 vs 130) | 15/0 | 0.0001 |
| 2 | 0.608 (126 vs 130) | 13/0 | 0.0002 |

The domain-adversarial term **roughly halves the NN-recall gain** (57.0% →
38.0% median) while pooled-collapsed damage stays about the same or
slightly smaller (median −16 → −12 events) — a real, consistent, 3-seed
effect in the intended direction, just not enough to clear the gate. On
`benchmark/` dev the attenuation is clearer: seed 1 lands within noise
entirely (52/70, −1), something no Stage 14 seed did (all three were at
−8). On `ptbp1_arena`, Stage 17 has *higher* pooled accuracy than Stage 14
despite lower raw NN recall -- consistent with the same story: less of the
"call everything real as NN" behavior, more of the actual class signal
surviving alongside it.

## Verdict: FAIL, but a genuine partial mitigation — not a null result

Unlike Stage 16 (which fully erased the shortcut's effect by removing the
confound in the data), Stage 17 attacks the confound architecturally on the
*same* confounded data and gets a partial, consistent, 3-seed effect in the
right direction: less of the shortcut's symptom (smaller NN-recall
overshoot) and less of its cost (smaller regression), never full attenuation
of either. Two design choices flagged in the pre-registration as
simplifications are the natural next things to try before concluding the
architecture doesn't work:

1. **`grl_lambda` ramp schedule.** Fixed at 1.0 throughout training here,
   rather than Ganin-Lempitsky's schedule (ramping 0→1 over training so the
   domain adversary doesn't dominate early gradients before the main task
   has learned anything useful to protect). A ramp is the standard fix when
   a domain-adversarial term is directionally correct but underpowered,
   which is exactly this result's shape.
2. **`--domain-lambda` magnitude.** Tried only at 1.0 (fixed before training
   per the pre-registration, matching `--diff-lambda`'s default scale). Not
   swept — a stronger weight is untried.

**Combining with Stage 16's data fix** (train the domain-adversarial
architecture on Stage 16's multi-label mix rather than Stage 14's
single-label one) is the other pre-registered optional follow-up
(Step 4 note) — not run here, since Stage 16 already independently passes
the "no regression" bar on its own; the interesting question there would be
whether the domain term also helps decorrelated-but-imperfectly-mixed real
data (Stage 16 still slightly hurt A5/RI recall even though pooled accuracy
held), not the shortcut this stage targets.

**For whoever picks this up next**: of the two attacks in this plan, Stage
16 (change the data) fully closes the gap Stage 14 opened; Stage 17
(change the architecture) only partially does, on the harder single-label
data. Neither is folded into `e4_union2.json`. If NN recall is revisited
again, the ramp-schedule variant of Stage 17 combined with Stage 16's data
is the natural next experiment, not a sixth attempt from scratch.

