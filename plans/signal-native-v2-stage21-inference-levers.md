# Stage 21 — Inference-side levers: plan

Written 2026-09-16. Parent plan: `plans/signal-native-v2-stage18-22-accuracy-plan.md`
§Stage 21. Independent of Stage 20's outcome (Stage 20.3: all three fine-tune
arms FAILED the gate, see `plans/signal-native-v2-stage20-real-finetune.md`
and `plans/signal-native-v2-resume-10.md`) — Stage 21 was never blocked on it
and runs regardless. No retraining anywhere in this stage; everything either
reuses checkpoints already on disk or scores posteriors already cached.

Order: **21.1 → 21.3 → 21.2** (cheapest/most mechanical first, the v1+v2
combination last since it's the only step touching v1's code and needing a
separate conda env / GPU run).

## 21.1 — Selective calling (mechanical, ~half day)

1. Add `--min-confidence THRESH` to `dsignal/predict.py` and
   `dsignal/score_benchmark.py`: predictions with max posterior probability
   below `THRESH` are abstained (`pred="NA"`) — excluded from accuracy,
   tallied separately as a coverage loss. *(mechanical)*
2. `score_benchmark.py`: print an accuracy-vs-coverage table computed
   post-hoc from the `p_*` columns it already writes — report by top-K%
   retained (matching the framing already used: top-50%/top-25%), not a raw
   probability cutoff. No retraining needed; this can run over per-event
   TSVs already on disk. *(mechanical)*
3. Re-score the current headline single-model recipe and `e4_union2` on all
   three evals (`benchmark_new` dev primary, `benchmark/` dev guard,
   `ptbp1_arena` confirmatory) with the new curve. Sanity check: it should
   reproduce the numbers already quoted in the parent plan (`benchmark/` dev
   top-50% = 94%, top-25% = 100%; `benchmark_new` dev flat below max-p 0.8) —
   if it doesn't, the mechanism is wrong before anything else is trusted.
   *(mechanical)*
4. Build → verify → commit.

### 21.1 Results (2026-09-16)

`--min-confidence` landed in `dsignal/predict.py` (shared `abstain()` helper)
and `dsignal/score_benchmark.py` (`score()` forces `pred="NA"` below
threshold; `report()` excludes `"NA"` rows from every accuracy figure and
prints the abstained count as a coverage loss instead of scoring it wrong).
`score_benchmark.py` also gained `--coverage-curve` (prints the table
inline after a live scoring pass) and `--from-tsv` (prints it from an
already-written per-event TSV with no checkpoint/BAMs at all, confirming the
"post-hoc from `p_*`, no re-scoring" property literally).

**Sanity check (step 3) passes**: re-running the curve on
`runs/benchmark_events_e4_union2_dev.tsv` (`benchmark/` dev, `e4_union2`)
via `--from-tsv` reproduces the parent plan's quoted top-50%/top-25%
exactly — **94.3% / 100.0%** vs the quoted 94%/100%. The mechanism is
trusted.

Full top-K% table, both models (headline single model
`runs/geomwin_iv30_refit`, and `e4_union2`), all three evals:

| eval (n) | model | 100% | 75% | 50% | 25% |
|---|---|---:|---:|---:|---:|
| `benchmark/` dev (70) | geomwin_iv30_refit | 75.7% | 82.7% | 97.1% | 100.0% |
| `benchmark/` dev (70) | e4_union2 | 87.1% | 92.3% | 94.3% | 100.0% |
| `benchmark_new` dev (242) | geomwin_iv30_refit | 48.3% | 55.5% | 59.5% | 75.0% |
| `benchmark_new` dev (242) | e4_union2 | 58.7% | 62.1% | 67.8% | 75.0% |
| `ptbp1_arena` (2422) | geomwin_iv30_refit | 49.8% | 54.1% | 59.3% | 68.6% |
| `ptbp1_arena` (2422) | e4_union2 | 48.5% | 54.6% | 61.6% | 75.1% |

(Collapsed accuracy; `benchmark/` dev has no NN truth so its curve is the
one directly comparable across models. `benchmark_new`/arena numbers
qualitatively match the prior "flat below max-p 0.8" read for
`benchmark_new` — the curve rises but stays well under `benchmark/`'s,
consistent with NN being the dominant, confidence-uncorrelated error there.)

Per-event TSVs written this step: `runs/arena_geomwin_iv30_refit.tsv` (new —
the headline single model had not been scored on the arena before); the
`benchmark/`/`benchmark_new`/`e4_union2`-arena TSVs already existed on disk
and were reused unchanged via `--from-tsv`.

Selective calling is a real, monotone lever on every eval measured, largest
where NN is the error mode's minority (`benchmark/`, no NN truth at all) and
smaller but still positive on the two NN-bearing evals — consistent with
confidence correlating with correctness generally, but NN specifically being
mis-called with the same overconfidence as everything else.

## 21.3 — Operating point recheck (mechanical, ~half day, no new training)

1. Run `dsignal.prior_sweep` on dev for the three seeds already on disk
   (`runs/union2_base_s{0,1,2}` / `runs/scale_union2_n12480`) — one pass,
   every prior setting evaluated analytically from the cached per-fold
   posteriors, no re-extraction or retraining. *(mechanical)*
2. Compare the 3-seed argmax `nn_weight` against the standing operating
   point (0.1); keep 0.1 unless a 3-seed-consistent argmax beats it.
   *(mechanical write-up)*
3. Build → verify → commit; fold the outcome into
   `plans/signal-native-v2-resume-10.md`.

### 21.3 Results (2026-09-16)

Ran `dsignal.prior_sweep` (no code change needed — it already caches
per-fold posteriors and evaluates every setting analytically off them) for
all three seed-matched `npz_union2`/n12480 checkpoints on `benchmark_new`
dev (the primary target), `--tta-condition-swap`,
`--train-labels sim/experiment/labels_union2.n12480.tsv`. Collapsed accuracy
per setting (oracle `nn_weight=0` excluded from the argmax search — not
deployable):

| setting | seed 0 (`scale_union2_n12480`) | seed 1 (`union2_base_s1`) | seed 2 (`union2_base_s2`) |
|---|---:|---:|---:|
| uncorrected | 55.0% | 54.5% | 55.4% |
| uniform / nn_weight=1 | 55.0% | 53.7% | **57.4%** |
| nn_weight=0.5 | 56.2% | **54.5%** | 55.8% |
| nn_weight=0.25 | **56.6%** | 54.1% | 57.0% |
| nn_weight=0.1 (standing) | 53.7% | 52.9% | 55.4% |
| nn_weight=0.05 | 52.9% | 51.2% | 53.7% |

Argmax (excluding oracle): seed 0 → `nn_weight=0.25` (56.6%), seed 1 →
tied `uncorrected`/`nn_weight=0.5` (54.5%), seed 2 → tied
`uniform`/`nn_weight=1` (57.4%). **No value is the argmax in more than one
seed** — three seeds, three different answers, none of them 0.1 but no two
of them agreeing with each other either. Per the pre-registered rule
("keep 0.1 unless a 3-seed-consistent argmax beats it"), **0.1 stays.** The
gap from 0.1 to each seed's own argmax (2.9 / 1.6 / 2.0 points, ≈4-7 of 242
events) is in the same range flagged elsewhere on this branch as
within-seed noise, and no single McNemar crosses significance at n=242
against the seed's own uncorrected baseline (p=0.503 seed 0, p=1.000 seed 1,
p=0.062 seed 2 — full curves in
`runs/prior_sweep_union2_s{0,1,2}_bnew_dev.tsv`).

Folded into `plans/signal-native-v2-resume-10.md`.

## 21.2 — v1+v2 combination: mean of posteriors (~1 day)

**Pre-registered rule (locked in before any scoring): mean of posteriors.**
Simple average of v1 and v2's probabilities in the collapsed 4-class space
(A, CSE, NN, RI), argmax. No free parameter, nothing to tune or
second-guess after seeing results. (Considered and rejected: an RI-rescue
fallback rule keyed on v2's RI-vs-A margin against a fixed τ — targets v1's
known RI strength, 89% vs v2's 75% on the arena, but carries a free
parameter that would need fixing a priori without looking at outcomes, a
weaker guarantee than having none at all. Not pursued, not even as a
secondary/exploratory report, per this stage's decision.)

**Constraint found while scoping this**: the shipped v1 checkpoints
(`data/trained_nets/<mode>/trained.pth`) were trained with `--a --cse`
collapsing (per `README.md`'s example `dsEnsemble.py` invocation), so v1's
released nets can only produce 4-class posteriors (A, CSE, NN, RI) — they
cannot separate A3/A5 or CE/SE. The combination therefore happens in that
collapsed space; v2's own 5-class posteriors are collapsed to match
(`p_A = p_A3 + p_A5`) rather than retraining v1.

Steps:

1. Extend `dsEnsemble.py`: capture the softmax it already computes at
   `dsEnsemble.py:144` (`probs = F.softmax(allout, dim=1)`) and currently
   discards after printing only the argmax class. Add `--out TSV`: writes
   one row per event — `csv`, `region`, `truth` (derived from the `--csv`
   filename via the same `CSV_LABEL` mapping `dsignal.score_benchmark`
   uses — import it, don't duplicate it), `p_A`, `p_CSE`, `p_NN`, `p_RI`,
   `pred` — keyed `(csv, region)` exactly like `dsignal.benchmark_split`'s
   comparison functions expect, so the output is directly joinable/scoreable
   with v2's per-event TSVs without a format shim. Existing stdout-only
   behavior is unchanged when `--out` is omitted (backward compatible with
   any existing caller). *(Opus/Sonnet — this is v1's inference path; the
   class-space/`--a --cse` matching has to be exactly right or the
   probabilities are meaningless)*
2. Run the v1 six-net ensemble (local GPU, `DSv1` env, `--zoom -1 --a --cse`,
   per the README's documented invocation) with `--out` over the three eval
   sets: `benchmark_new/{A3,A5,ES,IR,NN}.csv`, `benchmark/{A3,A5,ES,IR}.csv`
   (no NN file there — `benchmark/` has no NN truth), and
   `ptbp1_arena/candidates/{A3,A5,ES,IR,NN}.csv`. *(mechanical)*
3. Write a small combine script: load v2's already-scored per-event
   posteriors for the current headline model (`e4_union2` primary, or
   whichever is designated "v2-only" for this comparison) on the same three
   sets, collapse `p_A3+p_A5 -> p_A`, join with v1's output on
   `(csv, region)`, average the two collapsed-space posterior vectors,
   argmax, write a combined per-event TSV in the same schema `score_benchmark`
   produces so `dsignal.benchmark_split compare` works on it unchanged.
   *(Sonnet)*
4. Score: `ptbp1_arena` is primary (n=2,422 — dev's n=70/242 lacks power for
   this kind of comparison per the branch's own noise-ceiling numbers),
   `benchmark_new` dev confirms. Paired McNemar, v1+v2 vs v2-only and v1+v2
   vs v1-only, both directions reported. v2-only remains the preferred
   headline result regardless of outcome, per the standing user decision
   recorded in the parent plan's Context section — v1+v2 is reported
   alongside it, not proposed as a replacement. *(mechanical)*
5. Build → verify → commit; write up in this file's Results section (create
   one, following this branch's convention: pre-registration above, Results
   appended below, nothing above the line edited except retractions next to
   what they retract).

### 21.2 Results (2026-09-16)

**v1-only reproduces its own known numbers exactly**, confirming the new
`dsEnsemble.py --out` path changed nothing about v1's predictions: 123/140
= 87.9% over all of `benchmark/` (the historical six-net-ensemble figure)
and 60/70 dev on `benchmark/split.tsv`, matching `CLAUDE.md`'s quoted
numbers bit-for-bit.

`e4_union2` was used as v2-only (the current dev-best ensemble, per
`CLAUDE.md`). Pooled collapsed accuracy, all three evals:

| eval (n) | v1-only | v2-only (`e4_union2`) | v1+v2 (mean-of-posteriors) |
|---|---:|---:|---:|
| `ptbp1_arena` (2422) — **primary** | 36.5% | 48.5% | 37.8% |
| `benchmark_new` dev (242) — **confirm** | 48.8% | 58.7% | 50.4% |
| `benchmark/` dev (70) — extra, no NN truth | 85.7% | 87.1% | 87.1% |

Paired McNemar (`dsignal.benchmark_split compare --by-class`):

| comparison | arena p | arena Δ | bnew dev p | bnew dev Δ | bench dev p | bench dev Δ |
|---|---:|---:|---:|---:|---:|---:|
| v1+v2 vs v2-only | <0.0001 | **−10.7 pt** | 0.0091 | **−8.3 pt** | 1.0000 | 0.0 pt |
| v1+v2 vs v1-only | <0.0001 | +1.3 pt | 0.2891 | +1.6 pt | 1.0000 | +1.4 pt |

**v2-only significantly beats v1+v2 on both the primary and confirmatory
evals** (both p<0.01); **v1+v2 beats v1-only only marginally**, significant
on the arena's large n but not at n=242/70. `benchmark/` dev is a wash in
both directions — expected, since it carries no NN truth, v1's single
biggest weak point.

Per-class recall on the arena (primary) explains the shape: mean-of-posteriors
does deliver the RI gain the rejected "RI-rescue" alternative was chasing
(v1+v2 90.5% vs v2-only 75.0% vs v1-only 89.4% — v1's known RI strength
comes through even without a dedicated rule for it) and small gains on
A3/A5/CSE over v1-only, but **NN craters** (v1+v2 3.9% vs v2-only 16.9%,
v1-only 4.4%) because v1 is confidently wrong on NN essentially always (its
softmax is close to one-hot, so a wrong-but-confident v1 vote dominates the
average) and there is no NN truth in v1's own training data to correct it.
Net macro-recall over the 5 classes: v2-only 54.7%, v1+v2 42.9%, v1-only
41.1% — the averaging recovers some of what v1 alone loses but does not
approach v2 alone.

**Reading**: the pre-registered rule works exactly as specified and is not a
bug — it is a real, if modest and mostly non-significant, improvement over
v1 alone, bought at a significant cost relative to v2 alone. This confirms
rather than overturns the standing decision: **v2-only remains the
preferred headline result.** v1+v2 is reported here alongside it, not
proposed as a replacement, per the parent plan's Context section.

Files: `dsEnsemble.py` (`--out`, gated on `--a --cse`), new
`dsignal/combine_v1v2.py`. Per-event TSVs: `runs/v1_{bnew,bench,arena}_
{A3,A5,ES,IR,NN}.tsv` (fresh v1 posteriors, one per truth CSV),
`runs/v1_only_{bnew,bench,arena}.tsv` (concatenated, for the v1-only
McNemar arm), `runs/combined_v1v2_{bnew_dev,bench_dev,arena}.tsv` (the
combination itself).

## Verification

Same standing conventions: build → verify → commit each step; never `git
push`; never create/switch branches unprompted; every benchmark comparison
gets a McNemar test; `benchmark_new` test (243) stays untouched — nothing in
this stage touches it, since everything here scores dev/arena on already-
trained checkpoints.
