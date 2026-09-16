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

## Verification

Same standing conventions: build → verify → commit each step; never `git
push`; never create/switch branches unprompted; every benchmark comparison
gets a McNemar test; `benchmark_new` test (243) stays untouched — nothing in
this stage touches it, since everything here scores dev/arena on already-
trained checkpoints.
