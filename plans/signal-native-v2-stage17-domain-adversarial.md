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

(filled in after scoring)
