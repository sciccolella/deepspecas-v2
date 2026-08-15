# signal-native-v2 — prior-corrected inference (Phase 4 probe)

Tests one hypothesis about the zero-shot sim->real gap, using the *existing*
fold checkpoints. No retraining, no pasteur. Branch: `signal-native-v2`.

## Hypothesis

The dominant benchmark error is false NN: 41/140 under the fixed windows
(`runs/windowfix_allaug`), 29/140 under the old ones (`runs/best`). NN is
540/1260 = 43% of the training design and **0% of the benchmark**. If that
prior mismatch is the cause rather than a topology failure, a logit adjustment
at inference should recover most of the loss for free.

## Key design point

Run the identical sweep on **both** checkpoint sets — `runs/best` (old windows,
70/140 = 50.0%) and `runs/windowfix_allaug` (new windows, 60/140 = 42.9%).
Neither number alone separates "the window fix hurt transfer" from "prior
mismatch always dominated and the window fix is transfer-neutral".

## Steps

### Step 1 — checkpoint audit *(cheap tier)*
Confirm what `fold*.pt` carries. **Done**: `model_state`, `model_config`,
`scaler`, `classes`, `feature_config`, `feature_subset`, `fold`,
`held_out_macro_f1` — no training class counts, so the training prior has to
come from a labels TSV. Add `train_class_counts` to future checkpoints in
`train.py` (cheap, no effect on existing ones).

### Step 2 — implement the correction *(strong tier)*
`p'(y|x) ∝ p(y|x) · (p_target(y) / p_train(y))^tau`, equivalent to adding
`tau·log(target/train)` to the logits. Applied **per fold before the softmax
average** — averaging first and correcting after is a different, wrong
estimator. Opt-in only; the default inference path must not move.

Three target priors:
- `uniform` — domain-agnostic and deployable;
- an NN-weight sweep (scalar penalty on NN, 1.0 -> 0) giving the accuracy
  curve. A curve that stays flat until NN is nearly excluded means the false
  NNs are *confident*, which is a different diagnosis than a marginal one;
- `benchmark` (NN = 0) — an explicit **oracle upper bound**, not deployable:
  it presumes the truth set contains no negatives.

### Step 3 — run and read *(cheap to run, strong to read)*
Both checkpoint sets. Per-CSV accuracy, confusion matrices, McNemar against
each set's own uncorrected baseline. Also report what the correction does to
the *other* classes: if downweighting NN just converts false NNs into CSE->A3
errors, the prior story is incomplete and that has to be said.

### Step 4 — write up and commit *(cheap tier)*

## Verification

- [ ] `score_benchmark.py` with no new flags reproduces 70/140 and 60/140 exactly
- [ ] correction applied per fold, not post-average (checked against a manual
      two-fold computation)
- [ ] both checkpoint sets swept, McNemar reported per set

## What each outcome means

- **Large jump on both sets** -> prior mismatch was always the dominant error,
  the window fix is transfer-neutral, and Phase 4 is largely calibration.
- **Jump only on the new set** -> the window fix shifted NN's decision
  geometry and the 42.9% overstates its cost.
- **No movement** -> the errors are genuine; Phase 4 needs representation-level
  adaptation, not calibration.
