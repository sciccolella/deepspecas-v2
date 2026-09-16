# signal-native-v2 — Stage 22b: self-supervised pretraining on real windows

## Context

Stage 18 measured the sim→real gap directly and found it **diffuse**: a
real-vs-sim discriminator on the model's own features holds AUC 0.987–0.999
across every cohort, every class, every past simulation, with or without
geometry, and after removing the 30 most informative feature families
(`plans/signal-native-v2-stage18-domain-probe.md`, rule 1). Stage 20 then
tried to close the gap with real *labels* — three fine-tuning regimes (mixed
+ GRL, head-only, full-low-LR) on a gene-disjoint 3-cohort real pool — and all
three failed their pre-registered gate (`plans/signal-native-v2-stage20-real-finetune.md`;
resume-10 §"Stage 20.3 result"). Per the accuracy plan's own decision table
(`plans/signal-native-v2-stage18-22-accuracy-plan.md` §"Decisions the Stage 18
rules made", rule 1), a diffuse gap plus a failed real-supervision arm moves
**Stage 22b up next**, ahead of the gated re-simulation (Stage 19) and the
coverage-branch retest (22a, which depends on 19).

**What is different here from Stage 20**: self-supervision needs real
*windows*, not curator labels. That means the training pool is not capped at
the ~3.3k hand-curated real positives/NN this branch has been working with —
it can be any expressed-gene window from an already-downloaded, already
gene-vetted real BAM pair, at whatever scale the pretext tasks want. And
because no sim data enters this stage at all, there is no real-vs-sim signal
for the model to shortcut through (the exact failure mode Stage 15 found and
Stage 17's domain-adversarial term tried and failed to fix).

## Design

### Pretext tasks

Both operate on `dsignal.dataset.JunctionSetDataset`'s existing
`NormalizedRecord` → `build_features` pipeline; neither needs a new feature.

**1. Masked junction-feature reconstruction (MFM).** For each event, mask a
random ~30% of its junction tokens (never more than are present; a
zero-junction event contributes nothing to this loss). Masking replaces the
token's `token_proj` *output* embedding with a learned `mask_token`
parameter (BERT-style — masking the raw features would just teach the model
to invert `token_proj`, which is not the point). A `recon_head: Linear(d_model,
N_TOKEN_FEATURES)` reads the final-layer hidden state at each masked
position and predicts the token's true *scaled* feature vector (the same
tensor `JunctionSetDataset.__getitem__` already emits before masking). Loss:
MSE, averaged over masked non-padding positions in the batch. This directly
supervises the "infer a junction from the rest of the set + CLS" behavior the
architecture's docstring already claims as its inductive bias, on real data,
for the first time.

**2. Augmentation contrastive (SimCLR/NT-Xent).** Two independently-drawn
augmented views of the same window — reusing `apply_augmentations` verbatim
(so `flip`, `condition_swap`, `subsample`, `junction_dropout`, `jitter`, each
with its own RNG draw per view) — pooled through a `proj_head` (2-layer MLP,
d_model → d_model → 64, L2-normalized). NT-Xent loss over the batch: a
window's two views are the positive pair, every other window (and its two
views) in the batch are negatives. This teaches the encoder that these
nuisance transforms don't change a window's identity, using real data as the
anchor instead of sim.

Combined loss `L = alpha * L_mfm + beta * L_contrastive`. Three pretraining
variants run: MFM-only (beta=0), contrastive-only (alpha=0), both (alpha=beta=1)
— an ablation, not a hyperparameter search, so the eventual writeup can say
which task (if either) is doing the work.

### Data — no new BAM fetching

Only cohorts already flagged trainable in resume-10's table may be used:
**aqr, prpf4 (pair 3), u2af2 (pair 4)**. `ptbp1` and `tra2`/`benchmark_new`
stay excluded from the pretraining pool exactly as they are excluded from
every other training pool on this branch — for `ptbp1` because of the
92-gene overlap with `benchmark_new` dev (training exposure, even unlabeled,
would compromise the zero-shot claim and the model's eligibility for the
final test spend); for `tra2`/`benchmark_new` because they are the
evaluation sets themselves and self-supervised pretraining on the exact
windows being scored would let the model memorize their shape without ever
seeing a label — a real form of leakage, not a hypothetical one.

Two sources, combined:
- The ~3,305 already-extracted real npz for the three trainable cohorts
  (`sim/experiment/real_{pos,nn}_{aqr,prpf4,u2af2}`) — reused unlabeled
  (labels ignored entirely by the pretext losses), free.
- New: random expressed-gene windows, no catalogued event required, sampled
  from the same already-downloaded `aqr_work`/`prpf4_work`/`u2af2_work` BAMs
  on pasteur (resume-10: "pasteur holds prpf4_work/ and u2af2_work... next to
  ptbp1_work/aqr_work"), several per gene, gene-disjoint from every eval set
  (`benchmark`, `benchmark_new`, `ptbp1_arena`) via the existing
  `build_exclusion_set` machinery. Target ~20–50k windows total across the
  three cohorts to give the pretext tasks real scale.

New script `sim/sample_unlabeled_windows.py`: enumerate genes per cohort not
in the exclusion union, apply `sim/window.py`'s v3 sampler (event span +
U[0,100] bp/edge) directly to the gene span with no event anchor, K windows
per gene (K~3-5, different random offsets/subsets of the gene), write an
extraction CSV in the same `region,bam1,bam2` shape `sim/build_real_pos_extract_csv.py`
already emits. Extraction is `dsignal.extract` under `DSrender` on pasteur,
`limitram`/thread-pool-vars/`nice -n 19` per the branch's pasteur discipline;
only the resulting `.npz` come back.

### Architecture changes (`dsignal/model.py`)

The encoder pass is currently written out three times (`forward`,
`forward_parts`, `domain_logits`), each duplicating the CLS-embedding +
attention-bias + layer-stack block. Before adding a fourth near-copy for
pretraining:

1. Factor a shared `_encode(self, tokens, cls, s_rel, e_rel, pad_mask,
   coverage, need_weights) -> (x, attn)` returning the **full per-token
   hidden state** (not just pooled CLS, which the current private logic
   discards) after the layer stack. `forward`, `forward_parts`, and
   `domain_logits` are rewritten to call it and read `x[:, 0]` for pooled
   CLS. **Verification**: before/after this refactor, every existing
   checkpoint's forward/forward_parts/domain_logits output must be
   `torch.equal` on a fixed batch — this is a pure refactor and must ship as
   its own commit, checked, before any pretext code touches it.
2. Add (inert unless used): `self.mask_token = nn.Parameter(torch.zeros(d_model))`,
   `self.recon_head = nn.Linear(d_model, n_token_features)`,
   `self.proj_head = nn.Sequential(Linear(d_model, d_model), GELU(), Linear(d_model, 64))`.
   These are unused parameters on every classifier checkpoint — a
   `pretrain: bool` field is **not** added to `ModelConfig`; the pretext
   heads exist on every model but are simply never called outside
   `forward_pretrain`, so no existing checkpoint format, `to_dict()`, or
   `--init-from` compatibility check changes.
3. `forward_pretrain(self, tokens, cls, s_rel, e_rel, pad_mask, mask,
   coverage=None) -> (recon_pred [B,J,F] at masked positions, pooled [B,d_model])`:
   embeds tokens via `token_proj`, overwrites masked positions'
   *embeddings* with `mask_token`, runs `_encode`, returns `recon_head`
   applied to masked positions' final hidden states plus the pooled CLS (for
   the caller to run through `proj_head` per-view and compute NT-Xent
   itself, since that loss needs both views' pooled vectors together).

### Fine-tune hookup (`dsignal/train.py`)

`--init-from` (`train.py:251`, `load_parent`) asserts the parent's
`classes`/`feature_subset`/`feature_config` match — the right contract for
resuming a *classifier* checkpoint, wrong for an encoder-only SSL checkpoint
that has no classes or head at all. New flag **`--init-encoder-from PATH`**
(a single file — SSL pretraining has no folds — not a per-fold directory
like `--init-from`): loads only tensors whose name starts with
`token_proj.`/`cls_proj.`/`distance_bias.`/`relation_bias.`/`layers.`/`norm.`
into a freshly-constructed classifier `ModelConfig` (built the normal way
from CLI args, exactly as when no init flag is given); every head
(`head.*`/`type_head.*`/`diff_head.*`/`domain_head.*`) and `mask_token`/
`recon_head`/`proj_head` (not present in a classifier model's parameter
list, so nothing to load) stay randomly initialized. Mutually exclusive with
`--init-from` (argparse error if both given — they answer different
questions: "resume a classifier" vs "warm-start an encoder"). d_model/n_heads/
n_layers/d_ff/dropout/feature_config must be given explicitly and must match
what the SSL run used (asserted from the SSL checkpoint's own saved config,
same pattern as `load_parent`'s asserts).

**Verification**: construct a classifier model, load an SSL checkpoint via
`--init-encoder-from`, confirm every loaded tensor is `torch.equal` to the
SSL checkpoint's corresponding tensor, and every non-loaded tensor differs
from a second independently-seeded random init (i.e. genuinely random, not
accidentally zeroed or accidentally shared).

## Steps

| step | what | tier | cost |
|---|---|---|---|
| 22b.1 | `sim/sample_unlabeled_windows.py` (gene exclusion + v3-sampler windows, no event anchor), extraction CSVs, launch extraction on pasteur | Opus/Sonnet (exclusion + sampler logic), mechanical (launch, `limitram` discipline) | ~half day local + 2–4h pasteur unattended |
| 22b.2 | `model.py`: `_encode` refactor, bit-identity regression check | Opus/Sonnet | ~1 day |
| 22b.3 | `model.py`: `mask_token`/`recon_head`/`proj_head`/`forward_pretrain`; new SSL dataset (two-view + masking, wraps `JunctionSetDataset`'s normalize/augment/build_features pieces) and `dsignal/pretrain.py` CLI (MFM/contrastive/both, `--alpha`/`--beta`, held-out SSL-pool slice for loss monitoring only — never the benchmark) | Opus/Sonnet | ~2 days |
| 22b.4 | `train.py`: `--init-encoder-from`, mutual-exclusion with `--init-from`, wiring/identity checks | Opus/Sonnet | ~1 day |
| 22b.5 | 3 pretraining runs (MFM-only, contrastive-only, both), local GPU, monitor SSL-pool held-out loss to convergence | mechanical | ~half day |
| 22b.6 | 9 downstream fine-tunes: 3 pretraining variants × 3 seeds, identical recipe to `runs/scale_union2_n12480` (`npz_union2`, `--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`) plus `--init-encoder-from`; base control is the **already-measured** random-init numbers (128/130/134 of 242 dev, resume-10), not rerun | mechanical launch | ~1 day |
| 22b.7 | Score on `benchmark_new` dev (primary), `benchmark/` dev (guard), arena + ptbp1 (confirmatory — never in this stage's training pool, so still valid); McNemar each variant's 3 seeds against base; gate; decision doc; resume-10 update | Opus/Sonnet | ~1 day |

Total ≈ 1.5–2 weeks, matching the original accuracy plan's estimate for 22b.

## Pre-registered gate (fixed now, scored after 22b.6)

A pretraining variant **passes** iff, on `benchmark_new` dev:
- its median (of 3 seeds) collapsed accuracy beats the base's median with
  McNemar significance (p < 0.05) in **≥ 2 of 3 seeds** against the
  seed-matched base run, and
- no class's recall regresses beyond the established noise ceiling (10/242
  at this n) relative to base in the passing seeds, and
- the `benchmark/` dev guard shows no beyond-noise regression (5/70), and
- the pretraining loss (whichever of MFM/contrastive/both that variant used)
  actually converged on the held-out SSL-pool slice — a sanity check against
  reporting a variant whose pretext task never learned anything.

If a variant passes: fold it into the ensemble the way Stage 20 would have
(apply to all `e4_union2` members), and additionally try stacking it under
Stage 20's Arm A real-fine-tune recipe (SSL-encoder-init → sim training →
GRL real fine-tune) as a combined candidate — Arm A was the only Stage 20 arm
that moved arena NN cleanly; if a better base encoder removes the
mechanism that made it fail rule 3 (structural false-positive regression on
curated NN), that is worth knowing even though Arm A itself already has a
FAIL verdict on its own.

If no variant passes: record it as a second closed line for the "training
signal" axis (after Stage 20), and the accuracy plan's remaining evidence-order
item is Stage 19 (re-simulation) — currently gated on nothing but its own
25h pasteur cost.

Any model whose training pool (SSL pretraining **or** downstream fine-tune)
contains a `ptbp1_arena` window remains ineligible for the one remaining
`benchmark_new` test spend, per the standing rule.

## Verification (every step)

`dsignal.verify` after any new extraction; the `_encode` refactor's
bit-identity check before any pretext code is added; the
`--init-encoder-from` wiring check before any pretraining run is scored;
3 seeds and McNemar on every downstream comparison; `benchmark_new` test
untouched; build → verify → commit per step; pasteur discipline (`limitram`,
four thread-pool vars, `nice -n 19`, `/proc/loadavg` before/after); resume
doc updated at the end.

## Critical files

- New: `sim/sample_unlabeled_windows.py`, `dsignal/pretrain.py`.
- Modify: `dsignal/model.py` (`_encode` refactor, `mask_token`/`recon_head`/
  `proj_head`/`forward_pretrain`), `dsignal/train.py` (`--init-encoder-from`,
  mutual exclusion with `--init-from` at `train.py:251`/`load_parent`),
  `dsignal/dataset.py` (new SSL dataset class, reusing `apply_augmentations`,
  `normalize_record`, `build_features`).
- Reference only (no change expected): `sim/window.py` (v3 sampler reused
  directly), `sim/select_events.py`'s `build_exclusion_set` (gene exclusion,
  reused), `sim/build_real_pos_extract_csv.py` (CSV-shape pattern to follow).

## Status

Approved 2026-09-16; saved here per the `CLAUDE.md` workflow. Implementation
starts in a separate session, 22b.1 and 22b.2 first (independent of each
other — data sampling doesn't block on the model refactor and vice versa),
then 22b.3/22b.4 (which both need 22b.2's `_encode`), then 22b.5 → 22b.6 →
22b.7 in order.
