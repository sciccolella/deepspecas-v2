"""Torch dataset over extractor .npz records, with augmentations (Phase 3
steps 3-4, top-level plan items 3.2 / 3.3).

Emits ``(tokens [J, F], cls [E], label)`` per event, plus the window-relative
junction coordinates the model's relative-position and relation attention
biases are computed from. Augmentations act on the *normalized record* (see
``dsignal.features.NormalizedRecord``) and features are rebuilt afterwards, so
an augmentation is never a post-hoc edit of already-derived features -- flipping
a record really does recompute donor/acceptor groups from mirrored coordinates.

Junction counts are heavy-tailed (median 12, p99 ~131, max 480 on the 1260-event
set), so ``--max-junctions`` caps the token count keeping the highest-count
junctions; the discarded tail is summarized into the CLS vector by
``features.build_features`` rather than dropped silently. Events with zero
junctions exist (9/1260) and must work: they produce a CLS-only forward pass.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset

from dsignal.extract import load_record
from dsignal.features import (
    N_EVENT_FEATURES,
    N_TOKEN_FEATURES,
    FeatureConfig,
    NormalizedRecord,
    build_features,
    mirror,
    normalize_record,
)
from sim.gene_split import gene_id_of

CLASSES = ("A3", "A5", "CSE", "NN", "RI")

#: the flip augmentation is only label-preserving under this map
_FLIP_LABEL = {"A3": "A5", "A5": "A3"}


@dataclass
class AugmentConfig:
    """Each augmentation independently switchable, all off by default so the
    first run is the control and every contribution is measurable."""

    flip: bool = False
    condition_swap: bool = False
    subsample: bool = False
    junction_dropout: bool = False
    jitter: bool = False

    #: p ~ U(subsample_low, 1) reads retained
    subsample_low: float = 0.3
    #: probability of dropping any one below-median-count junction
    dropout_prob: float = 0.2
    #: retained window fraction ~ U(jitter_low, 1)
    jitter_low: float = 0.6
    #: per-augmentation probability of firing on a given sample
    prob: float = 0.5

    @property
    def any_enabled(self) -> bool:
        return any((self.flip, self.condition_swap, self.subsample, self.junction_dropout, self.jitter))

    @classmethod
    def from_names(cls, names, **kw):
        """AugmentConfig.from_names(["flip", "jitter"]); "all" enables every one."""
        names = list(names or [])
        if "all" in names:
            return cls(flip=True, condition_swap=True, subsample=True, junction_dropout=True, jitter=True, **kw)
        known = {"flip", "condition_swap", "subsample", "junction_dropout", "jitter"}
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"unknown augmentation(s): {sorted(unknown)}")
        return cls(**{n: True for n in names}, **kw)


# --------------------------------------------------------------------------
# augmentations, all operating on a NormalizedRecord
# --------------------------------------------------------------------------


def _replace(norm: NormalizedRecord, **kw) -> NormalizedRecord:
    fields = {
        f: getattr(norm, f)
        for f in (
            "n", "cov_1", "cov_2", "s_rel", "e_rel", "count_1", "count_2",
            "nhw_1", "nhw_2", "motif", "min_anchor", "strand", "flipped",
        )
    }
    fields.update(kw)
    return NormalizedRecord(**fields)


def augment_condition_swap(norm: NormalizedRecord) -> NormalizedRecord:
    """Exchange condition 1 and condition 2. Label-invariant: the label is the
    event *type*, not the direction of the PSI change."""
    return _replace(
        norm,
        cov_1=norm.cov_2, cov_2=norm.cov_1,
        count_1=norm.count_2, count_2=norm.count_1,
        nhw_1=norm.nhw_2, nhw_2=norm.nhw_1,
    )


def augment_subsample(norm: NormalizedRecord, rng: np.random.Generator, low: float) -> NormalizedRecord:
    """Binomially thin junction counts and scale coverage by the same factor --
    depth robustness. Coverage is scaled deterministically rather than thinned
    read-by-read (we no longer have reads at this point); that is an
    approximation, exact in expectation."""
    p = float(rng.uniform(low, 1.0))
    count_1 = rng.binomial(norm.count_1.astype(np.int64), p).astype(np.float64)
    count_2 = rng.binomial(norm.count_2.astype(np.int64), p).astype(np.float64)
    return _replace(
        norm,
        cov_1=norm.cov_1 * p, cov_2=norm.cov_2 * p,
        count_1=count_1, count_2=count_2,
        nhw_1=norm.nhw_1 * p, nhw_2=norm.nhw_2 * p,
    )


def augment_junction_dropout(norm: NormalizedRecord, rng: np.random.Generator, prob: float) -> NormalizedRecord:
    """Drop low-count junctions (below the median total count) at random.
    High-count junctions are never dropped -- losing the dominant junction
    would change the event, not perturb it."""
    j = len(norm.s_rel)
    if j < 3:
        return norm
    total = norm.count_1 + norm.count_2
    candidate = total < np.median(total)
    drop = candidate & (rng.random(j) < prob)
    if not drop.any():
        return norm
    keep = ~drop
    return _replace(
        norm,
        s_rel=norm.s_rel[keep], e_rel=norm.e_rel[keep],
        count_1=norm.count_1[keep], count_2=norm.count_2[keep],
        nhw_1=norm.nhw_1[keep], nhw_2=norm.nhw_2[keep],
        motif=norm.motif[keep], min_anchor=norm.min_anchor[keep],
    )


def augment_jitter(norm: NormalizedRecord, rng: np.random.Generator, low: float) -> NormalizedRecord:
    """Re-crop to a random sub-window, complementing the Phase-2 window
    randomization at the record level. Junctions with no overlap at all with
    the new window are dropped; partially-overlapping ones are kept, since
    that is what the extractor itself produces (a read overlapping the window
    can splice clean out of it)."""
    n = norm.n
    if n < 10:
        return norm
    frac = float(rng.uniform(low, 1.0))
    new_n = max(int(n * frac), 1)
    a = int(rng.integers(0, n - new_n + 1))
    b = a + new_n

    s_rel = norm.s_rel - a
    e_rel = norm.e_rel - a
    keep = (e_rel >= 0) & (s_rel < new_n)
    return _replace(
        norm,
        n=new_n,
        cov_1=norm.cov_1[a:b], cov_2=norm.cov_2[a:b],
        s_rel=s_rel[keep], e_rel=e_rel[keep],
        count_1=norm.count_1[keep], count_2=norm.count_2[keep],
        nhw_1=norm.nhw_1[keep], nhw_2=norm.nhw_2[keep],
        motif=norm.motif[keep], min_anchor=norm.min_anchor[keep],
    )


def apply_augmentations(norm: NormalizedRecord, label: str, rng: np.random.Generator, aug: AugmentConfig):
    """Returns the augmented record and its (possibly swapped) label."""
    if aug.flip and rng.random() < aug.prob:
        norm = mirror(norm)
        label = _FLIP_LABEL.get(label, label)
    if aug.condition_swap and rng.random() < aug.prob:
        norm = augment_condition_swap(norm)
    if aug.jitter and rng.random() < aug.prob:
        norm = augment_jitter(norm, rng, aug.jitter_low)
    if aug.junction_dropout and rng.random() < aug.prob:
        norm = augment_junction_dropout(norm, rng, aug.dropout_prob)
    if aug.subsample and rng.random() < aug.prob:
        norm = augment_subsample(norm, rng, aug.subsample_low)
    return norm, label


# --------------------------------------------------------------------------
# feature scaling
# --------------------------------------------------------------------------


class FeatureScaler:
    """Per-column standardization, fit on *training-fold* records only and
    carried in the checkpoint so inference (including the real benchmark)
    applies exactly the transform the model was trained under."""

    def __init__(self, token_mean=None, token_std=None, cls_mean=None, cls_std=None):
        self.token_mean = token_mean
        self.token_std = token_std
        self.cls_mean = cls_mean
        self.cls_std = cls_std

    @property
    def fitted(self) -> bool:
        return self.token_mean is not None

    def fit(self, token_batches, cls_batches) -> "FeatureScaler":
        tokens = np.concatenate([t for t in token_batches if len(t)], axis=0) if token_batches else None
        if tokens is None or len(tokens) == 0:
            tokens = np.zeros((1, N_TOKEN_FEATURES), dtype=np.float32)
        cls = np.stack(cls_batches, axis=0)
        self.token_mean = tokens.mean(axis=0)
        self.token_std = np.maximum(tokens.std(axis=0), 1e-6)
        self.cls_mean = cls.mean(axis=0)
        self.cls_std = np.maximum(cls.std(axis=0), 1e-6)
        return self

    def transform(self, tokens: np.ndarray, cls: np.ndarray):
        if not self.fitted:
            return tokens, cls
        if len(tokens):
            tokens = (tokens - self.token_mean) / self.token_std
        cls = (cls - self.cls_mean) / self.cls_std
        return tokens.astype(np.float32), cls.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "token_mean": self.token_mean, "token_std": self.token_std,
            "cls_mean": self.cls_mean, "cls_std": self.cls_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureScaler":
        return cls(**{k: np.asarray(v) for k, v in state.items()})


# --------------------------------------------------------------------------
# geometry control (plan Verification item 2)
# --------------------------------------------------------------------------

from dsignal.baseline_gbm import _GEOMETRY_EVENT_FEATURES, _GEOMETRY_TOKEN_FEATURES  # noqa: E402
from dsignal.features import EVENT_FEATURE_NAMES, TOKEN_FEATURE_NAMES  # noqa: E402

GEOMETRY_TOKEN_COLS = np.array([TOKEN_FEATURE_NAMES.index(n) for n in _GEOMETRY_TOKEN_FEATURES])
GEOMETRY_CLS_COLS = np.array([EVENT_FEATURE_NAMES.index(n) for n in _GEOMETRY_EVENT_FEATURES])


#: Token count the geometry control emits, fixed and independent of the record.
GEOMETRY_N_TOKENS = 4


def apply_feature_subset(tokens: np.ndarray, cls: np.ndarray, subset: str):
    """'geometry' keeps window geometry and nothing else (the control that must
    score at chance); 'no-geometry' zeroes the geometry columns, which carry a
    Phase-2 simulation artifact -- see dsignal/baseline_gbm.py.

    The geometry control **replaces the token set** with a fixed number of
    identical zero tokens rather than zeroing token columns in place. Zeroing
    columns left the *number* of junction tokens and their relative spacing
    (via the model's position/relation biases) intact, and junction count is
    structural information, not geometry -- that leak, not window width, is why
    the control originally scored 0.415 macro-F1 while width alone scores 0.253.
    With the token set replaced, the control sees the CLS geometry features and
    nothing else. Callers must pass the meta coordinates through
    `apply_feature_subset_meta` so they match the emitted token count.
    """
    if subset == "all":
        return tokens, cls
    tokens, cls = tokens.copy(), cls.copy()
    if subset == "no-geometry":
        if len(tokens):
            tokens[:, GEOMETRY_TOKEN_COLS] = 0.0
        cls[GEOMETRY_CLS_COLS] = 0.0
    elif subset == "geometry":
        keep_c = np.zeros(N_EVENT_FEATURES, dtype=bool)
        keep_c[GEOMETRY_CLS_COLS] = True
        cls[~keep_c] = 0.0
        tokens = np.zeros((GEOMETRY_N_TOKENS, N_TOKEN_FEATURES), dtype=np.float32)
    else:
        raise ValueError(f"unknown feature subset {subset!r}")
    return tokens, cls


def apply_feature_subset_meta(meta: dict, subset: str) -> dict:
    """Junction coordinates matching what `apply_feature_subset` emitted -- the
    model derives its relative-position and relation attention biases from
    these, so under the geometry control they must be as content-free as the
    dummy tokens they belong to."""
    if subset != "geometry":
        return meta
    zeros = np.zeros(GEOMETRY_N_TOKENS, dtype=np.int64)
    return {**meta, "s_rel": zeros, "e_rel": zeros.copy()}


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def read_labels(labels_path: str, npz_dir: str):
    """[(ix, event_id, label, gene_id, npz_path)] for every labelled record
    that actually has an .npz on disk."""
    index = {}
    for name in os.listdir(npz_dir):
        if name.endswith(".npz"):
            index[name.split(".", 1)[0]] = pjoin(npz_dir, name)
    out = []
    with open(labels_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            path = index.get(row["ix"])
            if path is not None:
                out.append((row["ix"], row["event_id"], row["label"], gene_id_of(row["event_id"]), path))
    return out


class JunctionSetDataset(Dataset):
    def __init__(
        self,
        entries,
        cfg: FeatureConfig | None = None,
        augment: AugmentConfig | None = None,
        scaler: FeatureScaler | None = None,
        classes=CLASSES,
        feature_subset: str = "no-geometry",
        cache: bool = True,
        seed: int = 0,
    ):
        self.entries = list(entries)
        self.cfg = cfg or FeatureConfig(max_junctions=256)
        self.augment = augment or AugmentConfig()
        self.scaler = scaler
        self.classes = list(classes)
        self.class_to_int = {c: i for i, c in enumerate(self.classes)}
        self.feature_subset = feature_subset
        self.cache = cache
        self._cache: dict = {}
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.entries)

    @property
    def labels(self):
        return [e[2] for e in self.entries]

    @property
    def gene_ids(self):
        return [e[3] for e in self.entries]

    def _normalized(self, i: int) -> NormalizedRecord:
        if self.cache and i in self._cache:
            return self._cache[i]
        norm = normalize_record(load_record(self.entries[i][4]), self.cfg)
        if self.cache:
            self._cache[i] = norm
        return norm

    def set_epoch(self, epoch: int) -> None:
        """Augmentation RNG is seeded per (epoch, index), so a run is
        reproducible but each epoch sees different draws."""
        self.epoch = epoch

    def __getitem__(self, i: int):
        ix, event_id, label, gene_id, path = self.entries[i]
        norm = self._normalized(i)

        if self.augment.any_enabled:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            norm, label = apply_augmentations(norm, label, rng, self.augment)

        tokens, cls, meta = build_features(cfg=self.cfg, norm=norm)
        tokens, cls = apply_feature_subset(tokens, cls, self.feature_subset)
        meta = apply_feature_subset_meta(meta, self.feature_subset)
        if self.scaler is not None:
            tokens, cls = self.scaler.transform(tokens, cls)

        return {
            "tokens": torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)),
            "cls": torch.from_numpy(np.ascontiguousarray(cls, dtype=np.float32)),
            "s_rel": torch.from_numpy(meta["s_rel"].astype(np.int64)),
            "e_rel": torch.from_numpy(meta["e_rel"].astype(np.int64)),
            "label": self.class_to_int.get(label, -1),
            "event_id": event_id,
            "ix": ix,
        }

    def raw_features(self, i: int):
        """Unaugmented, unscaled (tokens, cls) -- used to fit the scaler."""
        norm = self._normalized(i)
        tokens, cls, _ = build_features(cfg=self.cfg, norm=norm)
        return apply_feature_subset(tokens, cls, self.feature_subset)

    def fit_scaler(self) -> FeatureScaler:
        toks, clss = [], []
        for i in range(len(self)):
            t, c = self.raw_features(i)
            toks.append(t)
            clss.append(c)
        self.scaler = FeatureScaler().fit(toks, clss)
        return self.scaler


def collate(batch):
    """Pad to the batch's longest junction set; ``pad_mask`` is True where a
    token is padding. Coordinates are padded with 0 and masked out, never read."""
    b = len(batch)
    max_j = max(int(item["tokens"].shape[0]) for item in batch)
    max_j = max(max_j, 1)  # a batch of only zero-junction events still needs a slot
    f = N_TOKEN_FEATURES

    tokens = torch.zeros(b, max_j, f, dtype=torch.float32)
    s_rel = torch.zeros(b, max_j, dtype=torch.long)
    e_rel = torch.zeros(b, max_j, dtype=torch.long)
    pad_mask = torch.ones(b, max_j, dtype=torch.bool)

    for i, item in enumerate(batch):
        j = int(item["tokens"].shape[0])
        if j:
            tokens[i, :j] = item["tokens"]
            s_rel[i, :j] = item["s_rel"]
            e_rel[i, :j] = item["e_rel"]
            pad_mask[i, :j] = False

    return {
        "tokens": tokens,
        "cls": torch.stack([item["cls"] for item in batch]),
        "s_rel": s_rel,
        "e_rel": e_rel,
        "pad_mask": pad_mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "event_id": [item["event_id"] for item in batch],
        "ix": [item["ix"] for item in batch],
    }
