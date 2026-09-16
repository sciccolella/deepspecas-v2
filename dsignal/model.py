"""`JunctionSetTransformer` -- the signal-native model (Phase 3 step 3,
top-level plan item 3.2).

A junction set is a *set*, not a sequence: there is no meaningful left-to-right
order, but there is rich pairwise structure (which junctions share a donor,
which are nested inside which, how far apart they sit). So this is a plain
pre-norm encoder over junction tokens with two learned additive attention
biases carrying exactly that structure:

* **relative position bias** -- signed ``log10(|midpoint distance| + 1)``
  bucketized into 32 buckets, per head. Raw base-pair distance cannot work
  here: junction separations span four orders of magnitude (tens of bp to
  megabases), so a linear encoding would collapse the entire short range.
* **relation bias** -- a learned per-head scalar for each of the discrete
  relations ``identical`` / ``shares_donor`` / ``shares_acceptor`` / ``nested``
  / ``overlapping`` / ``disjoint``. This is the inductive bias that puts
  A3 ("shares donor, group fractions shift between conditions"), A5 (the
  mirror) and CSE (nested/skipping structure) directly within attention's
  reach, instead of something the model must rediscover from coordinates.

The CLS token is projected from the event-level feature vector rather than
being a free parameter, so window-level context (coverage moments, pooled
topology summaries, the dropped-junction tail) enters the same way. It is never
masked, which also means events with zero junctions are a valid CLS-only
forward pass rather than a degenerate all-padding batch.

Attention weights over junctions are returned on request -- this is the
`--explain` story that `deepSpecas.py predict` still raises
NotImplementedError for.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsignal.features import N_EVENT_FEATURES, N_TOKEN_FEATURES

#: discrete pairwise relations; order is the embedding's index order
RELATIONS = ("identical", "shares_donor", "shares_acceptor", "nested", "overlapping", "disjoint")
N_RELATIONS = len(RELATIONS)

#: half the buckets for each sign of the midpoint offset
N_DISTANCE_BUCKETS = 32
#: log10(distance + 1) is clamped here before bucketizing; 10^7 bp covers any
#: intra-chromosomal separation we will see
_MAX_LOG_DISTANCE = 7.0

#: Position of NN in dsignal.dataset.CLASSES, and the structural types in that
#: same order with NN removed. Derived, not written down twice: the composition
#: in `compose_two_head` has to agree with CLASSES exactly or the 5-vector is
#: silently permuted. Imported lazily inside the function that needs it to keep
#: model.py free of a dataset import cycle.
def _class_layout():
    from dsignal.dataset import CLASSES

    return list(CLASSES).index("NN"), tuple(c for c in CLASSES if c != "NN")


NN_INDEX, TYPE_CLASSES = _class_layout()


class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward (Ganin &
    Lempitsky 2015). This is the whole mechanism a domain-adversarial term
    needs: the domain head still gets a clean signal to learn "is this
    real or simulated" from the pooled embedding, but every gradient it
    sends back through that embedding into the encoder is flipped, so the
    encoder is pushed to make the embedding *worse* for that task instead
    of better -- the shortcut Stage 15 confirmed (`pooled` -> "looks real"
    -> NN) is what this directly opposes."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


@dataclass
class ModelConfig:
    n_token_features: int = N_TOKEN_FEATURES
    n_event_features: int = N_EVENT_FEATURES
    n_classes: int = 5
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.2
    #: two-head factorization (plan Stage 1.3). False keeps the single 5-way
    #: head, which is what every checkpoint before 2026-08-17 carries -- the
    #: field defaults so those still load.
    two_head: bool = False
    #: structural types the type head ranges over, in CLASSES order minus NN
    n_types: int = 4
    #: ablation switches (plan Stage 8, B3). Default True keeps every checkpoint
    #: before this field existed loading unchanged.
    use_distance_bias: bool = True
    use_relation_bias: bool = True
    #: Stage 12 / Track D: bins in the coverage branch's input, 0 = no branch.
    #: Defaults to 0 so every checkpoint written before this field loads and
    #: behaves exactly as it did.
    coverage_bins: int = 0
    #: channels in the coverage encoder's conv stack
    coverage_channels: int = 64
    #: Stage 17: gradient-reversal domain head against the real-vs-simulated
    #: shortcut (Stage 15). Defaults False so every checkpoint before this
    #: field existed loads and behaves exactly as it did -- `domain_logits`
    #: is simply unavailable on those models rather than silently wrong.
    domain_adversarial: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def compose_two_head(type_logits: torch.Tensor, diff_logit: torch.Tensor) -> torch.Tensor:
    """Two heads -> 5-class **log probabilities** in CLASSES order.

    ``p(NN) = 1 - p_diff`` and ``p(c) = p_diff * p(c | differential)``, done in
    log space with logsigmoid so nothing underflows. NN sits at index 3 of
    ``dsignal.dataset.CLASSES`` (A3, A5, CSE, NN, RI), so the four type columns
    wrap around it -- the constant here is checked against CLASSES by
    dsignal.verify_features rather than trusted.
    """
    log_p_diff = F.logsigmoid(diff_logit).unsqueeze(1)          # log p(differential)
    log_p_not = F.logsigmoid(-diff_logit).unsqueeze(1)          # log (1 - p)
    log_type = F.log_softmax(type_logits, dim=1) + log_p_diff   # [B, 4], types in TYPE order
    return torch.cat([log_type[:, :NN_INDEX], log_p_not, log_type[:, NN_INDEX:]], dim=1)


def pairwise_relations(s_rel: torch.Tensor, e_rel: torch.Tensor) -> torch.Tensor:
    """[B, J, J] relation index per junction pair, per RELATIONS.

    Checked in priority order: identical coordinates first (otherwise a pair
    sharing both ends would be counted as sharing only its donor), then the
    two single-end shares, then containment, then bare overlap.
    """
    si, sj = s_rel.unsqueeze(2), s_rel.unsqueeze(1)
    ei, ej = e_rel.unsqueeze(2), e_rel.unsqueeze(1)

    same_s, same_e = si == sj, ei == ej
    nested = ((si >= sj) & (ei <= ej)) | ((sj >= si) & (ej <= ei))
    overlapping = (si <= ej) & (ei >= sj)

    rel = torch.full_like(si, RELATIONS.index("disjoint"))
    rel = torch.where(overlapping, torch.full_like(rel, RELATIONS.index("overlapping")), rel)
    rel = torch.where(nested, torch.full_like(rel, RELATIONS.index("nested")), rel)
    rel = torch.where(same_e, torch.full_like(rel, RELATIONS.index("shares_acceptor")), rel)
    rel = torch.where(same_s, torch.full_like(rel, RELATIONS.index("shares_donor")), rel)
    rel = torch.where(same_s & same_e, torch.full_like(rel, RELATIONS.index("identical")), rel)
    return rel


def distance_buckets(s_rel: torch.Tensor, e_rel: torch.Tensor) -> torch.Tensor:
    """[B, J, J] signed log-distance bucket between junction midpoints.

    Buckets [0, 16) are "j is left of i", [16, 32) "j is right of i", each
    spaced logarithmically so a 50bp and a 500kb separation are distinguishable
    without either swamping the other.
    """
    mid = (s_rel + e_rel).float() / 2.0
    delta = mid.unsqueeze(1) - mid.unsqueeze(2)  # [B, J, J], j minus i
    half = N_DISTANCE_BUCKETS // 2
    log_d = torch.log10(delta.abs() + 1.0).clamp(0.0, _MAX_LOG_DISTANCE)
    idx = (log_d / _MAX_LOG_DISTANCE * (half - 1)).round().long()
    return torch.where(delta < 0, idx, idx + half)


class BiasedSelfAttention(nn.Module):
    """Multi-head self-attention with an additive [B, H, L, L] bias.

    Written out rather than using ``nn.MultiheadAttention`` because the biases
    are per-head and per-pair and have to be added to the logits *before*
    softmax, which the stock module only supports via mask-shaped hacks.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias=None, pad_mask=None, need_weights=False):
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        logits = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        if bias is not None:
            logits = logits + bias
        if pad_mask is not None:
            # a large finite penalty, not -inf: a fully-padded *query* row would
            # otherwise softmax to NaN and poison the batch. Padded queries are
            # discarded downstream anyway; padded keys must not be attended to.
            logits = logits.masked_fill(pad_mask[:, None, None, :], -1e9)

        attn = logits.softmax(dim=-1)
        out = self.dropout(attn) @ v
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.proj(out), (attn if need_weights else None)


class EncoderLayer(nn.Module):
    """Pre-norm block: norm -> attention -> residual, norm -> FFN -> residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, bias=None, pad_mask=None, need_weights=False):
        h, attn = self.attn(self.norm1(x), bias, pad_mask, need_weights)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn


class CoverageEncoder(nn.Module):
    """Binned log-coverage ``[B, 2, bins]`` -> one ``d_model`` vector.

    A dilated 1D conv stack, not a transformer: coverage along a window *is*
    a sequence, and what matters about it is local shape (where an intron's
    coverage steps up, how sharp the boundary is) at several scales at once.
    Dilations 1/2/4/8 reach ~half the window by the last layer, which is the
    span an RI-vs-CSE distinction needs, without the quadratic cost of
    attention over 128 positions that carry no set structure.

    The two conditions are the two input channels and stay aligned throughout,
    so the branch can see a *difference* in profile between conditions -- the
    part of the signal the event-level moments summarize away.
    """

    def __init__(self, bins: int, d_model: int, channels: int = 64, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = 2
        for dilation in (1, 2, 4, 8):
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.GroupNorm(1, channels),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        # mean and max pooling: mean carries "how much intronic signal overall",
        # max carries "is there a sharp localized feature somewhere"
        self.proj = nn.Sequential(
            nn.Linear(2 * channels, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )

    def forward(self, coverage: torch.Tensor) -> torch.Tensor:
        h = self.conv(coverage)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


def batch_inputs(batch: dict, device) -> dict:
    """The model's inputs from a collated batch, moved to ``device``.

    ``coverage`` is present only when the dataset was built with
    ``coverage_bins`` > 0, so callers that predate the branch keep working
    unchanged and a coverage-less batch never reaches a coverage model by
    accident -- the model raises instead.
    """
    out = {
        "tokens": batch["tokens"].to(device),
        "cls": batch["cls"].to(device),
        "s_rel": batch["s_rel"].to(device),
        "e_rel": batch["e_rel"].to(device),
        "pad_mask": batch["pad_mask"].to(device),
    }
    if batch.get("coverage") is not None:
        out["coverage"] = batch["coverage"].to(device)
    return out


class JunctionSetTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.token_proj = nn.Sequential(nn.Linear(c.n_token_features, c.d_model), nn.LayerNorm(c.d_model))
        self.cls_proj = nn.Sequential(nn.Linear(c.n_event_features, c.d_model), nn.LayerNorm(c.d_model))

        # one bias table per head, shared across layers: the relations are
        # properties of the junction set, not of a given depth
        self.distance_bias = nn.Embedding(N_DISTANCE_BUCKETS, c.n_heads)
        self.relation_bias = nn.Embedding(N_RELATIONS, c.n_heads)
        nn.init.zeros_(self.distance_bias.weight)
        nn.init.zeros_(self.relation_bias.weight)

        self.coverage_encoder = (
            CoverageEncoder(c.coverage_bins, c.d_model, c.coverage_channels, c.dropout)
            if c.coverage_bins
            else None
        )

        self.layers = nn.ModuleList(EncoderLayer(c) for _ in range(c.n_layers))
        self.norm = nn.LayerNorm(c.d_model)
        if c.two_head:
            # "what structure is this" and "did it change" are different
            # questions with different domain shifts -- see forward_parts
            self.type_head = nn.Linear(c.d_model, c.n_types)
            self.diff_head = nn.Linear(c.d_model, 1)
        else:
            self.head = nn.Linear(c.d_model, c.n_classes)
        if c.domain_adversarial:
            self.domain_head = nn.Linear(c.d_model, 1)

    def _attention_bias(self, s_rel, e_rel, pad_mask):
        """[B, H, L, L] with L = J + 1; row/col 0 is the CLS token, which
        participates in no junction-pair relation and gets zero bias."""
        b, j = s_rel.shape
        h = self.cfg.n_heads
        if j == 0:
            return torch.zeros(b, h, 1, 1, device=s_rel.device)

        bias = torch.zeros(b, j, j, self.cfg.n_heads, device=s_rel.device)
        if self.cfg.use_relation_bias:
            bias = bias + self.relation_bias(pairwise_relations(s_rel, e_rel))
        if self.cfg.use_distance_bias:
            bias = bias + self.distance_bias(distance_buckets(s_rel, e_rel))
        bias = bias.permute(0, 3, 1, 2)  # [B, H, J, J]

        out = torch.zeros(b, h, j + 1, j + 1, device=bias.device, dtype=bias.dtype)
        out[:, :, 1:, 1:] = bias
        return out

    def _cls_embedding(self, cls, coverage):
        """The CLS token: the event vector projection, plus the coverage
        branch when it is enabled.

        Added into CLS rather than appended as a sixth token on purpose. CLS is
        the only position that is never masked and is what the heads read, so
        the branch reaches the decision directly; as a token it would instead
        have to compete for attention with the junction set and would need a
        relation/distance bias it has no meaningful value for.
        """
        h = self.cls_proj(cls)
        if self.coverage_encoder is not None:
            if coverage is None:
                raise ValueError(
                    "this model was configured with coverage_bins > 0 but the batch "
                    "carries no 'coverage' -- build the dataset with coverage_bins set"
                )
            h = h + self.coverage_encoder(coverage)
        return h

    def _encode(self, tokens, cls, s_rel, e_rel, pad_mask, coverage=None, need_weights: bool = False):
        """Shared encoder pass -- CLS + junction-token embedding, attention
        bias, the layer stack, final norm -- used by `forward`,
        `forward_parts` and `domain_logits` alike (previously each wrote this
        block out separately; see plan Stage 22b.2). Returns ``(x [B, J+1,
        D], attn or None)``: ``x[:, 0]`` is what those three methods call
        "pooled", ``x[:, 1:]`` is the per-junction final hidden state
        `forward_pretrain`'s masked-reconstruction head reads -- the thing
        the pre-refactor private logic discarded by norming only the CLS
        row. ``self.norm`` is applied to the whole sequence rather than just
        position 0, which is exactly equivalent for the CLS row (LayerNorm
        normalizes each position independently over the feature dim) and is
        the bit-identity check this method shipped with.
        """
        b, j, _ = tokens.shape
        x = torch.cat([self._cls_embedding(cls, coverage).unsqueeze(1), self.token_proj(tokens)], dim=1)

        cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=pad_mask.device)
        full_pad = torch.cat([cls_pad, pad_mask], dim=1)
        bias = self._attention_bias(s_rel, e_rel, pad_mask)

        attns = []
        for layer in self.layers:
            x, attn = layer(x, bias, full_pad, need_weights)
            if need_weights:
                attns.append(attn)

        x = self.norm(x)
        return x, (torch.stack(attns, dim=1) if need_weights else None)

    def forward(self, tokens, cls, s_rel, e_rel, pad_mask, need_weights: bool = False,
                coverage=None):
        """tokens [B, J, F], cls [B, E], s_rel/e_rel [B, J], pad_mask [B, J]
        (True = padding). Returns (logits [B, C], attention or None).

        Under ``two_head`` the returned tensor is the **composed 5-class log
        probability**, not a raw logit vector. That is deliberate and exact:
        ``softmax(log p) == p`` for any proper distribution, so every existing
        caller -- argmax in train.evaluate, ``logits.softmax`` in
        predict.fold_probs, the prior correction downstream of it -- keeps
        working and keeps meaning the same thing.
        """
        x, attn = self._encode(tokens, cls, s_rel, e_rel, pad_mask, coverage, need_weights)
        pooled = x[:, 0]
        if self.cfg.two_head:
            type_logits, diff_logit = self.type_head(pooled), self.diff_head(pooled).squeeze(-1)
            logits = compose_two_head(type_logits, diff_logit)
        else:
            logits = self.head(pooled)
        if not need_weights:
            return logits, None
        # CLS's attention over the junction tokens, averaged over heads --
        # "which junctions drove this call"
        return logits, attn[:, :, :, 0, 1:].mean(dim=2)

    def forward_parts(self, tokens, cls, s_rel, e_rel, pad_mask, coverage=None):
        """``(type_logits [B, T], diff_logit [B])`` -- the two heads before
        composition, which is what the training loss needs: the type head is
        supervised only where a structural type exists, and the differential
        head on every event. Two-head models only.

        Why factor at all (plan Stage 1.3, hypothesis 1): "which structure is
        in this window" is something the simulation gets right and which is
        largely depth-robust, while "did it change between conditions" is where
        the whole domain shift lives -- the simulation trains positives at
        |dPSI| ~ U(0.3, 0.8) against negatives pinned at exactly 0, and 81.6%
        of real curated events sit below 0.30. Splitting them lets the NN
        operating point be set on the deployment argument (dsignal/prior_sweep)
        instead of being entangled with the type decision.

        *Measured (2026-08-17, runs/geomwin_twohead.* vs runs/geomwin_innerval_refit.*,
        same data, protocol and hyperparameters):* a null on transfer, and
        hypothesis 1 is **not** supported by it.

            sim-CV macro-F1     0.853 -> 0.858
            dev, uncorrected    49/70 -> 47/70   (p = 0.727), NN false
                                positives 7 -> 13
            dev, +TTA +the pre-registered nn_weight 0.1
                                53/70 -> 53/70   (p = 1.000), 4 events
                                differ in each direction

        So the factorization neither fixes the confident NN false positives nor
        costs anything once the operating point is set. What it does change is
        how the model *responds* to that operating point: uncorrected it calls
        NN roughly twice as often, and the prior correction then buys 8 events
        (65.7% -> 75.7%, McNemar p = 0.016 against its own uncorrected run)
        where the single head buys 4. The decision "did this change" is now a
        single scalar that a deployment argument can move, which is worth
        having; it is just not, on this evidence, a better classifier. That
        points back at the training distribution -- positives at |dPSI| ~
        U(0.3, 0.8) against negatives pinned at exactly 0 -- rather than at the
        head, i.e. plan Stage 3.1, not another architecture.
        """
        if not self.cfg.two_head:
            raise ValueError("forward_parts is only defined for two_head models")
        x, _ = self._encode(tokens, cls, s_rel, e_rel, pad_mask, coverage, need_weights=False)
        pooled = x[:, 0]
        return self.type_head(pooled), self.diff_head(pooled).squeeze(-1)

    def domain_logits(self, tokens, cls, s_rel, e_rel, pad_mask, coverage=None, grl_lambda: float = 1.0):
        """``[B]`` domain logit (real=1 vs simulated=0), off the same pooled
        CLS embedding `forward`/`forward_parts` read, through a gradient-
        reversal layer (Stage 17 design doc,
        plans/signal-native-v2-stage17-domain-adversarial.md).

        Recomputes the encoder pass rather than sharing `forward`'s
        activations -- the same duplicate-pass pattern `forward_parts`
        already uses -- so `forward`'s contract (return shape, call sites in
        predict.py/score_benchmark.py/train.evaluate) is untouched. The
        cost is an extra encoder pass per training step when the domain
        term is enabled; inference never calls this method at all.

        ``grl_lambda`` only scales the *backward* gradient (see
        `_GradientReversal`); it has no effect on the value returned or on
        the forward computation, so calling this with any `grl_lambda` on a
        model in `eval()`/`no_grad()` context is a no-op probe, and the
        loss this feeds is controlled separately by `--domain-lambda`
        (dsignal/train.py's `two_head_loss`-style loss weight, not this
        scale)."""
        if not self.cfg.domain_adversarial:
            raise ValueError("domain_logits is only defined for domain_adversarial models")
        x, _ = self._encode(tokens, cls, s_rel, e_rel, pad_mask, coverage, need_weights=False)
        pooled = x[:, 0]
        return self.domain_head(gradient_reversal(pooled, grl_lambda)).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
