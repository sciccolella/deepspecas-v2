"""Annotation-derived labels for real BAM windows (Stage 12 / Track A, step A0).

The premise, from `plans/signal-native-v2-real-supervision.md`: the model's
five classes are *event types*, and `sim/events/*_strict.ioe` already defines
those geometrically. What curation actually decided was something else --
whether the event is *there* in this sample. So real labelled data needs no
curation, only

    label  = the class of the uniquely-assigned catalogued event in the window
    present = a BAM-derived support criterion, not a model and not a dPSI

and this module implements exactly those two, plus the assignment rule that
turns "several catalogued events overlap this window" into one label or an
explicit abstention.

## The two forms of an event

Presence here means *alternatively spliced in this sample*, i.e. both
competing forms carry evidence -- not merely "annotated" and not merely
"expressed". Each SUPPA type contributes two forms, read straight off the
event_id's coordinate grammar (strict mode, `-b S`), all coordinates 1-based
exon boundaries:

    SE:  e1end-e2start:e2end-e3start   inclusion {(e1end,e2start),(e2end,e3start)}
                                       exclusion {(e1end,e3start)}
    A3:  e1end-aLong:e1end-aShort      two acceptors sharing a donor
    A5:  dLong-e2start:dShort-e2start  two donors sharing an acceptor
    RI:  e1start:e1end-e2start:e2end   spliced {(e1end,e2start)} vs retained
                                       (intronic coverage, no junction)

An intron derived from a pair of exon boundaries (a, b) is [a+1, b-1] in the
1-based inclusive convention STAR's `jI` uses -- but that offset is
*calibrated empirically* by `calibrate_offset` rather than trusted, because a
silent off-by-one here would look exactly like "the annotation doesn't match
the data".

## Deliberate non-goals

* No dPSI, no p-value, no caller output: differential usage is a different
  question from event type, and Stage 12 measured that caller significance
  does not separate real events from junk anyway (67.2% purity, flat under
  every stringency cut).
* No model prediction, ever, anywhere in this path.

## The shortcut hazard

A support-based presence filter uses signal the model itself sees, so
real-derived NN risks being trivially separable ("no junctions here"). Two
mitigations, both required by A0: NN windows must clear the same expression
floor as positives (`min_window_cov`), and a control model trained only on
the presence-filter quantities must be reported alongside -- if that control
already separates NN, the real NN class is a shortcut and cannot be used as
evidence about NN recall.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from sim.catalogue import Event, build_catalogue

# --- defaults, declared before the gate was run (Stage 12 / A1) ---
MIN_JUNC_READS = 5  # per form, summed over both conditions
MIN_CONTAINMENT = 0.5  # fraction of the event span that must lie inside the window
RI_MIN_INTRON_FRAC = 0.10  # median intron cov / median flanking exon cov
RI_FLANK = 100  # bp of flanking exon used as the RI denominator
MIN_WINDOW_COV = 1.0  # mean per-base coverage floor, both classes alike

AMBIGUOUS = "AMBIG"
NN = "NN"


@dataclass(frozen=True)
class EventForms:
    """The two competing forms of one catalogued event, as intron sets."""

    event: Event
    form_a: FrozenSet[Tuple[int, int]]
    form_b: FrozenSet[Tuple[int, int]]
    ri_intron: Optional[Tuple[int, int]] = None  # RI only: retention is coverage

    @property
    def label(self) -> str:
        return self.event.label


def _coord_pairs(event_id: str) -> List[List[int]]:
    """The event_id's middle coordinate fields, each split into its integers.

    `gene;TYPE:chrom:field[:field...]:strand` -- returns one list per field.
    """
    _, rest = event_id.split(";", 1)
    parts = rest.split(":")
    return [[int(x) for x in field.split("-")] for field in parts[2:-1]]


def event_forms(event: Event) -> Optional[EventForms]:
    """Intron sets for the event's two forms, or None if the grammar is
    unexpected (defensive: the catalogue is large and third-party)."""
    f = _coord_pairs(event.event_id)
    t = event.event_type
    try:
        if t == "SE":
            (e1end, e2start), (e2end, e3start) = f
            return EventForms(
                event,
                form_a=frozenset({(e1end, e2start), (e2end, e3start)}),
                form_b=frozenset({(e1end, e3start)}),
            )
        if t in ("A3", "A5"):
            (a1, b1), (a2, b2) = f
            return EventForms(event, form_a=frozenset({(a1, b1)}), form_b=frozenset({(a2, b2)}))
        if t == "RI":
            (_e1start,), (e1end, e2start), (_e2end,) = f
            return EventForms(
                event,
                form_a=frozenset({(e1end, e2start)}),
                form_b=frozenset(),
                ri_intron=(e1end, e2start),
            )
    except (ValueError, TypeError):
        return None
    return None


class CatalogueIndex:
    """Per-chromosome sorted spans, queried by window overlap."""

    def __init__(self, events: Sequence[Event]):
        self._by_chrom: Dict[str, list] = {}
        for ev in events:
            forms = event_forms(ev)
            if forms is None:
                continue
            self._by_chrom.setdefault(ev.chrom, []).append(
                (ev.region_start, ev.region_end, forms)
            )
        for chrom in self._by_chrom:
            self._by_chrom[chrom].sort(key=lambda r: r[0])
        self._starts = {c: [r[0] for r in v] for c, v in self._by_chrom.items()}
        # longest span per chromosome bounds how far left a hit can start
        self._max_span = {
            c: max((r[1] - r[0] for r in v), default=0) for c, v in self._by_chrom.items()
        }

    @classmethod
    def from_dir(cls, events_dir: str) -> "CatalogueIndex":
        return cls(build_catalogue(events_dir))

    def overlapping(
        self, chrom: str, start: int, end: int, min_containment: float = MIN_CONTAINMENT
    ) -> List[EventForms]:
        rows = self._by_chrom.get(chrom)
        if not rows:
            return []
        lo = bisect.bisect_left(self._starts[chrom], start - self._max_span[chrom] - 1)
        out = []
        for s, e, forms in rows[lo:]:
            if s > end:
                break
            span = max(1, e - s)
            inside = min(e, end) - max(s, start)
            if inside / span >= min_containment:
                out.append(forms)
        return out


# --- BAM-side evidence -------------------------------------------------------


def junction_counts(record: dict, offset: Tuple[int, int] = (1, -1)) -> Dict[Tuple[int, int], int]:
    """Observed junctions as {(exon_end, next_exon_start): reads}, summed over
    both conditions. `offset` maps an annotation exon-boundary pair to the
    intron coordinates the extractor reports; see `calibrate_offset`."""
    da, db = offset
    counts: Dict[Tuple[int, int], int] = {}
    for key in ("junctions_1", "junctions_2"):
        arr = record[key]
        for row in arr:
            k = (int(row["start"]) - da, int(row["end"]) - db)
            counts[k] = counts.get(k, 0) + int(row["count"])
    return counts


def calibrate_offset(
    records: Sequence[dict],
    index: "CatalogueIndex",
    candidates: Sequence[Tuple[int, int]] = ((1, -1), (0, 0), (1, 0), (0, -1)),
) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], int]]:
    """Pick the annotation->observed coordinate offset that matches the most
    catalogued junctions. Guards against a silent off-by-one masquerading as
    "the annotation doesn't fit the data"."""
    scores = {}
    for off in candidates:
        hits = 0
        for rec in records:
            obs = set(junction_counts(rec, off))
            for forms in index.overlapping(rec["chrom"], rec["start"], rec["end"]):
                hits += len(obs & (forms.form_a | forms.form_b))
        scores[off] = hits
    best = max(scores, key=scores.get)
    return best, scores


def _form_support(counts: Dict[Tuple[int, int], int], form: FrozenSet[Tuple[int, int]]) -> int:
    """A form is only as supported as its weakest constituent junction."""
    if not form:
        return 0
    return min(counts.get(j, 0) for j in form)


def _ri_retention_frac(record: dict, intron: Tuple[int, int], flank: int = RI_FLANK) -> float:
    """Median intronic coverage over median flanking-exon coverage, both
    conditions pooled. Window-local and annotation-free once the intron is
    given -- the same shape as `audit_noise`'s `intron_cov_frac`."""
    start, end = record["start"], record["end"]
    cov = record["cov_1"].astype(np.float64) + record["cov_2"].astype(np.float64)
    a, b = intron
    i0, i1 = a - start, b - start  # intron interior, window-relative
    i0, i1 = max(0, i0), min(len(cov), i1)
    if i1 - i0 < 10:
        return 0.0
    left = cov[max(0, i0 - flank) : i0]
    right = cov[i1 : min(len(cov), i1 + flank)]
    flanking = np.concatenate([left, right]) if left.size or right.size else np.array([0.0])
    denom = float(np.median(flanking))
    if denom <= 0:
        return 0.0
    return float(np.median(cov[i0:i1])) / denom


@dataclass
class Presence:
    forms: EventForms
    support_a: int
    support_b: int
    ri_frac: float
    present: bool

    @property
    def strength(self) -> float:
        """Deterministic tie-break when several events are present at once."""
        return float(min(self.support_a, self.support_b) + max(self.support_a, self.support_b) / 1e6)


def event_presence(
    record: dict,
    forms: EventForms,
    counts: Dict[Tuple[int, int], int],
    min_reads: int = MIN_JUNC_READS,
    ri_min_frac: float = RI_MIN_INTRON_FRAC,
) -> Presence:
    """Both competing forms must carry evidence. For RI the retained form is
    intronic coverage rather than a junction, so it uses a coverage ratio."""
    sa = _form_support(counts, forms.form_a)
    if forms.ri_intron is not None:
        frac = _ri_retention_frac(record, forms.ri_intron)
        present = sa >= min_reads and frac >= ri_min_frac
        # the retained form has no junction; report its evidence as the ratio
        return Presence(forms, sa, 0, frac, present)
    sb = _form_support(counts, forms.form_b)
    return Presence(forms, sa, sb, 0.0, sa >= min_reads and sb >= min_reads)


def window_mean_cov(record: dict) -> float:
    cov = record["cov_1"].astype(np.float64) + record["cov_2"].astype(np.float64)
    return float(cov.mean()) if cov.size else 0.0


@dataclass
class LabelResult:
    label: str  # A3 / A5 / CSE / RI / NN / AMBIG
    resolved: str  # same, but AMBIG broken by support instead of abstaining
    n_candidates: int
    n_present: int
    mean_cov: float
    detail: List[Presence]


def label_window(
    record: dict,
    index: CatalogueIndex,
    offset: Tuple[int, int] = (1, -1),
    min_reads: int = MIN_JUNC_READS,
    min_containment: float = MIN_CONTAINMENT,
    ri_min_frac: float = RI_MIN_INTRON_FRAC,
    min_window_cov: float = MIN_WINDOW_COV,
) -> LabelResult:
    """The A0 assignment rule.

    One present event -> its class. Several agreeing -> that class. Several
    disagreeing -> abstain (`AMBIG`), with `resolved` carrying the
    highest-support choice for the callers that would rather have a guess.
    None present, but the window is expressed -> NN. Not expressed -> AMBIG,
    because "no reads" is not evidence of "no event" and must not be allowed
    to become free NN training data.
    """
    mean_cov = window_mean_cov(record)
    cands = index.overlapping(record["chrom"], record["start"], record["end"], min_containment)
    counts = junction_counts(record, offset)
    pres = [
        event_presence(record, f, counts, min_reads=min_reads, ri_min_frac=ri_min_frac)
        for f in cands
    ]
    present = [p for p in pres if p.present]

    if not present:
        label = NN if mean_cov >= min_window_cov else AMBIGUOUS
        return LabelResult(label, label, len(cands), 0, mean_cov, pres)

    labels = {p.forms.label for p in present}
    best = max(present, key=lambda p: p.strength).forms.label
    if len(labels) == 1:
        only = labels.pop()
        return LabelResult(only, only, len(cands), len(present), mean_cov, pres)
    return LabelResult(AMBIGUOUS, best, len(cands), len(present), mean_cov, pres)
