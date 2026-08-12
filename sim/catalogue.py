"""Build a labelled AS-event catalogue from SUPPA `.ioe` files.

Per plans/signal-native-v2.md Phase 2 step 1: labels are grounded in real
annotation via `suppa.py generateEvents`, never invented. This module only
*reads* SUPPA's output -- it does not touch the GTF directly.

SUPPA event_id format (strict boundary mode `-b S`):
    GENE_ID;TYPE:CHROM:coord[:coord...]:STRAND
Coordinate fields vary by event type (SE/A3/A5 use two hyphenated pairs, RI
uses three colon-separated positions with one hyphenated pair, MX uses four
hyphenated pairs, AF/AL use two). We don't hand-parse each type's coordinate
grammar; the region span only needs min/max over every integer present.

Usage:
    from sim.catalogue import build_catalogue
    events = build_catalogue("sim/events", types=("SE", "A3", "A5", "RI"))
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterable, List

# SUPPA event type -> deepSpecas label. Collapsed 4-class headline: SE -> CSE.
# MX/AF/AL are intentionally excluded from the labelled catalogue (per plan:
# "MX/AF/AL -> NN or excluded, decided at build time" -- excluded here; their
# transcripts still count as ordinary background isoforms of the gene).
EVENT_TYPE_LABEL = {
    "SE": "CSE",
    "A3": "A3",
    "A5": "A5",
    "RI": "RI",
}

_INT_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Event:
    event_id: str
    gene_id: str
    event_type: str  # raw SUPPA type, e.g. "SE"
    label: str  # deepSpecas label, e.g. "CSE"
    chrom: str
    strand: str
    region_start: int
    region_end: int
    inclusion_transcripts: FrozenSet[str]
    exclusion_transcripts: FrozenSet[str]

    @property
    def total_transcripts(self) -> FrozenSet[str]:
        return self.inclusion_transcripts | self.exclusion_transcripts


def _parse_event_id(event_id: str):
    """Returns (gene_id, event_type, chrom, strand, region_start, region_end)."""
    gene_id, rest = event_id.split(";", 1)
    event_type, coord_field = rest.split(":", 1)
    # strand is the trailing field
    parts = coord_field.split(":")
    strand = parts[-1]
    chrom = parts[0]
    coord_parts = parts[1:-1]
    ints = [int(x) for p in coord_parts for x in _INT_RE.findall(p)]
    if not ints:
        raise ValueError(f"could not parse coordinates from event_id: {event_id!r}")
    return gene_id, event_type, chrom, strand, min(ints), max(ints)


def load_ioe(path: str) -> Iterable[dict]:
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _events_from_ioe(path: str, event_type: str) -> Iterable[Event]:
    label = EVENT_TYPE_LABEL.get(event_type)
    if label is None:
        return
    for row in load_ioe(path):
        event_id = row["event_id"]
        gene_id, parsed_type, chrom, strand, start, end = _parse_event_id(event_id)
        assert parsed_type == event_type, (event_id, parsed_type, event_type)
        inclusion = frozenset(
            t for t in row["alternative_transcripts"].split(",") if t
        )
        total = frozenset(t for t in row["total_transcripts"].split(",") if t)
        exclusion = total - inclusion
        if not inclusion or not exclusion:
            # degenerate event: nothing to contrast against -> not usable
            continue
        yield Event(
            event_id=event_id,
            gene_id=gene_id,
            event_type=event_type,
            label=label,
            chrom=chrom,
            strand=strand,
            region_start=start,
            region_end=end,
            inclusion_transcripts=inclusion,
            exclusion_transcripts=exclusion,
        )


def build_catalogue(
    events_dir: str,
    types: Iterable[str] = ("SE", "A3", "A5", "RI"),
    suffix: str = "_strict",
) -> List[Event]:
    events_dir = Path(events_dir)
    events: List[Event] = []
    for event_type in types:
        path = events_dir / f"events_{event_type}{suffix}.ioe"
        events.extend(_events_from_ioe(str(path), event_type))
    return events


def _build_argparser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--types", nargs="+", default=["SE", "A3", "A5", "RI"])
    return ap


def main(argv=None) -> None:
    import sys
    from collections import Counter

    args = _build_argparser().parse_args(argv)
    events = build_catalogue(args.events_dir, types=args.types)
    by_label = Counter(e.label for e in events)
    by_gene = Counter(e.gene_id for e in events)
    print(f"loaded {len(events)} events, {len(by_gene)} distinct genes", file=sys.stderr)
    for label, n in sorted(by_label.items()):
        print(f"  {label}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
