"""Rigorous input-PSI vs. post-STAR-observed-PSI check for a simulated design.

Written for the pilot, but it is now run on **every** batch -- see the
`sim/psi_recovery_*.tsv` outputs, one per noise layer and per scale-up. It is
one of the standing simulation-design gates.

Unlike the earlier "top junction share" proxy used during the pilot smoke
test, this uses each event's *real* junction identity, derived directly
from the SUPPA event_id coordinate fields (not a heuristic), and compares
observed junction/coverage counts from `dsignal.extract.extract_event`
against the `psi1`/`psi2` targets baked into `sim/design.tsv` by
`sim/abundance.py`.

Coordinate semantics, confirmed against SUPPA2's own event_generation
source (github.com/comprna/SUPPA lib/event.py):

- SE: event_id pairs are exon boundaries (exonEnd, exonStart) for the two
  flanking junctions of the skipped exon: (C1_e, A_s) and (A_e, C2_s).
  Convert exon boundary pair (a, b) -> junction (a+1, b-1) (1-based first/
  last intronic base, matching STAR's jI convention). Inclusion = both
  flanking junctions; exclusion = the direct C1->C2 skip junction, derived
  as (pair1[0]+1, pair2[1]-1) (not itself present in the event_id).
- A3/A5: event_id has two exon-boundary pairs, `incl_pair` (listed first)
  and `excl_pair` (listed second) -- SUPPA always writes the
  alternative_transcripts-defining pair first. Same (a+1, b-1) conversion
  as SE. Inclusion/exclusion are each a single junction.
- RI: event_id format is `exon1_start:exon1_end-exon2_start:exon2_end`.
  The middle hyphenated pair (exon1_end, exon2_start) is *already* the
  intron's (start, end) -- no +1/-1 offset (per SUPPA source: "no offset
  applied... represents the intron between two exons"). Inclusion
  (alternative_transcripts) = intron retention, i.e. no splice junction at
  all; estimated from continuous (non-split) read coverage over the
  intron's interior, not a junction lookup. Exclusion = the spliced
  junction at the middle pair.

Usage:
    python -m sim.psi_recovery_check \
        --design sim/design.tsv --events-dir sim/events \
        --bam1 sim/experiment/aligned/c1.bam \
        --bam2 sim/experiment/aligned/c2.bam \
        --out sim/psi_recovery.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from typing import List, Tuple

import numpy as np

from dsignal.extract import ReadFilters, extract_event
from sim.catalogue import build_catalogue

_INT_RE = re.compile(r"\d+")


def _parse_pairs(event_id: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Returns (event_type, raw coordinate pairs), preserving event_id order."""
    _, rest = event_id.split(";", 1)
    event_type, coord_field = rest.split(":", 1)
    parts = coord_field.split(":")[1:-1]  # drop leading chrom, trailing strand
    pairs = []
    for p in parts:
        ints = [int(x) for x in _INT_RE.findall(p)]
        if len(ints) == 2:
            pairs.append((ints[0], ints[1]))
        elif len(ints) == 1:
            pairs.append((ints[0],))
        else:
            raise ValueError(f"unexpected coordinate field {p!r} in {event_id!r}")
    return event_type, pairs


def _exon_pair_to_junction(pair: Tuple[int, int]) -> Tuple[int, int]:
    a, b = pair
    return (a + 1, b - 1)


def event_junctions(event_id: str):
    """Returns a dict describing how to score this event:
    {"kind": "junction", "incl": [(s,e),...], "excl": [(s,e),...]}
    or, for RI, {"kind": "retention", "excl": [(s,e)], "intron": (s,e)}.
    """
    event_type, pairs = _parse_pairs(event_id)
    if event_type == "SE":
        p1, p2 = pairs
        j1, j2 = _exon_pair_to_junction(p1), _exon_pair_to_junction(p2)
        skip = _exon_pair_to_junction((p1[0], p2[1]))
        return {"kind": "junction", "incl": [j1, j2], "excl": [skip]}
    if event_type in ("A3", "A5"):
        incl_pair, excl_pair = pairs
        return {
            "kind": "junction",
            "incl": [_exon_pair_to_junction(incl_pair)],
            "excl": [_exon_pair_to_junction(excl_pair)],
        }
    if event_type == "RI":
        _pos1, mid_pair, _pos2 = pairs
        # Empirically (checked against real STAR junctions), the middle pair
        # is exon-boundary style like SE/A3/A5, not pre-offset intron bounds
        # as SUPPA's own docs/comments claim -- verified off by exactly 1 on
        # each side against real jI-tagged junctions in the pilot BAMs.
        intron = _exon_pair_to_junction(mid_pair)
        return {"kind": "retention", "excl": [intron], "intron": intron}
    raise ValueError(f"unsupported event type {event_type!r}")


def _junction_lookup(junction_arr: np.ndarray) -> dict:
    return {
        (int(r["start"]), int(r["end"])): float(r["nh_weighted_count"])
        for r in junction_arr
    }


def _observed_psi(spec: dict, junctions: np.ndarray, cov: np.ndarray, region_start: int):
    lut = _junction_lookup(junctions)
    excl_count = sum(lut.get(j, 0.0) for j in spec["excl"])
    if spec["kind"] == "junction":
        incl_count = sum(lut.get(j, 0.0) for j in spec["incl"]) / len(spec["incl"])
        denom = incl_count + excl_count
        psi = incl_count / denom if denom > 0 else float("nan")
        return psi, incl_count, excl_count
    # retention: inclusion = non-split coverage over the intron interior
    s, e = spec["intron"]
    lo = max(s - region_start + 5, 0)
    hi = min(e - region_start - 5, len(cov))
    if hi <= lo:
        lo, hi = max(s - region_start, 0), min(e - region_start, len(cov))
    incl_count = float(np.median(cov[lo:hi])) if hi > lo else 0.0
    denom = incl_count + excl_count
    psi = incl_count / denom if denom > 0 else float("nan")
    return psi, incl_count, excl_count


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--bam1", required=True)
    ap.add_argument("--bam2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    catalogue = {e.event_id: e for e in build_catalogue(args.events_dir)}

    with open(args.design) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    filters = ReadFilters()
    out_rows = []
    for row in rows:
        if row["kind"] != "positive":
            continue
        event_id = row["event_id"]
        if event_id not in catalogue:
            print(f"WARN: {event_id} not in catalogue, skipping", file=sys.stderr)
            continue
        spec = event_junctions(event_id)
        region = (row["chrom"], int(row["region_start"]), int(row["region_end"]))
        record = extract_event(region, args.bam1, args.bam2, filters=filters)

        obs_psi1, incl1, excl1 = _observed_psi(
            spec, record["junctions_1"], record["cov_1"], region[1]
        )
        obs_psi2, incl2, excl2 = _observed_psi(
            spec, record["junctions_2"], record["cov_2"], region[1]
        )
        target_psi1, target_psi2 = float(row["psi1"]), float(row["psi2"])

        out_rows.append(
            dict(
                event_id=event_id,
                label=row["label"],
                event_type=event_id.split(";", 1)[1].split(":", 1)[0],
                target_psi1=target_psi1,
                obs_psi1=obs_psi1,
                err1=obs_psi1 - target_psi1 if not np.isnan(obs_psi1) else float("nan"),
                target_psi2=target_psi2,
                obs_psi2=obs_psi2,
                err2=obs_psi2 - target_psi2 if not np.isnan(obs_psi2) else float("nan"),
                incl1=incl1,
                excl1=excl1,
                incl2=incl2,
                excl2=excl2,
            )
        )
        print(
            f"{event_id[:60]:60s} label={row['label']:4s} "
            f"target=({target_psi1:.2f},{target_psi2:.2f}) "
            f"obs=({obs_psi1:.2f},{obs_psi2:.2f})",
            file=sys.stderr,
        )

    fields = list(out_rows[0].keys()) if out_rows else []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    errs = np.array(
        [r["err1"] for r in out_rows if not np.isnan(r["err1"])]
        + [r["err2"] for r in out_rows if not np.isnan(r["err2"])]
    )
    n_nan = sum(np.isnan(r["err1"]) + np.isnan(r["err2"]) for r in out_rows)
    print(f"\n{len(out_rows)} events scored, {n_nan} psi values undefined (no reads)", file=sys.stderr)
    if len(errs):
        print(
            f"abs error: mean={np.abs(errs).mean():.4f} median={np.median(np.abs(errs)):.4f} "
            f"p90={np.quantile(np.abs(errs), 0.9):.4f} max={np.abs(errs).max():.4f}",
            file=sys.stderr,
        )
        from collections import defaultdict

        by_label = defaultdict(list)
        for r in out_rows:
            for e in (r["err1"], r["err2"]):
                if not np.isnan(e):
                    by_label[r["label"]].append(abs(e))
        for label, vals in sorted(by_label.items()):
            print(f"  {label}: mean_abs_err={np.mean(vals):.4f} n={len(vals)}", file=sys.stderr)


if __name__ == "__main__":
    main()
