"""Predict Gate A from existing npz, before any pasteur spend (Stage 19).

Adds ``sim/build_tail_transcripts``'s tail arcs straight onto already-extracted
simulated records -- the junction arrays gain what an ideal aligner would
report for the tail reads -- so ``sim.tail_stats`` (A1), ``dsignal.domain_probe``
(A2) and ``sim.tail_stats --a3`` (A3) can be scored locally on e.g. b12-old.

It is an *upper bound on realism of the pipeline, not a substitute for it*:

- no reference: canonical arcs get STAR motif 1 (+) / 2 (-), the rest 0, and no
  end is snapped (so no annotated-site sharing either -- ``shared`` is only a
  flag here);
- no STAR: every tail read aligns and keeps its junction (STAR soft-clips some
  short or non-canonical anchors instead, which ``TAIL_RATE_SCALE`` absorbs);
- coverage and deletion arrays are left untouched (a tail read adds <= 200 bp
  of coverage);
- arcs drawn for *other* windows never reach this one.

PCR duplication is applied exactly as the polyester fork does it in
expectation (per fragment: 1 copy w.p. 1 - rate, else 2 + Poisson(lambda)).
The record strand is re-voted after the tail is added, as the extractor would.

Usage::

    python -m sim.tail_overlay --npz-dir sim/experiment/npz_union2 \\
        --labels sim/experiment/labels_union2.tsv --ix-prefix b12_ --seed 12 \\
        --out-dir runs/stage19/overlay_b12
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from os.path import join as pjoin

import numpy as np

from dsignal.extract import JUNCTION_DTYPE, _MOTIF_STRAND, load_record, save_record, strand_vote
from sim.build_tail_transcripts import READ_LEN, TailModel, sample_window, window_rng

OVERLAY_STREAM = 1919  # PCR draws, independent of the generator's own stream


def _pcr_copies(rng, rate: float, lam: float) -> int:
    return 1 if rng.random() >= rate else 2 + int(rng.poisson(lam))


def overlay_record(record: dict, arcs: list, rng, pcr_rate: float, pcr_lambda: float) -> dict:
    tables = []
    for key in ("junctions_1", "junctions_2"):
        tables.append({(int(j["start"]), int(j["end"])): [int(j["count"]), float(j["nh_weighted_count"]),
                                                          int(j["motif"]), int(j["min_anchor"])]
                       for j in record[key]})
    for arc in arcs:
        motif = (1 if arc.strand == "+" else 2) if arc.canonical else 0
        for sample, lf in arc.fragments:
            x = lf if lf < READ_LEN else lf - READ_LEN
            anchor = min(x, READ_LEN - x)
            copies = _pcr_copies(rng, pcr_rate, pcr_lambda)
            t = tables[0 if sample < 3 else 1]
            v = t.get((arc.a, arc.b))
            if v is None:
                t[(arc.a, arc.b)] = [copies, float(copies), motif, anchor]
            else:
                v[0] += copies
                v[1] += copies
                v[3] = min(v[3], anchor)
    out = dict(record)
    for key, t in zip(("junctions_1", "junctions_2"), tables):
        arr = np.zeros(len(t), dtype=JUNCTION_DTYPE)
        for i, ((a, b), (c, w, m, anc)) in enumerate(sorted(t.items())):
            arr[i] = (a, b, c, w, m, _MOTIF_STRAND.get(m, "?").encode(), anc)
        out[key] = arr
    out["strand"] = strand_vote(out["junctions_1"], out["junctions_2"])
    for c in ("1", "2"):
        out[f"stats_{c}"] = {**record[f"stats_{c}"], "n_junctions": len(out[f"junctions_{c}"])}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ix-prefix", default="", help="keep rows whose ix starts with this (e.g. b12_)")
    ap.add_argument("--model", default="sim/tail_model.json")
    ap.add_argument("--seed", type=int, required=True, help="the batch seed the generator would use")
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    model = TailModel.load(args.model)
    pcr = model.raw["pcr"]
    with open(args.labels) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["ix"].startswith(args.ix_prefix)]
    index = {n.split(".", 1)[0]: n for n in os.listdir(args.npz_dir) if n.endswith(".npz")}
    os.makedirs(pjoin(args.out_dir, "npz"), exist_ok=True)
    n_arcs = 0
    with open(pjoin(args.out_dir, "labels.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            name = index.get(r["ix"])
            if name is None:
                continue
            rec = load_record(pjoin(args.npz_dir, name))
            row_index = int(r["ix"][len(args.ix_prefix):]) if args.ix_prefix else int(r["ix"].rsplit("_", 1)[-1])
            arcs = sample_window(model, window_rng(args.seed, row_index), rec["chrom"], rec["start"],
                                 rec["end"], rec["strand"], None, None, args.rate_scale)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, OVERLAY_STREAM, row_index]))
            save_record(overlay_record(rec, arcs, rng, pcr["rate"], pcr["lambda"]), pjoin(args.out_dir, "npz", name))
            w.writerow(r)
            n_arcs += len(arcs)
    print(f"{len(rows)} records, {n_arcs} tail arcs added (rate scale {args.rate_scale}) -> {args.out_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
