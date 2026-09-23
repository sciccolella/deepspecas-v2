"""Stage 19 junction-tail layer: sparse spurious-junction transcripts per window.

plans/signal-native-v2-stage19-junction-tail.md, step 19.2. Additive to the
existing realisation: the base count matrix, FASTA and polyester seed are
untouched; this writes a *second* FASTA + count matrix that ``run_batch.sh``
feeds to a second polyester call (seed ``batch_seed + 100000``), whose reads
are renamed ``tail<k>`` and concatenated onto each sample before STAR.

The model (``sim/tail_model.json``, fit by ``sim/fit_tail_model.py`` on
aqr/prpf4/u2af2 pos+nn only) is a **width-conditional per-window** model,
not the plan's per-kb rate -- see that module's docstring for the measurement
that forced the change.
Per design row (every kind, label-blind), in its own RNG stream
``SeedSequence([seed, 19, row_index])`` so windows never perturb each other:

1. N ~ the fitted per-window tail-count pmf of the window's width bin, times
   ``--rate-scale`` (``TAIL_RATE_SCALE``; stochastic rounding).
2. Per arc: a location (inside / one end out / both out) from that bin's mix,
   an intron length from the bin's per-location pmf truncated to what is
   feasible at this width, and a uniform placement consistent with both.
3. With P(shared | location), one end moves to the nearest annotated exon
   boundary of the right kind (``exon_structure.tsv``; an in-window end stays in
   the window), and the arc takes that transcript's strand. Otherwise the
   strand is the window's with P(same strand | location), else the opposite.
4. With P(canonical | location), every free end is snapped to the nearest
   GT...AG (+) or CT...AC (-) dinucleotide within +-30 bp in the reference;
   otherwise the arc is kept only if it is *not* any STAR motif. An arc equal
   to an annotated intron, or already emitted for this window, is redrawn.
5. T fragments ~ ``fragments_pmf`` (the PCR-deconvolved 1/2/3 split), each in
   a uniform-random one of the 6 samples. One-condition presence therefore
   follows from T and is checked, not forced.

Each *fragment* is its own 200 bp transcript, ``ref[a-1-lf : a-1] +
ref[b : b+200-lf]``. Deviation from the plan's 500 bp centred design, for
yield: polyester's ``generate_fragments`` keeps a transcript whole when the
drawn fragment length (N(250, 25)) exceeds it, and ``get_reads`` takes 100 bp
from each end -- so a 200 bp transcript yields exactly one read spanning the
junction, with its anchor set by ``lf`` (drawn from the fitted per-read anchor
pmf, junction in either mate). A centred 500 bp transcript loses ~28% of
fragments to insert-only junctions and makes the anchor a geometric accident.

Outputs (``--out-dir``): ``tail.fa``, ``tail_count_matrix.tsv`` (rows in FASTA
order -- polyester matches by position), ``tail_junctions.tsv`` (one row per
arc, for the round-trip checks and Gate A bookkeeping).

Usage::

    python -m sim.build_tail_transcripts build --design design.geomwin.tsv \\
        --exon-structure exon_structure.tsv --reference ref.fa \\
        --model sim/tail_model.json --seed 12 --rate-scale 1.0 --out-dir tail/
    python -m sim.build_tail_transcripts verify --out-dir tail/ \\
        --reference ref.fa --exon-structure exon_structure.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

import numpy as np

SAMPLES = ("cond1_rep1", "cond1_rep2", "cond1_rep3", "cond2_rep1", "cond2_rep2", "cond2_rep3")
READ_LEN = 100
FRAG_LEN = 2 * READ_LEN  # < N(250, 25) in 97.7% of draws -> kept whole by polyester
SNAP_BP = 30
MIN_INTRON = 21  # STAR alignIntronMin
MAX_TRIES = 25
STREAM = 19

# STAR motif codes by (left dinucleotide, right dinucleotide) on the + strand
# genomic sequence, as in dsignal.extract._DINUC_MOTIF.
STAR_MOTIFS = {("GT", "AG"): 1, ("CT", "AC"): 2, ("GC", "AG"): 3, ("CT", "GC"): 4,
               ("AT", "AC"): 5, ("GT", "AT"): 6}
CANON = {"+": ("GT", "AG"), "-": ("CT", "AC")}


class TailModel:
    def __init__(self, d: dict):
        self.raw = d
        self.edges = np.asarray(d["log10_len_edges"], dtype=float)
        self.bins = d["width_bins"]
        self.w_hi = [b["log10_width_hi"] for b in self.bins]
        self.n_pmf = [np.asarray(b["n_pmf"], dtype=float) for b in self.bins]
        self.loc_names = ("in", "out1", "out2")
        self.loc_pmf = [np.asarray([b["loc_pmf"][l] for l in self.loc_names], dtype=float) for b in self.bins]
        self.len_pmf = [{l: np.asarray(b["len_pmf"][l], dtype=float) for l in self.loc_names} for b in self.bins]
        self.by_loc = d["by_loc"]
        self.frag_pmf = np.asarray(d["fragments_pmf"], dtype=float)
        self.anchor_min = int(d["anchor"]["min"])
        self.anchor_pmf = np.asarray(d["anchor"]["pmf"], dtype=float)

    @classmethod
    def load(cls, path: str) -> "TailModel":
        with open(path) as f:
            return cls(json.load(f))

    def width_bin(self, width: int) -> int:
        lw = np.log10(max(width, 1))
        for i, hi in enumerate(self.w_hi):
            if hi is None or lw < hi:
                return i
        return len(self.w_hi) - 1


def _as_model(model) -> TailModel:
    return model if isinstance(model, TailModel) else TailModel(model)


def location(a: int, b: int, s: int, e: int) -> str:
    from sim.tail_stats import location as _loc

    return _loc(a, b, s, e)


def _sample_len(pmf: np.ndarray, edges: np.ndarray, lo: float, hi: float, rng) -> int | None:
    """A length in [lo, hi] bp from a log10-binned pmf truncated to that range
    (uniform in log within a bin), or None if the range carries no mass."""
    llo, lhi = np.log10(max(lo, 1.0)), np.log10(max(hi, 1.0))
    a, b = np.maximum(edges[:-1], llo), np.minimum(edges[1:], lhi)
    width = np.clip(b - a, 0.0, None)
    full = edges[1:] - edges[:-1]
    w = pmf * np.where(full > 0, width / full, 0.0)
    if w.sum() <= 0:
        return None
    k = rng.choice(len(w), p=w / w.sum())
    return int(round(10 ** rng.uniform(a[k], b[k])))


def sample_arc_geometry(model, rng: np.random.Generator, win_start: int, win_end: int) -> tuple:
    """(a, b): 1-based inclusive intron bounds of one tail arc for the window's
    1-based (win_start, win_end]. Location, then a length feasible for it, then
    a uniform placement consistent with both."""
    m = _as_model(model)
    s, e = win_start, win_end
    W = max(e - s, 1)
    i = m.width_bin(W)
    for _ in range(MAX_TRIES):
        li = int(rng.choice(3, p=m.loc_pmf[i]))
        loc = m.loc_names[li]
        pmf = m.len_pmf[i][loc]
        if loc == "in":
            L = _sample_len(pmf, m.edges, MIN_INTRON, W, rng)
            if L is None:
                continue
            a = int(rng.integers(s + 1, e - L + 2))
        elif loc == "out1":
            L = _sample_len(pmf, m.edges, MIN_INTRON, 10 ** m.edges[-1], rng)
            if L is None:
                continue
            if rng.random() < 0.5:  # left end inside, right end past the window
                a = int(rng.integers(max(s + 1, e - L + 2), e + 1))
            else:  # right end inside, left end before the window
                b = int(rng.integers(s + 1, min(e, s + L - 1) + 1))
                a = b - L + 1
        else:
            L = _sample_len(pmf, m.edges, W + 2, 10 ** m.edges[-1], rng)
            if L is None:
                continue
            a = int(rng.integers(e + 2 - L, s + 1))
        return a, a + L - 1
    raise RuntimeError(f"no feasible tail arc for a {W} bp window")


# --------------------------------------------------------------------------
# annotation: exon boundaries and annotated introns


@dataclass
class Sites:
    """Per chrom, sorted annotated intron-left bounds (exon_end + 1) and
    intron-right bounds (exon_start - 1), each with a strand."""

    left: dict = field(default_factory=dict)   # chrom -> (positions list, strands list)
    right: dict = field(default_factory=dict)
    introns: set = field(default_factory=set)  # (chrom, a, b)

    @classmethod
    def from_exon_structure(cls, path: str) -> "Sites":
        from sim.gtf_exon_structure import load_exon_structure

        left, right, introns = {}, {}, set()
        for _gene, txs in load_exon_structure(path).items():
            for _tx, (chrom, strand, exons) in txs.items():
                for i in range(len(exons) - 1):
                    a, b = exons[i][1] + 1, exons[i + 1][0] - 1
                    if b - a + 1 < MIN_INTRON:
                        continue
                    left.setdefault(chrom, set()).add((a, strand))
                    right.setdefault(chrom, set()).add((b, strand))
                    introns.add((chrom, a, b))
        out = cls(introns=introns)
        for src, dst in ((left, out.left), (right, out.right)):
            for chrom, pts in src.items():
                pts = sorted(pts)
                dst[chrom] = ([p for p, _ in pts], [s for _, s in pts])
        return out

    def nearest(self, which: str, chrom: str, pos: int, lo: int, hi: int):
        """Nearest (position, strand) of kind ``which`` in [lo, hi], or None."""
        table = (self.left if which == "left" else self.right).get(chrom)
        if not table or lo > hi:
            return None
        pos_list, strands = table
        i0, i1 = bisect_left(pos_list, lo), bisect_right(pos_list, hi)
        if i0 >= i1:
            return None
        j = bisect_left(pos_list, pos, i0, i1)
        best = None
        for c in (j - 1, j):
            if i0 <= c < i1 and (best is None or abs(pos_list[c] - pos) < abs(pos_list[best] - pos)):
                best = c
        return pos_list[best], strands[best]


# --------------------------------------------------------------------------
# reference


class Ref:
    def __init__(self, path: str):
        import pysam

        self.fa = pysam.FastaFile(path)
        self.lengths = dict(zip(self.fa.references, self.fa.lengths))

    def seq(self, chrom: str, start0: int, end0: int) -> str:
        return self.fa.fetch(chrom, max(start0, 0), end0).upper()

    def left_dinuc(self, chrom: str, a: int) -> str:  # 1-based [a, a+1]
        return self.seq(chrom, a - 1, a + 1)

    def right_dinuc(self, chrom: str, b: int) -> str:  # 1-based [b-1, b]
        return self.seq(chrom, b - 2, b)

    def motif(self, chrom: str, a: int, b: int) -> int:
        return STAR_MOTIFS.get((self.left_dinuc(chrom, a), self.right_dinuc(chrom, b)), 0)

    def snap(self, chrom: str, pos: int, side: str, dinuc: str):
        """Nearest pos' within +-SNAP_BP whose dinucleotide on ``side`` is
        ``dinuc``, or None."""
        lo = pos - SNAP_BP
        if side == "left":  # need ref[p-1 : p+1] == dinuc, 0-based
            window = self.seq(chrom, lo - 1, pos + SNAP_BP + 1)
            cands = [lo + i for i in range(len(window) - 1) if window[i:i + 2] == dinuc]
        else:  # need ref[p-2 : p] == dinuc
            window = self.seq(chrom, lo - 2, pos + SNAP_BP)
            cands = [lo + i for i in range(len(window) - 1) if window[i:i + 2] == dinuc]
        if not cands:
            return None
        return min(cands, key=lambda p: (abs(p - pos), p))


# --------------------------------------------------------------------------
# generation


@dataclass
class Arc:
    a: int
    b: int
    loc: str
    strand: str          # "+", "-", or "." for an intended non-canonical arc
    canonical: bool      # intended
    shared: bool         # realised (an end sits on an annotated boundary)
    fragments: list      # [(sample_index, left_flank_len)]
    motif: int = -1      # from the reference, -1 when built without one


def _stochastic_round(x: float, rng) -> int:
    base = int(np.floor(x))
    return base + int(rng.random() < x - base)


def _try_share(arc_a, arc_b, loc, s, e, chrom, sites: Sites, rng):
    """Move one end onto an annotated boundary. Returns (a, b, strand, moved_end) or None."""
    L = arc_b - arc_a + 1
    a_in, b_in = s < arc_a <= e, s < arc_b <= e
    if loc == "in":
        ends = ["left", "right"]
    elif loc == "out1":
        ends = ["left"] if a_in else ["right"]
    else:
        ends = ["left", "right"]
    end = ends[int(rng.integers(len(ends)))]
    reach = max(1000, L // 2)
    if end == "left":
        if a_in:
            lo, hi = s + 1, min(e, arc_b - MIN_INTRON + 1)
        else:
            lo, hi = arc_a - reach, min(s, arc_b - MIN_INTRON + 1)
        hit = sites.nearest("left", chrom, arc_a, lo, hi)
        return None if hit is None else (hit[0], arc_b, hit[1], "left")
    if b_in:
        lo, hi = max(s + 1, arc_a + MIN_INTRON - 1), e
    else:
        lo, hi = max(e + 1, arc_a + MIN_INTRON - 1), arc_b + reach
    hit = sites.nearest("right", chrom, arc_b, lo, hi)
    return None if hit is None else (arc_a, hit[0], hit[1], "right")


def sample_window(model: TailModel, rng: np.random.Generator, chrom: str, win_start: int,
                  win_end: int, win_strand: str, sites: Sites | None, ref: Ref | None,
                  rate_scale: float = 1.0) -> list:
    """Every tail arc (with its fragments) for one window."""
    pmf = model.n_pmf[model.width_bin(win_end - win_start)]
    n = int(rng.choice(len(pmf), p=pmf))
    if rate_scale != 1.0:
        n = _stochastic_round(n * rate_scale, rng)
    chrom_len = ref.lengths.get(chrom) if ref is not None else None
    arcs, seen = [], set()
    for _ in range(n):
        for _try in range(MAX_TRIES):
            a, b = sample_arc_geometry(model, rng, win_start, win_end)
            loc = location(a, b, win_start, win_end)
            p = model.by_loc[loc]
            want_shared = rng.random() < p["p_shared"]
            canonical = rng.random() < p["p_canonical"]
            same = rng.random() < p["p_same_strand_given_canonical"]
            strand = win_strand if win_strand in "+-" else ("+" if rng.random() < 0.5 else "-")
            if not same:
                strand = "-" if strand == "+" else "+"
            shared = left_fixed = right_fixed = False
            if want_shared and sites is not None:
                hit = _try_share(a, b, loc, win_start, win_end, chrom, sites, rng)
                if hit is not None:
                    a, b, strand, moved = hit
                    shared = True
                    left_fixed, right_fixed = moved == "left", moved == "right"
            if ref is not None:
                if canonical:
                    ld, rd = CANON[strand]
                    if not left_fixed:
                        na = ref.snap(chrom, a, "left", ld)
                        if na is None:
                            continue
                        a = na
                    if not right_fixed:
                        nb = ref.snap(chrom, b, "right", rd)
                        if nb is None:
                            continue
                        b = nb
                    if ref.motif(chrom, a, b) == 0:
                        continue  # the shared end was an annotated site with an odd dinucleotide
                elif ref.motif(chrom, a, b) != 0:
                    continue
            if not canonical:
                strand = "."
            if b - a + 1 < MIN_INTRON:
                continue
            if chrom_len is not None and (a - FRAG_LEN < 1 or b + FRAG_LEN > chrom_len):
                continue
            if sites is not None and (chrom, a, b) in sites.introns:
                continue
            if (a, b) in seen:
                continue
            break
        else:
            continue  # gave up on this arc; the window keeps n - 1
        seen.add((a, b))
        loc = location(a, b, win_start, win_end)
        T = 1 + int(rng.choice(len(model.frag_pmf), p=model.frag_pmf))
        frags = []
        for _f in range(T):
            sample = int(rng.integers(len(SAMPLES)))
            anchor = model.anchor_min + int(rng.choice(len(model.anchor_pmf), p=model.anchor_pmf))
            x = anchor if rng.random() < 0.5 else READ_LEN - anchor   # left part of the spanning read
            lf = x if rng.random() < 0.5 else READ_LEN + x            # junction in the 5' or the 3' mate
            frags.append((sample, lf))
        motif = ref.motif(chrom, a, b) if ref is not None else -1
        arcs.append(Arc(a, b, loc, strand, canonical, shared, frags, motif))
    return arcs


def window_rng(seed: int, row_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, STREAM, row_index]))


def fragment_seq(ref: Ref, chrom: str, a: int, b: int, lf: int) -> str:
    """200 bp: the lf exonic bases ending at a-1, then 200-lf starting at b+1."""
    return ref.seq(chrom, a - 1 - lf, a - 1) + ref.seq(chrom, b, b + FRAG_LEN - lf)


def _read_design(path: str) -> list:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cmd_build(args) -> None:
    model = TailModel.load(args.model)
    sites = Sites.from_exon_structure(args.exon_structure)
    ref = Ref(args.reference)
    rows = _read_design(args.design)
    os.makedirs(args.out_dir, exist_ok=True)
    fa_path = os.path.join(args.out_dir, "tail.fa")
    cm_path = os.path.join(args.out_dir, "tail_count_matrix.tsv")
    jx_path = os.path.join(args.out_dir, "tail_junctions.tsv")
    n_arcs = n_frags = 0
    with open(fa_path, "w") as fa, open(cm_path, "w") as cm, open(jx_path, "w") as jx:
        cm.write("\t" + "\t".join(SAMPLES) + "\n")
        jx.write("ix\tevent_id\tchrom\tstart\tend\tloc\tstrand\tcanonical\tshared\tmotif\t"
                 "n_fragments\tsamples\tleft_flanks\n")
        for i, row in enumerate(rows):
            chrom, s, e = row["chrom"], int(row["win_start"]), int(row["win_end"])
            rng = window_rng(args.seed, i)
            arcs = sample_window(model, rng, chrom, s, e, row.get("strand", "?"), sites, ref, args.rate_scale)
            for j, arc in enumerate(arcs):
                for k, (sample, lf) in enumerate(arc.fragments):
                    tid = f"TAIL_{i:05d}_{j}_{k}"
                    fa.write(f">{tid}\n{fragment_seq(ref, chrom, arc.a, arc.b, lf)}\n")
                    counts = ["0"] * len(SAMPLES)
                    counts[sample] = "1"
                    cm.write(tid + "\t" + "\t".join(counts) + "\n")
                    n_frags += 1
                jx.write(f"{i:05d}\t{row['event_id']}\t{chrom}\t{arc.a}\t{arc.b}\t{arc.loc}\t{arc.strand}\t"
                         f"{int(arc.canonical)}\t{int(arc.shared)}\t{arc.motif}\t{len(arc.fragments)}\t"
                         f"{','.join(str(f[0]) for f in arc.fragments)}\t"
                         f"{','.join(str(f[1]) for f in arc.fragments)}\n")
                n_arcs += 1
    print(f"{len(rows)} windows -> {n_arcs} tail arcs, {n_frags} fragments "
          f"(rate scale {args.rate_scale}); wrote {fa_path}, {cm_path}, {jx_path}", file=sys.stderr)


def cmd_verify(args) -> None:
    """The 19.2 round-trip checks. Exit 1 on any failure."""
    ref = Ref(args.reference)
    sites = Sites.from_exon_structure(args.exon_structure) if args.exon_structure else None
    fa = {}
    name = None
    with open(os.path.join(args.out_dir, "tail.fa")) as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].strip()
            else:
                fa[name] = fa.get(name, "") + line.strip()
    with open(os.path.join(args.out_dir, "tail_count_matrix.tsv")) as f:
        header = f.readline().rstrip("\n").split("\t")[1:]
        cm_ids = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cm_ids.append(parts[0])
            if sum(int(x) for x in parts[1:]) != 1:
                raise SystemExit(f"FAIL: {parts[0]} does not hold exactly one fragment")
    if tuple(header) != SAMPLES:
        raise SystemExit(f"FAIL: count-matrix columns {header}")
    if cm_ids != list(fa):
        raise SystemExit("FAIL: count-matrix rows are not in FASTA order")
    bad = {"rederive": 0, "canonical": 0, "noncanonical": 0, "annotated": 0, "length": 0}
    n = 0
    arc_rank: dict = {}
    with open(os.path.join(args.out_dir, "tail_junctions.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n += 1
            chrom, a, b = r["chrom"], int(r["start"]), int(r["end"])
            # arcs of a window are written in order: j is this row's rank within its ix
            j = arc_rank[r["ix"]] = arc_rank.get(r["ix"], -1) + 1
            for k, lf in enumerate(int(x) for x in r["left_flanks"].split(",")):
                seq = fa.get(f"TAIL_{r['ix']}_{j}_{k}", "")
                if len(seq) != FRAG_LEN or seq[:lf] != ref.seq(chrom, a - 1 - lf, a - 1) \
                        or seq[lf:] != ref.seq(chrom, b, b + FRAG_LEN - lf):
                    bad["rederive"] += 1
            m = ref.motif(chrom, a, b)
            if r["canonical"] == "1":
                # snapped ends are exactly GT..AG / CT..AC; an end moved onto an
                # annotated site keeps that site's own motif (e.g. GC..AG), which
                # is still a STAR motif > 0
                dinucs = (ref.left_dinuc(chrom, a), ref.right_dinuc(chrom, b))
                if m == 0 or (r["shared"] == "0" and dinucs != CANON[r["strand"]]):
                    bad["canonical"] += 1
            elif m != 0:
                bad["noncanonical"] += 1
            if sites is not None and (chrom, a, b) in sites.introns:
                bad["annotated"] += 1
            if b - a + 1 < MIN_INTRON:
                bad["length"] += 1
    print(f"verified {n} arcs, {len(fa)} fragments: {bad}", file=sys.stderr)
    if any(bad.values()):
        raise SystemExit(1)
    print("tail round-trip: PASS", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--design", required=True, help="the design whose windows are extracted")
    b.add_argument("--exon-structure", required=True)
    b.add_argument("--reference", required=True)
    b.add_argument("--model", default="sim/tail_model.json")
    b.add_argument("--seed", type=int, required=True, help="the batch seed")
    b.add_argument("--rate-scale", type=float, default=1.0)
    b.add_argument("--out-dir", required=True)
    b.set_defaults(func=cmd_build)
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--reference", required=True)
    v.add_argument("--exon-structure", default=None)
    v.set_defaults(func=cmd_verify)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
