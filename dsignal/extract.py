"""Signal-native extractor: junction set + base-resolution coverage.

Reuses the fetch+CIGAR-walk shape from ``plot2s.get_coords_from_bam`` but keeps
junction identity intact instead of collapsing every N-op into a depth-of-N
array. Never touches a GTF: strand comes from ``jM`` (STAR's intron-motif tag,
with the +20 "annotated junction" bit masked off) or, as a fallback, from
reference dinucleotides at the intron boundaries. See
plans/signal-native-v2.md Phase 1.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from os import makedirs
from os.path import join as pjoin
from typing import Optional

import numpy as np
import pysam

# --- CIGAR op codes (pysam/htslib convention) ---
_OP_M = 0
_OP_D = 2
_OP_N = 3
_OP_EQ = 7
_OP_X = 8

_COV_OPS = frozenset((_OP_M, _OP_EQ, _OP_X))
_DEL_OP = _OP_D
_SKIP_OP = _OP_N
_REF_CONSUMING_OPS = _COV_OPS | {_DEL_OP, _SKIP_OP}

# STAR jM intron motif -> strand, once the +20 "annotated in the index's GTF"
# bit is masked off with `% 20`. 0 = non-canonical / unknown.
_MOTIF_STRAND = {0: "?", 1: "+", 2: "-", 3: "+", 4: "-", 5: "+", 6: "-"}

# Reference dinucleotide (donor, acceptor) -> STAR motif code, for the
# CIGAR-walk fallback when jI/jM tags are absent or ignored.
_DINUC_MOTIF = {
    ("GT", "AG"): 1,
    ("CT", "AC"): 2,
    ("GC", "AG"): 3,
    ("CT", "GC"): 4,
    ("AT", "AC"): 5,
    ("GT", "AT"): 6,
}

JUNCTION_DTYPE = np.dtype(
    [
        ("start", np.int64),  # 1-based, first intronic base (jI convention)
        ("end", np.int64),  # 1-based, last intronic base
        ("count", np.int32),
        ("nh_weighted_count", np.float32),
        ("motif", np.int8),  # 0-6, see _MOTIF_STRAND; -1 = unknown (no tags, no ref)
        ("strand", "S1"),  # b"+", b"-", b"?"
        ("min_anchor", np.int32),
    ]
)


@dataclass
class ReadFilters:
    """Read-level knobs, on by default. `min_anchor` is applied per junction
    occurrence (a read can have one short-anchor junction among several
    solid ones); `max_nh`/`min_mapq` drop the whole read, coverage included.
    """

    min_anchor: int = 5
    max_nh: Optional[int] = 10
    min_mapq: int = 0

    def keep_read(self, read: pysam.AlignedSegment, nh: int) -> bool:
        if self.min_mapq and read.mapping_quality < self.min_mapq:
            return False
        if self.max_nh is not None and nh > self.max_nh:
            return False
        return True


@dataclass
class ConditionRecord:
    cov: np.ndarray
    delcov: np.ndarray
    junctions: np.ndarray
    stats: dict = field(default_factory=dict)


def parse_region(string: str) -> tuple:
    m = re.match(r"(?P<chr>[\w.]+):(?P<start>\d+)-(?P<end>\d+)", string)
    if not m:
        raise ValueError(f"unable to parse region {string!r}")
    return (m.group("chr"), int(m.group("start")), int(m.group("end")))


def _tag_junctions(read: pysam.AlignedSegment):
    """(start_1b, end_1b, motif) per N-op from jI/jM, in CIGAR order, with
    the +20 annotated-junction bit masked off. None if the tags are absent
    (e.g. not a STAR BAM); [] if the read has no junctions."""
    try:
        jI = read.get_tag("jI")
        jM = read.get_tag("jM")
    except KeyError:
        return None
    if len(jI) < 2 or jI[0] == -1:
        return []
    out = []
    for i in range(0, len(jI), 2):
        start, end = int(jI[i]), int(jI[i + 1])
        motif_raw = int(jM[i // 2])
        motif = motif_raw % 20 if motif_raw >= 0 else -1
        out.append((start, end, motif))
    return out


def _ref_motif(ref: pysam.FastaFile, chrom: str, op_start0: int, op_end0: int) -> int:
    donor = ref.fetch(chrom, op_start0, op_start0 + 2).upper()
    acceptor = ref.fetch(chrom, op_end0 - 2, op_end0).upper()
    return _DINUC_MOTIF.get((donor, acceptor), 0)


def extract_condition(
    bam: pysam.AlignmentFile,
    region: tuple,
    ref: Optional[pysam.FastaFile] = None,
    filters: ReadFilters = None,
    use_tags: bool = True,
) -> ConditionRecord:
    """Single-condition pass: base-resolution coverage/deletion arrays plus a
    junction table keyed by (start, end). One `bam.fetch` pass, mirroring the
    difference-array trick in `plot2s.get_coords_from_bam` (op boundaries
    scattered as +1/-1, recovered with cumsum) but M/=/X, D and N are kept in
    three separate arrays instead of merging D into "coverage"."""
    if filters is None:
        filters = ReadFilters()

    chrom, start, end = region
    n = end - start

    cov_diff = np.zeros(n + 1, dtype=np.int64)
    del_diff = np.zeros(n + 1, dtype=np.int64)
    skip_diff = np.zeros(n + 1, dtype=np.int64)

    junctions: dict = {}
    n_reads_total = 0
    n_reads_kept = 0

    for read in bam.fetch(chrom, start, end):
        cigartuples = read.cigartuples
        if cigartuples is None or read.is_unmapped:
            continue
        n_reads_total += 1

        try:
            nh = int(read.get_tag("NH"))
        except KeyError:
            nh = 1

        if not filters.keep_read(read, nh):
            continue
        n_reads_kept += 1

        tag_junctions = _tag_junctions(read) if use_tags else None

        pos = read.reference_start
        n_idx = 0
        for i, (op, oplen) in enumerate(cigartuples):
            op_start, op_end = pos, pos + oplen

            if op in _COV_OPS or op == _DEL_OP or op == _SKIP_OP:
                s, e = max(op_start - start, 0), min(op_end - start, n)
                if e > s:
                    diff = (
                        skip_diff
                        if op == _SKIP_OP
                        else (del_diff if op == _DEL_OP else cov_diff)
                    )
                    diff[s] += 1
                    diff[e] -= 1

            if op == _SKIP_OP:
                anchor_left = (
                    cigartuples[i - 1][1]
                    if i > 0 and cigartuples[i - 1][0] in _COV_OPS
                    else 0
                )
                anchor_right = (
                    cigartuples[i + 1][1]
                    if i + 1 < len(cigartuples) and cigartuples[i + 1][0] in _COV_OPS
                    else 0
                )
                anchor = min(anchor_left, anchor_right)

                start_1b, end_1b = op_start + 1, op_end
                if tag_junctions is not None and n_idx < len(tag_junctions):
                    start_1b, end_1b, motif = tag_junctions[n_idx]
                elif ref is not None:
                    motif = _ref_motif(ref, chrom, op_start, op_end)
                else:
                    motif = -1
                n_idx += 1

                if anchor >= filters.min_anchor:
                    key = (start_1b, end_1b)
                    rec = junctions.setdefault(
                        key,
                        {"count": 0, "nh_weighted_count": 0.0, "motif": motif, "min_anchor": anchor},
                    )
                    rec["count"] += 1
                    rec["nh_weighted_count"] += 1.0 / nh
                    rec["min_anchor"] = min(rec["min_anchor"], anchor)
                    if rec["motif"] == -1 and motif != -1:
                        rec["motif"] = motif

            if op in _REF_CONSUMING_OPS:
                pos = op_end

    covs = np.cumsum(cov_diff[:n]) if n else np.array([], dtype=np.int64)
    delcovs = np.cumsum(del_diff[:n]) if n else np.array([], dtype=np.int64)
    skips = np.cumsum(skip_diff[:n]) if n else np.array([], dtype=np.int64)

    junction_arr = _junctions_to_array(junctions)

    stats = {
        "n_reads_total": n_reads_total,
        "n_reads_kept": n_reads_kept,
        "cov_mean": float(covs.mean()) if n else 0.0,
        "cov_max": int(covs.max()) if n else 0,
        "cov_median": float(np.median(covs)) if n else 0.0,
        "skip_max": int(skips.max()) if n else 0,
        "n_junctions": len(junctions),
    }

    return ConditionRecord(
        cov=covs.astype(np.uint32),
        delcov=delcovs.astype(np.uint32),
        junctions=junction_arr,
        stats=stats,
    )


def _junctions_to_array(junctions: dict) -> np.ndarray:
    arr = np.zeros(len(junctions), dtype=JUNCTION_DTYPE)
    for i, ((start, end), rec) in enumerate(sorted(junctions.items())):
        strand = _MOTIF_STRAND.get(rec["motif"], "?")
        arr[i] = (
            start,
            end,
            rec["count"],
            rec["nh_weighted_count"],
            rec["motif"],
            strand.encode(),
            rec["min_anchor"],
        )
    return arr


def strand_vote(junctions_1: np.ndarray, junctions_2: np.ndarray) -> str:
    """NH-weighted majority vote across both conditions' junctions. This is
    the GTF-free replacement for annotation-derived strand."""
    votes = {"+": 0.0, "-": 0.0}
    for arr in (junctions_1, junctions_2):
        for row in arr:
            s = row["strand"].decode()
            if s in votes:
                votes[s] += float(row["nh_weighted_count"])
    if votes["+"] == 0.0 and votes["-"] == 0.0:
        return "?"
    return "+" if votes["+"] >= votes["-"] else "-"


def extract_event(
    region: tuple,
    bam1_path: str,
    bam2_path: str,
    ref_path: Optional[str] = None,
    filters: ReadFilters = None,
    use_tags: bool = True,
) -> dict:
    ref = pysam.FastaFile(ref_path) if ref_path else None
    try:
        with pysam.AlignmentFile(bam1_path, "rb") as bam1, pysam.AlignmentFile(
            bam2_path, "rb"
        ) as bam2:
            c1 = extract_condition(bam1, region, ref, filters, use_tags)
            c2 = extract_condition(bam2, region, ref, filters, use_tags)
    finally:
        if ref is not None:
            ref.close()

    chrom, start, end = region
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "strand": strand_vote(c1.junctions, c2.junctions),
        "cov_1": c1.cov,
        "delcov_1": c1.delcov,
        "junctions_1": c1.junctions,
        "stats_1": c1.stats,
        "cov_2": c2.cov,
        "delcov_2": c2.delcov,
        "junctions_2": c2.junctions,
        "stats_2": c2.stats,
    }


def save_record(record: dict, path: str) -> None:
    np.savez_compressed(
        path,
        chrom=record["chrom"],
        start=record["start"],
        end=record["end"],
        strand=record["strand"],
        cov_1=record["cov_1"],
        delcov_1=record["delcov_1"],
        junctions_1=record["junctions_1"],
        cov_2=record["cov_2"],
        delcov_2=record["delcov_2"],
        junctions_2=record["junctions_2"],
        stats_1_json=json.dumps(record["stats_1"]),
        stats_2_json=json.dumps(record["stats_2"]),
    )


def load_record(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "chrom": str(data["chrom"]),
        "start": int(data["start"]),
        "end": int(data["end"]),
        "strand": str(data["strand"]),
        "cov_1": data["cov_1"],
        "delcov_1": data["delcov_1"],
        "junctions_1": data["junctions_1"],
        "cov_2": data["cov_2"],
        "delcov_2": data["delcov_2"],
        "junctions_2": data["junctions_2"],
        "stats_1": json.loads(str(data["stats_1_json"])),
        "stats_2": json.loads(str(data["stats_2_json"])),
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract junction set + base-resolution coverage per event "
        "(dsignal Phase 1). Never reads a GTF."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="region,bam1,bam2 CSV, no header (same format dsEnsemble.py takes)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--ref",
        default=None,
        help="reference FASTA (indexed with .fai) for CIGAR-walk motif lookup; "
        "used as a fallback when jI/jM tags are absent, and by --no-tags",
    )
    ap.add_argument(
        "--no-tags",
        action="store_true",
        help="ignore jI/jM tags, always derive junctions from the CIGAR walk "
        "(+ --ref for motif); for cross-checking against the tag path",
    )
    ap.add_argument("--min-anchor", type=int, default=5)
    ap.add_argument("--max-nh", type=int, default=10)
    ap.add_argument("--min-mapq", type=int, default=0)
    return ap


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    makedirs(args.out_dir, exist_ok=True)
    filters = ReadFilters(
        min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq
    )

    with open(args.csv) as f:
        rows = [r for r in csv.reader(f) if r]

    for ix, row in enumerate(rows):
        region_str, bam1_path, bam2_path = row[0], row[1], row[2]
        region = parse_region(region_str)
        record = extract_event(
            region,
            bam1_path,
            bam2_path,
            ref_path=args.ref,
            filters=filters,
            use_tags=not args.no_tags,
        )
        safe_name = region_str.replace(":", "_").replace("-", "_")
        out_path = pjoin(args.out_dir, f"{ix:05d}.{safe_name}.npz")
        save_record(record, out_path)
        print(
            f"[{ix}] {region_str}: strand={record['strand']} "
            f"junctions={record['stats_1']['n_junctions']}/{record['stats_2']['n_junctions']} "
            f"reads_kept={record['stats_1']['n_reads_kept']}/{record['stats_2']['n_reads_kept']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
