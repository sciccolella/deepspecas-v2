"""Inject small indels into polyester's simulated reads (Stage 19.2).

polyester writes substitution errors only (``error_model = "illumina5"``), so
every simulated BAM has ``delcov`` ~ 0 and deletion coverage became a pure
real-vs-sim tell (Stage 18.4). This is a seeded per-sample post-processor
over the FASTA pairs polyester wrote -- base and tail reads alike -- run by
``sim/run_batch.sh`` before STAR when ``INDEL_RATE`` > 0. Off by default, so
the old behaviour is byte-identical (the step is skipped, not run as a no-op).

Per read base, a deletion starts with probability ``--del-rate`` and an
insertion with ``--ins-rate``; lengths are Geometric(p = 0.6) capped at 5
(``sim.fit_tail_model.indel_len_pmf``). The rates come from
``sim/tail_model.json`` (``indel.del_rate_per_base``: the fit cohorts' pooled
sum(delcov) / sum(cov) divided by the mean length; insertion rate set equal,
an assumption since coverage arrays cannot see insertions).

Known, recorded inaccuracies:

- a deletion *shortens* the read by k (the read cannot be extended without the
  source fragment); an insertion is trimmed back to the original length, as a
  fixed-cycle sequencer would;
- PCR duplicates (separate reads in polyester's output) receive independent
  indels;
- mates are processed independently (they are in separate files).

Event positions are drawn as geometric gaps over the concatenated base stream
of a file, so the cost is one pass over the file and ~no RNG work per read.

Usage::

    python -m sim.inject_indels --reads-dir reads/ --seed 12 \\
        --model sim/tail_model.json [--del-rate R --ins-rate R]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from sim.fit_tail_model import INDEL_MAX, INDEL_P, indel_len_pmf

STREAM = (19, 7)
BASES = np.array(list("ACGT"))


def read_fasta_records(path: str):
    name, chunks = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line.rstrip("\n"), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


class EventStream:
    """Geometric gaps between indel events over a concatenated base stream."""

    def __init__(self, rng, del_rate: float, ins_rate: float):
        self.rng = rng
        self.rate = del_rate + ins_rate
        self.p_del = del_rate / self.rate if self.rate > 0 else 0.0
        self.len_pmf = indel_len_pmf(INDEL_P, INDEL_MAX)
        self.next = self._gap() if self.rate > 0 else np.inf

    def _gap(self) -> int:
        return int(self.rng.geometric(self.rate)) - 1  # bases skipped before the next event

    def events_in(self, n: int) -> list:
        """Events among the next n bases: [(offset, is_del, length)]."""
        out, pos = [], 0
        while self.next < n - pos:
            pos += int(self.next)
            is_del = self.rng.random() < self.p_del
            k = 1 + int(self.rng.choice(len(self.len_pmf), p=self.len_pmf))
            out.append((pos, is_del, k))
            pos += 1
            self.next = self._gap()
        self.next -= n - pos
        return out


def mutate(seq: str, events: list, rng) -> str:
    if not events:
        return seq
    n = len(seq)
    out, cursor = [], 0
    for pos, is_del, k in events:
        if pos < cursor:
            continue  # inside a previous deletion
        out.append(seq[cursor:pos])
        if is_del:
            cursor = min(pos + k, n)
        else:
            out.append("".join(BASES[rng.integers(0, 4, size=k)]))
            cursor = pos
    out.append(seq[cursor:])
    new = "".join(out)
    return new[:n]  # insertions are trimmed back to the read length


def process_file(path_in: str, path_out: str, rng, del_rate: float, ins_rate: float) -> dict:
    ev = EventStream(rng, del_rate, ins_rate)
    n_reads = n_touched = n_del = n_ins = 0
    with open(path_out, "w") as out:
        for name, seq in read_fasta_records(path_in):
            events = ev.events_in(len(seq))
            if events:
                n_touched += 1
                n_del += sum(1 for e in events if e[1])
                n_ins += sum(1 for e in events if not e[1])
                seq = mutate(seq, events, rng)
            out.write(f"{name}\n{seq}\n")
            n_reads += 1
    return {"reads": n_reads, "touched": n_touched, "deletions": n_del, "insertions": n_ins}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads-dir", required=True, help="holds sample_0{1..6}_{1,2}.fasta; rewritten in place")
    ap.add_argument("--seed", type=int, required=True, help="the batch seed")
    ap.add_argument("--model", default="sim/tail_model.json")
    ap.add_argument("--del-rate", type=float, default=None, help="default: the model's indel.del_rate_per_base")
    ap.add_argument("--ins-rate", type=float, default=None, help="default: the model's indel.ins_rate_per_base")
    args = ap.parse_args(argv)
    with open(args.model) as f:
        ind = json.load(f)["indel"]
    del_rate = ind["del_rate_per_base"] if args.del_rate is None else args.del_rate
    ins_rate = ind["ins_rate_per_base"] if args.ins_rate is None else args.ins_rate
    for s in range(1, 7):
        for mate in (1, 2):
            path = os.path.join(args.reads_dir, f"sample_{s:02d}_{mate}.fasta")
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, *STREAM, s, mate]))
            tmp = path + ".indel.tmp"
            st = process_file(path, tmp, rng, del_rate, ins_rate)
            os.replace(tmp, path)
            print(f"{path}: {st['reads']} reads, {st['touched']} touched, {st['deletions']} del, "
                  f"{st['insertions']} ins (rates {del_rate:.3e}/{ins_rate:.3e})", file=sys.stderr)


if __name__ == "__main__":
    main()
