"""Reorder a FASTA so its records match a transcript-id list **positionally**.

``polyester::simulate_experiment_countmat`` matches the count matrix to the
FASTA by *row position*, not by name (it only checks
``nrow(readmat) == length(transcripts)``), so a FASTA assembled from three
sources -- gffread's spliced transcripts, ``build_premrna_fasta.py``'s
``PREMRNA_*`` and ``novel_junctions.py``'s ``NOVEL_*`` -- has to be put back
into ``transcript_ids.txt`` order before it is handed over. Getting this wrong
silently assigns every transcript someone else's expression.

Usage:
    python sim/order_fasta.py <ids.txt> <in.fa> <out.fa>
"""

import sys


def read_fasta(path):
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, chunks
                # gffread appends attributes after the id; keep the first token
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, chunks


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    ids_path, in_path, out_path = argv[:3]

    with open(ids_path) as fh:
        ids = [line.strip() for line in fh if line.strip()]

    records = {}
    for name, chunks in read_fasta(in_path):
        if name in records:
            raise SystemExit(f"duplicate FASTA record: {name}")
        records[name] = chunks

    missing = [i for i in ids if i not in records]
    if missing:
        raise SystemExit(
            f"{len(missing)} ids in {ids_path} have no FASTA record "
            f"(first: {missing[:5]})"
        )
    extra = len(records) - len(ids)

    with open(out_path, "w") as out:
        for i in ids:
            out.write(f">{i}\n")
            out.writelines(records[i])

    print(
        f"wrote {out_path}: {len(ids)} records in {ids_path} order "
        f"({extra} unused records dropped)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
