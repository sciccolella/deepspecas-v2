"""Final test-spend step 2: assert no candidate's training pool touches a
benchmark_new test gene (plans/signal-native-v2-final-test-spend.md).

A test row's genes are the union of (a) every GTF gene whose name equals the
curated ``gene_symbol`` and (b) every GTF gene whose span overlaps the region
on the same chromosome (strand-agnostic), so a symbol that fails to map cannot
hide a hit. Training genes are the ENSG prefix of each label ``event_id``.
Also asserts no ``ptbp1`` in any kind/event_id of any pool.

    python -m dsignal.check_test_eligibility
"""
import csv
import gzip
import re
import sys
from collections import defaultdict

GTF = "/datasets/deepSpecas-v2/ref/Homo_sapiens.GRCh38.99.gtf.gz"
E = "sim/experiment/"
POOLS = {
    "e4_union2 (n12480 + nn15)": [E + "labels_union2.n12480.tsv", E + "labels_union2.n12480.nn15.tsv"],
    "stage20 arm A (union2_real3)": [E + "labels_union2_real3.tsv"],
    "SSL both (sim n12480 + real_ssl_pool)": [E + "labels_union2.n12480.tsv", E + "real_ssl_pool/labels.tsv"],
}


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    genes, by_name = [], defaultdict(set)
    with gzip.open(GTF, "rt") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            c = ln.split("\t")
            if c[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', c[8]).group(1)
            m = re.search(r'gene_name "([^"]+)"', c[8])
            genes.append((c[0], int(c[3]), int(c[4]), gid))
            if m:
                by_name[m.group(1)].add(gid)
    # split.tsv buckets by final truth, curation.tsv by source csv: key by region
    syms = defaultdict(set)
    for r in rows("benchmark_new/curation.tsv"):
        syms[r["region"]].add(r["gene_symbol"])
    test = [r for r in rows("benchmark_new/split.tsv") if r["half"] == "test"]
    print(f"{len(test)} test rows")
    test_genes, unmapped = defaultdict(set), 0
    for r in test:
        ch, rest = r["region"].split(":")
        s, e = map(int, rest.split("-"))
        g = set().union(*(by_name.get(x, set()) for x in syms[r["region"]]))
        g |= {gid for c, a, b, gid in genes if c == ch and a <= e and b >= s}
        if not g:
            unmapped += 1
        for x in g:
            test_genes[x].add(r["region"])
    print(f"{len(test_genes)} distinct test genes; {unmapped} rows mapped to no gene")
    ok = True
    for name, files in POOLS.items():
        pool, ptbp1 = set(), 0
        for p in files:
            for r in rows(p):
                pool.add(r["event_id"].split(";")[0])
                ptbp1 += "ptbp1" in (r["kind"] + r["event_id"] + r["ix"]).lower()
        hit = pool & set(test_genes)
        print(f"{name}: {len(pool)} train genes, overlap with test genes = {len(hit)}, ptbp1 rows = {ptbp1}")
        if hit or ptbp1:
            ok = False
            print("  INELIGIBLE:", sorted(hit)[:20])
    print("literal criterion (all training genes):", "ELIGIBLE" if ok else "FAIL")
    # Deviation approved 2026-10-05 (plans/signal-native-v2-final-test-spend.md,
    # Results): only real-derived rows (kind startswith "real") and ptbp1 count.
    ok = True
    for name, files in POOLS.items():
        real = {r["event_id"].split(";")[0] for p in files for r in rows(p) if r["kind"].startswith("real")}
        hit = real & set(test_genes)
        print(f"{name}: real genes {len(real)}, overlap with test genes = {len(hit)}")
        ok &= not hit
    print("real-only criterion:", "ELIGIBLE" if ok else "some candidates flagged (see above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
