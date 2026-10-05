"""Recompute every headline number in RESULTS.md from the committed per-event
TSVs. Reads files only -- nothing is scored and no BAM or checkpoint is touched.

    python scripts/check_results.py
"""
import csv
from collections import Counter
from math import comb

R = "runs/"


def load(path, split=None, half=None):
    rows = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        key = (r["csv"], r["region"])
        if half and split.get(key) != half:
            continue
        rows[key] = r
    return rows


def split_of(path):
    return {(r["csv"], r["region"]): r["half"] for r in csv.DictReader(open(path), delimiter="\t")}


def mcnemar(a, b):
    """exact two-sided; returns (b_only_correct, a_only_correct, p) -- a, b dicts of key->0/1"""
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return only_a, only_b, p


def acc(rows, col="correct_collapsed"):
    ok = {k: int(r[col]) for k, r in rows.items()}
    return ok, sum(ok.values()), len(ok)


def show(name, rows, col="correct_collapsed"):
    ok, k, n = acc(rows, col)
    print(f"  {name:<34} {k}/{n} = {100 * k / n:.1f}%")
    return ok


sp_b, sp_n = split_of("benchmark/split.tsv"), split_of("benchmark_new/split.tsv")

print("benchmark/ (140 curated events, collapsed A3+A5->A)")
for half in ("dev", "test"):
    print(f" {half}:")
    e3 = show("E3 (16 members)", load(R + f"benchmark_events_e3_{half}.tsv"))
    st = show("single model (stack)", load(R + f"benchmark_events_stack_{half}.tsv"))
    v1 = show("v1 six-net ensemble", load(R + "benchmark_events_v1_ensemble.tsv", sp_b, half))
    v1s = show("v1 single squishstack", load(R + "benchmark_events_v1_squishstack.tsv", sp_b, half))
    print("  E3 vs v1 ensemble   v2-only %d / v1-only %d, p=%.3f" % mcnemar(e3, v1))

print("\nbenchmark_new test (243 events)")
e4_rows = load(R + "bnew_test_e4_union2.tsv")
v1_rows = load(R + "bnew_test_v1_ensemble.tsv", sp_n, "test")
e4 = show("e4_union2, collapsed", e4_rows)
v1 = show("v1 six-net, collapsed", v1_rows)
show("e4_union2, 5-class", e4_rows, "correct")
print("  e4 vs v1            v2-only %d / v1-only %d, p=%.2g" % mcnemar(e4, v1))
nn = [k for k, r in e4_rows.items() if r["truth"] != "NN"]
print("  excluding NN truth: e4 %d/%d, v1 %d/%d" % (sum(e4[k] for k in nn), len(nn), sum(v1[k] for k in nn), len(nn)))
print("  per truth class (collapsed): class n  v1  e4  v2-only/v1-only")
for c in ("A3", "A5", "CSE", "NN", "RI"):
    ks = [k for k, r in e4_rows.items() if r["truth"] == c]
    a = {k: e4[k] for k in ks}
    b = {k: v1[k] for k in ks}
    print(f"    {c:<4}{len(ks):>4} {sum(b.values()):>4} {sum(a.values()):>4}   %d/%d p=%.3g" % mcnemar(a, b))
cm = Counter((r["truth"], r["pred"]) for r in e4_rows.values())
print("  NN truth predicted as:", {p: cm[("NN", p)] for p in ("A3", "A5", "CSE", "NN", "RI")})
print("  v2 errors (collapsed): %d, of which NN truth: %d" % (
    sum(1 - e4[k] for k in e4), sum(1 - e4[k] for k in e4 if e4_rows[k]["truth"] == "NN")))
dev_e4 = show("e4_union2 dev (n=242)", load(R + "bnew_dev_e4_union2.tsv"))
dev_v1 = show("v1 six-net dev (n=242)", load(R + "bnew_dev_v1_ensemble.tsv"))
