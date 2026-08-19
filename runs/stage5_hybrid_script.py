import csv, sys
sys.path.insert(0, ".")
from dsignal.benchmark_split import load_split
from dsignal.score_benchmark import CSV_LABEL, collapse

split = load_split()
CLASSES = ("A3", "A5", "CSE", "NN", "RI")

def load_v1_crossover_probs():
    out = {}
    for stem in ("A3", "A5", "ES", "IR"):
        with open(f"runs/v1_crossover/pred/{stem}.csv") as f:
            for row in csv.DictReader(f):
                probs = [float(row[c]) for c in CLASSES]
                out[(stem, row["region"])] = probs
    return out

def load_v2_probs(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            probs = [float(row[f"p_{c}"]) for c in CLASSES]
            out[(row["csv"], row["region"])] = probs
    return out

def load_truth_pred(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["csv"], row["region"])] = row["pred"]
    return out

v1_probs = load_v1_crossover_probs()
v2_probs = load_v2_probs("runs/benchmark_events_e3_dev.tsv")

keys = [k for k in v2_probs if split.get(k) == "dev"]
print(f"n={len(keys)}")

# hybrid: image-free E3 + v1 crossover net, mean_prob
n_correct = 0
for k in keys:
    stem, region = k
    truth = CSV_LABEL[stem]
    p1 = v1_probs.get(k)
    p2 = v2_probs[k]
    if p1 is None:
        print("missing v1 crossover pred for", k)
        continue
    hybrid = [(a + b) / 2 for a, b in zip(p1, p2)]
    pred = CLASSES[hybrid.index(max(hybrid))]
    if collapse(pred) == collapse(truth):
        n_correct += 1
print(f"image-free E3 + v1-crossover hybrid, dev: {n_correct}/{len(keys)} = {n_correct/len(keys):.1%} collapsed")

# oracle unions
v1_single_pred = load_truth_pred("runs/benchmark_events_v1_squishstack.tsv")
v1_ens_pred = load_truth_pred("runs/benchmark_events_v1_ensemble.tsv")
v2_pred = load_truth_pred("runs/benchmark_events_e3_dev.tsv")

for name, v1d in (("v1 squishstack single net", v1_single_pred), ("v1 six-net ensemble", v1_ens_pred)):
    n_oracle = 0
    for k in keys:
        stem, region = k
        truth = CSV_LABEL[stem]
        p_v2 = v2_pred.get(k)
        p_v1 = v1d.get(k)
        ok = (p_v2 is not None and collapse(p_v2) == collapse(truth)) or \
             (p_v1 is not None and collapse(p_v1) == collapse(truth))
        if ok:
            n_oracle += 1
    print(f"oracle union (image-free E3 OR {name}), dev: {n_oracle}/{len(keys)} = {n_oracle/len(keys):.1%} collapsed")
