"""Score Stage 19 Gate A locally from the pilot's returned files.

plans/signal-native-v2-stage19-junction-tail.md, "Pre-registered gates". The
gate text is frozen; this module only mechanizes it. Inputs are what comes
back from pasteur under ``--pilot-dir`` (``<tag>/{npz,labels.tsv,
psi_recovery.tsv,logs/}``, tags ``b12_p0``, ``b1_p0`` and the P1 tag).

- **A0** ``sim.npz_diff``: b12_p0 vs ``batches_union2/b12``, b1_p0 vs
  ``batches_geomwin/b1`` (the npz that went into ``npz_union2``), arrays and
  ``stats_*`` equal, labels byte-identical. A failure is a pipeline bug.
- **A1** ``sim.tail_stats`` on P1 against the frozen ranges (``GATE_A1``).
  Also printed, not gated: [min, max] over the fit cohorts' pos+nn, raw and
  width-matched to P1 (the 19.0/19.2 findings the user rules on).
- **A2** ``dsignal.domain_probe`` built twice (real pool vs b12-old, vs P1),
  class-matched; each of the four tells' |MW-AUC - 0.5| must at least halve.
  Evaluated against the frozen b12-old column of the gate table (verbatim)
  and, for reference, against the b12-old build measured here.
- **A3** tail-only LightGBM balanced accuracy on P1 <= 0.30.
- **A4** ``dsignal.verify`` (its pasteur log: exit 0 and both parity lines N/N),
  ``sim.psi_recovery_check`` (event PSI unaffected: every label's mean
  |error| within 0.01 of b12-old -- threshold fixed 2026-09-23, before any
  pilot data, since the gate names the check but no number),
  ``sim.width_leak_check`` (reported; windows are identical by A0), and
  ``dsignal.verify_features`` on the P1 npz.

Usage::

    python -m sim.stage19_gate_a --pilot-dir sim/experiment/batches_stage19 --p1 b12_p1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from os.path import join as pjoin

import numpy as np

E = "sim/experiment"
REAL_POOL = [f"{E}/real_pos_ptbp1", f"{E}/real_nn_ptbp1", f"{E}/real_pos_aqr", f"{E}/real_nn_aqr",
             "runs/real_dev_npz", "runs/real_tra2_npz"]
PROBE_CKPT = "runs/scale_union2_n12480/fold0.pt"
# Gate A2's table, verbatim (commit 0e11fe204)
A2_FROZEN = {"mean__log1p_count_2": 0.026, "log1p_n_junctions": 0.853,
             "mean__log1p_min_anchor": 0.974, "wmean__present_2": 0.062}
PSI_TOL = 0.01


def run(cmd: list, log: str | None = None) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)
    out = p.stdout + p.stderr
    if log:
        with open(log, "w") as f:
            f.write(out)
    return p.returncode, out


def a0(pilot: str, out_dir: str) -> dict:
    res = {}
    for tag, old in (("b12_p0", f"{E}/batches_union2/b12"), ("b1_p0", f"{E}/batches_geomwin/b1")):
        rc, out = run(["sim.npz_diff", f"{old}/npz", pjoin(pilot, tag, "npz"),
                       "--labels", f"{old}/labels.tsv", pjoin(pilot, tag, "labels.tsv")],
                      pjoin(out_dir, f"a0_{tag}.log"))
        res[tag] = {"pass": rc == 0, "summary": out.strip().splitlines()[0] if out.strip() else ""}
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    return res


def a1_a3(pilot: str, p1: str, out_dir: str) -> dict:
    js = pjoin(out_dir, "tail_stats.json")
    fit = [f"{c}={E}/real_pos_{c}/npz+{E}/real_nn_{c}/npz" for c in ("aqr", "prpf4", "u2af2")]
    run(["sim.tail_stats", "--max-n", "0", "--set", f"p1={pilot}/{p1}/npz",
         "--set", f"b12_old={E}/batches_union2/b12/npz", *sum([["--set", s] for s in fit], []),
         "--ranges-from", "aqr", "prpf4", "u2af2", "--width-match", "p1", "--gate-a1", "p1", "--a3", "p1",
         "--json-out", js], pjoin(out_dir, "tail_stats.log"))
    with open(js) as f:
        t = json.load(f)
    g = t["gate_a1"]["p1"]
    return {"A1": {"pass": g["PASS"], **{k: v for k, v in g.items() if k != "PASS"}},
            "A1_context_fit_cohort_ranges": t["ranges_from"]["ranges"],
            "A1_context_width_matched_ranges": t["ranges_from"]["width_matched"]["ranges"],
            "A3": t["sets"]["p1"]["a3"]}


def a2(pilot: str, p1: str, out_dir: str) -> dict:
    labels_old = pjoin(out_dir, "labels_b12_old.tsv")
    with open(f"{E}/batches_union2/b12/labels.tsv") as f, open(labels_old, "w") as g:
        g.write(f.read())
    feats = {}
    for name, npz, labels in (("old", f"{E}/batches_union2/b12/npz", labels_old),
                              ("new", pjoin(pilot, p1, "npz"), pjoin(pilot, p1, "labels.tsv"))):
        d = pjoin(out_dir, f"probe_{name}")
        run(["dsignal.domain_probe", "build", "--ckpt", PROBE_CKPT, "--real", *REAL_POOL,
             "--sim-npz", npz, "--sim-labels", labels, "--out", pjoin(d, "table.npz")], d + ".build.log")
        run(["dsignal.domain_probe", "real-vs-sim", "--table", pjoin(d, "table.npz"),
             "--removal-k", "0", "--out-dir", d], d + ".rvs.log")
        with open(pjoin(d, "features.tsv")) as f:
            feats[name] = {r["feature"]: float(r["mw_auc"]) for r in csv.DictReader(f, delimiter="\t")}
        with open(pjoin(d, "summary.json")) as f:
            feats[name + "_pooled_auc"] = json.load(f).get("pooled_auc")
    res = {"tells": {}, "pooled_auc": {"old": feats["old_pooled_auc"], "new": feats["new_pooled_auc"]}}
    ok_frozen = ok_measured = True
    for k, frozen in A2_FROZEN.items():
        new, old = feats["new"][k], feats["old"][k]
        pf = abs(new - 0.5) <= abs(frozen - 0.5) / 2
        pm = abs(new - 0.5) <= abs(old - 0.5) / 2
        ok_frozen &= pf
        ok_measured &= pm
        res["tells"][k] = {"frozen_old": frozen, "measured_old": old, "new": new,
                           "pass_vs_frozen": pf, "pass_vs_measured": pm}
    res["pass"] = ok_frozen
    res["pass_vs_measured_old"] = ok_measured
    return res


def _psi_by_label(path: str) -> dict:
    by = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k in ("err1", "err2"):
                try:
                    by.setdefault(r["label"], []).append(abs(float(r[k])))
                except ValueError:
                    pass
    return {k: float(np.mean(v)) for k, v in by.items()}


def a4(pilot: str, p1: str, out_dir: str) -> dict:
    res = {}
    vlog = pjoin(pilot, p1, "logs", "verify.log")
    text = open(vlog).read() if os.path.exists(vlog) else ""
    parity = [l for l in text.splitlines() if l.startswith(("[coverage parity]", "[junction parity]"))]
    full = all(l.split("]")[1].split()[0].split("/")[0] == l.split("]")[1].split()[0].split("/")[1]
               for l in parity) and len(parity) == 2
    res["verify"] = {"pass": bool(text) and "verify exit 0" in text and full, "parity": parity,
                     "log": vlog if text else "missing -- run `stage19_pilot.sh gates <tag>` on pasteur"}
    old = _psi_by_label(f"{E}/batches_union2/b12/psi_recovery.tsv")
    new = _psi_by_label(pjoin(pilot, p1, "psi_recovery.tsv"))
    res["psi_recovery"] = {"old": old, "new": new, "tol": PSI_TOL,
                           "pass": all(abs(new[k] - old[k]) <= PSI_TOL for k in old)}
    rc, out = run(["sim.width_leak_check", "--npz-dir", pjoin(pilot, p1, "npz"),
                   "--labels", pjoin(pilot, p1, "labels.tsv")], pjoin(out_dir, "width_leak.log"))
    res["width_leak"] = {"reported": [l for l in out.splitlines() if "accuracy" in l][-1:] or out[-200:]}
    rc, out = run(["dsignal.verify_features", "--npz-dir", pjoin(pilot, p1, "npz")],
                  pjoin(out_dir, "verify_features.log"))
    res["verify_features"] = {"pass": rc == 0 and "ALL CHECKS PASSED" in out}
    res["pass"] = all(res[k]["pass"] for k in ("verify", "psi_recovery", "verify_features"))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", default=f"{E}/batches_stage19")
    ap.add_argument("--p1", default="b12_p1", help="the P1 (or P1') tag to score A1-A4 on")
    ap.add_argument("--out-dir", default="runs/stage19/gate_a")
    ap.add_argument("--skip-a0", action="store_true")
    args = ap.parse_args(argv)
    out = pjoin(args.out_dir, args.p1)
    os.makedirs(out, exist_ok=True)
    result = {"p1": args.p1}
    if not args.skip_a0:
        result["A0"] = a0(args.pilot_dir, out)
    result.update(a1_a3(args.pilot_dir, args.p1, out))
    result["A2"] = a2(args.pilot_dir, args.p1, out)
    result["A4"] = a4(args.pilot_dir, args.p1, out)
    result["A3"]["pass"] = bool(result["A3"]["pass"])
    rules = [r for r in ("A0", "A1", "A2", "A3", "A4") if r in result]
    result["GATE_A"] = "PASS" if all(result[r]["pass"] for r in rules) else "FAIL"
    with open(pjoin(out, "gate_a.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    for r in rules:
        print(f"{r}: {'PASS' if result[r]['pass'] else 'FAIL'}")
    for k, v in result["A2"]["tells"].items():
        print(f"   A2 {k:>24}: old {v['frozen_old']:.3f} (measured {v['measured_old']:.3f}) -> new {v['new']:.3f}"
              f"   {'ok' if v['pass_vs_frozen'] else 'NOT halved'}")
    for k, v in result["A1"].items():
        if isinstance(v, dict):
            print(f"   A1 {k:>20}: {v['value']:.3f} in {v['range']} -> {'ok' if v['pass'] else 'OUT'}")
    print(f"   A3 balanced accuracy {result['A3']['balanced_accuracy']:.3f}")
    print(f"GATE A ({args.p1}): {result['GATE_A']}  -> {pjoin(out, 'gate_a.json')}")


if __name__ == "__main__":
    main()
