#!/usr/bin/env python
"""Stage 8 / Track A2: build SUPPA2 transcript-TPM matrices from salmon quant.sf
files. SUPPA2's psiPerEvent takes one expression file per condition (rows =
transcript ID, columns = replicate TPM, tab-separated). The header row must
have exactly n_samples fields (no leading blank cell) -- lib.tools.ExpressionParser
sets min_fields = len(header_fields) + 1, so a header with a blank first cell
(n_samples + 1 fields) makes every data row look one field short.

Usage: make_tpm_matrix.py <salmon_dir> <out_dir>
  <salmon_dir> must contain cond1_rep1/, cond1_rep2/, cond2_rep1/, cond2_rep2/,
  each with a quant.sf (Name, Length, EffectiveLength, TPM, NumReads).
Writes <out_dir>/cond1_tpm.tsv and <out_dir>/cond2_tpm.tsv.
"""
import sys
import pandas as pd

SAMPLES = {
    "cond1": ["cond1_rep1", "cond1_rep2"],
    "cond2": ["cond2_rep1", "cond2_rep2"],
}


def main():
    salmon_dir, out_dir = sys.argv[1], sys.argv[2]
    for cond, reps in SAMPLES.items():
        cols = {}
        for rep in reps:
            df = pd.read_csv(f"{salmon_dir}/{rep}/quant.sf", sep="\t", usecols=["Name", "TPM"])
            cols[rep] = df.set_index("Name")["TPM"]
        mat = pd.DataFrame(cols)
        out_path = f"{out_dir}/{cond}_tpm.tsv"
        with open(out_path, "w") as f:
            f.write("\t".join(mat.columns) + "\n")
            mat.to_csv(f, sep="\t", header=False)
        print(f"wrote {out_path}: {mat.shape[0]} transcripts x {mat.shape[1]} replicates")


if __name__ == "__main__":
    main()
