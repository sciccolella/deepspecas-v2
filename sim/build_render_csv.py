"""Build the 4-column render CSV plot2s.py wants: region,bam1,bam2,label.

Row identity between extract.geomwin.csv, design.geomwin.tsv and
labels.geomwin.tsv is checked by rowcheck.py first -- these files are joined by
position, which is exactly the kind of silent misalignment this branch has been
bitten by before.
"""
import csv

out = open("sim/render_geomwin.csv", "w")
n = 0
for b in ["b1", "b2", "b3", "b4", "b5"]:
    ex = [l.strip() for l in open(f"sim/batches/{b}/extract.geomwin.csv") if l.strip()]
    lab = list(csv.DictReader(open(f"sim/batches/{b}/labels.geomwin.tsv"), delimiter="\t"))
    assert len(ex) == len(lab), (b, len(ex), len(lab))
    for row, lr in zip(ex, lab):
        out.write(f"{row},{lr['label']}\n")
        n += 1
out.close()
print(f"wrote sim/render_geomwin.csv, {n} events")
