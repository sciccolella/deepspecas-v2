import re
import sys

ids_path, gtf_path, out_path = sys.argv[1:4]
ids = set(l.strip() for l in open(ids_path))
with open(gtf_path) as f, open(out_path, "w") as out:
    for line in f:
        if line.startswith("#"):
            continue
        m = re.search(r'transcript_id "([^"]+)"', line)
        if m and m.group(1) in ids:
            out.write(line)
