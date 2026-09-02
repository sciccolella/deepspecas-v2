#!/usr/bin/env python
"""Stage 9 / Stage 0.1: ingest the manually-curated review TSVs
(``benchmark_new_uncurated/reviews/review_{A3,A5,ES,IR}.tsv``) into one flat
``curation.tsv``, one row per reviewed event, **all 573 retained**.

Columns::

    csv_source, region, gene_symbol, rmats_pvalue, rmats_incleveldiff,
    suppa_dpsi, suppa_pvalue, v1_pred, v2_pred, review_raw, truth, ambiguous,
    notes_raw, trash, borderline, low_signal, complex, similar_signal

Normalisation rules
--------------------

``Review`` -> ``truth`` / ``ambiguous``: a row is ambiguous when ``Review``
contains ``/`` or ``+`` (multi-label, 87 rows) or is blank (1 row, in
``review_ES.tsv``). ``truth`` is written only for the 485 unambiguous rows.

``Notes`` -> flag columns: upper-cased and split on ``,``, with ``.``
normalised to ``,`` first -- ``review_ES.tsv`` has one row noted
``BORDERLINE.TRASH`` that would otherwise parse as a single unrecognised
token. Each flag is then set by substring match against the resulting tokens:
``TRASH``, ``BORDERLINE``, ``LOW SIGNAL``, ``COMPLEX``, and ``SIMILAR
SIGNAL``/``SAME SIGNAL`` (folded into one ``similar_signal`` column).
Unrecognised tokens (``SUS REGION``, ``NO SIGNAL``, ``VERY FEW READS``, a
stray ``REGION chr2:...``) are not dropped -- ``notes_raw`` carries the
original, unmodified ``Notes`` text regardless of what was recognised.

``trash`` is three-valued: ``1`` / ``0`` / ``NA``, not derived from
``Notes`` alone. The user curated ``review_A5.tsv`` first and only started
using the ``TRASH`` note partway through it: its first ``TRASH`` row is row
18 of 52 (index 17, 0-based), and rows 1-17 already carry other flags
(``BORDERLINE``, ``COMPLEX``, ``SIMILAR SIGNAL``) but never ``TRASH`` --
i.e. the marker itself hadn't been adopted yet, not that these rows were
skipped. Treating a blank ``TRASH`` token there as a real negative would be
wrong, so those 17 rows get ``trash = NA`` (only the ``trash`` column --
the other four flags are computed normally, since those markers were
already in use). This prefix length is a **written-down constant**, not
re-derived by scanning for the first ``TRASH`` row: that heuristic would
silently mislabel a file whose genuinely clean prefix happens to be long.
The other three files annotate throughout, with clean rows before their
first ``TRASH`` row, so their blanks are real negatives:
``review_A3.tsv`` first ``TRASH`` at row 33 of 71 (index 32),
``review_ES.tsv`` at row 5 of 373 (index 4),
``review_IR.tsv`` at row 15 of 77 (index 14).
If the user later annotates the 17, flipping ``UNANNOTATED_PREFIX`` to ``{}``
is the whole change.

Usage::

    python benchmark_build/ingest_curation.py \\
        benchmark_new_uncurated/reviews benchmark_new_uncurated \\
        benchmark_new/curation.tsv
"""
from __future__ import annotations

import csv
import sys
from os.path import basename, splitext

CLASSES = ("A3", "A5", "CSE", "RI", "NN")

#: csv_source -> number of leading rows (0-based, exclusive) for which the
#: TRASH marker was not yet in use. See module docstring.
UNANNOTATED_PREFIX = {"A5": 17}

FIELDS = [
    "csv_source", "region", "gene_symbol", "rmats_pvalue", "rmats_incleveldiff",
    "suppa_dpsi", "suppa_pvalue", "v1_pred", "v2_pred", "review_raw", "truth",
    "ambiguous", "notes_raw", "trash", "borderline", "low_signal", "complex",
    "similar_signal",
]


def _notes_tokens(notes: str) -> list[str]:
    return [t.strip() for t in notes.upper().replace(".", ",").split(",") if t.strip()]


def _flags(tokens: list[str]) -> dict:
    def has(*keywords):
        return int(any(kw in tok for tok in tokens for kw in keywords))

    return {
        "trash": has("TRASH"),
        "borderline": has("BORDERLINE"),
        "low_signal": has("LOW SIGNAL"),
        "complex": has("COMPLEX"),
        "similar_signal": has("SIMILAR SIGNAL", "SAME SIGNAL"),
    }


def ingest(review_dir: str, csv_dir: str) -> list[dict]:
    rows_out = []
    seen_regions = {}
    for stem in ("A3", "A5", "ES", "IR"):
        review_path = f"{review_dir}/review_{stem}.tsv"
        with open(review_path, newline="") as f:
            review_rows = list(csv.DictReader(f, delimiter="\t"))

        candidate_path = f"{csv_dir}/{stem}.csv"
        with open(candidate_path, newline="") as f:
            candidate_regions = {line.split(",")[0] for line in f if line.strip()}

        prefix = UNANNOTATED_PREFIX.get(stem, 0)
        for i, r in enumerate(review_rows):
            region = r["region"]
            if region not in candidate_regions:
                raise AssertionError(f"{review_path}:{i}: region {region!r} not in {candidate_path}")
            if region in seen_regions:
                raise AssertionError(
                    f"region {region!r} appears in both {seen_regions[region]!r} and {review_path!r}"
                )
            seen_regions[region] = review_path

            review_raw = r["Review"].strip()
            ambiguous = int("/" in review_raw or "+" in review_raw or review_raw == "")
            truth = "" if ambiguous else review_raw
            if truth and truth not in CLASSES:
                raise AssertionError(f"{review_path}:{i}: unrecognised truth {truth!r}")

            notes_raw = r["Notes"]
            flags = _flags(_notes_tokens(notes_raw))
            if i < prefix:
                flags["trash"] = "NA"

            rows_out.append({
                "csv_source": stem,
                "region": region,
                "gene_symbol": r["gene_symbol"],
                "rmats_pvalue": r["rmats_pvalue"],
                "rmats_incleveldiff": r["rmats_incleveldiff"],
                "suppa_dpsi": r["suppa_dpsi"],
                "suppa_pvalue": r["suppa_pvalue"],
                "v1_pred": r["v1_pred"],
                "v2_pred": r["v2_pred"],
                "review_raw": review_raw,
                "truth": truth,
                "ambiguous": ambiguous,
                "notes_raw": notes_raw,
                **flags,
            })
    return rows_out


def main() -> None:
    review_dir, csv_dir, out_path = sys.argv[1:4]
    rows = ingest(review_dir, csv_dir)

    assert len(rows) == 573, f"expected 573 rows, got {len(rows)}"

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    trash_all = Counter(r["trash"] for r in rows)
    trash_unambig = Counter(r["trash"] for r in rows if not r["ambiguous"])
    n_ambig = sum(r["ambiguous"] for r in rows)
    print(f"wrote {out_path}: {len(rows)} rows, {n_ambig} ambiguous, {len(rows) - n_ambig} unambiguous")
    print(f"  trash tally (unambiguous): {dict(trash_unambig)}")
    print(f"  trash tally (all 573):     {dict(trash_all)}")


if __name__ == "__main__":
    main()
