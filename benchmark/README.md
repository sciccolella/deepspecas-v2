# Manually curated benchmark

We considered the dataset provided in Best, A., James, K., Dalgliesh, C. et al. Human Tra2 proteins jointly control a CHEK1 splicing switch among alternative and constitutive target exons. Nat Commun 5, 4760 (2014). https://doi.org/10.1038/ncomms5760
[SRA BioProject ID: `PRJNA255099`]

To build a trustworthy truth set, we relied on two state-of-the-art tools for the differential quantification of AS events, namely [rMATS](https://doi.org/10.1073/pnas.1419161111) and [SUPPA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1417-1). 

We considered the alternative splicing events reported by both tools and 
we compare the two sets of significant events (i.e., those events with reported p-value < 0.05). 
Any event reported as significant by both tools is then considered a potential true candidate and added to the truth set.

A total of 62 alternative acceptor (A3), 54 alternative donor (A5), and 41 intron retention events resulted from this analysis.
(The consensus count for exon skipping is not recorded here; only the post-curation count below is.)
Although this truth set is assembled using the consensus of two accurate tools
we manually inspected their classification and found several regions
with debatable callings.

We thus provide the lists of manually curated events in the files:
- [Exon Skipping](ES.txt) 65 events
- [Alternative acceptor](A3.txt) 28 events
- [Alternative donor](A5.txt) 30 events
- [Intron retention](IR.txt) 17 events

## Files the tools actually read

The `.txt` files above are the human-readable lists. Every tool in this repository
consumes the matching **CSV** files instead — `A3.csv`, `A5.csv`, `ES.csv`, `IR.csv`,
140 events in total — which are headerless and formatted as:

```
region,BAM_path_condition1,BAM_path_condition2
```

with paths relative to the repository root (`benchmark/c1.bam`, `benchmark/c2.bam`). The
CSV basename is the label: `ES` means SUPPA's `SE` (`CSE` in v2's class list) and `IR`
means `RI`.

`events.bed` holds the same 140 regions in BED form. **Its intervals are identical to the
region column of the four CSVs** — verified by direct comparison, not by assumption. That
identity is what makes it possible to ask what the curated window *is* relative to the
underlying event span, which is how the v2 extraction-window geometry was derived
(`python -m dsignal.audit_noise padding`).

Note that the benchmark contains **no negative (NN) events** — it is a truth set of four
positive classes only. Any NN prediction against it is a false positive by construction,
and NN recall cannot be measured here.

## The BAM files, and one measurement caveat

We provide the bam files of the aligned reads from `PRJNA255099`, in which only the reads overlapping the regions of interest are retained,
divided in two conditions:
- [Condition 1](c1.bam) comprised of samples `SRR1513329`, `SRR1513330`, `SRR1513331`
- [Condition 2](c2.bam) comprised of samples `SRR1513332`, `SRR1513333`, `SRR1513334`

**Consequence of that subsetting:** because the BAMs are effectively
`samtools view -L events.bed`, reads that fall genuinely *outside* the windows are
absent. Any quantity measured on these BAMs that depends on off-window signal —
background fraction, antisense fraction — is therefore a **lower bound** on the real
value, and comparisons against simulated data along those axes are directional only.

Because the three SRA runs are merged per condition, read names carry a library prefix
(`SRR1513329.6348221`). Anything scoping per-library statistics (e.g. duplicate rate)
must split on that prefix — and must not assume it exists, since simulated reads have no
such prefix.

Lastly in `igv_images/[ES|A3|A5|IR]` we provide the `igv` screenshot of the curated images.