#!/usr/bin/env Rscript
# Drive polyester's simulate_experiment_countmat from our own per-transcript
# per-replicate count matrix (sim/build_experiment.py), bypassing polyester's
# / ASimulatoR's own event generator entirely -- per plans/signal-native-v2.md
# Phase 2 step 4. This is layer-1 only (real STAR alignment); PCR-duplicate
# and adapter-contamination noise (layer 5, ASimulatoR extras pcr_rate /
# adapter_contamination) are deliberately NOT wired in yet -- added later,
# one layer at a time, each scored against the benchmark.
#
# Usage:
#   Rscript sim/run_polyester.R <transcripts.fa> <count_matrix.tsv> <outdir> \
#       <fraglen> <fragsd> <readlen> <seed>

suppressMessages(library(polyester))
suppressMessages(library(Biostrings))

args <- commandArgs(trailingOnly = TRUE)
fasta_path <- args[1]
count_matrix_path <- args[2]
outdir <- args[3]
fraglen <- as.numeric(args[4])
fragsd <- as.numeric(args[5])
readlen <- as.numeric(args[6])
seed <- as.integer(args[7])

countmat <- as.matrix(read.table(
  count_matrix_path, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE
))
storage.mode(countmat) <- "integer"

fasta <- readDNAStringSet(fasta_path)
tx_ids <- sub("\\s.*$", "", names(fasta))
stopifnot(all(rownames(countmat) %in% tx_ids))

# reorder+write a FASTA matching countmat row order exactly -- polyester
# requires readmat row i to correspond to the i-th sequence in the FASTA.
ordered <- fasta[match(rownames(countmat), tx_ids)]
names(ordered) <- rownames(countmat)
ordered_fasta_path <- file.path(dirname(fasta_path), "transcripts.ordered.fa")
writeXStringSet(ordered, ordered_fasta_path)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

simulate_experiment_countmat(
  fasta = ordered_fasta_path,
  readmat = countmat,
  outdir = outdir,
  paired = TRUE,
  seed = seed,
  readlen = readlen,
  fraglen = fraglen,
  fragsd = fragsd,
  error_model = "illumina5",
  strand_specific = FALSE,
  write_info = TRUE
)

cat("done, wrote reads to", outdir, "\n")
