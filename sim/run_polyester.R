#!/usr/bin/env Rscript
# Drive polyester's simulate_experiment_countmat from our own per-transcript
# per-replicate count matrix (sim/build_experiment.py), bypassing polyester's
# / ASimulatoR's own event generator entirely -- per plans/signal-native-v2.md
# Phase 2 step 4.
#
# Noise layer 5 (plans/signal-native-v2-noise-layer5.md): PCR duplicates and
# adapter contamination, both provided by the `biomedbigdata/polyester` fork
# (ASimulatoR's) via extra args threaded through `simulate_experiment_countmat`'s
# `...` -> `.check_extras` -> `sgseq` -> `generate_fragments`/`get_reads`.
# Requires that fork installed (plain Bioconductor polyester doesn't have
# these args); optional here, off by default so a layer-1-4 invocation with
# no new args is unchanged.
#
# Usage:
#   Rscript sim/run_polyester.R <transcripts.fa> <count_matrix.tsv> <outdir> \
#       <fraglen> <fragsd> <readlen> <seed> \
#       [pcr_rate] [pcr_lambda] [adapter_contamination] [adapter_sequence]
#
#   pcr_rate: fraction of fragments seeded for PCR duplication, "NA"/omitted
#     to disable (layer-5 off). pcr_lambda: Poisson rate for extra copies
#     per seeded fragment (default 1, only used if pcr_rate is set).
#   adapter_contamination: "TRUE"/"FALSE" (default FALSE). adapter_sequence:
#     override sequence, empty/omitted uses polyester's own default.

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
pcr_rate <- if (length(args) >= 8 && !is.na(args[8]) && args[8] != "") as.numeric(args[8]) else NA
pcr_lambda <- if (length(args) >= 9 && !is.na(args[9]) && args[9] != "") as.numeric(args[9]) else 1
adapter_contamination <- length(args) >= 10 && toupper(args[10]) == "TRUE"
adapter_sequence <- if (length(args) >= 11 && args[11] != "") args[11] else NULL

countmat <- as.matrix(read.table(
  count_matrix_path, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE
))
storage.mode(countmat) <- "integer"

# The ASimulatoR polyester fork's simulate_experiment_countmat has
# `stopifnot(class(readmat) == "matrix")`. Under R>=4, class() on any plain
# matrix always returns c("matrix", "array") regardless of explicit
# class<- assignment (R synthesizes this implicit class for any object with
# a length-2 dim attribute) -- so that check is unconditionally broken on
# this R version (confirmed interactively: `class(m) <- "matrix"; class(m)`
# still prints c("matrix","array")), independent of our own usage. Rather
# than patch the installed package file, define a local copy of the
# function with that one check loosened to is.matrix() -- everything else
# is byte-identical to the fork's own polyester:::simulate_experiment_countmat
# body (confirmed by deparsing it directly).
simulate_experiment_countmat_patched <- function(fasta = NULL, gtf = NULL, seqpath = NULL,
    readmat, outdir = ".", paired = TRUE, seed = NULL, ncores = 1L, ...) {
  extras <- list(...)
  if (!is.null(seed)) set.seed(seed)
  if (!is.null(fasta) & is.null(gtf) & is.null(seqpath)) {
    transcripts <- readDNAStringSet(fasta)
  } else if (is.null(fasta) & !is.null(gtf) & !is.null(seqpath)) {
    transcripts <- polyester::seq_gtf(gtf, seqpath, ...)
  } else {
    stop("must provide either fasta or both gtf and seqpath")
  }
  if (is.null(names(transcripts))) {
    num_trans <- length(transcripts)
    digits <- nchar(num_trans)
    fmt <- paste0("%s%0", digits, "d")
    names(transcripts) <- sprintf(fmt, "transcript", 1:num_trans)
  }
  stopifnot(is.matrix(readmat))
  stopifnot(nrow(readmat) == length(transcripts))
  extras <- polyester:::.check_extras(extras, paired, total.n = ncol(readmat))
  sysoutdir <- gsub(" ", "\\\\ ", outdir)
  if (.Platform$OS.type == "windows") {
    shell(paste("mkdir", sysoutdir))
  } else {
    system(paste("mkdir -p", sysoutdir))
  }
  polyester:::sgseq(readmat, transcripts, paired, outdir, extras, ncores = ncores)
}

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

sim_args <- list(
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
if (!is.na(pcr_rate)) {
  sim_args$pcr_rate <- pcr_rate
  sim_args$pcr_lambda <- pcr_lambda
  cat("noise layer 5: pcr_rate =", pcr_rate, "pcr_lambda =", pcr_lambda, "\n")
}
if (adapter_contamination) {
  sim_args$adapter_contamination <- TRUE
  if (!is.null(adapter_sequence)) sim_args$adapter_sequence <- adapter_sequence
  cat("noise layer 5: adapter_contamination = TRUE, adapter_sequence =",
      if (is.null(adapter_sequence)) "(package default)" else adapter_sequence, "\n")

  # Second bug in the ASimulatoR fork (the first is the class(readmat) check
  # above). In get_reads()'s short-fragment branch, after the adapter is
  # appended to every short fragment it trims any that now exceed readlen:
  #
  #   subseq(outShort[width(outShort) > readlen], start = (readlen + 1)) <- ""
  #
  # If that selection is *empty* -- there were short fragments, but none long
  # enough to overrun readlen even after padding -- `subseq<-` calls xscat()
  # with zero sequences against a one-sequence value and dies with
  # "xscat() cannot mix arguments made of 0 sequence with arguments made of 1
  # or more sequences". At fraglen=250/fragsd=25 short fragments are already a
  # ~6-sd event, so whether any of them clears readlen is luck of the seed:
  # scale-up batches 1-4 happened to have one, batch 5 did not and crashed
  # after 25 minutes of read generation.
  #
  # Guard the selection instead of disabling adapter contamination, so every
  # batch keeps an identical noise model. Patched by text substitution on the
  # deparsed function, asserted to match exactly once so a future package
  # version silently changing the line is a loud failure, not a silent no-op.
  .src <- deparse(polyester:::get_reads)
  .needle <- "subseq(outShort[width(outShort) > readlen], start = (readlen + "
  .hits <- grep(.needle, .src, fixed = TRUE)
  if (length(.hits) != 1L) {
    stop("polyester:::get_reads no longer matches the expected adapter-trim ",
         "line (", length(.hits), " matches); re-check the fork before patching")
  }
  .src[.hits] <- paste0(
    "if (any(width(outShort) > readlen)) ", .src[.hits]
  )
  .patched <- eval(parse(text = paste(.src, collapse = "\n")))
  environment(.patched) <- getNamespace("polyester")
  assignInNamespace("get_reads", .patched, ns = "polyester")
  cat("noise layer 5: patched polyester:::get_reads (empty adapter-trim selection)\n")
}

do.call(simulate_experiment_countmat_patched, sim_args)

cat("done, wrote reads to", outdir, "\n")
