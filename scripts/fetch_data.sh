#!/usr/bin/env bash
# Fetch the benchmark BAMs (+ .bai) that are deliberately not in git, and verify
# them against scripts/data.sha256. See DATA.md for what each file is.
#
#   scripts/fetch_data.sh                      # download from $DEEPSPECAS_DATA_URL
#   scripts/fetch_data.sh --from DIR           # copy from a checkout that has them
#                                              # (e.g. ~/code/deepSpecas)
#   scripts/fetch_data.sh --verify             # only check what is already present
#   scripts/fetch_data.sh --set arena          # also/instead the ptbp1 arena BAMs (~840 MB)
#
# Sets: benchmark (35 MB), benchmark_new (145 MB), benchmark_new_uncurated
# (165 MB), arena (ptbp1_arena, ~840 MB). Default = the first three.
#
# Download URL: PLACEHOLDER until the BAMs are uploaded (Zenodo/release). Files
# are expected at  $DEEPSPECAS_DATA_URL/<dir>__<file>  e.g. .../benchmark__c1.bam
set -euo pipefail
cd "$(dirname "$0")/.."

DEEPSPECAS_DATA_URL="${DEEPSPECAS_DATA_URL:-https://zenodo.org/records/PLACEHOLDER/files}"
mode=download; from=""; sets=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) mode=copy; from="$2"; shift 2;;
    --verify) mode=verify; shift;;
    --set) sets+=("$2"); shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ ${#sets[@]} -eq 0 ] && sets=(benchmark benchmark_new benchmark_new_uncurated)

dirs=()
for s in "${sets[@]}"; do
  case "$s" in arena) dirs+=(ptbp1_arena);; *) dirs+=("$s");; esac
done

if [ "$mode" = download ] && [[ "$DEEPSPECAS_DATA_URL" == *PLACEHOLDER* ]]; then
  echo "DEEPSPECAS_DATA_URL is still the placeholder (the BAMs have not been uploaded yet)." >&2
  echo "Set DEEPSPECAS_DATA_URL, or use --from <checkout with the BAMs>." >&2
  exit 1
fi

fail=0
for d in "${dirs[@]}"; do
  while read -r sum path; do
    [[ "$path" == "$d"/* ]] || continue
    if [ "$mode" != verify ] && [ ! -f "$path" ]; then
      mkdir -p "$d"
      if [ "$mode" = copy ]; then cp -v "$from/$path" "$path"
      else curl -fL --retry 3 -o "$path" "$DEEPSPECAS_DATA_URL/${d}__$(basename "$path")"; fi
    fi
    if [ -f "$path" ] && echo "$sum  $path" | sha256sum -c --quiet -; then echo "ok       $path"
    else echo "MISSING/BAD $path" >&2; fail=1; fi
  done < scripts/data.sha256
done
exit $fail
