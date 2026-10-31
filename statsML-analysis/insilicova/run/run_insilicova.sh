#!/usr/bin/env bash
# Paper InSilicoVA run. Usage:  cd insilicova/run && bash run_insilicova.sh
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RUN_DIR/../.." && pwd)"

# --- EDIT THESE (set your own paths) ---------------------------------------
VENV="${CHAMPS_VENV:-$REPO/.venv}"   # existing local venv (repo-local .venv per README); set your own path or CHAMPS_VENV
RSCRIPT="${RSCRIPT:-Rscript}"   # leave as "Rscript" if it is on PATH; otherwise e.g. /path/to/R/bin/Rscript
# On clusters you may need e.g.:  module load R  (and a Java module, InSilicoVAx needs Java)
# ---------------------------------------------------------------------------

cd "$RUN_DIR"

# 1) Python preprocessing: fixed paper split + Feb 2026 release -> data/feb2026/processed/
source "$VENV/bin/activate"
python "$REPO/src/data/CHAMPS_train_split.py" \
  --data-dir "$REPO/data/feb2026" \
  --split-json "$REPO/data/infant_child_split.json" \
  --output-dir "$REPO/data/feb2026/processed"

# 2) InSilicoVA (R). Rscript is machine-wide; a full path is only needed if it is not on PATH.
command -v "$RSCRIPT" >/dev/null || { echo "Rscript not found: set RSCRIPT=/full/path/to/Rscript"; exit 1; }
command -v java >/dev/null || echo "WARNING: java not on PATH (InSilicoVAx needs it)"
# Install what is missing: CRAN deps first, then InSilicoVAx from the bundled source tarball (R CMD INSTALL).
# Needs a working Java (JDK) for rJava; on macOS:  brew install openjdk  then  sudo R CMD javareconf
"$RSCRIPT" -e '
need <- c("rJava", "coda", "ggplot2", "InterVA5", "tidyverse", "jsonlite")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss)) { message("Installing from CRAN: ", paste(miss, collapse = ", ")); install.packages(miss, repos = "https://cloud.r-project.org") }
if (!requireNamespace("InSilicoVAx", quietly = TRUE)) {
  message("Installing InSilicoVAx from source tarball")
  install.packages("'"$REPO"'/insilicova/InSilicoVAx_1.4.2.tar.gz", repos = NULL, type = "source")
}
if (!requireNamespace("InSilicoVAx", quietly = TRUE)) stop("InSilicoVAx still not available (check Java / rJava)")
'
"$RSCRIPT" insilico_paper.R

echo "Done: $REPO/output/paper/ml/infant_child_insilicova_pred.jsonl"
