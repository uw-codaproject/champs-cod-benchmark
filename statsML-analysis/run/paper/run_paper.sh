#!/usr/bin/env bash
# Run the paper LightGBM and Elastic Net models (all 5 feature settings each), sequentially.
# Usage (from anywhere):   bash run/paper/run_paper.sh [all|lgbm|enet|figures]      (default: all)
#   figures = bootstrap CIs + all paper figures/tables from the saved predictions (needs output/paper/ml/*_pred.jsonl and the LLM results in output/paper/llm/)
# Python: uses $CHAMPS_VENV/bin/python if CHAMPS_VENV is set, else <repo>/.venv/bin/python, else `python` on PATH.
# Outputs: output/paper/infant_child_{lgbm,enet}{.json,_best_params.json,_pred.jsonl}; figures -> output/paper/bootstrap/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if   [ -n "${CHAMPS_VENV:-}" ];   then PY="$CHAMPS_VENV/bin/python"
elif [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"
else PY="python"; fi
command -v "$PY" >/dev/null || [ -x "$PY" ] || { echo "No python found ($PY). Set CHAMPS_VENV=/path/to/venv"; exit 1; }

for f in data/feb2026/CHAMPS_deid_decode_results.csv data/infant_child_split.json; do
  [ -e "$f" ] || { echo "MISSING $f (see README, 'Data setup')"; exit 1; }
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
WHAT="${1:-all}"
echo "Python: $($PY --version 2>&1)   Started: $(date)"

if [ "$WHAT" = all ] || [ "$WHAT" = lgbm ]; then "$PY" run/paper/lgbm_paper.py; fi
if [ "$WHAT" = all ] || [ "$WHAT" = enet ]; then "$PY" run/paper/enet_paper.py; fi
if [ "$WHAT" = figures ]; then
  "$PY" run/paper/bootstrap_paper.py --n-boot 1000
  "$PY" run/paper/cause_metric_figure.py
fi

echo "Finished: $(date)   Outputs in output/paper/"
