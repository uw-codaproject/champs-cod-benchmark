# CHAMPS cause-of-death benchmark: LightGBM, Elastic Net and InSilicoVA

Code to reproduce the supervised-model and InSilicoVA results of the paper on the CHAMPS infant + child cohort
(2,191 cases: 1,939 train / 252 test, fixed split in `data/infant_child_split.json`; 46 grouped underlying causes,
28 of them present in the test set). The LLM experiments (gpt-oss-20b / gpt-oss-120b) live in a companion repository;
this repository only *consumes* their predictions in the bootstrap / figure step.

Five cumulative input configurations are evaluated: **S** structured VA, **+D** demographics, **+N** VA narratives,
**+M** MITS (lab + TAC), **+C** clinical abstraction. InSilicoVA is run on S and +D only (it cannot use the other inputs).

---

## Repository structure

```
.
├── README.md
├── requirements.txt              # Python dependencies (minimum versions)
├── requirements-frozen.txt       # exact versions used on the cluster for the reported LightGBM / Elastic Net runs
├── data/                         # NOT tracked (data use agreement), except the split
│   ├── infant_child_split.json   # fixed train/test case ids (shared by every method)
│   ├── feb2026/                  # CHAMPS Level-2 release (see "Data setup")
│   │   └── processed/            # InSilicoVA inputs, written by src/data/CHAMPS_train_split.py
│   └── may2026/                  # fallback location of the two derived (non-release) input files
├── output/                       # NOT tracked; everything the code writes
│   └── paper/
│       ├── ml/                   # <model>_pred.jsonl for every method (LightGBM, Elastic Net, InSilicoVA, ...)
│       ├── llm/                  # LLM results from the companion repo: unzipped <release>/run_*/ folders or the zip (input of the bootstrap only)
│       └── bootstrap/            # figures and tables written by the bootstrap script
├── src/                          # shared Python package (imported as `src.*`, repo root on sys.path)
│   ├── config_paper.py           # data file paths for the paper runs
│   ├── pipeline.py               # feature construction per configuration + model pipeline (CV search, predictions)
│   ├── metrics.py                # top-1/top-3, micro/macro-F1, Anywhere-in-Chain
│   └── data/
│       ├── constants.py          # column names, target columns, columns dropped from VA
│       ├── cleaning.py           # VA / demographics / lab / TAC cleaning
│       ├── clean_child_abs.py    # clinical-abstraction cleaning
│       └── CHAMPS_train_split.py # merges the release files; writes the InSilicoVA train/test CSVs from the fixed split
├── run/paper/                    # LightGBM / Elastic Net / bootstrap entry points
│   ├── common.py                 # shared setup (feature settings, 5-fold CV, cohort check, output writing)
│   ├── lgbm_paper.py             # LightGBM
│   ├── enet_paper.py             # Elastic Net
│   ├── run_paper.sh              # runs LightGBM + Elastic Net (all 5 configurations each)
│   ├── merge_paper_outputs.py    # only if you run configurations as separate jobs (--setting): merges the pieces
│   ├── bootstrap_paper.py        # shared-resample bootstrap, CIs, pairwise tests, all figures
│   └── cause_metric_figure.py    # class-wise precision / recall figure (LightGBM vs gpt-oss-120b)
└── insilicova/
    ├── InSilicoVAx_1.4.2.tar.gz  # InSilicoVA R package (modified build used for the paper); install from source
    ├── src/clean_champs.R        # VA -> InSilicoVA symptom encoding
    └── run/
        ├── insilico_paper.R      # trains / predicts both settings, writes the prediction JSONL
        └── run_insilicova.sh     # one-command driver (preprocessing + R install checks + run)
```

### Conventions shared with other methods (read this before merging repositories)

* **Case ids and split.** Every method uses the cases and order of `data/infant_child_split.json`
  (`{"train": [...], "test": [...]}`, 1,939 / 252 `champs_deid` values). Do not re-split.
* **Labels.** Underlying cause = `UC_champs_group_desc` (46 grouped causes), **lower-cased**
  (`injury` and `Injury` in the raw release are the same cause). Chain labels for Anywhere-in-Chain are the underlying cause,
  `IC_champs_group_desc` and `Morbid_Cond_01..08_champs_group_desc`.
* **Prediction file.** Each method writes `output/paper/ml/infant_child_<method>_pred.jsonl`, one JSON record per
  (input configuration, test case), 252 per configuration:
  ```json
  {"feature_setting": "demographics, structured-va",
   "champs_deid": "...",
   "true_label": {"type": "UC_champs_group_desc", "name": "malnutrition"},
   "auxiliary_labels": [{"type": "IC_champs_group_desc", "name": "sepsis"}, ...],
   "predicted_labels": {"top_1": "malnutrition", "top_3": ["...", "...", "..."],
                        "all": [{"name": "malnutrition", "probability": 0.92}, ...]}}
  ```
  `feature_setting` is one of `structured-va`, `demographics, structured-va`, `demographics, structured-va, va-narrative`,
  `demographics, structured-va, va-narrative, mits-lab, mits-tac`,
  `demographics, structured-va, va-narrative, mits-lab, mits-tac, clinical-abstraction`.
  `all` lists all 46 causes sorted by probability.
* **Paths.** Scripts locate the repo root from their own file location, so they can be started from any directory
  (the R scripts expect to be started from `insilicova/run/`, which `run_insilicova.sh` takes care of).
* **Python package name.** The shared code is the top-level package `src`. If the other repository also has a `src/`,
  either keep both under separate sub-directories and adjust the `sys.path` line in `run/paper/common.py`, or rename one package.

---

## Data setup

The CHAMPS Level-2 de-identified data are available under a data use agreement
(<https://champshealth.org/causes-of-death/>) and are not distributed with this repository.
Place the release (v4.5.0, February 2026) in `data/feb2026/`:

| File | Used for |
|---|---|
| `CHAMPS_deid_decode_results.csv` | cause labels (underlying, immediate, comorbidities) |
| `CHAMPS_deid_basic_demographics.csv` | demographics, site, sex |
| `CHAMPS_deid_verbal_autopsy.csv` | structured VA |
| `CHAMPS_deid_lab_results.csv`, `CHAMPS_deid_tac_results.csv` | MITS |
| `CHAMPS_deid_child_abstraction_v1.csv` | clinical abstraction |

Two further inputs are derived by the authors and are **not** part of the CHAMPS release. They are read from
`data/feb2026/` if present, otherwise from `data/may2026/`:

* `CHAMPS_va_narrative_annotation_icd10_ranges.csv`: ICD-10-range indicators extracted from the VA narratives (the +N configuration;
  see the paper appendix on automated narrative annotation).
* `CHAMPS_child_abstraction_v1_var.json`: variable list for the clinical-abstraction cleaning (+C configuration).

`data/infant_child_split.json` is tracked.

## Python environment

```bash
uv venv .venv                    # or: python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt     # or: pip install -r requirements.txt
```

`requirements-frozen.txt` lists the exact versions used for the reported LightGBM / Elastic Net runs
(`uv pip install -r requirements-frozen.txt` to reproduce that environment; it does not include matplotlib).

---

## 1. LightGBM and Elastic Net

Both models use the 5 configurations above, grouped-cause labels, 5-fold cross-validated hyperparameter search
(scoring = accuracy) and seed 42. LightGBM: native handling of missing values, balanced class weights, randomized search
(50 iterations). Elastic Net: multinomial logistic regression (`saga`, elastic-net penalty), kNN (k=5) imputation,
grid search over `l1_ratio`. Each run checks that the cohort is 1,939 train / 252 test cases with 46 causes (28 in test)
and aborts otherwise.

```bash
export CHAMPS_VENV=$HOME/path/to/venv        # optional; default <repo>/.venv (its bin/python is used)
bash run/paper/run_paper.sh                  # LightGBM then Elastic Net, all 5 configurations each
bash run/paper/run_paper.sh lgbm             # or only one model: lgbm | enet
```

Outputs, in `output/paper/`: `infant_child_{lgbm,enet}.json` (metrics), `..._best_params.json`, and `..._pred.jsonl`
(1,260 records each = 5 configurations x 252 cases). Move or copy the two `_pred.jsonl` files into `output/paper/ml/`
to keep all methods together (the bootstrap script also finds them in `output/paper/`).

**Running configurations as separate jobs (e.g. a cluster job array; write your own scheduler script).** Each
configuration is independent (index 0..4 = S, +D, +N, +M, +C):

```bash
python run/paper/lgbm_paper.py --setting 0        # writes output/paper/infant_child_lgbm_s0*
python run/paper/enet_paper.py --setting 0
python run/paper/merge_paper_outputs.py           # after all five settings of both models have finished
python run/paper/lgbm_paper.py --smoke            # fast wiring test (2 folds, 2 iterations) -> output/paper_smoke/
```

Set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) when running several jobs on one node.

---

## 2. InSilicoVA

Needs R, a working Java (JDK 7+) linked to R (`rJava`), and a Python venv for the preprocessing step. One command, from `insilicova/run/`:

```bash
cd insilicova/run
bash run_insilicova.sh
```

It (1) activates `<repo>/.venv` (override with `CHAMPS_VENV=/path/to/venv`), (2) builds
`data/feb2026/processed/champs_infant_child_{train,test}.csv` from the Feb 2026 release and the fixed split
(`python src/data/CHAMPS_train_split.py --data-dir data/feb2026 --split-json data/infant_child_split.json --output-dir data/feb2026/processed`),
(3) installs any missing R packages (`rJava`, `coda`, `ggplot2`, `InterVA5`, `tidyverse`, `jsonlite`) from CRAN and
`InSilicoVAx` from the bundled tarball, and (4) runs `insilico_paper.R`. If `Rscript` is not on your `PATH`, set
`RSCRIPT=/full/path/to/Rscript`.

To install the R package by hand: `R CMD INSTALL insilicova/InSilicoVAx_1.4.2.tar.gz` (after installing the CRAN packages above).
On macOS, if `rJava` cannot find Java: `brew install openjdk`, then `sudo R CMD javareconf`. The official InSilicoVA installation guide
(CRAN / GitHub versions, Java requirements) is at <https://github.com/verbal-autopsy-software/InSilicoVA>; the Java setup is the same
for this build. For `java.lang.OutOfMemoryError: Java heap space`, start a fresh R session and set
`options(java.parameters = c("-XX:+UseConcMarkSweepGC"))` before loading the package (the script itself sets `-Xmx2g`).

Settings: structured VA (S) and demographics + structured VA (+D); 4,000 posterior draws, 2,000 burn-in. The script stops if the cohort
is not 1,939 / 252 or the class counts are not 46 / 28. Output: `output/paper/ml/infant_child_insilicova_pred.jsonl`
(504 records = 2 configurations x 252 cases). Training uses all 1,939 train cases (the 92 without a verbal autopsy keep their
cause label with empty VA fields).

## 3. Bootstrap, confidence intervals and figures

```bash
python run/paper/bootstrap_paper.py --n-boot 1000     # default is N_BOOT at the top of the file
```

Inputs: the three prediction files in `output/paper/ml/` (LightGBM, Elastic Net, InSilicoVA) and the LLM greedy-decoding results
from the companion repository, either as an unzipped release folder `output/paper/llm/<release>/run_*/` (preferred) or as a zip in
`output/paper/llm/*.zip` (10 runs = 2 models x 5 configurations; the LLM predictions are not produced by this repository).
The reported results use the release in which the gpt-oss-20b generations that reached the initial 4,096-new-token cap were regenerated
with larger caps (16,384, and 32,768 for three cases), so that only one gpt-oss-20b and one gpt-oss-120b generation remain without a valid answer.

One matrix of 1,000 x 252 case indices is drawn with a fixed seed (`--seed`, default 2026) and **every** method, configuration and metric is scored on the
same resamples; intervals are percentile 95% intervals; the point estimates are on the full test set. Pairwise differences use the same
resamples (`***`, `**`, `*` = 95%, 90%, 80% interval excludes 0). The whole run takes seconds and is deterministic.

Metrics: top-1 / top-3 accuracy under Single Cause (SC) and Anywhere-in-Chain (AiC: a hit if the top-k set intersects the case's chain);
macro-F1 (labels in truth or predictions; for AiC the truth is replaced by the prediction when the prediction is in the chain);
mean chance-corrected concordance (Murray et al. 2011, 46 prediction classes, averaged over the 28 test causes);
norm-AiC = |P ∩ C| / |C| (chain recall of the top-3), stratified by chain length; CSMF accuracy (Murray et al. 2011; underlying cause only):
the predicted CSMF is the column mean of the case-level probabilities, each case's probabilities first rescaled to sum to 1 (an LLM's top-3 probabilities count
as 0 for every other cause), and the true CSMF is the observed test-set composition. CSMF intervals are conditional on that composition: the true fractions
stay fixed and only the predictions are resampled. `csmf_soft_vs_hard.csv` compares this soft aggregation with hard top-1 counts.
Unparseable LLM outputs count as wrong predictions.
The pairwise figure panels are titled "Single Cause Top-1" and "Anywhere-in-Chain Top-1".
"Best configuration per model" (pairwise, site, sex) is the most complete configuration each model has (+C; S+D for InSilicoVA).

Outputs in `output/paper/bootstrap/`: `accuracy_bootstrap.pdf`, `accuracy_bootstrap_top3.pdf`, `accuracy_macro_f1.pdf`, `ccc_bootstrap.pdf`,
`pairwise_diff_bootstrap.pdf`, `site_breakdown_top1.pdf`, `sex_breakdown_top1.pdf`, plus `bootstrap_summary.csv` (all estimates and CIs),
`pairwise_top1_differences.csv`, `chain_length_table.csv`, `csmf_table.csv`, `csmf_soft_vs_hard.csv`, `site_breakdown_top1.csv`, `sex_breakdown_top1.csv` and `bootstrap_indices.npy`.

### Class-wise precision / recall figure

```bash
python run/paper/cause_metric_figure.py        # same inputs as bootstrap_paper.py
```

Writes `output/paper/bootstrap/cause_metric.pdf` and `cause_metric_table.csv` for LightGBM and gpt-oss-120b at the most complete configuration (+C)
under Anywhere-in-Chain. With P the top-1 prediction and V the whole causal chain, a predicted cause c is a true positive if c is in V and a false positive
otherwise; a cause in V that is not predicted is a false negative; precision = TP/(TP+FP), recall = TP/(TP+FN) (zero division counts as 0).
The 28 causes present in the test set are ordered by training-set frequency; bar thickness scales with log(1 + number of test cases whose chain contains the cause).

---

## Reproducibility notes

* Seeds are fixed (models: 42; bootstrap: 2026) and the bootstrap is bit-for-bit reproducible. LightGBM and Elastic Net refits can differ
  slightly in the last digits across library versions and thread counts; use `requirements-frozen.txt` to match the reported environment.
* InSilicoVA is a Gibbs sampler run with the package's default seed (`seed = 1` in `insilico2`); its predictions are written once to
  `infant_child_insilicova_pred.jsonl` and the bootstrap only reads that file.
