"""
Config for the PAPER / REBUTTAL runs (grouped-cause labels, infant+child cohort).

Differs from src/config.py (dev) in that it:
  * reads the CHAMPS Level-2 release v4.5.0 (2026-02-01) from data/feb2026
  * uses the universal split data/infant_child_split.json (1,939 train / 252 test)
  * takes the two project-derived files that are NOT part of a CHAMPS release
    (VA-narrative ICD-10 annotation matrix, child-abstraction variable JSON)
    from data/feb2026 if present, otherwise falls back to data/may2026.

src/config.py is intentionally left untouched so dev scripts keep working.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RELEASE_DIR = DATA_DIR / "feb2026"
FALLBACK_DIR = DATA_DIR / "may2026"


def _pick(name):
    """Prefer the Feb-2026 release folder; fall back to may2026 for derived files."""
    p = RELEASE_DIR / name
    return p if p.exists() else FALLBACK_DIR / name


LABEL_FILE = RELEASE_DIR / "CHAMPS_deid_decode_results.csv"
DEMOGRAPHICS_FILE = RELEASE_DIR / "CHAMPS_deid_basic_demographics.csv"
VA_FILE = RELEASE_DIR / "CHAMPS_deid_verbal_autopsy.csv"
MITS_LAB_FILE = RELEASE_DIR / "CHAMPS_deid_lab_results.csv"
MITS_TAC_FILE = RELEASE_DIR / "CHAMPS_deid_tac_results.csv"
CHILD_ABS_FILE = RELEASE_DIR / "CHAMPS_deid_child_abstraction_v1.csv"

# Derived (not part of the CHAMPS release)
VA_NARRATIVE_FILE = _pick("CHAMPS_va_narrative_annotation_icd10_ranges.csv")
CHILD_ABS_VAR_JSON = _pick("CHAMPS_child_abstraction_v1_var.json")

TRAIN_TEST_SPLIT_FILE = DATA_DIR / "infant_child_split.json"
