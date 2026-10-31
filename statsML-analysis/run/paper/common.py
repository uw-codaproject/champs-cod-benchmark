"""Shared setup for the paper/rebuttal ML runs (LightGBM, Elastic Net)."""
import os, sys, json, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]      # repo root, independent of cwd
sys.path.insert(0, str(ROOT))

from src import config_paper as config                       # noqa: E402
from src.pipeline import CHAMPSPipeline, CHAMPSModelPipeline  # noqa: E402

OUT_DIR = ROOT / "output" / "paper"

# Ablation phases 1-5 in the manuscript (S, +D, +N, +M, +C)
FEATURE_SETTINGS = [
    ["structured-va"],
    ["demographics", "structured-va"],
    ["demographics", "structured-va", "va-narrative"],
    ["demographics", "structured-va", "va-narrative", "mits-lab", "mits-tac"],
    ["demographics", "structured-va", "va-narrative", "mits-lab", "mits-tac", "clinical-abstraction"],
]
SETTING_NAMES = ["S", "S+D", "S+D+N", "S+D+N+M", "S+D+N+M+C"]

AGE_GROUPS = [
    "Infant (28 days to less than 12 months)",
    "Child (12 months to less than 60 Months)",
]

CV_FOLDS = 5          # manuscript: 5-fold cross-validated search
EXPECTED = dict(n_train=1939, n_test=252, n_classes=46, n_test_classes=28)


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--setting", type=int, default=None, choices=range(5),
                    help="Run only feature setting i (0..4 = S, S+D, S+D+N, S+D+N+M, S+D+N+M+C). "
                         "Default: run all five sequentially.")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast wiring test: 2 CV folds, 2 search iterations, writes to output/paper_smoke.")
    return ap.parse_args()


def build_pipeline(n_neighbors):
    """n_neighbors=None -> NaNs kept (LightGBM native handling); 5 -> kNN imputation (Elastic Net)."""
    pipe = CHAMPSPipeline(
        config=config,
        allowed_age_groups=AGE_GROUPS,
        min_class_samples=1,
        n_neighbors=n_neighbors,
    ).run()

    # Fail fast if the data release / split is not the one used in the manuscript.
    tr = pipe.data["train"]["y"]["main"]
    te = pipe.data["test"]["y"]["main"]
    got = dict(n_train=len(tr), n_test=len(te),
               n_classes=len(pipe.label_encoder.classes_),
               n_test_classes=te.nunique())
    print("COHORT CHECK  expected:", EXPECTED, " got:", got, flush=True)
    if got != EXPECTED:
        raise RuntimeError(f"Cohort mismatch: expected {EXPECTED}, got {got}. "
                           "Check data/feb2026 and data/infant_child_split.json.")
    return pipe


def run_paper_model(pipe, model, model_name, tag, tune_params, search_type, n_iter, args):
    smoke = args.smoke
    out = (ROOT / "output" / "paper_smoke") if smoke else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    idxs = [args.setting] if args.setting is not None else ([0] if smoke else list(range(5)))
    suffix = f"_s{args.setting}" if args.setting is not None else ""
    base = out / f"infant_child_{tag}{suffix}"

    runner = CHAMPSModelPipeline(
        champs_pipeline=pipe,
        feature_settings=[FEATURE_SETTINGS[i] for i in idxs],
        model=model,
        fixed_param_list=None,
        tune_params=tune_params,
        search_type=search_type,
        n_iter_search=2 if smoke else n_iter,
        cv_folds=2 if smoke else CV_FOLDS,
        scoring="accuracy",
        save_metrics_path=str(base) + ".json",
        save_params_path=str(base) + "_best_params.json",
        save_preds_path=str(base) + "_pred.jsonl",
        model_name=model_name,
        verbose=True,
    )
    runner.run_all(top_k=3)
