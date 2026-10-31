"""
PAPER run: LightGBM, infant+child cohort, grouped causes (UC_champs_group_desc).
Native missing-value handling (no imputation), balanced class weights,
5-fold CV randomized search (50 iterations, seed 42, scoring=accuracy).

  python run/paper/lgbm_paper.py                 # all 5 feature settings
  python run/paper/lgbm_paper.py --setting 2     # one setting (one configuration; see README)
  python run/paper/lgbm_paper.py --smoke         # quick wiring test
"""
from common import parse_args, build_pipeline, run_paper_model
from scipy.stats import randint, uniform
from lightgbm import LGBMClassifier


def main():
    args = parse_args(__doc__)
    pipe = build_pipeline(n_neighbors=None)

    param_dist = {
        "num_leaves": randint(31, 500),
        "max_depth": [-1] + list(range(5, 13, 2)),
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": randint(200, 1000),
        "min_child_samples": randint(10, 200),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "min_split_gain": [0.0, 0.01, 0.1],
    }
    # n_jobs=1 inside each fit: parallelism comes from the CV search (n_jobs=-1),
    # avoiding nested thread oversubscription. Does not change the fitted model.
    model = LGBMClassifier(objective="multiclass", class_weight="balanced",
                           random_state=42, n_jobs=1, verbosity=-1)

    run_paper_model(pipe, model, "LightGBM", "lgbm", param_dist, "random", 50, args)


if __name__ == "__main__":
    main()
