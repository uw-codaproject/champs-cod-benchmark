"""
PAPER run: Elastic Net (multinomial logistic regression, saga), infant+child cohort,
grouped causes. kNN imputation (k=5) + scaling inside the pipeline, balanced class
weights, 5-fold CV grid search over l1_ratio (accuracy).

  python run/paper/enet_paper.py                 # all 5 feature settings
  python run/paper/enet_paper.py --setting 2     # one setting (one configuration; see README)
  python run/paper/enet_paper.py --smoke         # quick wiring test
"""
from common import parse_args, build_pipeline, run_paper_model
from sklearn.linear_model import LogisticRegression


def main():
    args = parse_args(__doc__)
    pipe = build_pipeline(n_neighbors=5)

    tune = {"l1_ratio": [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0]}
    model = LogisticRegression(class_weight="balanced", solver="saga", max_iter=1000,
                               random_state=42, penalty="elasticnet")

    run_paper_model(pipe, model, "Elastic Net", "enet", tune, "grid", None, args)


if __name__ == "__main__":
    main()
