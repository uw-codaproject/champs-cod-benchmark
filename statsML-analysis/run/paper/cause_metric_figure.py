#!/usr/bin/env python3
"""Class-wise precision / recall figure (LightGBM vs gpt-oss-120b, full clinical configuration, Anywhere-in-Chain).

    python run/paper/cause_metric_figure.py
Writes output/paper/bootstrap/cause_metric.pdf and cause_metric_table.csv.

Definitions follow the paper (Evaluation / Individual-level metrics), with P = top-1 prediction and V = whole causal chain
(Anywhere-in-Chain): c in P and c in V is a true positive, c in P and c not in V a false positive, c in V and c not in P a
false negative; precision = TP/(TP+FP), recall = TP/(TP+FN). Zero-division counts as 0.
Causes are the 28 present in the test set, ordered by training-set frequency (top to bottom). Bar thickness is
proportional to log(1 + number of test cases whose chain contains the cause) (maximum 0.4 of a row per bar).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_paper as bp

MODELS = [("LightGBM", "LightGBM"), ("gpt-oss-120b", "gpt-oss-120b")]
SETTING = 4
BLUE, ORANGE, ALPHA = "#1f77b4", "#ff7f0e", 0.95
# tie-break for causes with equal training frequency (keeps the order of the previous version of this figure)
PREV_ORDER = ["malnutrition", "malaria", "lower respiratory infections", "congenital birth defects", "diarrheal diseases", "hiv",
              "sepsis", "undetermined", "neonatal preterm birth complications", "injury", "other infections",
              "other respiratory disease", "sickle cell disorders", "poisoning", "other neonatal disorders", "other", "cancer",
              "tuberculosis", "other endocrine, metabolic, blood, and immune disorders", "meningitis/encephalitis",
              "other neurological disorders", "liver disease", "other nutritional deficiencies", "congenital infection",
              "paralytic ileus and intestinal obstruction", "syphilis", "perinatal asphyxia/hypoxia", "epilepsy"]


def short(s, n=30):
    return s if len(s) <= n else s[:n] + "..."


def main():
    test_ids, truth, chain, demo, preds, probs, _ = bp.load_all()
    split = bp.json.load(open(bp.SPLIT_JSON))
    lab = pd.read_csv(bp.DECODE_CSV, low_memory=False).drop_duplicates("champs_deid").set_index("champs_deid")
    train_n = pd.Series([bp.norm(lab.loc[i, "UC_champs_group_desc"]) for i in split["train"]]).value_counts()
    truth_a = np.array(truth)
    causes = sorted(set(truth), key=lambda c: (-train_n.get(c, 0), PREV_ORDER.index(c) if c in PREV_ORDER else 99))
    support = {c: int((truth_a == c).sum()) for c in causes}
    rows = []
    for name, m in MODELS:
        top1 = np.array([r[0] if r else bp.NONE for r in preds[(m, SETTING)]])
        for c in causes:
            y = truth_a == c
            inV = np.array([c in V for V in chain])      # c belongs to the case's validation set V (the whole causal chain)
            pred_c = top1 == c                            # c in P (top-1 prediction)
            tp = int((pred_c & inV).sum())                # c in P and c in V
            rows.append(dict(model=name, cause=c, train_n=int(train_n.get(c, 0)), support=support[c], n_chain=int(inV.sum()),
                             precision=tp / pred_c.sum() if pred_c.sum() else 0.0,
                             recall=tp / inV.sum() if inV.sum() else 0.0))
    tab = pd.DataFrame(rows)
    tab.to_csv(bp.OUT_DIR / "cause_metric_table.csv", index=False)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(15.89, 6.89), sharey=True)
    fig.subplots_adjust(left=0.215, right=0.995, top=0.985, bottom=0.05, wspace=0.03)
    nch = tab[tab.model == MODELS[0][0]].set_index("cause").n_chain.to_dict()   # cases whose chain contains the cause (recall denominator)
    smax = max(nch.values())
    h = {c: 0.4 * np.log1p(nch[c]) / np.log1p(smax) for c in causes}
    y = np.arange(len(causes))
    for ax, (name, _) in zip(axes, MODELS):
        t = tab[tab.model == name].set_index("cause").loc[causes]
        hh = np.array([h[c] for c in causes])
        ax.barh(y - hh / 2, t.recall.values, height=hh, color=ORANGE, alpha=ALPHA, label="recall")
        ax.barh(y + hh / 2, t.precision.values, height=hh, color=BLUE, alpha=ALPHA, label="precision")
        ax.set_xlim(0, 1.05); ax.set_ylim(len(causes) - 1 + 1.55, -1.8)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(axis="y", labelsize=14); ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        hl, lb = ax.get_legend_handles_labels()
        order = [lb.index("precision"), lb.index("recall")]
        ax.legend([hl[i] for i in order], [lb[i] for i in order], loc="lower right", fontsize=16, framealpha=0.8)
        t1 = ax.text(0.03, 0.03, "Model: ", transform=ax.transAxes, fontsize=14, style="italic", va="bottom", ha="left", zorder=3)
        fig.canvas.draw()
        bb = t1.get_window_extent().transformed(ax.transAxes.inverted())
        t2 = ax.text(bb.x1, 0.03, name, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left", zorder=3)
        fig.canvas.draw()
        b2 = t2.get_window_extent().transformed(ax.transAxes.inverted())
        ax.add_patch(plt.Rectangle((0.015, 0.012), b2.x1 + 0.015 - 0.015, 0.065, transform=ax.transAxes,
                                   fc="#faf9f5", ec="0.8", alpha=0.6, zorder=2, clip_on=False))
    axes[0].set_yticks(y); axes[0].set_yticklabels([short(c) for c in causes])
    fig.savefig(bp.OUT_DIR / "cause_metric.pdf")
    print(tab[tab.cause.isin(["malnutrition", "lower respiratory infections", "malaria"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
