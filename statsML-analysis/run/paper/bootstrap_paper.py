"""Paired case-level bootstrap for the CHAMPS paper / rebuttal figures.

One resample of the 252 test cases is drawn per replicate and EVERY metric of EVERY method and
input configuration is computed on that same resample (shared index matrix, fixed seed).
CIs are percentile intervals.  Point estimates (bar heights) are computed on the full test set.

Inputs (all paper runs, nothing is retrained):
  output/paper/ml/infant_child_{lgbm,enet,insilicova}_pred.jsonl  (also accepted: lgbm/enet directly in output/paper/)
  output/paper/llm/<release>/run_*/  (10 greedy gpt-oss runs, unzipped; preferred) or output/paper/llm/*.zip (read straight from the zip)
  data/infant_child_split.json, data/feb2026/{CHAMPS_deid_decode_results,CHAMPS_deid_basic_demographics}.csv

Outputs (output/paper/bootstrap/):  figures as PDF + CSV tables + the saved index matrix.

Usage (repo root):   python run/paper/bootstrap_paper.py [--n-boot 1000] [--seed 2026]
Needs numpy, pandas, matplotlib only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ============================== SETTINGS ==============================
N_BOOT = 10          # <-- number of bootstrap resamples; set to 1000 for the final run (or --n-boot)
SEED = 2026          # fixed seed for the shared index matrix
# ======================================================================

REPO = Path(__file__).resolve().parents[2]
PAPER_OUT = REPO / "output" / "paper"
ML_DIRS = [PAPER_OUT / "ml", PAPER_OUT]   # prediction files are looked up here, in this order
LLM_ZIP_GLOB = str(REPO / "output" / "paper" / "llm" / "*.zip")
LLM_DIR_GLOB = str(REPO / "output" / "paper" / "llm" / "*" / "run_*" / "metrics_log.jsonl")   # unzipped release (preferred)
OUT_DIR = REPO / "output" / "paper" / "bootstrap"
SPLIT_JSON = REPO / "data" / "infant_child_split.json"
DECODE_CSV = REPO / "data" / "feb2026" / "CHAMPS_deid_decode_results.csv"
DEMO_CSV = REPO / "data" / "feb2026" / "CHAMPS_deid_basic_demographics.csv"

METHODS = ["InSilicoVA", "Elastic Net", "LightGBM", "gpt-oss-20b", "gpt-oss-120b"]
LLMS = {"gpt-oss-20b", "gpt-oss-120b"}
CONFIGS = ["Structured VA", "+ Basic\nDemographics", "+ VA\nnarratives", "+ MITS", "+ Clinical\nAbstraction"]
# Settings: 0=S, 1=+D, 2=+N, 3=+M, 4=+C.  InSilicoVA only has settings 0 and 1.
# "Best input data configuration per model" (pairwise / site / sex) = the most complete configuration
# each model has (this reproduces the rebuttal numbers, e.g. LightGBM +C and InSilicoVA S+D).
BEST_CONFIG = {"InSilicoVA": 1, "Elastic Net": 4, "LightGBM": 4, "gpt-oss-20b": 4, "gpt-oss-120b": 4}
N_PRED_CLASSES = 46  # size of the prediction space for chance-corrected concordance
NONE = "__none__"    # label used when an LLM output cannot be parsed (counted as a wrong prediction)

COLORS = {"InSilicoVA": "#ffd23f", "Elastic Net": "#c2704a", "LightGBM": "#8f4b32",
          "gpt-oss-20b": "#5a8ac6", "gpt-oss-120b": "#2f5da7"}

AUX_COLS = ["IC_champs_group_desc"] + [f"Morbid_Cond_0{i}_champs_group_desc" for i in range(1, 9)]
ML_FILES = {"InSilicoVA": "infant_child_insilicova_pred.jsonl", "Elastic Net": "infant_child_enet_pred.jsonl",
            "LightGBM": "infant_child_lgbm_pred.jsonl"}
ML_SETTINGS = {"structured-va": 0, "demographics, structured-va": 1,
               "demographics, structured-va, va-narrative": 2,
               "demographics, structured-va, va-narrative, mits-lab, mits-tac": 3,
               "demographics, structured-va, va-narrative, mits-lab, mits-tac, clinical-abstraction": 4}
LLM_SETTINGS = {frozenset({"structured_va"}): 0, frozenset({"structured_va", "demographics"}): 1,
                frozenset({"structured_va", "narrative", "demographics"}): 2,
                frozenset({"structured_va", "narrative", "demographics", "mits"}): 3,
                frozenset({"structured_va", "narrative", "demographics", "mits", "clinical_abstraction"}): 4}


def find_ml_file(fname: str) -> Path:
    for d in ML_DIRS:
        if (d / fname).exists():
            return d / fname
    raise FileNotFoundError(f"{fname} not found in {[str(d) for d in ML_DIRS]}")


def norm(s):
    return None if (s is None or pd.isna(s)) else str(s).strip().lower()


# ------------------------------------------------------------------ data
def load_all():
    split = json.load(open(SPLIT_JSON))
    test_ids = list(split["test"])
    assert len(test_ids) == 252 and len(set(test_ids)) == 252
    lab = pd.read_csv(DECODE_CSV, low_memory=False).drop_duplicates("champs_deid").set_index("champs_deid")
    truth = [norm(lab.loc[i, "UC_champs_group_desc"]) for i in test_ids]
    chain = [({t} | {norm(lab.loc[i, c]) for c in AUX_COLS}) - {None} for i, t in zip(test_ids, truth)]
    demo = pd.read_csv(DEMO_CSV, usecols=["champs_deid", "site_iso_code", "sex"]).drop_duplicates("champs_deid")
    demo = demo.set_index("champs_deid").loc[test_ids]

    preds: dict[tuple[str, int], list[list[str]]] = {}   # (method, setting) -> top-3 name lists in test order
    probs: dict[tuple[str, int], list[dict[str, float]]] = {}   # (method, setting) -> {cause: probability} per case
    for method, fname in ML_FILES.items():
        by = collections.defaultdict(dict)
        byp = collections.defaultdict(dict)
        for line in open(find_ml_file(fname)):
            r = json.loads(line)
            allp = sorted(r["predicted_labels"]["all"], key=lambda x: -x["probability"])
            assert len(allp) == N_PRED_CLASSES
            by[ML_SETTINGS[r["feature_setting"]]][r["champs_deid"]] = [x["name"] for x in allp[:3]]
            byp[ML_SETTINGS[r["feature_setting"]]][r["champs_deid"]] = {x["name"]: float(x["probability"]) for x in allp}
        for k, d in by.items():
            assert set(d) == set(test_ids), (method, k)
            preds[(method, k)] = [d[i] for i in test_ids]
            probs[(method, k)] = [byp[k][i] for i in test_ids]

    zips = sorted(glob.glob(LLM_ZIP_GLOB))
    run_dirs = sorted(glob.glob(LLM_DIR_GLOB))
    assert run_dirs or zips, f"no LLM results found at {LLM_DIR_GLOB} or {LLM_ZIP_GLOB}"
    if run_dirs:    # unzipped release folder (preferred): <folder>/run_*/{metrics_log,predictions}.jsonl
        z = None
        sources = [str(Path(m).parent) for m in run_dirs]
        read = lambda path: open(path, encoding="utf-8").read()
    else:           # zip release
        z = zipfile.ZipFile(zips[-1])
        sources = [n.rsplit("/", 1)[0] for n in z.namelist() if n.endswith("metrics_log.jsonl")]
        read = lambda path: z.read(path).decode()
    n_unparse = collections.Counter()
    for d in sources:
        ml = json.loads(read(d + "/metrics_log.jsonl").splitlines()[0])
        model = ml["model"].split("/")[-1]
        k = LLM_SETTINGS[frozenset(x.strip() for x in ml["inputs_included"].split(","))]
        recs = {}
        for line in read(d + "/predictions.jsonl").splitlines():
            r = json.loads(line)
            recs[r["champs_deid"]] = r
        assert set(recs) == set(test_ids)
        out, outp = [], []
        for i in test_ids:
            try:
                tc = json.loads(recs[i]["generation"])["top_causes"]
                names = [norm(x["cause_name"]) for x in tc]
                pr = [x.get("probability") for x in tc]
            except Exception:
                names, pr = [], []
                n_unparse[(model, k)] += 1
            seen, pd_ = [], {}
            for nme, pv in zip(names, pr):
                if nme is not None and nme not in seen:
                    seen.append(nme)
                    try:
                        pd_[nme] = float(pv)
                    except (TypeError, ValueError):
                        pd_[nme] = 0.0
            out.append(seen[:3])
            outp.append({c: pd_[c] for c in seen[:3]})   # top-3 probabilities; every other cause gets 0
        assert (model, k) not in preds, f"duplicate LLM run for {(model, k)}"
        preds[(model, k)] = out
        probs[(model, k)] = outp
    return test_ids, truth, chain, demo, preds, probs, n_unparse


# ------------------------------------------------------------------ per-case arrays
class Case:
    """Everything the metrics need, as arrays over the 252 test cases."""

    def __init__(self, truth, chain, preds, probs):
        names = set(truth) | {c for V in chain for c in V} | {NONE}
        for v in preds.values():
            names |= {c for r in v for c in r}
        for v in probs.values():
            names |= {c for r in v for c in r}
        self.vocab = {nme: j for j, nme in enumerate(sorted(names))}
        self.K = len(self.vocab)
        self.y = np.array([self.vocab[t] for t in truth])
        self.pi0 = np.bincount(self.y, minlength=self.K) / len(self.y)   # observed test-set cause fractions
        self.test_causes = np.array(sorted(set(self.y)))          # the 28 causes present in the test set
        self.chain_sz = np.array([len(V) for V in chain])
        self.items = {}
        for key, rows in preds.items():
            top1 = [r[0] if r else NONE for r in rows]
            p1 = np.array([self.vocab[t] for t in top1])
            sc1 = np.array([t == pp for t, pp in zip(truth, top1)])
            sc3 = np.array([t in r for t, r in zip(truth, rows)])
            aic1 = np.array([pp in V for pp, V in zip(top1, chain)])
            aic3 = np.array([any(c in V for c in r) for r, V in zip(rows, chain)])
            nrm = np.array([len(set(r) & V) / len(V) for r, V in zip(rows, chain)])
            y2 = np.where(aic1, p1, self.y)     # truth re-labelled to the prediction when it lies in the chain
            P = np.zeros((len(rows), self.K))     # predicted cause probabilities (zero outside what the method reports)
            for a, d in enumerate(probs[key]):
                for c, v in d.items():
                    P[a, self.vocab[c]] = v
            rs = P.sum(axis=1, keepdims=True)
            P = np.divide(P, rs, out=np.zeros_like(P), where=rs > 0)   # each case's probabilities sum to 1 (unparsed LLM output: all zero)
            self.items[key] = dict(p1=p1, y2=y2, SC1=sc1, SC3=sc3, AiC1=aic1, AiC3=aic3, normAiC=nrm, P=P)


# ------------------------------------------------------------------ metrics on one resample
def macro_f1(yt, yp, K):
    """sklearn-default macro-F1: average over labels that occur in y_true or y_pred."""
    ny = np.bincount(yt, minlength=K)
    npd = np.bincount(yp, minlength=K)
    tp = np.bincount(yt[yt == yp], minlength=K)
    den = ny + npd
    present = den > 0
    return float(np.mean(2 * tp[present] / den[present]))


def mean_ccc(y_true, yt, yp, causes, K, n_pred=N_PRED_CLASSES):
    """Mean chance-corrected concordance (Murray 2011) over the test causes.

    y_true: original labels (decides which of the 28 causes are present in the resample);
    yt: labels used for recall (== y_true for Single Cause; truth re-labelled to the prediction when it lies
    in the chain for Anywhere-in-Chain).  A present cause with no case under `yt` has recall 0."""
    present = np.bincount(y_true, minlength=K)[causes] > 0
    ny = np.bincount(yt, minlength=K)[causes]
    hit = np.bincount(yt[yt == yp], minlength=K)[causes]
    rec = np.where(ny > 0, hit / np.maximum(ny, 1), 0.0)[present]
    return float(np.mean((rec - 1 / n_pred) / (1 - 1 / n_pred)))


def csmf_accuracy(y_true, P, K, n_pred=N_PRED_CLASSES, pi=None):
    """CSMF accuracy (Murray 2011): 1 - sum_c |pi_hat_c - pi_c| / (2 (1 - min_c pi_c)).

    pi = true underlying-cause fractions, pi_hat = column means of the predicted probabilities (each case's probabilities
    are first rescaled to sum to 1; an LLM's top-3 probabilities count as 0 for every other cause).  The minimum runs over all `n_pred` (= 46) causes, so it is 0 whenever some cause has
    no case in the test set, which is always the case here (only 28 of the 46 causes occur in it).

    `pi` (optional) fixes the true fractions; the bootstrap passes the original test-set fractions, so that CSMF
    intervals are conditional on the observed test composition (only the predictions are resampled)."""
    if pi is None:
        pi = np.bincount(y_true, minlength=K) / len(y_true)
    pi_hat = P if P.ndim == 1 else P.mean(axis=0)       # P: per-case probabilities (n, K), or a ready-made CSMF vector
    min_pi = pi.min() if (pi > 0).sum() >= n_pred else 0.0
    return float(1 - np.abs(pi_hat - pi).sum() / (2 * (1 - min_pi)))


METRICS = ["top1", "top3", "macroF1", "ccc", "csmf"]
REGIMES = ["SC", "AiC"]


def metrics_on(case: Case, it: dict, ix: np.ndarray) -> dict:
    """All plotted metrics for one method/config on the cases `ix` (indices into the 252)."""
    K, y, p1 = case.K, case.y[ix], it["p1"][ix]
    y2 = it["y2"][ix]
    return {
        ("SC", "top1"): it["SC1"][ix].mean(), ("SC", "top3"): it["SC3"][ix].mean(),
        ("AiC", "top1"): it["AiC1"][ix].mean(), ("AiC", "top3"): it["AiC3"][ix].mean(),
        ("SC", "macroF1"): macro_f1(y, p1, K), ("AiC", "macroF1"): macro_f1(y2, p1, K),
        ("SC", "ccc"): mean_ccc(y, y, p1, case.test_causes, K), ("AiC", "ccc"): mean_ccc(y, y2, p1, case.test_causes, K),
        ("AiC", "normAiC"): it["normAiC"][ix].mean(),
        ("SC", "csmf"): csmf_accuracy(y, it["P"][ix], K, pi=case.pi0),   # underlying cause only; true fractions fixed at the observed ones
    }


def run_bootstrap(case: Case, n_boot: int, seed: int):
    n = len(case.y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))          # ONE index matrix shared by all methods and metrics
    full = np.arange(n)
    est, dist = {}, {}
    for key, it in case.items.items():
        est[key] = metrics_on(case, it, full)
        reps = [metrics_on(case, it, idx[b]) for b in range(n_boot)]
        dist[key] = {m: np.array([r[m] for r in reps]) for m in est[key]}
    return idx, est, dist


def pct_ci(x, level=95):
    a = (100 - level) / 2
    return float(np.percentile(x, a)), float(np.percentile(x, 100 - a))


# ------------------------------------------------------------------ tables
def summary_table(est, dist):
    rows = []
    for (method, k), e in est.items():
        for (reg, met), v in e.items():
            lo, hi = pct_ci(dist[(method, k)][(reg, met)])
            rows.append(dict(method=method, setting=k, regime=reg, metric=met, estimate=v, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def csmf_sensitivity(case: Case):
    """CSMF accuracy with soft aggregation (mean probabilities; reported in the paper) and hard aggregation
    (share of cases whose top-1 prediction is each cause), full test set, point estimates."""
    rows = []
    for (method, k), it in case.items.items():
        hard = np.bincount(it["p1"], minlength=case.K) / len(it["p1"])
        rows.append(dict(method=method, setting=k, soft=csmf_accuracy(case.y, it["P"], case.K),
                         hard=csmf_accuracy(case.y, hard, case.K)))
    return pd.DataFrame(rows)


def csmf_table(summ):
    """CSMF accuracy per method (rows) and input configuration (columns): 'estimate [lo, hi]'."""
    d = summ[summ.metric == "csmf"]
    rows = []
    for method in METHODS:
        row = {"method": method}
        for k in range(5):
            r = d[(d.method == method) & (d.setting == k)]
            row[f"config_{k}"] = "-" if r.empty else "{:.3f} [{:.3f}, {:.3f}]".format(*r.iloc[0][["estimate", "ci_lo", "ci_hi"]])
        rows.append(row)
    return pd.DataFrame(rows)


PAIRS = [  # top -> bottom order of the pairwise figure
    ("gpt-oss-120b", "Elastic Net"), ("Elastic Net", "gpt-oss-20b"), ("LightGBM", "gpt-oss-120b"),
    ("gpt-oss-120b", "gpt-oss-20b"), ("LightGBM", "Elastic Net"), ("LightGBM", "gpt-oss-20b"),
    ("gpt-oss-120b", "InSilicoVA"), ("LightGBM", "InSilicoVA"),
]


def stars(d):
    for level, s in [(95, "***"), (90, "**"), (80, "*")]:
        lo, hi = pct_ci(d, level)
        if lo > 0 or hi < 0:
            return s
    return ""


def pairwise_table(est, dist):
    rows = []
    for reg in REGIMES:
        for a, b in PAIRS:
            ka, kb = (a, BEST_CONFIG[a]), (b, BEST_CONFIG[b])
            d = dist[ka][(reg, "top1")] - dist[kb][(reg, "top1")]
            lo, hi = pct_ci(d)
            rows.append(dict(regime=reg, A=a, B=b, diff=est[ka][(reg, "top1")] - est[kb][(reg, "top1")],
                             ci_lo=lo, ci_hi=hi, stars=stars(d)))
    return pd.DataFrame(rows)


def chain_length_table(case: Case):
    rows = []
    bucket = np.minimum(case.chain_sz, 6)
    for method in METHODS:
        it = case.items[(method, BEST_CONFIG[method])]
        for v in range(1, 7):
            m = bucket == v
            if not m.any():
                continue
            rows.append(dict(method=method, V=("6+" if v == 6 else v), n=int(m.sum()), pct=100 * m.mean(),
                             AiC1=it["AiC1"][m].mean(), AiC3=it["AiC3"][m].mean(), normAiC=it["normAiC"][m].mean()))
        rows.append(dict(method=method, V="All", n=len(bucket), pct=100.0, AiC1=it["AiC1"].mean(),
                         AiC3=it["AiC3"].mean(), normAiC=it["normAiC"].mean()))
    return pd.DataFrame(rows)


SITES = {"ET": "Ethiopia", "KE": "Kenya", "SL": "Sierra Leone", "ZA": "South Africa"}


def group_table(case: Case, groups: pd.Series, order: list[str]):
    rows = []
    for method in METHODS:
        it = case.items[(method, BEST_CONFIG[method])]
        for g in order:
            m = (groups == g).to_numpy()
            rows.append(dict(method=method, group=g, n=int(m.sum()), SC1=it["SC1"][m].mean(), AiC1=it["AiC1"][m].mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plotting
pct0 = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
pct1 = pct0  # whole-number percent ticks (5%, 10%, ...)


def _bar(ax, x, h, w, method):
    hatched = method in LLMS
    c = COLORS[method]
    if hatched:
        plt.rcParams["hatch.linewidth"] = 0.8
        ax.bar(x, h, w, color=c, edgecolor="white", hatch="//", linewidth=0, zorder=2)
    else:
        ax.bar(x, h, w, color=c, edgecolor="none", linewidth=0, zorder=2)


def _legends(ax, extra=None, loc="upper left"):
    """Method legend, plus a second box (No Training [+ chance line]) placed just right of the first one."""
    handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    leg1 = ax.legend(handles=handles, loc=loc, fontsize=11, framealpha=0.9)
    ax.add_artist(leg1)
    ax.figure.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    second = [Patch(facecolor="#bbbbbb", edgecolor="white", hatch="//", label="No Training")]
    if extra is not None:
        second.append(extra)
    ax.legend(handles=second, loc="upper left", bbox_to_anchor=(bb.x1 + 0.002, bb.y1), fontsize=11, framealpha=0.9)


def grouped_bar_figure(est, dist, metric, ylabel, ylim, yticks, fmt, path, chance=False):
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.2), sharey=True)
    w = 0.15
    for ax, reg, title in zip(axes, REGIMES, ["Single Cause", "Anywhere-in-Chain"]):
        for mi, method in enumerate(METHODS):
            for k in range(5):
                if (method, k) not in est:
                    continue
                x = k + (mi - 2) * w
                h = est[(method, k)][(reg, metric)]
                lo, hi = pct_ci(dist[(method, k)][(reg, metric)])
                _bar(ax, x, h, w, method)
                ax.errorbar(x, h, yerr=[[max(h - lo, 0)], [max(hi - h, 0)]], fmt="none", ecolor="black",
                            elinewidth=1.3, capsize=3, capthick=1.3, zorder=4)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(5))
        ax.set_xticklabels(CONFIGS, fontsize=11)
        ax.set_xlim(-0.55, 4.55)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if chance:
            ax.axhline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    axes[0].set_ylabel(ylabel, fontsize=12)
    extra = Line2D([0], [0], color="red", linestyle="--", alpha=0.7, label="Chance (CCC=0)") if chance else None
    _legends(axes[0], extra=extra)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pairwise_figure(pw, path):
    colors = {"***": "#c0392b", "**": "#e67e22", "*": "#f5a623", "": "#aaaaaa"}
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.8), sharey=True)
    ypos = {p: len(PAIRS) - 1 - i for i, p in enumerate(PAIRS)}
    for ax, reg in zip(axes, REGIMES):
        sub = pw[pw.regime == reg]
        for _, r in sub.iterrows():
            y = ypos[(r.A, r.B)]
            c = colors[r.stars]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, linewidth=2.5, solid_capstyle="butt", zorder=3)
            ax.plot(r["diff"], y, "o", color=c, markersize=8, zorder=4)
            if r.stars:
                ax.text(r.ci_hi + 0.008, y, r.stars, color=c, va="center", fontsize=14, fontweight="bold")
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_title("Anywhere-in-Chain Top-1" if reg == "AiC" else "Single Cause Top-1", fontsize=13)
        ax.set_xlabel("Accuracy difference of A vs B  (A − B)", fontsize=12)
        ax.xaxis.set_major_formatter(pct0)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_ylim(-0.7, len(PAIRS) - 0.3)
    axes[0].set_yticks([ypos[p] for p in PAIRS])
    axes[0].set_yticklabels([f"{a}\nvs {b}" for a, b in PAIRS], fontsize=11)
    lo = min(pw.ci_lo.min() - 0.04, -0.12)
    for ax in axes:
        ax.set_xlim(lo, pw.ci_hi.max() + 0.09)
    handles = [Line2D([0], [0], color=colors["***"], marker="o", linewidth=2.5, label="***  95% bootstrap CI excludes 0"),
               Line2D([0], [0], color=colors["**"], marker="o", linewidth=2.5, label="**   90% bootstrap CI excludes 0"),
               Line2D([0], [0], color=colors["*"], marker="o", linewidth=2.5, label="*    80% bootstrap CI excludes 0"),
               Line2D([0], [0], color=colors[""], marker="o", linewidth=2.5, label="      CI spans zero")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=10.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def breakdown_figure(tab, order, labels, title, path, figsize):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    w = 0.16
    for ax, col, ttl in zip(axes, ["SC1", "AiC1"], ["Single Cause", "Anywhere-in-Chain"]):
        for mi, method in enumerate(METHODS):
            for gi, g in enumerate(order):
                r = tab[(tab.method == method) & (tab.group == g)].iloc[0]
                _bar(ax, gi + (mi - 2) * w, r[col], w, method)
        ax.set_title(ttl, fontsize=13)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(pct0)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axes[0].set_ylabel("Top-1 Accuracy", fontsize=12)
    handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    handles.append(Patch(facecolor="#bbbbbb", edgecolor="white", hatch="//", label="No Training"))
    axes[0].legend(handles=handles, loc="upper center", ncol=3, fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    test_ids, truth, chain, demo, preds, probs, n_unparse = load_all()
    case = Case(truth, chain, preds, probs)
    print(f"{len(preds)} method/config runs loaded; {len(test_ids)} test cases; {len(case.test_causes)} test causes; "
          f"LLM runs with unparseable outputs: {dict(n_unparse)}")
    print(f"|V|: mean {case.chain_sz.mean():.2f}, median {np.median(case.chain_sz):.0f}, max {case.chain_sz.max()}, "
          f"singletons {100 * np.mean(case.chain_sz == 1):.1f}%")

    idx, est, dist = run_bootstrap(case, args.n_boot, args.seed)
    np.save(out / "bootstrap_indices.npy", idx.astype(np.int16))
    print(f"bootstrap: B={args.n_boot}, seed={args.seed}, index matrix {idx.shape} saved")

    summ = summary_table(est, dist)
    summ.to_csv(out / "bootstrap_summary.csv", index=False)
    csmf_table(summ).to_csv(out / "csmf_table.csv", index=False)
    csmf_sensitivity(case).to_csv(out / "csmf_soft_vs_hard.csv", index=False)
    pw = pairwise_table(est, dist)
    pw.to_csv(out / "pairwise_top1_differences.csv", index=False)
    chain_length_table(case).to_csv(out / "chain_length_table.csv", index=False)
    site = demo["site_iso_code"].map(SITES)
    site_order = ["Ethiopia", "Kenya", "Sierra Leone", "South Africa"]
    st = group_table(case, site, site_order)
    st.to_csv(out / "site_breakdown_top1.csv", index=False)
    sx = group_table(case, demo["sex"], ["Female", "Male"])
    sx.to_csv(out / "sex_breakdown_top1.csv", index=False)

    grouped_bar_figure(est, dist, "top1", "Top-1 Accuracy", (0, 0.85), np.arange(0, 0.81, 0.1), pct0,
                       out / "accuracy_bootstrap.pdf")
    grouped_bar_figure(est, dist, "top3", "Top-3 Accuracy", (0, 1.02), np.arange(0, 1.01, 0.2), pct0,
                       out / "accuracy_bootstrap_top3.pdf")
    grouped_bar_figure(est, dist, "macroF1", "Macro-F1", (0, 0.39), np.arange(0, 0.36, 0.05), pct1,
                       out / "accuracy_macro_f1.pdf")
    grouped_bar_figure(est, dist, "ccc", "Mean Chance-Corrected Concordance", (-0.05, 0.39), np.arange(-0.05, 0.36, 0.05),
                       pct1, out / "ccc_bootstrap.pdf", chance=True)
    pairwise_figure(pw, out / "pairwise_diff_bootstrap.pdf")
    breakdown_figure(sx, ["Female", "Male"], [f"{g}\n(n={int((demo['sex'] == g).sum())})" for g in ["Female", "Male"]],
                     "Top-1 Accuracy by Sex (best config per model)", out / "sex_breakdown_top1.pdf", (11, 5.6))
    n_ng = int((demo["site_iso_code"] == "NG").sum())
    breakdown_figure(st, site_order, [f"{s}\n(n={int((site == s).sum())})" for s in site_order],
                     f"Top-1 Accuracy by Site (best config per model, NG excluded n={n_ng})",
                     out / "site_breakdown_top1.pdf", (15.6, 5.4))
    print("wrote figures + tables to", out)
    print("\nPairwise top-1 differences (best config per model):")
    print(pw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
