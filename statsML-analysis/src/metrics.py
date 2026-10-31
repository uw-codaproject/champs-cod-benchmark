# ========================================
# METRIC FUNCTIONS (MAIN + AUX)
# ========================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from datetime import datetime, timezone
from collections import Counter

# -------------------------------------------------
# Utility helpers
# -------------------------------------------------

def top_k_accuracy(y_true, y_proba, top_k=1):
    """
    Standard top-K accuracy: true label is in top-K predictions
    """
    top_k_preds = np.argsort(y_proba, axis=1)[:, -top_k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])


import numpy as np
import pandas as pd

# -------------------------------------------------
# Helper: check if prediction matches main or any auxiliary label
# -------------------------------------------------
def correct_under_main_or_aux(y_pred, y_test_main, other_target_test, label_encoder):
    """
    Boolean array: True if prediction equals main or any auxiliary label.
    Robust to:
      - string/object aux labels
      - pandas StringDtype
      - integer aux labels
      - unseen labels
      - case mismatches
    """
    correct = (y_pred == y_test_main)

    # Build lowercase class map once
    class_map = {str(name).lower(): idx for idx, name in enumerate(label_encoder.classes_)}

    for col in other_target_test.columns:
        y_aux = other_target_test[col]
        mask = y_aux.notna()
        if not mask.any():
            continue

        y_vals = y_aux[mask]

        # Integer dtype → use directly
        if pd.api.types.is_integer_dtype(y_vals):
            aux_enc = y_vals.to_numpy()

        # Object / string / StringDtype → lowercase and map
        else:
            y_vals = y_vals.astype(str).str.lower()
            aux_enc = np.array([class_map.get(v, -1) for v in y_vals])

        valid = aux_enc >= 0
        if not np.any(valid):
            continue

        idxs = np.where(mask)[0][valid]
        correct[idxs] |= (y_pred[idxs] == aux_enc[valid])

    return correct

# -------------------------------------------------
# Helper: check if a specific class appears in any auxiliary label
# -------------------------------------------------
def _aux_contains_class(c, other_target_test, label_encoder):
    """
    Returns boolean mask: True if class `c` appears in ANY auxiliary label.
    Robust to string/int dtypes and case mismatches.
    """
    mask_any = np.zeros(len(other_target_test), dtype=bool)
    class_name_lower = str(label_encoder.classes_[c]).lower()

    for col in other_target_test.columns:
        y_aux = other_target_test[col]
        mask = y_aux.notna()
        if not mask.any():
            continue

        y_vals = y_aux[mask]

        if pd.api.types.is_integer_dtype(y_vals):
            mask_any[np.where(mask)[0]] |= (y_vals.to_numpy() == c)
        else:
            y_vals_lower = y_vals.astype(str).str.lower()
            mask_any[np.where(mask)[0]] |= (y_vals_lower == class_name_lower).to_numpy()

    return mask_any

# -------------------------------------------------
# Top-K accuracy including main + auxiliary labels
# -------------------------------------------------
def top_k_accuracy_main_or_aux(y_test_main, y_proba, other_target_test, label_encoder, top_k=3):
    """
    Top-K accuracy where prediction is correct if ANY of
    main or auxiliary labels appear in top-K predictions.
    Handles string/int/NaN dtypes robustly.
    """
    top_k_preds = np.argsort(y_proba, axis=1)[:, -top_k:]
    n_samples = len(y_test_main)
    correct = np.zeros(n_samples, dtype=bool)

    # Main labels
    for i in range(n_samples):
        if y_test_main[i] in top_k_preds[i]:
            correct[i] = True

    # Precompute lowercase class map
    class_map = {str(name).lower(): idx for idx, name in enumerate(label_encoder.classes_)}

    for col in other_target_test.columns:
        y_aux = other_target_test[col]
        mask = y_aux.notna()
        if not mask.any():
            continue

        y_vals = y_aux[mask]

        if pd.api.types.is_integer_dtype(y_vals):
            aux_enc = y_vals.to_numpy()
        else:
            y_vals_lower = y_vals.astype(str).str.lower()
            aux_enc = np.array([class_map.get(v, -1) for v in y_vals_lower])

        valid = aux_enc >= 0
        if not np.any(valid):
            continue

        idxs = np.where(mask)[0][valid]

        for idx, aux_label in zip(idxs, aux_enc[valid]):
            if aux_label in top_k_preds[idx]:
                correct[idx] = True

    return correct.mean()

# -------------------------------------------------
# Main-only metrics
# -------------------------------------------------

def calculate_all_metrics(y_true, y_pred, y_proba=None, n_classes=None, top_k=3):
    """
    Compute overall metrics for main-only evaluation
    """
    metrics = {}

    metrics["top1_accuracy"] = accuracy_score(y_true, y_pred)

    metrics[f"top{top_k}_accuracy"] = (
        top_k_accuracy(y_true, y_proba, top_k=top_k) if y_proba is not None else np.nan
    )

    # Macro / Micro precision, recall, F1
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return metrics


# -------------------------------------------------
# MAIN + AUX evaluation
# -------------------------------------------------

def evaluate_main_and_aux2(y_test_main, y_pred, y_proba, other_target_test, label_encoder, n_classes, top_k=3):
    """
    Evaluate metrics under main-only and main+aux regimes.
    Returns nested dictionary:
    {
        "main_only": {...},
        "main_plus_aux": {...}
    }
    """
    results = {}

    # Main-only metrics
    results["main_only"] = calculate_all_metrics(
        y_true=y_test_main,
        y_pred=y_pred,
        y_proba=y_proba,
        n_classes=n_classes,
        top_k=top_k
    )

    # Main + auxiliary
    correct = correct_under_main_or_aux(y_pred, y_test_main, other_target_test, label_encoder)
    main_aux = {}
    main_aux["top1_accuracy"] = correct.mean()

    if y_proba is not None:
        main_aux[f"top{top_k}_accuracy"] = top_k_accuracy_main_or_aux(
            y_test_main, y_proba, other_target_test, label_encoder, top_k=top_k
        )

    # Treat binary correctness as predictions for precision/recall/F1
    y_true_binary = np.ones_like(correct, dtype=int)
    y_pred_binary = correct.astype(int)

    main_aux["macro_precision"] = precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    main_aux["macro_recall"] = recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    main_aux["macro_f1"] = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)

    main_aux["micro_precision"] = precision_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
    main_aux["micro_recall"] = recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
    main_aux["micro_f1"] = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)

    results["main_plus_aux"] = main_aux
    return results


# -------------------------------------------------
# Per-class metrics
# -------------------------------------------------

def per_class_metrics_main_and_aux2(y_test_main, y_pred, other_target_test, label_encoder, n_classes):
    """
    Returns nested dict:
    {
        "main_only": {class_idx: {precision, recall, f1, cause_name}, ...},
        "main_plus_aux": {class_idx: {precision, recall, f1, cause_name}, ...}
    }
    """
    results = {"main_only": {}, "main_plus_aux": {}}

    for c in range(n_classes):
        cause_name = label_encoder.inverse_transform([c])[0]

        # ---------------- Main-only
        y_true_binary = (y_test_main == c)
        y_pred_binary = (y_pred == c)

        results["main_only"][c] = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "cause_name": cause_name
        }

        # ---------------- Main + Aux
        pred_is_c = (y_pred == c)
        correct = pred_is_c & ((y_test_main == c) | _aux_contains_class(c, other_target_test, label_encoder))

        results["main_plus_aux"][c] = {
            "precision": precision_score(y_true_binary, correct, zero_division=0),
            "recall": recall_score(y_true_binary, correct, zero_division=0),
            "f1": f1_score(y_true_binary, correct, zero_division=0),
            "cause_name": cause_name
        }

    return results


# -------------------------------------------------
# Pretty-print helpers
# -------------------------------------------------

def print_overall_metrics(eval_results, digits=4):
    for regime_key, title in [("main_only", "MAIN"), ("main_plus_aux", "MAIN + AUX")]:
        if regime_key not in eval_results:
            continue

        metrics = eval_results[regime_key]

        print("\n" + title)
        print("=" * len(title))

        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, float):
                print(f"{k:20s}: {v:.{digits}f}")
            else:
                print(f"{k:20s}: {v}")


def print_per_cause_metrics(per_class_results, digits=4, sort_by="f1_main_plus_aux"):
    """
    Print per-cause metrics as a table.
    Expects nested dict returned by per_class_metrics_main_and_aux2
    """
    rows = []

    n_classes = len(per_class_results["main_only"])
    for class_idx in range(n_classes):
        main_only = per_class_results["main_only"][class_idx]
        main_plus_aux = per_class_results["main_plus_aux"][class_idx]

        row = {
            "cause": main_only["cause_name"],
            "precision_main": main_only["precision"],
            "recall_main": main_only["recall"],
            "f1_main": main_only["f1"],
            "precision_chain": main_plus_aux["precision"],
            "recall_chain": main_plus_aux["recall"],
            "f1_chain": main_plus_aux["f1"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(digits)

    print("\nPER-CAUSE METRICS")
    print("=================")
    print(df.to_string(index=False))


def build_evaluation_json_standalone(
    eval_results,
    per_cause_results,
    feature_group,
    train_class_counts,
    test_class_counts,
    label_encoder,  # <--- pass the LabelEncoder here
    model_name=None,
    age_group_filter=None
):
    """
    Build JSON evaluation object for aligned train/test data using LabelEncoder for support counts.
    """
    # Convert Counter keys to int (numeric IDs) if needed
    train_class_counts_norm = {int(k): int(v) for k, v in train_class_counts.items()}
    test_class_counts_norm  = {int(k): int(v) for k, v in test_class_counts.items()}

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name if model_name else "model",
        "split": "test",
        "num_cases": sum(train_class_counts_norm.values()),
        "num_cases_test": sum(test_class_counts_norm.values()),
        "age_group_filter": age_group_filter,
        "inputs_included": feature_group,
        "metrics": {},
    }

    # -----------------------------
    # Overall main metrics
    # -----------------------------
    main_metrics = {k: float(v) for k, v in eval_results.get("main_only", {}).items()}
    out["metrics"].update(main_metrics)

    # -----------------------------
    # Per-class main metrics
    # -----------------------------
    per_class = {}
    for _, metrics in per_cause_results.get("main_only", {}).items():
        cause_name = metrics["cause_name"].strip()
        if cause_name in label_encoder.classes_:
            class_id = label_encoder.transform([cause_name])[0]
            support = train_class_counts_norm.get(class_id, 0)
        else:
            class_id = None
            support = 0

        per_class[cause_name] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "support": support,
        }
    out["metrics"]["per_class"] = per_class

    # -----------------------------
    # Decode / main_plus_aux metrics
    # -----------------------------
    decode_metrics = {k: float(v) for k, v in eval_results.get("main_plus_aux", {}).items()}
    decode_per_class = {}
    for _, metrics in per_cause_results.get("main_plus_aux", {}).items():
        cause_name = metrics["cause_name"].strip()
        if cause_name in label_encoder.classes_:
            class_id = label_encoder.transform([cause_name])[0]
            support = train_class_counts_norm.get(class_id, 0)
        else:
            class_id = None
            support = 0

        decode_per_class[cause_name] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "support": support,
        }
    decode_metrics["per_class"] = decode_per_class
    out["metrics"]["decode_metrics"] = decode_metrics

    return out
