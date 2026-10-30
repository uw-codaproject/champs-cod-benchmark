from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    normalized_predictions: List[str]
    raw_outputs: List[str]


class Evaluator:
    """Aggregates metrics for generative COD predictions."""

    def __init__(
        self,
        output_dir: str | Path,
        allowed_labels: Iterable[str],
    ):
        base_labels = list(allowed_labels)
        if not base_labels:
            raise ValueError("Evaluator requires non-empty allowed_labels.")
        self.label_list = base_labels
        self.output_dir = Path(output_dir).resolve()

    def evaluate(
        self,
        references: List[str],
        predictions: List[str],
        topk_predictions: List[List[str]],
        decode_targets: Optional[List[List[str]]] = None,
        raw_outputs: Optional[List[str]] = None,
    ) -> EvaluationResult:
        metrics: Dict[str, Any] = self._compute_classification_metrics(
            references,
            predictions,
            self.label_list,
            topk_predictions=topk_predictions,
        )
        self._emit_confusion_matrix(references, predictions, self.label_list)

        if decode_targets:
            decode_eval_refs: List[str] = []
            decode_eval_preds: List[str] = []
            decode_eval_topk: List[List[str]] = []
            for idx, (pred, ref_label, candidates) in enumerate(
                zip(predictions, references, decode_targets)
            ):
                candidate_set = {value for value in (candidates or [])}
                if pred in candidate_set:
                    decode_eval_refs.append(pred)
                else:
                    decode_eval_refs.append(ref_label)
                decode_eval_preds.append(pred)
                decode_eval_topk.append(topk_predictions[idx])
            metrics["decode_metrics"] = self._compute_classification_metrics(
                decode_eval_refs,
                decode_eval_preds,
                self.label_list,
                topk_predictions=decode_eval_topk,
            )

        return EvaluationResult(
            metrics=metrics,
            normalized_predictions=predictions,
            raw_outputs=raw_outputs or predictions
        )

    def _compute_classification_metrics(
        self,
        refs: List[str],
        preds: List[str],
        labels_for_report: List[str],
        topk_predictions: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        metrics["top1_accuracy"] = accuracy_score(refs, preds)
        metrics["top3_accuracy"] = self._compute_topk_accuracy(
            refs, topk_predictions or [[pred] for pred in preds], k=3
        )
        metrics["macro_precision"] = precision_score(
            refs, preds, average="macro", zero_division=0
        )
        metrics["micro_precision"] = precision_score(
            refs, preds, average="micro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            refs, preds, average="macro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            refs, preds, average="micro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(refs, preds, average="macro", zero_division=0)
        metrics["micro_f1"] = f1_score(refs, preds, average="micro", zero_division=0)

        p, r, f1, support = precision_recall_fscore_support(
            refs, preds, labels=labels_for_report, zero_division=0
        )
        metrics["per_class"] = {
            label: {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels_for_report)
        }
        return metrics

    def _compute_topk_accuracy(
        self, refs: List[str], topk_predictions: List[List[str]], k: int
    ) -> float:
        if not refs:
            return 0.0
        hits = 0
        for ref, preds in zip(refs, topk_predictions):
            if ref in (preds or [])[:k]:
                hits += 1
        return hits / len(refs)

    def _emit_confusion_matrix(
        self, refs: List[str], preds: List[str], labels: List[str]
    ) -> None:
        # Confusion matrix is for top-1 predictions only.
        label_set = [label for label in labels if label in refs or label in preds]
        if not label_set:
            return
        matrix = confusion_matrix(refs, preds, labels=label_set)
        # Save to CSV table file.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "confusion_matrix.csv"
        df = pd.DataFrame(matrix, index=label_set, columns=label_set)
        df.index.name = "ref/pred"
        df.to_csv(out_path, encoding="utf-8")