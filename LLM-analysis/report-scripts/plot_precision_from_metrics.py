import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics_line(path: Path, line_number: int) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"line_number={line_number} is out of range (1-{len(lines)})")
    idx = line_number - 1
    while idx < len(lines) and not lines[idx].lstrip().startswith("{"):
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"No JSON found at or after line {line_number}.")
    return json.loads(lines[idx])


def _collect_rows(per_class: dict) -> list[dict]:
    rows = []
    for cause, stats in per_class.items():
        rows.append(
            {
                "cause": cause,
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "support": int(stats.get("support", 0)),
            }
        )
    return rows


def build_metric_plot(
    rows: list[dict],
    metric_key: str,
    out_path: Path,
    title: str,
    color: str,
) -> None:
    # Sort by metric (descending), then by support (descending)
    rows = sorted(rows, key=lambda r: (r[metric_key], r["support"]), reverse=True)

    causes = [r["cause"] for r in rows]
    values = np.array([r[metric_key] for r in rows], dtype=float)
    supports = np.array([r["support"] for r in rows], dtype=float)

    if len(causes) == 0:
        raise ValueError("No per-class results found for plotting.")

    max_support = max(supports.max(), 1.0)
    line_widths = 0.5 + 5.5 * (supports / max_support)
    dot_sizes = 20 + 280 * (supports / max_support)

    fig_height = max(6, 0.28 * len(causes))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = np.arange(len(causes))

    for y, metric, lw, size in zip(y_positions, values, line_widths, dot_sizes):
        ax.plot([0, metric], [y, y], linewidth=lw, color=color, alpha=0.8)
        ax.scatter([metric], [y], s=size, color=color, alpha=0.9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(causes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(title.split()[-1])
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-class precision and recall from metrics_log.jsonl."
    )
    parser.add_argument(
        "--metrics-log",
        default="outputs/pipeline/metrics_log.jsonl",
        help="Path to metrics_log.jsonl",
    )
    parser.add_argument(
        "--line",
        type=int,
        default=47,
        help="1-based line number for the target JSON entry (line 47 for your request).",
    )
    parser.add_argument(
        "--out-precision",
        default="outputs/report/precision_line47.png",
        help="Output image path for precision.",
    )
    parser.add_argument(
        "--out-recall",
        default="outputs/report/recall_line47.png",
        help="Output image path for recall.",
    )
    args = parser.parse_args()

    metrics_log = Path(args.metrics_log)
    payload = _load_metrics_line(metrics_log, args.line)
    metrics = payload.get("metrics", {})
    per_class = metrics.get("per_class", {})
    if not per_class:
        raise ValueError("No per_class metrics found before decode_metrics.")

    rows = _collect_rows(per_class)
    build_metric_plot(
        rows,
        "precision",
        Path(args.out_precision),
        "Per-Class Precision (dot size = support)",
        "#1f77b4",
    )
    build_metric_plot(
        rows,
        "recall",
        Path(args.out_recall),
        "Per-Class Recall (dot size = support)",
        "#ff7f0e",
    )


if __name__ == "__main__":
    main()

