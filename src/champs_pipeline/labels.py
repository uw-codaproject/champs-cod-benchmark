from __future__ import annotations

import re
from pathlib import Path
from typing import List


def load_cause_labels(path: str | Path) -> List[str]:
    """
    Load allowed cause-of-death labels from a comma-separated or newline-separated file.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Cause label file not found: {resolved}")

    raw_text = resolved.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Label file is empty: {resolved}")

    # Heuristic: If there are multiple lines, assume it's line-separated.
    # This prevents labels containing commas from being split incorrectly.
    if "\n" in raw_text:
        labels = [line.strip() for line in raw_text.splitlines()]
    elif "," in raw_text:
        labels = [item.strip() for item in raw_text.split(",")]
    else:
        labels = [raw_text]

    cleaned: List[str] = []
    for label in labels:
        if not label:
            continue
        # Remove numeric prefixes like "13. " and trailing commas.
        normalized = re.sub(r"^\s*\d+\.\s*", "", label)
        normalized = normalized.rstrip(",").strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)

    labels = cleaned
    if not labels:
        raise ValueError(f"No labels found inside {resolved}")
    return labels

