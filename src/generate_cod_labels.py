from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set


def extract_unique_labels(csv_path: Path, column_name: str) -> List[str]:
    """
    Extract unique values from a specific column, ensuring they are capitalized.
    """
    labels: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column_name not in (reader.fieldnames or []):
            raise ValueError(f"Column '{column_name}' not found in CSV.")
        
        for row in reader:
            val = row.get(column_name, "")
            if not val:
                continue
            
            # Clean and ensure it starts with a capital letter
            clean = val.strip()
            if clean:
                # Ensure the first character is uppercase, keep rest as is
                # or use capitalize() if the whole string should be lowered first.
                # User said "Each category should start with a capitalized letter".
                formatted = clean[0].upper() + clean[1:]
                labels.add(formatted)
                
    return sorted(list(labels))


def write_labels(labels: List[str], output_path: Path) -> None:
    """
    Write labels line by line without indices.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unique cause-of-death labels from the decode results CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/CHAMPS_de_identified_data_new/CHAMPS_deid_decode_results.csv"
        ),
        help="Path to CHAMPS_deid_decode_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/CHAMPS_de_identified_data_new/outputs"),
        help="Directory where to save the generated label files.",
    )
    args = parser.parse_args()

    # Define targets
    targets = [
        ("UC_champs_group_desc", "cod_labels_group_new.txt"),
        ("Underlying_Cause", "cod_labels_detailed_new.txt"),
    ]

    for col, filename in targets:
        output_file = args.output_dir / filename
        labels = extract_unique_labels(args.input, col)
        write_labels(labels, output_file)
        print(f"Wrote {len(labels)} labels from '{col}' to {output_file.resolve()}")


if __name__ == "__main__":
    main()
