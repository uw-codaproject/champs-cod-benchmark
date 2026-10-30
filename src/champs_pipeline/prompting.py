from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .config import PromptConfig
import pandas as pd

class PromptBuilder:
    """Constructs model-ready prompts from tabular + narrative inputs."""

    def __init__(
        self,
        config: PromptConfig,
        allowed_causes: Optional[List[str]] = None,
        format_instructions: Optional[str] = None,
    ):
        self.config = config
        self.allowed_causes = allowed_causes or []
        self.format_instructions = format_instructions

    def build_user_payload(self, example: Dict[str, Any]) -> str:
        """Constructs the user-facing case prompt based on enabled sections."""
        # 1. Gather all requested data into a structured dictionary
        case_data: Dict[str, Any] = {
            "champs_deid": str(example.get("case_id") or example.get("champs_deid") or "UNKNOWN_CASE_ID").strip(),
        }
        
        sections = self.config.include_sections or []
        for key in sections:
            if key == "diagnosis_standard":
                value = self.config.diagnosis_standard
            elif key == "narrative":
                raw_val = example.get(key)
                value = str(raw_val).strip() if not self._is_missing(raw_val) else "Narrative not provided."
            else:
                value = example.get(key)
            
            if not self._is_missing(value):
                case_data[key] = value

        # 2. Format the dictionary into a markdown-friendly string (flat structure)
        parts: List[str] = []
        
        # Pull Diagnosis Standard to the very top if requested
        if "diagnosis_standard" in case_data:
            parts.append(f"## DIAGNOSIS STANDARD\n{case_data.pop('diagnosis_standard')}")

        parts.append("## INPUT")
        parts.append(f"- case_id: {case_data.pop('champs_deid')}")
        
        if "narrative" in case_data:
            parts.append(f"- narrative:\n{self._indent_block(case_data.pop('narrative'), indent='  ')}")

        # Render all remaining sections flat
        # Show demographics early for readability if present
        if "demographics" in case_data:
            demo = case_data.pop("demographics")
            parts.append("- demographics:")
            labels = demo.get("__labels__", {}) if isinstance(demo, dict) else {}
            used_labels = set()
            for k, v in demo.items():
                if k == "__labels__":
                    continue
                label = labels.get(k, k)
                if label in used_labels:
                    label = f"{label} ({k})"
                used_labels.add(label)
                parts.append(f"    {label}: {v}")
        
        # Process remaining sections
        for key in list(case_data.keys()):
            value = case_data.pop(key)
            
            if key == "clinical_abstraction":
                parts.append("- clinical_abstraction:")
                if isinstance(value, dict):
                    for k, v in value.items(): parts.append(f"    {k}: {v}")
                else:
                    parts.append(self._indent_block(str(value), indent='    '))
            
            elif key == "mits":
                parts.append("- mits:")
                if isinstance(value, dict):
                    for subkey in ["lab_results", "tac_results"]:
                        results = value.get(subkey) or []
                        if results:
                            parts.append(f"    {subkey}:")
                            for item in results:
                                desc = item.get("description") or item.get("field")
                                parts.append(f"      - {desc}: {item.get('value')}")
            
            elif key in {"structured_va", "va"}:
                responses = value.get("responses") if isinstance(value, dict) else (value if isinstance(value, list) else [])
                if responses:
                    parts.append(f"- va_responses: {len(responses)} entries")
                    for resp in responses:
                        parts.append(f"    - {resp.get('question', '')} => {resp.get('answer', '')}")
            else:
                parts.append(f"- {key}:")
                parts.append(self._indent_block(str(value), indent='    '))

        return "\n".join(parts)

    def prepare_system_prompt(self) -> str:
        """Produces the rendered system prompt with allowed causes injected."""
        system_prompt = self.config.system_prompt
        if self.allowed_causes:
            system_prompt = system_prompt.replace("{allowed_causes}", ", ".join(self.allowed_causes))
        return system_prompt

    def _indent_block(self, text: str, indent: str = "  ") -> str:
        cleaned = (text or "").replace("\\n", "\n")
        return "\n".join(f"{indent}{line}" for line in cleaned.splitlines() or [""])

    def _is_missing(self, value: Any) -> bool:
        if value is None: return True
        if isinstance(value, str): return value.strip() == ""
        if pd is not None:
            try:
                if pd.isna(value): return True
            except Exception: pass
        return False
