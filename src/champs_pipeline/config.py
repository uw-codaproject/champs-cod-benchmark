from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    root_dir: str = "data/CHAMPS_de_identified_data_new"
    narrative_file: str = "CHAMPS_deid_verbal_autopsy.csv"
    label_file: Optional[str] = None
    decode_file: str = "CHAMPS_deid_decode_results.csv"
    demographics_file: str = "CHAMPS_deid_basic_demographics.csv"
    structured_subset_file: Optional[str] = "outputs/structured_subset.json"
    id_column: str = "champs_deid"
    narrative_column: str = "id10476_deid"
    label_column: str = "Underlying_Cause"
    immediate_label_column: Optional[str] = "Immediate_COD"
    comorbidity_columns: List[str] = field(
        default_factory=lambda: [
            "Morbid_Condition_01",
            "Morbid_Condition_02",
            "Morbid_Condition_03",
            "Morbid_Condition_04",
            "Morbid_Condition_05",
            "Morbid_Condition_06",
            "Morbid_Condition_07",
            "Morbid_Condition_08",
        ]
    )
    map_icd_codes: bool = True
    label_schema: Optional[str] = None
    label_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stratify_column: Optional[str] = None
    age_group_column: str = "age_group"
    include_age_groups: Optional[List[str]] = field(default_factory=list)
    age_group_keywords: Optional[List[str]] = field(
        default_factory=lambda: ["infant", "child"]
    )
    age_group_required: bool = True
    text_min_chars: int = 30
    max_samples: Optional[int] = None
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42
    clinical_abstraction_dir: Optional[str] = "ClinicalAbstraction/ChildAbst"
    clinical_abstraction_glob: str = "*.csv"
    clinical_abstraction_case_id_col: Optional[str] = None
    clinical_abstraction_text_cols: Optional[List[str]] = None
    mits_lab_file: str = "CHAMPS_deid_lab_results.csv"
    mits_tac_file: str = "CHAMPS_deid_tac_results.csv"
    mits_lab_prefixes: List[str] = field(
        default_factory=lambda: ["bld_", "csf_", "HIV_DNA_", "Mal_", "TB_"]
    )
    mits_tac_prefixes: List[str] = field(
        default_factory=lambda: ["bld_", "bld_sp_", "CSF_", "lung_", "rect_swab_", "NP_OP_"]
    )
    question_map: Dict[str, str] = field(default_factory=dict)
    lab_test_map: Dict[str, str] = field(default_factory=dict)
    tac_assay_map: Dict[str, str] = field(default_factory=dict)
    demo_cols: List[str] = field(default_factory=list)
    va_cols: List[str] = field(default_factory=list)

    def root_path(self) -> Path:
        return Path(self.root_dir).resolve()

    def narrative_path(self) -> Path:
        return self.root_path() / self.narrative_file

    def label_path(self) -> Path:
        if self.label_file:
            return self.root_path() / self.label_file
        return self.root_path() / self.decode_file


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-14B-Instruct"
    revision: Optional[str] = None
    provider: str = "local"  # "local" | "api"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "API_KEY_ENV"
    dtype: str = "bfloat16"
    use_4bit: bool = True
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9
    num_models: int = 2
    inference_batch_size: int = 4
    multi_agent: bool = False


@dataclass
class PromptConfig:
    system_prompt: str = (
        "You are an expert physician participating in the CHAMPS consortium. "
        "Given a verbal autopsy narrative, determine the most likely underlying "
        "cause of death strictly from the provided list of CHAMPS categories."
    )
    schema_guidance: Dict[str, str] = field(default_factory=dict)
    allowed_causes: Optional[List[str] | str] = None
    decision_prompt: Optional[str] = None
    specialist_prompt: Optional[str] = None
    agent_strategy: str = "disease"
    diagnosis_standard: Optional[str] = None
    include_sections: Optional[List[str]] = None
    role_agents: List[Dict[str, Any]] = field(default_factory=list)
    role_specialist_prompt: Optional[str] = None
    panel_chair_prompt: Optional[str] = None
    disease_decision_prompt: Optional[str] = None
    disease_specialist_prompt: Optional[str] = None


@dataclass
class EvalConfig:
    output_dir: str = "outputs"

    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        data_raw = dict(raw.get("data", {}) or {})
        label_schema = data_raw.get("label_schema")
        label_schemas = data_raw.get("label_schemas") or {}
        if label_schema and isinstance(label_schemas, dict):
            schema_overrides = label_schemas.get(label_schema)
            if isinstance(schema_overrides, dict):
                data_raw = {**data_raw, **schema_overrides}
        data_cfg = DataConfig(**data_raw)

        model_cfg = ModelConfig(**raw.get("model", {}))

        prompt_raw = dict(raw.get("prompt", {}) or {})
        prompt_cfg = PromptConfig(**prompt_raw)

        schema_key = data_cfg.label_schema or "detailed"
        if prompt_cfg.schema_guidance:
            guidance = prompt_cfg.schema_guidance.get(schema_key)
            if guidance:
                if "{schema_guidance}" in prompt_cfg.system_prompt:
                    prompt_cfg.system_prompt = prompt_cfg.system_prompt.replace(
                        "{schema_guidance}", guidance.strip()
                    )
                else:
                    prompt_cfg.system_prompt = (
                        prompt_cfg.system_prompt.rstrip()
                        + "\n\n"
                        + guidance.strip()
                    )

        if prompt_cfg.agent_strategy not in {"disease", "role"}:
            prompt_cfg.agent_strategy = "disease"

        eval_cfg = EvalConfig(**raw.get("evaluation", {}))

        # Normalize allowed causes to List[str] or None
        prompt_cfg.allowed_causes = cls._normalize_list(prompt_cfg.allowed_causes)

        return cls(
            data=data_cfg,
            model=model_cfg,
            prompt=prompt_cfg,
            evaluation=eval_cfg,
        )

    @staticmethod
    def _normalize_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else None
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            return cleaned or None
        return None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
