"""
CHAMPS LLM pipeline package.

Provides data loading, prompt construction, inference, and evaluation tools
for experimenting with DeepSeek-R1-Distill-Qwen-7B on cause-of-death tasks.
"""

from .config import PipelineConfig, DataConfig, ModelConfig, PromptConfig, EvalConfig
from .data import CHAMPSDatasetBuilder, PostProcessor
from .prompting import PromptBuilder
from .evaluation import Evaluator, EvaluationResult
from .labels import load_cause_labels

__all__ = [
    "PipelineConfig",
    "DataConfig",
    "ModelConfig",
    "PromptConfig",
    "EvalConfig",
    "CHAMPSDatasetBuilder",
    "PostProcessor",
    "PromptBuilder",
    "Evaluator",
    "EvaluationResult",
    "load_cause_labels",
]
