from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import pandas as pd
from tqdm.auto import tqdm
import yaml
from champs_pipeline import (
    CHAMPSDatasetBuilder,
    Evaluator,
    DataConfig,
    PipelineConfig,
    PromptBuilder,
    load_cause_labels,
    PostProcessor,
)
from champs_pipeline.agents import MultiAgentOrchestrator
from champs_pipeline.model_runner import ModelRunner, ModelRunnerPool
from concurrent.futures import ThreadPoolExecutor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CHAMPS cause-of-death prediction pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to run (train/validation/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples to use from the split.",
    )
    parser.add_argument(
        "--export-prompts",
        action="store_true",
        help="Save prompts + labels to disk for remote inference.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Prepare data and prompts but skip model inference.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Randomly sample a fraction of the split before running (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=None,
        help="Override number of models/runners to use for inference.",
    )
    parser.add_argument(
        "--label-schema",
        type=str,
        default=None,
        help="Override data label schema (e.g., detailed).",
    )
    parser.add_argument(
        "--include-sections",
        type=str,
        default=None,
        help="Comma-separated prompt sections to include (overrides config).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (e.g., openai/gpt-oss-120b).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Split the split's cases across N independent jobs. Each job runs "
            "in its own process so results stay reproducible; merge afterwards "
            "with --merge-shards."
        ),
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Which shard this job handles, 0-based (case i => i %% num_shards).",
    )
    parser.add_argument(
        "--merge-shards",
        type=str,
        default=None,
        help=(
            "Comma-separated shard run dirs to merge. Runs no inference: reuses "
            "each case's saved generation and re-scores the full split. Pass the "
            "same config/section flags the shards used."
        ),
    )
    return parser.parse_args()


def save_prompt_example(path: Path, prompts: Dict[str, str]) -> None:
    """Save one or more prompts to a markdown file for inspection."""
    markdown = ""
    for key, value in prompts.items():
        markdown += f"# {key}\n\n{value}\n\n"
    path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _load_env()
    # Experimental: force deterministic kernels when CHAMPS_DETERMINISTIC=1.
    # No-op otherwise. Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in the env.
    if os.getenv("CHAMPS_DETERMINISTIC") == "1":
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        print("[determinism] use_deterministic_algorithms(warn_only=True) enabled", flush=True)
    config = PipelineConfig.from_yaml(args.config)
    if args.label_schema:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("data", {})["label_schema"] = args.label_schema
        config = PipelineConfig.from_dict(raw)
    if args.num_models is not None:
        config.model.num_models = args.num_models
    if args.include_sections is not None:
        sections = [s.strip() for s in args.include_sections.split(",") if s.strip()]
        config.prompt.include_sections = sections or None
    if args.model_name:
        config.model.name = args.model_name
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    base_output_dir = Path(config.evaluation.output_dir)
    output_dir = base_output_dir / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_builder = CHAMPSDatasetBuilder(config.data)
    dataset_dict = data_builder.build_splits(limit=args.limit)
    splits_dir = output_dir / "splits"
    try:
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df in dataset_dict.items():
            split_path = splits_dir / f"{split_name}.csv"
            split_df.to_csv(split_path, index=False)
        print(f"[Dataset] Saved split CSVs to {splits_dir}")
    except OSError as exc:
        print(f"[WARN] Failed to save split CSVs: {exc}")
    pipeline_copy_path = output_dir / "pipeline.yaml"
    pipeline_copy_path.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[Config] Saved pipeline config to {pipeline_copy_path}")

    train_df = dataset_dict.get("train", pd.DataFrame())
    label_values = train_df.get("label", pd.Series(dtype=object)).tolist()
    immediate_values = train_df.get("immediate_label", pd.Series(dtype=object)).tolist()
    comorbidity_values = train_df.get("comorbidity_labels", pd.Series(dtype=object)).tolist()

    # De-duplicate labels case-insensitively but keep first-seen casing.
    label_map: Dict[str, str] = {}
    for label in label_values:
        _add_label_to_map(label_map, label)
    for label in immediate_values:
        _add_label_to_map(label_map, label)
    for label in comorbidity_values:
        _add_label_to_map(label_map, label)
    base_causes = sorted(label_map.values())
    focus_list = config.prompt.allowed_causes
    focus_base: List[str] = []
    if focus_list:
        focus_base = _normalize_focus_list(focus_list)
        allowed_causes = list(focus_base)
    else:
        allowed_causes = list(base_causes)
    print(f"Allowed causes: len={len(allowed_causes)}, class={allowed_causes}")
    if args.split not in dataset_dict:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(dataset_dict.keys())}")

    dataset_df = dataset_dict[args.split]
    print(f"Dataset size before sampling: {len(dataset_df)}")
    if args.sample_frac:
        if not 0 < args.sample_frac <= 1:
            raise ValueError("--sample-frac must be between 0 and 1.")
        dataset_df = dataset_df.sample(
            frac=args.sample_frac, random_state=config.data.random_seed
        )
        print(f"Dataset size after sample_frac={args.sample_frac}: {len(dataset_df)}")
    if args.limit:
        dataset_df = dataset_df.head(args.limit)
        print(f"Dataset size after limit={args.limit}: {len(dataset_df)}")
    final_dataset_path = output_dir / "final_dataset.csv"
    try:
        dataset_df.to_csv(final_dataset_path, index=False)
        print(f"[Dataset] Saved final dataset to {final_dataset_path}")
    except OSError as exc:
        print(f"[WARN] Failed to save final dataset CSV: {exc}")
    _print_label_distribution(dataset_df)
    prompt_builder = PromptBuilder(
        config.prompt,
        allowed_causes=allowed_causes,
    )
    # Ensure system prompt has allowed_causes injected before use everywhere.
    config.prompt.system_prompt = prompt_builder.prepare_system_prompt()
    records = []
    for record in dataset_df.to_dict(orient="records"):
        user_payload = prompt_builder.build_user_payload(record)
        records.append({**record, "user_payload": user_payload, "prompt": user_payload})

    # Case-level sharding across independent jobs. The dataset, its order and
    # every prompt are built identically in all shards; only the subset each
    # job generates for differs. Selection is index-based (i % num_shards) so
    # it does not depend on timing, and the shards partition the split exactly.
    if args.num_shards > 1 and not args.merge_shards:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError(
                f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
            )
        total_cases = len(records)
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_id]
        print(
            f"[Shard] {args.shard_id}/{args.num_shards}: "
            f"{len(records)} of {total_cases} cases"
        )

    prompts = [record["user_payload"] for record in records]
    # write the first system + user payload to a file for inspection
    generations: List[str] = []
    first_prompt = True

    use_multi_agent = getattr(config.model, "multi_agent", False)
    if args.merge_shards:
        # Re-score a full split from per-shard generations. No GPU work: the
        # shards already produced each case's raw output, so this only has to
        # reassemble them in the canonical record order and run the normal
        # post-processing/eval path.
        id_column = config.data.id_column
        shard_dirs = [d.strip() for d in args.merge_shards.split(",") if d.strip()]
        gen_by_id: Dict[str, str] = {}
        for shard_dir in shard_dirs:
            preds_path = Path(shard_dir) / "predictions.jsonl"
            n_before = len(gen_by_id)
            with open(preds_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    case_id = row[id_column]
                    if case_id in gen_by_id:
                        raise ValueError(
                            f"Case {case_id} appears in more than one shard; "
                            "shards must partition the split."
                        )
                    gen_by_id[case_id] = row["generation"]
            print(f"[Merge] {shard_dir}: {len(gen_by_id) - n_before} cases")

        missing = [r[id_column] for r in records if r[id_column] not in gen_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(records)} cases missing from the shard "
                f"outputs (e.g. {missing[:3]}). Did every shard job finish?"
            )
        extra = len(gen_by_id) - len(records)
        if extra:
            raise ValueError(
                f"Shard outputs hold {extra} case(s) not in this split; the "
                "merge flags must match the flags the shards ran with."
            )
        generations = [gen_by_id[r[id_column]] for r in records]
        print(f"[Merge] reassembled {len(generations)} cases from "
              f"{len(shard_dirs)} shards")
    elif use_multi_agent:
        orchestrator = MultiAgentOrchestrator(config.model, config.prompt, allowed_causes)
        orchestrator.save_markdown(output_dir / "graph.md")
        
        # Batch size for cross-case parallelism.
        # We can run as many cases in parallel as we have runners (GPUs).
        num_models = config.model.num_models if config.model.num_models > 0 else 1
        
        print(f"[Pipeline] Classifying {len(records)} cases in chunks of {num_models}...")
        for i in tqdm(range(0, len(records), num_models), desc="Classifying cases"):
            chunk = records[i : i + num_models]
            chunk_payloads = [r["user_payload"] for r in chunk]
            results = orchestrator.classify_batch(chunk_payloads)
            
            for (pred, prompts_dict), record in zip(results, chunk):
                generations.append(pred)
                if first_prompt:
                    save_prompt_example(output_dir / "prompt.md", prompts_dict)
                    first_prompt = False
    else:
        pool = ModelRunnerPool(config.model, config.prompt)
        pool.load()

        # Batch size for cross-case parallelism.
        num_models = config.model.num_models if config.model.num_models > 0 else 1

        print(f"[Pipeline] Classifying {len(records)} cases using single-agent pool (size={num_models})...")

        # Static (deterministic) sharding for reproducibility with parallelism:
        # case i is ALWAYS handled by replica (i % num_models), in fixed order,
        # so each case's output is stable run-to-run regardless of thread timing.
        # (A dynamic pool.get() queue assigns cases to whichever GPU is free
        # first, which is timing-dependent and therefore non-reproducible.)
        generations = [None] * len(records)
        pinned_runners = [pool.get() for _ in range(num_models)]

        # Warm up each replica SEQUENTIALLY before parallelizing. The gpt-oss
        # MoE router uses a Triton kernel that is JIT-compiled/loaded on first
        # use; two shard threads hitting that concurrently races on the shared
        # Triton cache and triggers "illegal memory access". Compiling once per
        # runner up front makes the later parallel launches load cached binaries.
        if records:
            warmup_prompt = records[0]["user_payload"]
            for wi, runner in enumerate(pinned_runners):
                print(f"[warmup] compiling kernels on replica {wi}...", flush=True)
                runner.predict_prompt(warmup_prompt)

        def run_shard(shard_id: int) -> None:
            runner = pinned_runners[shard_id]
            idxs = list(range(shard_id, len(records), num_models))
            if not idxs:
                return
            shard_prompts = [records[i]["user_payload"] for i in idxs]
            gens, ptexts = runner.predict_batches(shard_prompts)
            for i, g in zip(idxs, gens):
                generations[i] = g
            if shard_id == 0 and ptexts:
                save_prompt_example(output_dir / "prompt_0.md", {"Single Agent": ptexts[0]})

        try:
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                list(tqdm(executor.map(run_shard, range(num_models)),
                          total=num_models, desc="Classifying shards"))
        finally:
            for runner in pinned_runners:
                pool.put(runner)

    post_processor = PostProcessor(allowed_causes)
    (
        generations,
        normalized_references,
        normalized_predictions,
        normalized_topk,
        normalized_decode_targets,
    ) = post_processor.post_process_generations_and_references(records, generations)
    evaluator = Evaluator(output_dir, allowed_labels=post_processor.allowed_labels)
    parsed_generations = [None for _ in generations]
    result = evaluator.evaluate(
        normalized_references,
        normalized_predictions,
        topk_predictions=normalized_topk,
        decode_targets=normalized_decode_targets,
        raw_outputs=generations,
    )

    log_run_metrics(
        output_dir=output_dir,
        config_path=args.config,
        model_name=config.model.name,
        split=args.split,
        num_cases=len(records),
        age_group_filter=_describe_age_filter(config.data),
        inputs_included=", ".join(config.prompt.include_sections or []),
        metrics=result.metrics,
    )

    save_predictions(
        records,
        generations,
        parsed_generations,
        normalized_predictions,
        output_dir,
        args.split,
        config.data.id_column,
    )

    print("Evaluation metrics:")
    for metric, value in result.metrics.items():
        if metric == "per_class":
            continue
        if isinstance(value, dict):
            if metric == "decode_metrics":
                value = {k: v for k, v in value.items() if k != "per_class"}
            print(f"- {metric}: {json.dumps(value)}")
            continue
        try:
            print(f"- {metric}: {value:.4f}")
        except (TypeError, ValueError):
            print(f"- {metric}: {value}")


def log_run_metrics(
    output_dir: Path,
    config_path: str,
    model_name: str,
    split: str,
    num_cases: int,
    metrics: Dict[str, Any],
    age_group_filter: str = "",
    inputs_included: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics_log.jsonl"
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": str(config_path),
        "model": model_name,
        "split": split,
        "num_cases": num_cases,
        "age_group_filter": age_group_filter,
        "inputs_included": inputs_included,
        "metrics": metrics,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"Appended metrics to {log_path}")


def save_predictions(
    records,
    generations,
    parsed_generations,
    normalized_predictions,
    output_dir: Path,
    split_name: str,
    id_column: str,
) -> None:
    if not generations:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for example, raw, parsed, normalized in zip(
            records,
            generations,
            parsed_generations,
            normalized_predictions,
        ):
            record: Dict[str, Any] = {
                id_column: example[id_column],
                "reference_label": example.get("label", ""),
                "reference_immediate": example.get("immediate_label", ""),
                "reference_comorbidities": example.get("comorbidity_labels", []),
                "prompt": example["prompt"],
                "generation": raw,
                "predicted_label": normalized,
            }
            if parsed is not None:
                record["parsed_generation"] = parsed
            f.write(json.dumps(record) + "\n")
    print(f"Wrote predictions to {path}")


def _resolve_focus_label(name: str, allowed: List[str]) -> str:
    normalized_lookup = {label.strip().lower(): label for label in allowed}
    target = str(name).strip().lower()
    if target in normalized_lookup:
        return normalized_lookup[target]
    raise ValueError(
        f"Focus label '{name}' not found in allowed cause list. "
        "Please ensure it exactly matches a cod_labels entry."
    )


def _normalize_focus_list(value: Any) -> List[str]:
    if value is None:
        return []
    
    # If it's a single string, check if it's a file or just one label.
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        path = Path(stripped)
        if path.exists() and path.is_file():
            return load_cause_labels(path)
        return [stripped]
    
    # If it's a list, check if the FIRST element is a file path.
    # This handles the case where the YAML parser wrapped the string in a list.
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], str):
            path = Path(value[0].strip())
            if path.exists() and path.is_file():
                return load_cause_labels(path)
        return [str(v).strip() for v in value if str(v).strip()]
    
    return []


def _describe_age_filter(data_cfg: DataConfig) -> str:
    if data_cfg.include_age_groups:
        return ", ".join(data_cfg.include_age_groups)
    if data_cfg.age_group_keywords:
        return ", ".join(data_cfg.age_group_keywords)
    return "none"

def _add_label_to_map(label_map: Dict[str, str], label: Any) -> None:
    if label is None:
        return
    if isinstance(label, (list, tuple, set)):
        for item in label:
            _add_label_to_map(label_map, item)
        return
    if isinstance(label, str):
        cleaned = label.strip()
        if not cleaned:
            return
    else:
        if pd.isna(label):
            return
        cleaned = str(label).strip()
        if not cleaned:
            return
    key = cleaned.lower()
    if key not in label_map:
        label_map[key] = cleaned

def _print_label_distribution(df: pd.DataFrame) -> None:
    if "label" not in df.columns or df.empty:
        print(f"[INFO] Dataframe is empty or has no label column.")
    counts = df["label"].value_counts(dropna=False)
    total = len(df)
    print(f"[INFO] Label distribution (n={total}):")
    for label, count in counts.items():
        pct = (count / total) * 100 if total else 0.0
        print(f"  - {label}: {count} ({pct:.1f}%)")

def _load_env() -> None:
    """
    Load .env from repo root (one level above src). Lightweight, no external deps.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        print(f"[INFO] .env file not found at {env_path}")
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

if __name__ == "__main__":
    main()
