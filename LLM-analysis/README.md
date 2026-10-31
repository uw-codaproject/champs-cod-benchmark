# LLM-analysis

Code for the LLM-based analysis in our NeurIPS 2026 Evaluations and Datasets
Track paper. The statistical and
machine learning baselines (LightGBM, Elastic Net, InSilicoVA) and the CHAMPS
preprocessing for them are in the companion folder.

LLM-based cause-of-death (COD) classification for the CHAMPS (Child Health and Mortality Prevention Surveillance) network. The pipeline ingests verbal-autopsy narratives, structured VA responses, demographics, MITS lab/TAC results, and clinical abstraction records, then uses local LLMs to predict the underlying cause of death.

Run every command below from this folder (`LLM-analysis/`).

### Environment and compute


- **Hardware:** NVIDIA H200 (141 GB), one GPU per job, 8 CPUs and 100–200 GB
  host RAM per job, Rocky 9, Slurm.
- **Compute:** 34.73 H200 GPU-hours for the released set (40 inference shard
  jobs, 34.63 h; 10 merge jobs, 0.10 h).
- **Software:** install `requirements-freeze.txt`, the exact environment of the
  runs (Python 3.11.14, torch 2.9.1, transformers 4.57.6, tokenizers 0.22.1,
  triton 3.5.1, kernels 0.11.7, numpy 1.26.4).

  ```bash
  conda create -p /path/to/envs/champs-gpu python=3.11.14
  conda activate /path/to/envs/champs-gpu
  pip install -r requirements-freeze.txt
  ```

- **Model weights,** pinned by Hugging Face revision. Download both snapshots
  first and run with `HF_HUB_OFFLINE=1`, so a newer upload cannot replace them.

  | model | revision |
  |-------|----------|
  | `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
  | `openai/gpt-oss-20b`  | `6cee5e81ee83917806bbde320786a8fb61efebee` |

### Data

The CHAMPS de-identified data are not in this repository (`data/` is
git-ignored). Access is granted by the CHAMPS network on application:
https://champshealth.org/data/. Place the release under
`data/CHAMPS_de_identified_data_new/` with the file names given in
`configs/pipeline.yaml` (`data.narrative_file`, `data.decode_file`,
`data.demographics_file`, `data.mits_lab_file`, `data.mits_tac_file`,
`data.clinical_abstraction_dir`). The 252-case test split is rebuilt from the
data by `CHAMPSDatasetBuilder` (`random_seed: 31`, `test_size: 0.2`), so no
split file is needed.

### Running

Each configuration runs as four single-GPU shard jobs, each in its own process
with one model replica (`--num-models 1`). Case `i` goes to shard `i % 4`. A
merge job then joins the four shard run directories and scores the full split;
it runs no inference. For example, cfg5:

```bash
export HF_HOME=/path/to/huggingface HF_HUB_CACHE=/path/to/huggingface/transformers
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# one job per shard, s = 0..3; each writes outputs/pipeline/run_<timestamp>/
python src/run_pipeline.py --split test --model-name openai/gpt-oss-120b \
    --num-models 1 --label-schema group --include-sections structured_va \
    --num-shards 4 --shard-id $s

# after all four finish, with the same model and section flags
python src/run_pipeline.py --split test --model-name openai/gpt-oss-120b \
    --label-schema group --include-sections structured_va \
    --merge-shards outputs/pipeline/run_A,outputs/pipeline/run_B,outputs/pipeline/run_C,outputs/pipeline/run_D
```

Shards take about 2 h (120b) or 4.5 h (20b). The merge fails if shards overlap,
a case is missing, or the flags differ from the shards'.


## Pipeline overview

`src/run_pipeline.py` orchestrates the full workflow:

1. **Data prep** – `CHAMPSDatasetBuilder` merges verbal-autopsy narratives with decode results, demographics, MITS lab/TAC findings, and clinical abstraction records. Two label schemas are supported: `detailed` (ICD-level causes with automatic ICD-code-to-name mapping) and `group` (CHAMPS grouped cause categories). Age-group filtering (e.g. infants, children) is configurable.
2. **Prompting** – `PromptBuilder` assembles each case into a chat prompt. The prompt can include any combination of input sections (`narrative`, `structured_va`, `demographics`, `mits`, `clinical_abstraction`, `diagnosis_standard`) controlled by `prompt.include_sections`. Schema-specific clinical guidance and a diagnosis-standard rubric are injected into the system prompt.
3. **Inference** – Two execution modes:
   - **Single-agent**: `ModelRunnerPool` dispatches cases across one or more model instances. Supports local Hugging Face models (with optional 4-bit quantization) and remote OpenAI-compatible APIs.
   - **Multi-agent** (LangGraph): `MultiAgentOrchestrator` runs either a *disease* strategy (per-cause specialist agents + decision agent) or a *role* strategy (six domain-expert panelists + panel chair). Enable with `model.multi_agent: true`.
4. **Post-processing** – `PostProcessor` extracts top-k cause predictions from JSON generations, normalizes labels, and resolves aliases.
5. **Evaluation** – `Evaluator` computes top-1/top-3 accuracy, macro/micro precision/recall/F1, per-class metrics, and a confusion matrix. When the decode reference matches a top-k prediction, decode metrics (measuring anywhere-in-chain) are also reported.

## Project structure

```
LLM-analysis/
├── configs/
│   └── pipeline.yaml              # main configuration
├── src/
│   ├── run_pipeline.py            # entry point
│   ├── test_model.py              # model smoke test
│   ├── match_icd_labels.py        # ICD code → name mapping utility
│   ├── generate_cod_labels.py     # extract COD labels from decode CSV
│   ├── debug_counts.py            # debug dataset filtering
│   └── champs_pipeline/
│       ├── config.py              # dataclass configs (Pipeline/Data/Model/Prompt/Eval)
│       ├── data.py                # CHAMPSDatasetBuilder, PostProcessor
│       ├── prompting.py           # PromptBuilder
│       ├── model_runner.py        # ModelRunner, ModelRunnerPool
│       ├── evaluation.py          # Evaluator
│       ├── labels.py              # load_cause_labels
│       └── agents.py              # MultiAgentOrchestrator (LangGraph)
├── report-scripts/
│   └── plot_precision_from_metrics.py
├── run_all_jobs.sh                # legacy launcher (multi-replica, NOT reproducible)
├── example.slurm                  # example SLURM job script
├── requirements-freeze.txt        # exact environment of the released runs
├── requirements-core.txt
├── requirements-gpu.txt
├── requirements-langchain.txt
└── data/                          # CHAMPS de-identified data (not tracked)
```

## Setup

```bash
conda create -n champs-llm python=3.11
conda activate champs-llm
pip install -r requirements-core.txt        # core HF / transformers / pandas stack
pip install -r requirements-gpu.txt         # optional: GPU-validated torch + bitsandbytes
pip install -r requirements-langchain.txt   # optional: LangChain (for multi-agent mode)
```

To reproduce the released results, install `requirements-freeze.txt`  — it is the exact environment the runs used, down to the patch version. See [Environment and compute](#environment-and-compute).

Model weights are fetched at runtime from Hugging Face (or via an API endpoint). For API-based models, create a `.env` file in `LLM-analysis/` with your key:

```
API_KEY_ENV=your-key-here
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config (default `configs/pipeline.yaml`) |
| `--split` | Dataset split: `train`, `validation`, or `test` |
| `--limit` | Max samples from the split |
| `--sample-frac` | Random fraction of the split (e.g. `0.1`) |
| `--skip-generation` | Prepare data/prompts but skip inference |
| `--export-prompts` | Save prompts to disk |
| `--model-name` | Override model name |
| `--label-schema` | Override label schema (`detailed` or `group`) |
| `--include-sections` | Comma-separated prompt sections to include |
| `--num-models` | Override number of parallel model runners |
| `--num-shards`, `--shard-id` | Run shard `shard-id` of `num-shards` (case `i` goes to shard `i % num-shards`) |
| `--merge-shards` | Comma-separated shard run dirs to merge and score; runs no inference |

## Outputs

Each run writes to `outputs/pipeline/run_<timestamp>/`:

| File | Contents |
|------|----------|
| `predictions.jsonl` | case_id, reference label, prompt, raw generation, predicted label |
| `confusion_matrix.csv` | per-class confusion matrix |
| `metrics_log.jsonl` | appended metrics record for cross-run comparison |
| `splits/` | train/validation/test CSVs as used for this run |
| `final_dataset.csv` | the exact subset fed to inference |
| `pipeline.yaml` | snapshot of the config used |
| `prompt*.md` | rendered prompt examples for inspection |
| `graph.md` | Mermaid diagram of the multi-agent graph (multi-agent only) |

## Customizing the pipeline

All configuration lives in `configs/pipeline.yaml`:

- **Data sources** – point to narrative, decode, demographics, MITS lab, MITS TAC, and clinical abstraction files. `lab_test_map` and `tac_assay_map` provide human-readable names for coded lab/TAC columns.
- **Label schema** – `detailed` maps ICD codes to cause names; `group` uses CHAMPS grouped cause descriptions. Both schemas support underlying, immediate, and comorbidity columns.
- **Age filtering** – restrict to specific age groups (e.g. infants, children) via `include_age_groups` or keyword matching.
- **Prompt sections** – control which evidence types are shown to the model via `prompt.include_sections`.
- **Diagnosis standard** – a rubric (Level 1/2/3 certainty) for malnutrition, malaria, LRI, congenital defects, diarrheal diseases, HIV, sepsis, and neonatal preterm complications, injected into the prompt.
- **Multi-agent** – set `model.multi_agent: true` and choose `prompt.agent_strategy`: `disease` (per-cause specialists) or `role` (expert panelists with configurable personas under `prompt.role_agents`).
- **Model** – switch between local HF models and remote APIs (`provider: local` vs `api`), adjust decoding hyperparameters, quantization, and parallelism.


## Citation

Paper: https://openreview.net/forum?id=8qRMs3zaOJ

Authors: 

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=8qRMs3zaOJ}
}
```

## License and third-party assets

The code in this repository is released under the MIT License (`LICENSE`),
copyright University of Washington.
That license covers this code only — not the models it calls, and not the
data it reads. Each third-party asset keeps its own terms:

| Asset | Terms | Notes |
|-------|-------|-------|
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | Apache License 2.0, plus the usage policy shipped in the model repository (`USAGE_POLICY`) | Used for inference only, unmodified, at the revisions pinned in [Environment and compute](#environment-and-compute) |
| CHAMPS de-identified data | CHAMPS data access agreement | Not redistributed here; obtained on application to the CHAMPS network. See [Data](#data) |

The CHAMPS release used here is the network's **de-identified** data set,
obtained under its data access agreement and not included in this repository
— `data/` is git-ignored and no case-level data, identifiers, or model
outputs over real cases are tracked here. Anyone reproducing this work needs
their own access agreement; the pipeline cannot run without it.
