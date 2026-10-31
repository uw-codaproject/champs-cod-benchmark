#!/bin/bash
#SBATCH --job-name=champs-llm
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Example of an ad-hoc multi-replica run. This is NOT the reproducible path:
# multi-replica inference inside one process is not verified reproducible. To
# reproduce the released results, see "Reproducing the paper results" in
# README.md.

module load conda
conda activate "${CHAMPS_CONDA_ENV:?set CHAMPS_CONDA_ENV to your conda env path}"
export HF_HOME="${CHAMPS_HF_HOME:?set CHAMPS_HF_HOME to your Hugging Face cache root}"
export HF_HUB_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/run_pipeline.py --split test --model-name openai/gpt-oss-120b --num-models 4 --label-schema group --include-sections structured_va,narrative,demographics,mits,clinical_abstraction
