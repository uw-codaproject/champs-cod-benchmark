#!/bin/bash
set -euo pipefail

# Generate and submit one multi-replica job per model x label schema x section
# combo.
#
# NOTE: this is the legacy launcher and it is NOT verified reproducible. It runs
# several model replicas inside one process (--num-models > 1). Cases are pinned
# to replicas by index, but replicas share one process and its GPU kernels, and
# only one replica per process was verified to give identical outputs across
# runs. The results released with the paper were produced by single-replica
# shard jobs instead; see "Reproducing the paper results" in README.md.

cd "$(dirname "$0")"

: "${CHAMPS_CONDA_ENV:?set CHAMPS_CONDA_ENV to your conda env path, e.g. /path/to/envs/champs-gpu}"
: "${CHAMPS_HF_HOME:?set CHAMPS_HF_HOME to your Hugging Face cache root}"
: "${CHAMPS_MAIL_USER:=}"

ENV_BLOCK="module load conda
conda activate ${CHAMPS_CONDA_ENV}
export HF_HOME=${CHAMPS_HF_HOME}
export HF_HUB_CACHE=${CHAMPS_HF_HOME}/transformers
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

if [[ -n "${CHAMPS_MAIL_USER}" ]]; then
  MAIL_BLOCK="#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${CHAMPS_MAIL_USER}"
else
  MAIL_BLOCK=""
fi

models=("openai/gpt-oss-120b" "openai/gpt-oss-20b")
label_schemas=("group")
sections=(
  "structured_va,narrative,demographics,mits,clinical_abstraction"
  "structured_va,narrative,demographics,mits"
  "structured_va,narrative,demographics"
  "structured_va,demographics"
  "structured_va"
)

job_idx=1
for model in "${models[@]}"; do
  # 120b OOMs at 4 replicas/GPU (eager attention on long inputs); halve it.
  # 20b is small (~13GB/copy) so keep full parallelism.
  if [[ "${model}" == *120b* ]]; then
    num_models=2
  else
    num_models=4
  fi
  for schema in "${label_schemas[@]}"; do
    for include in "${sections[@]}"; do
      job_file="job${job_idx}.slurm"
      cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --job-name=champs-llm-job${job_idx}
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
${MAIL_BLOCK}

${ENV_BLOCK}
python src/run_pipeline.py --split test --model-name ${model} --num-models ${num_models} --label-schema ${schema} --include-sections ${include}
EOF

      sbatch "${job_file}"
      job_idx=$((job_idx + 1))
    done
  done
done

