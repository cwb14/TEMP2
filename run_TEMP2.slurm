#!/bin/bash
#SBATCH --job-name=TEMP2_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=151680M
#SBATCH --time=4-00:00:00
#SBATCH --output=TEMP2_%A_%a.out
#SBATCH --error=TEMP2_%A_%a.err
#SBATCH --partition=shared
#SBATCH --array=0-344%10

set -euo pipefail

# -------------------------
# User inputs
# -------------------------
PREFIX_LIST="prefixes.txt"   # one prefix per line
REF="ref.fa"
CONS="LTRRT.consensus2.fa"
TARGET_BED="ref.fa.out2.bed6"
BWA_INDEX="bwa_index/ref.fa"

# -------------------------
# Pick this task's prefix
# -------------------------
prefix="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PREFIX_LIST")"
if [[ -z "${prefix:-}" ]]; then
  echo "ERROR: No prefix found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${PREFIX_LIST}" >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# -------------------------
# Quiet environment setup
# -------------------------
# Suppress module noise (including xalt "not unloaded") if module exists
if command -v module >/dev/null 2>&1; then
  module --force purge >/dev/null 2>&1 || true
fi

# Robust conda activate without 'conda init' requirement
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # common install path
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  # fallback: try conda base dynamically if conda is on PATH
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found and conda.sh not located." >&2
    exit 1
  fi
fi

conda activate TEMP2 >/dev/null 2>&1

threads="${SLURM_CPUS_PER_TASK:-1}"
outdir="${prefix}_TEMP2"

# -------------------------
# Detect input naming scheme
# -------------------------
left=""
right=""
expected_bed=""

if [[ -f "${prefix}_paired_1.fq.gz" && -f "${prefix}_paired_2.fq.gz" ]]; then
  left="${prefix}_paired_1.fq.gz"
  right="${prefix}_paired_2.fq.gz"
  expected_bed="${outdir}/${prefix}_paired.insertion.bed"
elif [[ -f "${prefix}_1.fastq.gz" && -f "${prefix}_2.fastq.gz" ]]; then
  left="${prefix}_1.fastq.gz"
  right="${prefix}_2.fastq.gz"
  expected_bed="${outdir}/${prefix}.insertion.bed"
else
  echo "SKIP: ${prefix} (no recognized FASTQ pairs found)" >&2
  exit 0
fi

# -------------------------
# Pickup-where-left-off
# -------------------------
if [[ -f "${expected_bed}" ]]; then
  linecount="$(wc -l < "${expected_bed}" || echo 0)"
  if [[ "${linecount}" -gt 1 ]]; then
    echo "SKIP: ${prefix} (already complete: ${expected_bed}, lines=${linecount})"
    exit 0
  fi
fi

# -------------------------
# Run TEMP2
# -------------------------
echo "RUN: ${prefix}  inputs=(${left}, ${right})  threads=${threads}"

bash TEMP2/TEMP2 insertion2 \
  -l "${left}" \
  -r "${right}" \
  -g "${REF}" \
  -R "${CONS}" \
  -t "${TARGET_BED}" \
  -o "${outdir}" \
  -I "${BWA_INDEX}" \
  -c "${threads}"

# Cleanup temp only if it exists
rm -rf "${outdir}/tmpTEMP2/" 2>/dev/null || true

echo "DONE: ${prefix}"

