#!/bin/bash
##############################################################################
# TEMPLATE: Generic TSV Dataset Execution Script
# USAGE: sbatch --export=DATASET_PATH='datasets/tsv/YOUR_DATASET.tsv',OUTPUT_DIR='results/your_output' run_bc_tsv_template.sbatch
# OR: Edit DATASET_PATH and OUTPUT_DIR below and run: sbatch run_bc_tsv_template.sbatch
##############################################################################

#SBATCH --job-name=roadNet-CA
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ========== USER CONFIGURATION (Edit these) ==========
DATASET_PATH="${DATASET_PATH:-/data/Dataset/gobardhan-datasets/roadNet-CA_adj.tsv}"     # ← CHANGE THIS
OUTPUT_DIR="${OUTPUT_DIR:-results/}"                              # ← CHANGE THIS
BATCH_SIZE="${BATCH_SIZE:-32}"                                          # ← OPTIONAL: Batch size

# ====================================================

set -euo pipefail

PROJECT_DIR="/data/gobardhan/LSDM-V3-Back_Optimized"
cd "${PROJECT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Load required modules
module load gcc-12.3.0 cuda-12.3 2>/dev/null || true

OUTPUT_FILE="${OUTPUT_DIR}/$(basename "${DATASET_PATH}" .tsv)_BC.txt"

echo "======================================================================"
echo "BC Computation - TSV Dataset"
echo "======================================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${HOSTNAME}"
echo "Dataset:      ${DATASET_PATH}"
echo "Output:       ${OUTPUT_FILE}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Start Time:   $(date)"
echo ""
nvidia-smi
echo ""

# Execute
/usr/bin/time -v ./bc_cuda "${DATASET_PATH}" ${BATCH_SIZE} > "${OUTPUT_FILE}" 2>&1

echo ""
echo "======================================================================"
echo "Execution Completed!"
echo "Output saved to: ${OUTPUT_FILE}"
echo "======================================================================"
echo ""
echo "=== TIMING SUMMARY ==="
grep -E "Total init|Total forward|Total backward|Total kernel|End-to-end" "${OUTPUT_FILE}" 2>/dev/null || true
echo ""
echo "=== SAMPLE BC VALUES (last 10) ==="
tail -n 10 "${OUTPUT_FILE}" 2>/dev/null || true
