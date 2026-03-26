#!/bin/bash
##############################################################################
# TEMPLATE: MTX Dataset Execution Script (Auto-converts to TSV)
# USAGE: sbatch --export=MTX_PATH='datasets/mtx/YOUR_DATASET.mtx',OUTPUT_DIR='results/your_output' run_bc_mtx_template.sbatch
# OR: Edit MTX_PATH and OUTPUT_DIR below and run: sbatch run_bc_mtx_template.sbatch
##############################################################################

#SBATCH --job-name=bc_mtx
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# ========== USER CONFIGURATION (Edit these) ==========
MTX_PATH="${MTX_PATH:-/data/Dataset/SuiteSparse/soc-LiveJournal1/soc-LiveJournal1.mtx}"                         # ← CHANGE THIS
OUTPUT_DIR="${OUTPUT_DIR:-results/bc_mtx}"                              # ← CHANGE THIS
BATCH_SIZE="${BATCH_SIZE:-32}"                                          # ← OPTIONAL: Batch size
TEMP_DIR="${TEMP_DIR:-/tmp}"                                            # ← Temp directory for TSV conversion

# ====================================================

set -euo pipefail

PROJECT_DIR="/data/gobardhan/LSDM-v3-bck-opt"
cd "${PROJECT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Load required modules
module load gcc-12.3.0 cuda-12.3 python3 2>/dev/null || true

DATASET_NAME=$(basename "${MTX_PATH}" .mtx)
TSV_FILE="${TEMP_DIR}/${DATASET_NAME}_${SLURM_JOB_ID}.tsv"
OUTPUT_FILE="${OUTPUT_DIR}/bc_${DATASET_NAME}_${SLURM_JOB_ID}.txt"

echo "======================================================================"
echo "BC Computation - MTX Dataset (Auto-converted to TSV)"
echo "======================================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${HOSTNAME}"
echo "MTX File:     ${MTX_PATH}"
echo "TSV File:     ${TSV_FILE}"
echo "Output:       ${OUTPUT_FILE}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Start Time:   $(date)"
echo ""
nvidia-smi
echo ""

# Step 1: Convert MTX to TSV
echo "Step 1: Converting MTX to TSV..."
python3 "${PROJECT_DIR}/mtx_to_tsv.py" "${MTX_PATH}" "${TSV_FILE}"
echo "✓ Conversion complete"
echo ""

# Step 2: Execute BC computation
echo "Step 2: Running BC computation..."
/usr/bin/time -v ./bc_cuda "${TSV_FILE}" ${BATCH_SIZE} > "${OUTPUT_FILE}" 2>&1

# Step 3: Cleanup temporary file
echo ""
echo "Step 3: Cleaning up temporary TSV file..."
rm -f "${TSV_FILE}"
echo "✓ Cleanup complete"

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
