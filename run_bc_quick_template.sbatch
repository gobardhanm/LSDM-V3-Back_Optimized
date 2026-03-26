#!/bin/bash
##############################################################################
# QUICK TEMPLATE: Execute Any Dataset (TSV or auto-convert from MTX)
# 
# USAGE EXAMPLES:
#   # Method 1: Edit below and just run
#   sbatch run_bc_quick_template.sbatch
#
#   # Method 2: Pass via environment variables
#   sbatch --export=INPUT_FILE='datasets/tsv/graph1.tsv',OUTPUT_DIR='results/test1',BATCH_SIZE='8' run_bc_quick_template.sbatch
#
#   # Method 3: Modify and submit with name tag
#   sbatch --export=INPUT_FILE='datasets/mtx/graph1.mtx',OUTPUT_DIR='results/mtx_test' run_bc_quick_template.sbatch
##############################################################################

#SBATCH --job-name=bc_quick
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# ========== EDIT THESE VALUES ==========
INPUT_FILE="${INPUT_FILE:-datasets/tsv/com-DBLP_from_mtx.tsv}"          # TSV or MTX file path
OUTPUT_DIR="${OUTPUT_DIR:-results/my_experiment}"                        # Where to save results
BATCH_SIZE="${BATCH_SIZE:-16}"                                          # BC batch size
JOB_NAME="${JOB_NAME:-bc_run}"                                          # For output filename
# ========================================

set -euo pipefail

PROJECT_DIR="/data/gobardhan/LSDM-v3-bck-opt"
cd "${PROJECT_DIR}"

mkdir -p "${OUTPUT_DIR}"
module load gcc-12.3.0 cuda-12.3 python3 2>/dev/null || true

# Determine file type and prepare for execution
if [[ "${INPUT_FILE}" == *.tsv ]]; then
    INPUT_FOR_BC="${INPUT_FILE}"
    CLEANUP_FILE=""
    FILE_TYPE="TSV"
elif [[ "${INPUT_FILE}" == *.mtx ]]; then
    FILE_TYPE="MTX (auto-converting)"
    INPUT_FOR_BC="/tmp/$(basename ${INPUT_FILE} .mtx)_${SLURM_JOB_ID}.tsv"
    CLEANUP_FILE="${INPUT_FOR_BC}"
    python3 mtx_to_tsv.py "${INPUT_FILE}" "${INPUT_FOR_BC}"
else
    echo "ERROR: File must be .tsv or .mtx"
    exit 1
fi

OUTPUT_FILE="${OUTPUT_DIR}/${JOB_NAME}_${SLURM_JOB_ID}.txt"

echo "======================================================================"
echo "BC Computation - Quick Run"
echo "======================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${HOSTNAME}"
echo "Input File:    ${INPUT_FILE} (${FILE_TYPE})"
echo "Output:        ${OUTPUT_FILE}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Start:         $(date)"
echo ""
nvidia-smi
echo ""

/usr/bin/time -v ./bc_cuda "${INPUT_FOR_BC}" ${BATCH_SIZE} > "${OUTPUT_FILE}" 2>&1

# Cleanup if we created a temporary TSV
[ -n "${CLEANUP_FILE}" ] && rm -f "${CLEANUP_FILE}"

echo ""
echo "======================================================================"
echo "✓ Completed!"
echo "Output: ${OUTPUT_FILE}"
echo "======================================================================"
echo ""
grep -E "Total init|Total forward|Total backward|Total kernel|End-to-end" "${OUTPUT_FILE}"
