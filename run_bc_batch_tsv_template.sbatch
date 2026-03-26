#!/bin/bash
##############################################################################
# TEMPLATE: Batch Execute Multiple TSV Datasets
# USAGE: sbatch --export=OUTPUT_DIR='results/batch_run' run_bc_batch_tsv_template.sbatch
# OR: Edit OUTPUT_DIR below and run: sbatch run_bc_batch_tsv_template.sbatch
##############################################################################

#SBATCH --job-name=bc_batch_tsv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# ========== USER CONFIGURATION (Edit these) ==========
OUTPUT_DIR="${OUTPUT_DIR:-results/small-datasets-32}"                           # ← CHANGE THIS
BATCH_SIZE="${BATCH_SIZE:-32}"                                          # ← OPTIONAL: Batch size
DATASET_FOLDER="${DATASET_FOLDER:-/data/Dataset/GraphChallange}"                        # Folder containing TSV files

# ====================================================

set -euo pipefail

PROJECT_DIR="/data/gobardhan/LSDM-v3-bck-opt"
cd "${PROJECT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Load required modules
module load gcc-12.3.0 cuda-12.3 2>/dev/null || true

echo "======================================================================"
echo "Batch BC Computation - All TSV Datasets"
echo "======================================================================"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           ${HOSTNAME}"
echo "Dataset Folder: ${DATASET_FOLDER}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Start Time:     $(date)"
echo ""
nvidia-smi
echo ""

# Create summary file
SUMMARY_FILE="${OUTPUT_DIR}/batch_summary_${SLURM_JOB_ID}.csv"
echo "Dataset,Nodes,Edges,InitMS,ForwardMS,BackwardMS,TotalMS,WallTimeMS" > "${SUMMARY_FILE}"

# Counter for processed datasets
DATASET_COUNT=0

# Process each TSV file
for TSV_FILE in "${DATASET_FOLDER}"/*.tsv; do
    if [ ! -f "${TSV_FILE}" ]; then
        echo "No TSV files found in ${DATASET_FOLDER}"
        break
    fi
    
    DATASET_NAME=$(basename "${TSV_FILE}" .tsv)
    OUTPUT_FILE="${OUTPUT_DIR}/bc_${DATASET_NAME}_${SLURM_JOB_ID}.txt"
    
    DATASET_COUNT=$((DATASET_COUNT + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing: $(basename ${TSV_FILE})"
    echo "=========================================="
    echo "Output: ${OUTPUT_FILE}"
    echo ""
    
    /usr/bin/time -v ./bc_cuda "${TSV_FILE}" ${BATCH_SIZE} > "${OUTPUT_FILE}" 2>&1
    
    # Extract metrics
    NODES=$(grep "^Nodes:" "${OUTPUT_FILE}" | awk '{print $NF}' || echo "N/A")
    EDGES=$(grep "^Directed edges" "${OUTPUT_FILE}" | awk '{print $NF}' || echo "N/A")
    INIT_MS=$(grep "Total init" "${OUTPUT_FILE}" | awk '{print $(NF-1)}' | sed 's/ms//' || echo "N/A")
    FWD_MS=$(grep "Total forward" "${OUTPUT_FILE}" | awk '{print $(NF-1)}' | sed 's/ms//' || echo "N/A")
    BWD_MS=$(grep "Total backward" "${OUTPUT_FILE}" | awk '{print $(NF-1)}' | sed 's/ms//' || echo "N/A")
    TOT_MS=$(grep "Total kernel" "${OUTPUT_FILE}" | awk '{print $(NF-1)}' | sed 's/ms//' || echo "N/A")
    WALL_MS=$(grep "End-to-end" "${OUTPUT_FILE}" | awk '{print $(NF-1)}' | sed 's/ms//' || echo "N/A")
    
    echo "${DATASET_NAME},${NODES},${EDGES},${INIT_MS},${FWD_MS},${BWD_MS},${TOT_MS},${WALL_MS}" >> "${SUMMARY_FILE}"
    
    echo "✓ Completed - Init: ${INIT_MS}ms | Fwd: ${FWD_MS}ms | Bwd: ${BWD_MS}ms | Total: ${TOT_MS}ms"
done

echo ""
echo "======================================================================"
echo "Batch Execution Completed!"
echo "Datasets processed: ${DATASET_COUNT}"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Summary:          ${SUMMARY_FILE}"
echo "======================================================================"
echo ""
echo "=== BATCH SUMMARY ==="
cat "${SUMMARY_FILE}"
