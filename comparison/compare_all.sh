#!/bin/bash
##############################################################################
# TEMPLATE: Compare BC results between two implementations
# USAGE: Edit DATASET_NAMES below to list the datasets to compare,
#        or pass a single dataset: ./compare_all.sh <dataset_name>
#
# File naming conventions:
#   Your results:    <RESULTS_DIR>/<name>_BC.txt
#   cuGraph results: <CUGRAPH_DIR>/BC_<name>.txt
##############################################################################

# ========== USER CONFIGURATION ==========
RESULTS_DIR="${RESULTS_DIR:-results/small}"
CUGRAPH_DIR="${CUGRAPH_DIR:-/data/gobardhan/cuGraph/results}"
OUTPUT_DIR="${OUTPUT_DIR:-comparison}"
TOLERANCE_ABS="${TOLERANCE_ABS:-1e-3}"
TOLERANCE_REL="${TOLERANCE_REL:-1e-4}"
PROJECT_DIR="/data/gobardhan/LSDM-V3-Back_Optimized"

# ← ADD / REMOVE dataset names here (stem only, e.g. "ca-CondMat")
DATASET_NAMES=(
    "ca-CondMat"
    "email-Enron"
    "oregon1_010331"
    "oregon2_010331"
    "p2p-Gnutella25"
    "soc-Slashdot0902"
)
# =========================================

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# If a single dataset is passed as argument, override the list
if [ $# -ge 1 ]; then
    DATASET_NAMES=("$1")
fi

SUMMARY_FILE="${OUTPUT_DIR}/comparison_summary.txt"
echo "======================================================================" > "${SUMMARY_FILE}"
echo "BC COMPARISON SUMMARY" >> "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "======================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for ds in "${DATASET_NAMES[@]}"; do
    FILE_A="${RESULTS_DIR}/${ds}_BC.txt"
    FILE_B="${CUGRAPH_DIR}/BC_${ds}.txt"
    REPORT="${OUTPUT_DIR}/${ds}_comparison.txt"

    echo "--- Comparing: ${ds} ---"
    
    if [ ! -f "${FILE_A}" ]; then
        echo "  ⚠ SKIP: ${FILE_A} not found"
        echo "[SKIP] ${ds}: ${FILE_A} not found" >> "${SUMMARY_FILE}"
        continue
    fi
    if [ ! -f "${FILE_B}" ]; then
        echo "  ⚠ SKIP: ${FILE_B} not found"
        echo "[SKIP] ${ds}: ${FILE_B} not found" >> "${SUMMARY_FILE}"
        continue
    fi
    
    python3 compare_bc.py "${FILE_A}" "${FILE_B}" \
        --tolerance-abs "${TOLERANCE_ABS}" \
        --tolerance-rel "${TOLERANCE_REL}" \
        --output "${REPORT}"
    
    # Extract key line for summary
    RESULT_LINE=$(grep -E "RESULT:" "${REPORT}" 2>/dev/null | head -1)
    MAX_ABS=$(grep "Max absolute" "${REPORT}" 2>/dev/null | head -1)
    MAX_REL=$(grep "Max relative" "${REPORT}" 2>/dev/null | head -1)
    echo "[${ds}]  ${RESULT_LINE}" >> "${SUMMARY_FILE}"
    echo "  ${MAX_ABS}" >> "${SUMMARY_FILE}"
    echo "  ${MAX_REL}" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
done

echo "" >> "${SUMMARY_FILE}"
echo "Reports saved in: ${OUTPUT_DIR}/" >> "${SUMMARY_FILE}"
echo "======================================================================" >> "${SUMMARY_FILE}"

echo ""
echo "=== OVERALL SUMMARY ==="
cat "${SUMMARY_FILE}"
