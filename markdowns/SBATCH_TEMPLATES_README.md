# BC-CUDA Optimization: SBATCH Template Guide

## 📂 Dataset Organization

All datasets are now organized by format in `datasets/` folder:

```
datasets/
├── tsv/                          # TSV format (3-column: node1 node2 weight)
│   ├── com-DBLP_from_mtx.tsv
│   ├── as-Skitter_from_mtx.tsv
│   ├── LiveJournal_from_mtx.tsv
│   ├── road_usa_from_mtx.tsv
│   ├── egoGplus_from_edges.tsv
│   ├── egoGplus_from_edges_remapped.tsv
│   └── graph1.tsv
├── mtx/                          # Matrix Market format
│   └── graph1.mtx
└── txt/                          # Text/node lists
    └── egoGplus_nodes_unique.txt
```

## 🎯 Available Templates

### Template 1: TSV Datasets (Main)
**File:** `run_bc_tsv_template.sbatch`

**Use when:** Running TSV format datasets directly

**Quick Start:**
```bash
# Option A: Edit and submit
edit run_bc_tsv_template.sbatch  # Change DATASET_PATH and OUTPUT_DIR
sbatch run_bc_tsv_template.sbatch

# Option B: Pass variables to sbatch
sbatch --export=DATASET_PATH='datasets/tsv/com-DBLP_from_mtx.tsv',OUTPUT_DIR='results/test1' \
       run_bc_tsv_template.sbatch
```

**Customizable Parameters:**
```bash
DATASET_PATH="datasets/tsv/com-DBLP_from_mtx.tsv"  # Change to any TSV file
OUTPUT_DIR="results/bc_tsv"                         # Change output folder
BATCH_SIZE="16"                                     # Change batch size (optional)
```

---

### Template 2: MTX Datasets (Auto-Convert)
**File:** `run_bc_mtx_template.sbatch`

**Use when:** Running MTX format files (auto-converts to TSV)

**Quick Start:**
```bash
# Option A: Edit and submit
edit run_bc_mtx_template.sbatch  # Change MTX_PATH and OUTPUT_DIR
sbatch run_bc_mtx_template.sbatch

# Option B: Use environment variables
sbatch --export=MTX_PATH='datasets/mtx/graph1.mtx',OUTPUT_DIR='results/mtx_run' \
       run_bc_mtx_template.sbatch
```

**Customizable Parameters:**
```bash
MTX_PATH="datasets/mtx/graph1.mtx"                  # Change MTX file
OUTPUT_DIR="results/bc_mtx"                         # Change output folder
BATCH_SIZE="16"                                     # Change batch size
TEMP_DIR="/tmp"                                     # Temporary directory for TSV conversion
```

---

### Template 3: Batch Execute (All TSV Files)
**File:** `run_bc_batch_tsv_template.sbatch`

**Use when:** Running all TSV datasets in sequence automatically

**Quick Start:**
```bash
# Option A: Edit and submit
edit run_bc_batch_tsv_template.sbatch  # Change OUTPUT_DIR only
sbatch run_bc_batch_tsv_template.sbatch

# Option B: Use environment variables
sbatch --export=OUTPUT_DIR='results/batch_full' run_bc_batch_tsv_template.sbatch
```

**Features:**
- Automatically discovers all `.tsv` files in `datasets/tsv/`
- Creates summary CSV with metrics for each dataset
- Saves individual output files for each dataset
- Generates comparison report

**Output Structure:**
```
results/batch_full/
├── batch_summary_<job_id>.csv          # Summary table
├── bc_com-DBLP_from_mtx_<job_id>.txt
├── bc_as-Skitter_from_mtx_<job_id>.txt
├── bc_graph1_<job_id>.txt
└── ... (all other TSV datasets)
```

**Customizable Parameters:**
```bash
OUTPUT_DIR="results/batch_run"                      # Change output folder
BATCH_SIZE="16"                                     # Change batch size
DATASET_FOLDER="datasets/tsv"                       # Alternative dataset folder
```

---

### Template 4: Quick Run (Any Format)
**File:** `run_bc_quick_template.sbatch`

**Use when:** Need flexibility - handles both TSV and MTX automatically

**Quick Start:**
```bash
# Option A: Edit line and submit
edit run_bc_quick_template.sbatch
sbatch run_bc_quick_template.sbatch

# Option B: One-liner with TSV
sbatch --export="INPUT_FILE='datasets/tsv/graph1.tsv',OUTPUT_DIR='results/quick1',BATCH_SIZE='8'" \
       run_bc_quick_template.sbatch

# Option C: One-liner with MTX (auto-converts)
sbatch --export="INPUT_FILE='datasets/mtx/graph1.mtx',OUTPUT_DIR='results/quick2'" \
       run_bc_quick_template.sbatch
```

**Customizable Parameters:**
```bash
INPUT_FILE="datasets/tsv/com-DBLP_from_mtx.tsv"    # TSV or MTX file
OUTPUT_DIR="results/my_experiment"                  # Change output folder
BATCH_SIZE="16"                                     # Change batch size
JOB_NAME="bc_run"                                   # Output filename prefix
```

---

## 📋 Common Use Cases

### Scenario 1: Run Single TSV Dataset
```bash
sbatch --export=DATASET_PATH='datasets/tsv/LiveJournal_from_mtx.tsv',\
OUTPUT_DIR='results/livejournal_test' run_bc_tsv_template.sbatch
```

### Scenario 2: Run Single MTX Dataset with Auto-Conversion
```bash
sbatch --export=MTX_PATH='datasets/mtx/graph1.mtx',\
OUTPUT_DIR='results/graph1_conversion_test' run_bc_mtx_template.sbatch
```

### Scenario 3: Run All TSV Datasets with Single Command
```bash
sbatch --export=OUTPUT_DIR='results/all_datasets' run_bc_batch_tsv_template.sbatch
```

### Scenario 4: Quick Test with Custom Batch Size
```bash
sbatch --export=INPUT_FILE='datasets/tsv/graph1.tsv',\
OUTPUT_DIR='results/quick_test',BATCH_SIZE='4' run_bc_quick_template.sbatch
```

### Scenario 5: Compare Multiple Batch Sizes (Run Template 4 Multiple Times)
```bash
for B in 1 4 8 16 32; do
  sbatch --export=INPUT_FILE='datasets/tsv/com-DBLP_from_mtx.tsv',\
  OUTPUT_DIR="results/batch_compare",BATCH_SIZE="$B",JOB_NAME="bc_b${B}" \
  run_bc_quick_template.sbatch
done
```

---

## 🎮 Job Management

### Submit a Job
```bash
sbatch run_bc_tsv_template.sbatch
```

### Check Job Status
```bash
# All your jobs
squeue -u $(whoami)

# Specific job
squeue -j 12345
```

### Monitor Real-Time Output
```bash
# Watch stdout (while running)
tail -f results/bc_tsv/bc_com-DBLP_from_mtx_12345.txt

# Check sbatch log
tail -f slurm-12345.out
```

### Cancel a Job
```bash
scancel 12345
```

### View Completed Job Info
```bash
sacct -j 12345 --format=JobID,State,TotalCPU,Elapsed
```

---

## 📊 Understanding Output Files

Each job produces:

1. **Main Output:** `<output_dir>/bc_<dataset>_<job_id>.txt`
   - Contains BC values for all nodes
   - Includes timing metrics

2. **Timing Information (in output file):**
   ```
   Total init kernel ms: X       ← Device initialization
   Total forward kernel ms: Y    ← BFS forward pass
   Total backward kernel ms: Z   ← Backward dependency accumulation
   Total kernel ms: X+Y+Z        ← Total GPU time
   End-to-end wall time ms: ...  ← Including host work & memcpy
   ```

3. **Batch Mode:** Additionally creates `batch_summary_<job_id>.csv`
   - Spreadsheet format with metrics for each dataset
   - Easy to import into Excel/Python for analysis

---

## ⚙️ Advanced: Modify Templates

### Increase GPU Time
Edit the `#SBATCH --time=02:00:00` line in any template:
- `--time=05:30:00` for 5.5 hours
- `--time=24:00:00` for 24 hours

### Use Different GPU
Change `#SBATCH --gres=gpu:1` to request specific GPUs:
- `#SBATCH --gres=gpu:V100:1` for V100 (if available)
- `#SBATCH --gres=gpu:2` for 2 GPUs

### Request Different Partition
Change `#SBATCH --partition=gpu` to:
- `#SBATCH --partition=gpu-long` for longer jobs
- `#SBATCH --partition=high-priority` (if available)

### Send Email Notifications
Add these lines after time specification:
```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@domain.com
```

---

## 🔍 Debugging

### Job Failed - Check Error Log
```bash
# View sbatch stderr output
cat slurm-<job_id>.err

# Or use sacct to see exit code
sacct -j <job_id> --format=JobID,State,ExitCode
```

### Module Loading Issues
Verify modules are available:
```bash
module list
module avail gcc
module avail cuda
```

### No GPU Available
```bash
# Check GPU availability
gpustat
# Or
nvidia-smi

# Check queue
squeue -p gpu
```

### Dataset Not Found
Check paths:
```bash
ls -lh datasets/tsv/
ls -lh datasets/mtx/
find datasets/ -type f
```

---

## 💡 Tips & Tricks

1. **Quick Dry-Run:** Test with small dataset first (e.g., `graph1.tsv`)
2. **Batch Size:** Larger B = more computation per batch, but needs more GPU memory
3. **Parallel Runs:** Submit different template jobs simultaneously (different datasets)
4. **Archive Results:** After verification, move results to safe location
5. **Performance Tracking:** Use `run_bc_batch_tsv_template.sbatch` to track speedups across datasets

---

## ✅ Workflow Summary

```
1. Choose a template (1, 2, 3, or 4)
2. Edit dataset path and output directory
3. (Optional) Edit batch size or SLURM parameters
4. Run: sbatch <template>
5. Monitor: squeue -u $(whoami)
6. View results: cat results/<output>/bc_*.txt
7. (For batch mode) Analyze: cat results/<output>/batch_summary_*.csv
```

**You're all set!** 🚀
