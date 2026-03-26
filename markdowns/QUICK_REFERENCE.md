# Quick Reference Card

## One-Liners

### Run Single TSV File
```bash
sbatch --export=DATASET_PATH='datasets/tsv/com-DBLP_from_mtx.tsv',OUTPUT_DIR='results/test1' \
       run_bc_tsv_template.sbatch
```

### Run Single MTX File (Auto-Convert)
```bash
sbatch --export=MTX_PATH='datasets/mtx/graph1.mtx',OUTPUT_DIR='results/test2' \
       run_bc_mtx_template.sbatch
```

### Run ALL TSV Datasets
```bash
sbatch --export=OUTPUT_DIR='results/alldata' run_bc_batch_tsv_template.sbatch
```

### Quick Run (Any Format)
```bash
sbatch --export=INPUT_FILE='datasets/tsv/graph1.tsv',OUTPUT_DIR='results/quick' \
       run_bc_quick_template.sbatch
```

---

## Check Progress

```bash
# See your running jobs
squeue -u $(whoami)

# Watch output live
tail -f results/test1/bc_com-DBLP_from_mtx_*.txt
```

---

## Datasets Location

| Format | Folder | Available Files |
|--------|--------|-----------------|
| TSV | `datasets/tsv/` | 7 files total |
| MTX | `datasets/mtx/` | graph1.mtx |
| TXT | `datasets/txt/` | egoGplus_nodes_unique.txt |

---

## Templates Reference

| Template | Use For | Time | Auto-Convert |
|----------|---------|------|--------------|
| `run_bc_tsv_template.sbatch` | Single TSV file | 2h | No |
| `run_bc_mtx_template.sbatch` | Single MTX file | 2h | Yes |
| `run_bc_batch_tsv_template.sbatch` | All TSV files | 6h | No |
| `run_bc_quick_template.sbatch` | Any format | 2h | Yes |

---

## Edit & Run

```bash
# 1. Edit template
vi run_bc_tsv_template.sbatch

# 2. Change these lines (line numbers ~20-23):
#    DATASET_PATH="YOUR_FILE"
#    OUTPUT_DIR="YOUR_OUTPUT"
#    BATCH_SIZE="16"  # optional

# 3. Save and submit
:wq
sbatch run_bc_tsv_template.sbatch

# 4. Check output
ls -lh results/YOUR_OUTPUT/
```

---

## Output Structure

```
results/
└── my_experiment/
    ├── bc_dataset_name_12345.txt      ← Main output
    ├── bc_dataset_name_12346.txt      ← Another run
    └── batch_summary_12347.csv        ← (Batch mode only)
```

---

## Extract Timing Metrics

```bash
# View all timing info for a run
grep "Total" results/test1/bc_*.txt

# Just backward time
grep "backward" results/test1/bc_*.txt

# Compare multiple runs
for f in results/test*/bc_*.txt; do
  echo "$f:"; grep "backward" "$f"; done
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| Module not found | `module load gcc-12.3.0 cuda-12.3` |
| File not found | Check path: `ls datasets/tsv/` |
| GPU out of memory | Reduce `BATCH_SIZE`, or run smaller dataset |
| Job stuck | `scancel <job_id>` |

---

## Next Steps

1. Choose a template (1, 2, 3, or 4)
2. Change dataset path and output folder
3. Run: `sbatch <template>`
4. Monitor: `squeue -u $(whoami)`
5. Done! Results in your output folder
