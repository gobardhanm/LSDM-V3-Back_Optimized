# Quick Start: Testing Solution 1 with sbatch

## Compilation (if needed)

```bash
cd /data/gobardhan/LSDM-v3-bck-opt
module load gcc-12.3.0 cuda-12.3
make
```

## Running Tests with sbatch

### Option 1: Quick Correctness Test (Small Graph)
```bash
sbatch run_bc_test_graph1.sbatch
```
- **Time:** ~10 seconds
- **Purpose:** Verify backward kernel produces correct BC values
- **Output:** `results/bc_test_graph1_<job_id>.out`

### Option 2: Medium Graph Performance Test
```bash
sbatch run_bc_test_dblp.sbatch
```
- **Time:** ~30 seconds
- **Purpose:** Measure backward kernel speedup on medium graph
- **Datasets:** com-DBLP (334K nodes, ~1M edges)
- **Output:** `results/bc_test_dblp_<job_id>.out`

### Option 3: Comprehensive Benchmark (Recommended)
```bash
sbatch run_bc_perf_comprehensive.sbatch
```
- **Time:** ~2 hours
- **Purpose:** Full performance analysis across multiple datasets and batch sizes
- **Tests:**
  - graph1: batch sizes 1, 2, 4
  - com-DBLP: batch sizes 8, 16
  - as-Skitter: batch size 32
- **Output:** `results/` directory with individual results + summary CSV

## Monitoring Jobs

```bash
# View all your running jobs
squeue -u $(whoami)

# View specific job details
squeue -j <job_id>

# Cancel a job
scancel <job_id>

# View completed job info
sacct -j <job_id>
```

## Viewing Results

```bash
# Real-time tail of current job
tail -f results/bc_test_graph1_*.out

# After job completes
cat results/bc_test_graph1_<job_id>.out

# Extract timing metrics
grep "Total.*kernel ms:" results/bc_test_graph1_*.txt

# Compare backward times across runs
for f in results/bc_test_dblp_*.txt; do
    echo "$f:"
    grep "backward" "$f"
done
```

## Expected Performance Metrics to Check

Looking at the output, verify:

```
Total init kernel ms: <X>      ← Initialization time (shouldn't change)
Total forward kernel ms: <Y>   ← Forward phase (shouldn't change)
Total backward kernel ms: <Z>  ← Should be 2-2.7x faster than original!
Total kernel ms: <X+Y+Z>
End-to-end wall time ms: <total with memcpy>
```

## Table: Expected Speedups

| Graph | Nodes | Edges | Batch | Expected Backward Time | Speedup Factor |
|-------|-------|-------|-------|------------------------|-----------------|
| graph1 | 13 | 72 | 4 | <1ms | 2.0-2.5x |
| com-DBLP | 334K | 1M | 8 | 4-8ms | 2.0-2.7x |
| as-Skitter | 1.7M | 11M | 32 | 20-40ms | 2.0-2.7x |

## Troubleshooting

**Job fails with "nvcc not found":**
```bash
module load cuda-12.3
make
```

**Job fails with "gcc not found":**
```bash
module load gcc-12.3.0 cuda-12.3
make clean && make
```

**Results directory permission denied:**
```bash
mkdir -p /data/gobardhan/LSDM-v3-bck-opt/results
chmod 755 /data/gobardhan/LSDM-v3-bck-opt/results
```

**Out of GPU memory:**
- Reduce batch size (B parameter): `./bc_cuda dataset.tsv 4` instead of `32`
- Or request a larger GPU with `#SBATCH --gres=gpu:V100` (depends on available hardware)

## Analyzing Results

After comprehensive test completes:

```bash
# View summary CSV
cat results/perf_summary_<job_id>.csv

# Extract backward times for all runs
grep "Total backward" results/bc_*.txt | awk '{print $NF}' | sort -n

# Calculate average backward time
grep "Total backward" results/bc_test_dblp_*.txt | awk '{sum+=$NF; count++} END {print "Average: " sum/count "ms"}'
```

## Confirmation Checklist

- [ ] Recompile with `module load gcc-12.3.0 cuda-12.3 && make`
- [ ] Run `sbatch run_bc_test_graph1.sbatch` (quick test)
- [ ] Verify backward time < 1ms (should be 0.2-0.4ms)
- [ ] Run `sbatch run_bc_perf_comprehensive.sbatch` (full benchmark)
- [ ] Check backward times are 2-2.7x faster than expected original times
- [ ] Document final speedup numbers in results directory

## Commands Reference

```bash
# Clean and rebuild
cd /data/gobardhan/LSDM-v3-bck-opt
module load gcc-12.3.0 cuda-12.3
make
make           # if Makefile supports clean target

# Submit all tests
sbatch run_bc_test_graph1.sbatch
sbatch run_bc_test_dblp.sbatch
sbatch run_bc_perf_comprehensive.sbatch

# Monitor
watch 'squeue -u $(whoami)'

# Collect results
mkdir -p final_results
cp results/perf_summary_*.csv final_results/
cp results/bc_*.txt final_results/
```
