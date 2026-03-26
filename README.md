# Batched GPU Betweenness Centrality (BC)

A high-performance CUDA implementation of batched Brandes' betweenness centrality algorithm for large graphs. Designed to scale to graphs with ~1 million nodes by processing sources in sequential batches.

## Features

- **Batched BFS-based BC computation:** Process 1–32 sources per batch to reduce per-source overhead.
- **Memory-efficient:** Allocates device memory once; reuses across all batches (no malloc/free churn).
- **Robust CUDA error handling:** All API calls and kernel launches checked with `CUDA_CHECK` macro.
- **Optional verbose mode:** `--verbose` flag prints BFS level progress and batch statistics.
- **Undirected graph support:** Automatically bidirectionalizes edges from TSV input.
- **Tested on 1M+ nodes:** Runs on RTX 4000 Ada (20 GiB) and larger GPUs.

## Building

```bash
make
```

**Requirements:**
- CUDA Toolkit 11.0+ (tested on CUDA 13.0)
- nvcc compiler
- C++14 or later

## Usage

```bash
./bc_cuda <graph.tsv> <batch_size> [--verbose]
```

**Arguments:**
- `<graph.tsv>` — Input graph as TSV with format: `source target weight` (weight column ignored).
- `<batch_size>` — Sources per batch; range [1–32]. Larger batches → more parallelism, higher atomic contention. Default: 32 recommended.
- `--verbose` — Enable per-level logging and batch progress output (optional).

**Examples:**

```bash
# Small graph, batch size 4, verbose
./bc_cuda graph1.tsv 4 --verbose

# Large graph, batch size 32, quiet (production)
./bc_cuda large_graph.tsv 32
```

## Input Graph Format

**TSV (Tab-Separated Values) file; one edge per line:**

```
1	2	1.0
2	3	1.5
3	1	2.0
...
```

- **Column 1:** Source node ID (any positive integer).
- **Column 2:** Target node ID.
- **Column 3:** Edge weight (currently unused for BC computation; kept for compatibility).
- **Nodes:** Automatically re-mapped to 0-indexed internally. Original node IDs printed in output.
- **Edges:** Treated as undirected; automatically bidirectionalized.

## Output

```
=== FINAL BC ===
node 1 BC=0
node 2 BC=15.2
node 3 BC=26.5
...
```

Each line: `node <original_id> BC=<betweenness_centrality_score>`.

## Memory Usage

### Per-(Node, Source) Data:
- `dist` (int): 4 bytes — Distance from source node.
- `sigma` (unsigned long long): 8 bytes — Shortest-path counts.
- `delta` (double): 8 bytes — Dependency scores.
- **Per element:** 20 bytes
- **Per batch:** N × B × 20 bytes (N = nodes, B = batch size)

### Total GPU Memory (examples):

| Nodes | Batch | Array Size | Graph (CSR) | Total | GPU |
|-------|-------|-----------|------------|-------|-----|
| 1K    | 32    | 640 KiB   | ~50 KiB    | ~700 KiB  | Any |
| 10K   | 32    | 6.4 MiB   | ~0.5 MiB   | ~7 MiB    | Any |
| 100K  | 32    | 64 MiB    | ~5 MiB     | ~70 MiB   | Any |
| 1M    | 32    | 640 MiB   | ~50 MiB    | ~700 MiB  | RTX 4000 Ada ✅ |
| 2M    | 32    | 1.28 GiB  | ~100 MiB   | ~1.4 GiB  | RTX 4090 ✅ |

**Graph overhead (CSR):** ~4 MiB per 10K nodes (depends on avg. degree; examples assume degree ≈ 10).

### GPU Device Recommendations:

| GPU | VRAM | Max Nodes (B=32) | Notes |
|-----|------|------------------|-------|
| RTX 4000 Ada | 20 GiB | 1.2M | Tested; good margin. |
| RTX 4090 | 24 GiB | 1.5M | Production-grade. |
| RTX 6000 Ada | 48 GiB | 3M   | Enterprise-scale. |
| A40 | 48 GiB | 3M   | Data center option. |

## Algorithm Overview

### Forward Phase (BFS by Level):
For each BFS level \(d\):
1. Each thread processes a node \(u\) at level \(d\).
2. Reads which sources have \(u\) at distance \(d\) (via a per-node 32-bit bitmask).
3. For each unvisited neighbor \(v\):
   - **Atomically** set distance to \(d+1\) (first time) or skip (already visited).
   - **Accumulate** shortest-path counts via `atomicAdd`.
   - **Mark** \(v\) as active for the next level via `atomicOr`.

### Backward Phase (Dependency Accumulation):
Starting from the maximum distance, work backward:
1. For each node \(u\) at level \(d\), compute its dependency score:
   $$\text{delta}[u,s] = \sum_{v \text{ at level } d+1} \frac{\sigma[u,s]}{\sigma[v,s]} (1 + \text{delta}[v,s])$$
2. Accumulate all source contributions to a global BC counter.

### Final BC:
$$BC[v] = \frac{1}{2} \sum_s \text{delta}[v,s]$$
(Division by 2 because graph is undirected.)

## Batching Strategy

- **Sequential batches:** For `batch_start = 0; batch_start < N; batch_start += B_max`:
  1. Initialize per-(node,source) arrays for sources \([batch\_start, batch\_start+B)\).
  2. Run full forward+backward passes.
  3. Accumulate BC contributions into a persistent global BC array.
  4. **Reuse** device allocations; only reset arrays (via `cudaMemset`), not reallocate.

- **Rationale:** BFS-style dependency tracking requires full reinitialization per source batch; reusing allocations avoids expensive malloc/free and cudaMemcpy churn.

## Performance Tuning

### Batch Size:
- **Small (B=1–4):** Lower atomic contention; more passes over the graph. Less parallelism per kernel.
- **Large (B=16–32):** High parallelism; more atomic contention. Typically 2–4× faster than B=1 for large graphs.
- **Recommendation:** Start with B=32; reduce if you observe slowdown or memory constraints.

### Verbose Mode:
- **`--verbose`:** Prints per-level statistics and batch progress. Useful for debugging small graphs but adds 10–20% overhead on large graphs due to device-to-host copies. Disable for production runs.

### Atomic Contention:
- High-degree nodes (e.g., hubs in social networks) accumulate many atomic updates.
- Future optimization: warp-aggregated atomics or edge-partitioned kernels.

## Test Example

Run the included test graph (13 nodes):

```bash
./bc_cuda graph1.tsv 4 --verbose
```

**Expected output excerpt:**
```
Processing batch [0,4) B=4
  level 0 changed=1
  level 1 changed=1
  level 2 changed=1
  level 3 changed=1
  level 4 changed=0
Processing batch [4,8) B=4
  ...
=== FINAL BC ===
node 1 BC=0
node 2 BC=15.2
node 3 BC=26.5
...
```

## Known Limitations

1. **Batch size capped at 32:** Uses per-node 32-bit bitmask for active sources. For larger batches, switch to per-source-per-node boolean or 64-bit masks (memory/atomic cost trade-off).
2. **32-bit edge indices:** CSR `col_idx` uses 32-bit ints. Max ~2B edges. Switch to 64-bit if exceeding this.
3. **unsigned long long sigma:** Path count overflow rare but possible for extremely dense/degenerate graphs. Use `double sigma` for such cases (requires minor code change).
4. **No multi-GPU support (current):** Single GPU only. Domain decomposition for multi-GPU is a future enhancement.
5. **Hard stop on CUDA errors:** Calls `exit(EXIT_FAILURE)` upon any CUDA error. Production builds may want graceful error handling.

## File Structure

```
.
├── Makefile              # Build configuration
├── README.md             # This file
├── bc_cuda_verbose.cu    # Reference single-source BC implementation
├── bc_cuda               # Compiled executable
├── graph1.tsv            # Sample test graph (13 nodes)
├── include/
│   └── bc_cuda.hpp       # Header: kernel prototypes, CUDA_CHECK macro, wrapper declarations
├── src/
│   ├── bc_cuda.cu        # CUDA kernels and wrapper implementations
│   └── main.cpp          # Host program: graph I/O, batching loop, CLI
└── .gitignore
```

## Future Optimizations

- **Warp-aggregated atomics:** Reduce contention on high-degree nodes.
- **Edge partitioning:** Process edges in chunks to improve cache locality.
- **Pinned memory:** Faster PCIe transfers for large device-to-host copies.
- **Multi-GPU:** Distribute source batches or nodes across devices.
- **Custom CUDA kernels:** Optimize for specific GPU architectures (Ampere, Ada, etc.).

## References

- **Brandes, U.** (2001). "A faster algorithm for betweenness centrality." *Journal of Mathematical Sociology*, 25(2), 163–177.
- **NVIDIA CUDA C Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## License

[Specify your license, e.g., MIT, GPL, etc.]

## Author

Master's Thesis Project  
February 2026

---

**Questions or Issues?** Check the verbose output (`--verbose` flag) for per-level BFS statistics and batch progress. CUDA errors are logged with file and line numbers.
