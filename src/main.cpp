#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "../include/bc_cuda.hpp"

using namespace std;

struct BatchStats {
    int batch_id = 0;
    int batch_start = 0;
    int batch_size = 0;
    int max_level = 0;
    int forward_kernel_calls = 0;
    int backward_kernel_calls = 0;
    float init_ms = 0.0f;
    float forward_ms = 0.0f;
    float backward_ms = 0.0f;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./bc_cuda <graph.tsv> <batch_size> [--verbose]\n";
        return 1;
    }

    string file = argv[1];
    int Bmax = atoi(argv[2]);
    bool verbose = false;
    for (int i = 3; i < argc; i++) if (string(argv[i]) == "--verbose") verbose = true;

    if (Bmax <= 0 || Bmax > 32) {
        cerr << "Batch size must be in [1,32]\n";
        return 1;
    }

    // Read TSV undirected
    vector<pair<int,int>> edges;
    int u,v;
    double w;
    int undirected_edges = 0;
    ifstream fin(file);
    if (!fin) { cerr << "Cannot open " << file << "\n"; return 1; }
    while (fin >> u >> v >> w) {
        undirected_edges++;
        edges.emplace_back(u, v);
        edges.emplace_back(v, u);
    }

    set<int> nodes;
    for (auto &e : edges) { nodes.insert(e.first); nodes.insert(e.second); }
    vector<int> node_list(nodes.begin(), nodes.end());
    int N = node_list.size();
    unordered_map<int,int> id;
    for (int i = 0; i < N; i++) id[node_list[i]] = i;

    vector<vector<int>> adj(N);
    for (auto &e : edges) adj[id[e.first]].push_back(id[e.second]);

    // CSR
    vector<int> row_ptr(N+1,0);
    for (int i = 0; i < N; i++) row_ptr[i+1] = row_ptr[i] + adj[i].size();
    vector<int> col_idx; col_idx.reserve(row_ptr.back());
    for (int i = 0; i < N; i++) for (int x : adj[i]) col_idx.push_back(x);

    int *d_row = nullptr, *d_col = nullptr, *d_dist = nullptr;
    double *d_sigma = nullptr;
    double *d_delta = nullptr;
    int *d_queue = nullptr;
    int *d_level_ptr = nullptr;
    int *d_added_level = nullptr;

    size_t row_bytes = (row_ptr.size()) * sizeof(int);
    size_t col_bytes = (col_idx.size()) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_row, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_col, col_bytes));

    CUDA_CHECK(cudaMemcpy(d_row, row_ptr.data(), row_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, col_idx.data(), col_bytes, cudaMemcpyHostToDevice));

    int stride = Bmax;
    size_t node_batch_elems = (size_t)N * stride;
    CUDA_CHECK(cudaMalloc(&d_dist, node_batch_elems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sigma, node_batch_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_delta, node_batch_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_queue, (size_t)N * Bmax * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_level_ptr, (size_t)(N + 5) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_added_level, N * sizeof(int)));

    vector<double> BC(N, 0.0);

    int block = 256;
    int grid = (N + block - 1) / block;

    // For edge-parallel backward kernel (Solution 1)
    int total_edges = (int)col_idx.size();
    int block_edge = 256;
    int grid_edge = (total_edges + block_edge - 1) / block_edge;

    cudaEvent_t evt_start = nullptr, evt_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&evt_start));
    CUDA_CHECK(cudaEventCreate(&evt_stop));

    vector<BatchStats> batch_stats;
    float total_init_ms = 0.0f;
    float total_forward_ms = 0.0f;
    float total_backward_ms = 0.0f;
    auto wall_start = chrono::steady_clock::now();

    int batch_id = 0;
    for (int batch_start = 0; batch_start < N; batch_start += Bmax) {
        int B = min(Bmax, N - batch_start);
        BatchStats stats;
        stats.batch_id = batch_id++;
        stats.batch_start = batch_start;
        stats.batch_size = B;

        // init device arrays
        CUDA_CHECK(cudaMemset(d_dist, 0xFF, node_batch_elems * sizeof(int))); // -1
        CUDA_CHECK(cudaMemset(d_sigma, 0, node_batch_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_delta, 0, node_batch_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_added_level, 0xFF, N * sizeof(int))); // -1
        
        vector<int> h_level_ptr(N + 5, 0);
        CUDA_CHECK(cudaMemcpy(d_level_ptr, h_level_ptr.data(), (N + 5) * sizeof(int), cudaMemcpyHostToDevice));

        // init sources for this batch
        int tblock = 128;
        int tgrid = (B + tblock - 1) / tblock;
        CUDA_CHECK(cudaEventRecord(evt_start));
        launch_init_sources(N, B, stride, batch_start, d_dist, d_sigma, d_queue, d_level_ptr, d_added_level, tgrid, tblock);
        CUDA_CHECK(cudaEventRecord(evt_stop));
        CUDA_CHECK(cudaEventSynchronize(evt_stop));
        CUDA_CHECK(cudaEventElapsedTime(&stats.init_ms, evt_start, evt_stop));

        CUDA_CHECK(cudaMemcpy(&h_level_ptr[1], d_level_ptr + 1, sizeof(int), cudaMemcpyDeviceToHost));
        h_level_ptr[2] = h_level_ptr[1];
        CUDA_CHECK(cudaMemcpy(d_level_ptr + 2, &h_level_ptr[1], sizeof(int), cudaMemcpyHostToDevice));

        if (verbose || stats.batch_id % 10 == 0) {
            cout << "Processing batch " << stats.batch_id << " [" << batch_start << "," << batch_start+B << ") B=" << B << "\n";
        }

        // Forward BFS levels
        int level = 0;
        while (true) {
            int start = h_level_ptr[level];
            int end = h_level_ptr[level+1];
            int size = end - start;
            if (size == 0) break;

            if (verbose) {
                cout << "  level " << level << " frontier=" << size << "\n";
            }

            grid = (size * 32 + block - 1) / block;

            float forward_iter_ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(evt_start));
            launch_forward_level(N, B, stride, d_row, d_col, d_dist, d_sigma, d_queue, start, size, d_level_ptr, d_added_level, level, grid, block);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(evt_stop));
            CUDA_CHECK(cudaEventSynchronize(evt_stop));
            CUDA_CHECK(cudaEventElapsedTime(&forward_iter_ms, evt_start, evt_stop));
            stats.forward_ms += forward_iter_ms;
            stats.forward_kernel_calls++;

            CUDA_CHECK(cudaMemcpy(&h_level_ptr[level+2], d_level_ptr + level + 2, sizeof(int), cudaMemcpyDeviceToHost));
            h_level_ptr[level+3] = h_level_ptr[level+2];
            CUDA_CHECK(cudaMemcpy(d_level_ptr + level + 3, &h_level_ptr[level+2], sizeof(int), cudaMemcpyHostToDevice));

            level++;
        }
        int max_level = level;
        stats.max_level = max_level;

        // Backward accumulation (using frontier-based backward kernel)
        for (int d = max_level-1; d >= 0; d--) {
            int start = h_level_ptr[d];
            int end = h_level_ptr[d+1];
            int size = end - start;
            if (size == 0) continue;

            grid = (size * 32 + block - 1) / block;

            float backward_iter_ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(evt_start));
            launch_backward_level_frontier(N, B, stride, d_row, d_col, d_dist, d_sigma, d_delta, d_queue, start, size, d, grid, block);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(evt_stop));
            CUDA_CHECK(cudaEventSynchronize(evt_stop));
            CUDA_CHECK(cudaEventElapsedTime(&backward_iter_ms, evt_start, evt_stop));
            stats.backward_ms += backward_iter_ms;
            stats.backward_kernel_calls++;
        }

        // copy delta back and accumulate BC for this batch
        vector<double> h_delta(node_batch_elems);
        CUDA_CHECK(cudaMemcpy(h_delta.data(), d_delta, node_batch_elems * sizeof(double), cudaMemcpyDeviceToHost));

        for (int node = 0; node < N; node++) {
            for (int s = 0; s < B; s++) {
                int src = batch_start + s;
                if (node == src) continue;
                BC[node] += h_delta[node*stride + s];
            }
        }

        total_init_ms += stats.init_ms;
        total_forward_ms += stats.forward_ms;
        total_backward_ms += stats.backward_ms;
        batch_stats.push_back(stats);
    }
    auto wall_end = chrono::steady_clock::now();

    // finalize
    for (int i = 0; i < N; i++) BC[i] /= 2.0;

    cout << fixed << setprecision(3);
    cout << "=== EXECUTION SUMMARY ===\n";
    cout << "Graph file: " << file << "\n";
    cout << "Nodes: " << N << "\n";
    cout << "Undirected edges: " << undirected_edges << "\n";
    cout << "Directed edges (CSR): " << col_idx.size() << "\n";
    cout << "Configured batch size: " << Bmax << "\n";
    cout << "Batches: " << batch_stats.size() << "\n";
    // cout << "Per-batch kernel execution time (ms):\n";
    // for (const auto& s : batch_stats) {
    //     cout << "  batch " << s.batch_id
    //          << " range=[" << s.batch_start << "," << (s.batch_start + s.batch_size) << ")"
    //          << " B=" << s.batch_size
    //          << " levels=" << s.max_level
    //          << " forward_kernels=" << s.forward_kernel_calls
    //          << " backward_kernels=" << s.backward_kernel_calls
    //          << " init_ms=" << s.init_ms
    //          << " forward_ms=" << s.forward_ms
    //          << " backward_ms=" << s.backward_ms
    //          << " total_kernel_ms=" << (s.init_ms + s.forward_ms + s.backward_ms)
    //          << "\n";
    // }

    double wall_ms = chrono::duration<double, milli>(wall_end - wall_start).count();
    cout << "Total init kernel ms: " << total_init_ms << "\n";
    cout << "Total forward kernel ms: " << total_forward_ms << "\n";
    cout << "Total backward kernel ms: " << total_backward_ms << "\n";
    cout << "Total kernel ms: " << (total_init_ms + total_forward_ms + total_backward_ms) << "\n";
    cout << "End-to-end wall time ms (includes memcpy + host work): " << wall_ms << "\n";

    cout << "=== FINAL BC ===\n";
    for (int i = 0; i < N; i++) cout << "node " << node_list[i] << " BC=" << BC[i] << "\n";

    CUDA_CHECK(cudaEventDestroy(evt_start));
    CUDA_CHECK(cudaEventDestroy(evt_stop));

    return 0;
}
 
