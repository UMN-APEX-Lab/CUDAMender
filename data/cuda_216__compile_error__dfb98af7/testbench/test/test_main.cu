#include "bfs_kernel.h"
#include <cassert>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <nvtx3/nvToolsExt.h>

// CPU BFS implementation for validation
void cpuBFS(const std::vector<int>& rowOffsets_h, 
           const std::vector<int>& colIndices_h,
           int source,
           std::vector<int>& distances_h) {
    int numNodes = rowOffsets_h.size() - 1;
    
    // Initialize distances
    std::fill(distances_h.begin(), distances_h.end(), -1);
    distances_h[source] = 0;
    
    // BFS using queue
    std::queue<int> queue;
    queue.push(source);
    
    while (!queue.empty()) {
        int u = queue.front();
        queue.pop();
        
        // Process all neighbors of u
        int start = rowOffsets_h[u];
        int end = rowOffsets_h[u + 1];
        
        for (int i = start; i < end; ++i) {
            int v = colIndices_h[i];
            
            // If neighbor v is unvisited
            if (distances_h[v] == -1) {
                distances_h[v] = distances_h[u] + 1;
                queue.push(v);
            }
        }
    }
}

void launch() {
    const int MAX_NODES = 1000000;
    const int MAX_EDGES = 10000000;
    const int NUM_TEST_CASES = 7;
    const int BLOCK_SIZE = 256;

    // Get CUDA device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    int maxBlocksPerSm = deviceProp.maxThreadsPerMultiProcessor / BLOCK_SIZE;
    int maxBlocks = maxBlocksPerSm * deviceProp.multiProcessorCount;

    auto generateConnectedGraph = [](
        std::vector<int>& rowOffsets_h,
        std::vector<int>& colIndices_h,
        int numNodes,
        int numEdges,
        std::mt19937& rng) -> void {

        // Assert input parameters
        assert(numNodes > 0);
        assert(numEdges >= numNodes - 1); // Need at least spanning tree edges
        assert(numNodes <= MAX_NODES);
        assert(numEdges <= MAX_EDGES);

        rowOffsets_h.resize(numNodes + 1);
        colIndices_h.clear();
        colIndices_h.reserve(numEdges);

        // Step 1: Create adjacency list representation
        std::vector<std::vector<int>> adjList(numNodes);

        // Step 2: Build spanning tree to ensure connectivity
        std::vector<int> nodes(numNodes);
        std::iota(nodes.begin(), nodes.end(), 0);
        std::shuffle(nodes.begin(), nodes.end(), rng);

        // Connect each node to a random previous node in shuffled order
        for (int i = 1; i < numNodes; ++i) {
            std::uniform_int_distribution<int> parentDist(0, i - 1);
            int parent = nodes[parentDist(rng)];
            int child = nodes[i];

            // Add bidirectional edges for connectivity
            adjList[parent].push_back(child);
            adjList[child].push_back(parent);
        }

        int edgesAdded = (numNodes - 1) * 2; // Each undirected edge = 2 directed edges

        // Step 3: Add remaining edges randomly
        std::uniform_int_distribution<int> nodeDist(0, numNodes - 1);

        while (edgesAdded < numEdges) {
            int u = nodeDist(rng);
            int v = nodeDist(rng);

            if (u != v) { // No self-loops
                // Check if edge already exists
                bool exists = std::find(adjList[u].begin(), adjList[u].end(), v) != adjList[u].end();
                if (!exists) {
                    adjList[u].push_back(v);
                    edgesAdded++;
                }
            }
        }

        // Step 4: Convert adjacency list to CSR format
        rowOffsets_h[0] = 0;
        for (int i = 0; i < numNodes; ++i) {
            // Sort neighbors for better cache locality
            std::sort(adjList[i].begin(), adjList[i].end());

            for (int neighbor : adjList[i]) {
                colIndices_h.push_back(neighbor);
            }
            rowOffsets_h[i + 1] = colIndices_h.size();
        }

        // Assert final graph structure
        assert(rowOffsets_h.size() == numNodes + 1);
        assert(rowOffsets_h[0] == 0);
        assert(colIndices_h.size() > 0);
        assert(colIndices_h.size() <= numEdges);

        // Validate all neighbors are within bounds
        for (int neighbor : colIndices_h) {
            assert(neighbor >= 0 && neighbor < numNodes);
        }
    };

    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Device pointers with _d suffix
    int *rowOffsets_d, *colIndices_d, *distances_d;
    int *frontier_d, *nextFrontier_d, *nextFrontierSize_d;

    CUDA_CHECK(cudaMallocAsync(&rowOffsets_d, (MAX_NODES + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&colIndices_d, MAX_EDGES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&distances_d, MAX_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&frontier_d, MAX_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nextFrontier_d, MAX_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nextFrontierSize_d, sizeof(int), stream));

    // Random number generator with fixed seed for reproducibility
    std::mt19937 rng(42); // Fixed seed

    // Test case configurations
    struct TestConfig {
        int numNodes;
        int numEdges;
    };

    const TestConfig testConfigs[NUM_TEST_CASES] = {
        {1000, 5000},       // Small graph
        {10000, 50000},      // Medium graph
        {50000, 200000},     // Large graph
        {100000, 500000},    // Very large graph
        {200000, 1000000},   // Huge graph
        {500000, 2000000},   // Massive graph
        {800000, 5000000}    // Maximum graph
    };

    // Assert test configurations are valid
    for (int i = 0; i < NUM_TEST_CASES; ++i) {
        assert(testConfigs[i].numNodes > 0);
        assert(testConfigs[i].numEdges >= testConfigs[i].numNodes - 1); // Ensure connectivity possible
        assert(testConfigs[i].numNodes <= MAX_NODES);
        assert(testConfigs[i].numEdges <= MAX_EDGES);
    }

    for (int i = 0; i < NUM_TEST_CASES; ++i) {
        int currentNodes = testConfigs[i].numNodes;
        int currentEdges = testConfigs[i].numEdges;

        // Assert test case parameters
        assert(currentNodes > 0 && currentNodes <= MAX_NODES);
        assert(currentEdges >= currentNodes - 1 && currentEdges <= MAX_EDGES);

        // Host vectors with _h suffix
        std::vector<int> rowOffsets_h, colIndices_h;
        generateConnectedGraph(rowOffsets_h, colIndices_h, currentNodes, currentEdges, rng);

        // Assert graph generation results
        assert(rowOffsets_h.size() == currentNodes + 1);
        assert(colIndices_h.size() > 0);
        assert(rowOffsets_h[0] == 0);

        // Random source node
        std::uniform_int_distribution<int> sourceDist(0, currentNodes - 1);
        int source = sourceDist(rng);

        // Assert source is valid
        assert(source >= 0 && source < currentNodes);

        std::vector<int> distances_h(currentNodes, -1);
        std::vector<int> cpuDistances_h(currentNodes, -1);

        int numNodes = rowOffsets_h.size() - 1;
        int numEdges = colIndices_h.size();

        // Assert BFS input parameters
        assert(numNodes == currentNodes);
        assert(numEdges > 0);
        assert(source >= 0 && source < numNodes);

        // Run CPU BFS for validation
        cpuBFS(rowOffsets_h, colIndices_h, source, cpuDistances_h);

        // Copy graph to device asynchronously
        CUDA_CHECK(cudaMemcpyAsync(rowOffsets_d, rowOffsets_h.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(colIndices_d, colIndices_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Initialize distances to -1
        std::vector<int> initDistances_h(numNodes, -1);
        initDistances_h[source] = 0;
        CUDA_CHECK(cudaMemcpyAsync(distances_d, initDistances_h.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Initialize frontier with source node
        int frontierSize = 1;
        CUDA_CHECK(cudaMemcpyAsync(frontier_d, &source, sizeof(int), cudaMemcpyHostToDevice, stream));

        int currentLevel = 0;
        int maxIterations = numNodes; // Prevent infinite loops
        int iterations = 0;

        while (frontierSize > 0) {
            // Assert loop bounds
            assert(iterations < maxIterations);
            assert(frontierSize > 0);
            assert(currentLevel >= 0);

            CUDA_CHECK(cudaMemsetAsync(nextFrontierSize_d, 0, sizeof(int), stream));

            // Calculate grid size based on device properties and frontier size
            int gridSize = std::min(maxBlocks, (frontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Assert grid configuration
            assert(gridSize > 0);
            assert(gridSize <= maxBlocks);

            // Prepare kernel parameters
            void* kernelArgs[] = {
                &rowOffsets_d,
                &colIndices_d,
                &frontier_d,
                &frontierSize,
                &currentLevel,
                &distances_d,
                &nextFrontier_d,
                &nextFrontierSize_d
            };

            // Launch kernel using cudaLaunchKernel
            CUDA_CHECK(cudaLaunchKernel(
                (void*)k_bfsKernel,
                dim3(gridSize),
                dim3(BLOCK_SIZE),
                kernelArgs,
                0,
                stream
            ));

            CUDA_CHECK(cudaMemcpyAsync(&frontierSize, nextFrontierSize_d, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Assert frontier size is reasonable
            assert(frontierSize >= 0);
            assert(frontierSize <= numNodes);

            std::swap(frontier_d, nextFrontier_d);
            currentLevel++;
            iterations++;
        }

        CUDA_CHECK(cudaMemcpyAsync(distances_h.data(), distances_d, numNodes * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Validate GPU results against CPU results
        for (int j = 0; j < numNodes; ++j) {
            assert(distances_h[j] == cpuDistances_h[j]);
        }

        assert(distances_h[source] == 0); // Source distance should be 0

        int reachableCount = 0;
        int maxDistance = 0;
        for (int j = 0; j < numNodes; ++j) {
            if (distances_h[j] != -1) {
                assert(distances_h[j] >= 0);
                assert(distances_h[j] < numNodes); // Distance can't exceed graph diameter
                reachableCount++;
                maxDistance = std::max(maxDistance, distances_h[j]);
            }
        }

        // Assert result validity
        assert(reachableCount > 0); // At least source should be reachable
        assert(reachableCount == numNodes);
        assert(maxDistance >= 0);
        assert(maxDistance < numNodes);
    }

    // Free device memory asynchronously
    CUDA_CHECK(cudaFreeAsync(rowOffsets_d, stream));
    CUDA_CHECK(cudaFreeAsync(colIndices_d, stream));
    CUDA_CHECK(cudaFreeAsync(distances_d, stream));
    CUDA_CHECK(cudaFreeAsync(frontier_d, stream));
    CUDA_CHECK(cudaFreeAsync(nextFrontier_d, stream));
    CUDA_CHECK(cudaFreeAsync(nextFrontierSize_d, stream));

    // Destroy stream
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void runBFS(
    int* rowOffsets_d, int* colIndices_d, int* distances_d,
    int* frontier_d, int* nextFrontier_d, int* nextFrontierSize_d,
    int numNodes, int source, int maxBlocks, int BLOCK_SIZE, cudaStream_t stream)
{
    std::vector<int> initDistances_h(numNodes, -1);
    initDistances_h[source] = 0;
    CUDA_CHECK(cudaMemcpyAsync(distances_d, initDistances_h.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice, stream));

    int frontierSize = 1;
    CUDA_CHECK(cudaMemcpyAsync(frontier_d, &source, sizeof(int), cudaMemcpyHostToDevice, stream));

    int currentLevel = 0;

    while (frontierSize > 0) {
        CUDA_CHECK(cudaMemsetAsync(nextFrontierSize_d, 0, sizeof(int), stream));

        int gridSize = std::min(maxBlocks, (frontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

        void* kernelArgs[] = {
            &rowOffsets_d, &colIndices_d, &frontier_d,
            &frontierSize, &currentLevel, &distances_d,
            &nextFrontier_d, &nextFrontierSize_d
        };

        CUDA_CHECK(cudaLaunchKernel(
            (void*)k_bfsKernel, dim3(gridSize), dim3(BLOCK_SIZE),
            kernelArgs, 0, stream));

        CUDA_CHECK(cudaMemcpyAsync(&frontierSize, nextFrontierSize_d, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::swap(frontier_d, nextFrontier_d);
        currentLevel++;
    }
}

void benchmark() {
    const int BENCH_NODES = 1000000;
    const int BENCH_EDGES = 8000000;
    const int BLOCK_SIZE = 256;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int maxBlocksPerSm = deviceProp.maxThreadsPerMultiProcessor / BLOCK_SIZE;
    int maxBlocks = maxBlocksPerSm * deviceProp.multiProcessorCount;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int *rowOffsets_d, *colIndices_d, *distances_d;
    int *frontier_d, *nextFrontier_d, *nextFrontierSize_d;

    CUDA_CHECK(cudaMallocAsync(&rowOffsets_d, (BENCH_NODES + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&colIndices_d, BENCH_EDGES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&distances_d, BENCH_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&frontier_d, BENCH_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nextFrontier_d, BENCH_NODES * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nextFrontierSize_d, sizeof(int), stream));

    std::mt19937 rng(123);

    // Generate a large connected graph
    std::vector<int> rowOffsets_h(BENCH_NODES + 1);
    std::vector<int> colIndices_h;
    colIndices_h.reserve(BENCH_EDGES);

    std::vector<std::vector<int>> adjList(BENCH_NODES);

    std::vector<int> nodes(BENCH_NODES);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::shuffle(nodes.begin(), nodes.end(), rng);

    for (int i = 1; i < BENCH_NODES; ++i) {
        std::uniform_int_distribution<int> parentDist(0, i - 1);
        int parent = nodes[parentDist(rng)];
        int child = nodes[i];
        adjList[parent].push_back(child);
        adjList[child].push_back(parent);
    }

    int edgesAdded = (BENCH_NODES - 1) * 2;
    std::uniform_int_distribution<int> nodeDist(0, BENCH_NODES - 1);

    while (edgesAdded < BENCH_EDGES) {
        int u = nodeDist(rng);
        int v = nodeDist(rng);
        if (u != v) {
            adjList[u].push_back(v);
            edgesAdded++;
        }
    }

    rowOffsets_h[0] = 0;
    for (int i = 0; i < BENCH_NODES; ++i) {
        std::sort(adjList[i].begin(), adjList[i].end());
        for (int neighbor : adjList[i]) {
            colIndices_h.push_back(neighbor);
        }
        rowOffsets_h[i + 1] = colIndices_h.size();
    }

    int numEdges = colIndices_h.size();

    CUDA_CHECK(cudaMemcpyAsync(rowOffsets_d, rowOffsets_h.data(), (BENCH_NODES + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(colIndices_d, colIndices_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int source = 0;

    // Warmup iterations
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        runBFS(rowOffsets_d, colIndices_d, distances_d,
               frontier_d, nextFrontier_d, nextFrontierSize_d,
               BENCH_NODES, source, maxBlocks, BLOCK_SIZE, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; ++i) {
        runBFS(rowOffsets_d, colIndices_d, distances_d,
               frontier_d, nextFrontier_d, nextFrontierSize_d,
               BENCH_NODES, source, maxBlocks, BLOCK_SIZE, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(rowOffsets_d, stream));
    CUDA_CHECK(cudaFreeAsync(colIndices_d, stream));
    CUDA_CHECK(cudaFreeAsync(distances_d, stream));
    CUDA_CHECK(cudaFreeAsync(frontier_d, stream));
    CUDA_CHECK(cudaFreeAsync(nextFrontier_d, stream));
    CUDA_CHECK(cudaFreeAsync(nextFrontierSize_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}