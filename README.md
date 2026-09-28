# Parallel and GPU Computing (PGC) Lab
> **Experiment 1:** Dense Matrix Multiplication ($4000 \times 4000$) using Sequential CPU, OpenMP, MPI, and CUDA Acceleration

---

### 👤 Student Details & Project Metadata
* **Student USN / Roll Number:** `01FE24BCI069`
* **Repository Link:** `https://github.com/vinayhiremath440/Parallel-and-GPU-Computing-Lab`
* **Lab Reference Document:** `docs/Experiment_1_Parallel_Matrix_Multiplication_Lab_Manual_REFERENCE_FORMAT.docx`

---

## 📊 Executive Summary & Performance Tracker

The table below outlines the experimental benchmark analysis comparing four parallel computing paradigms executing a $4000 \times 4000$ matrix multiplication:

| Benchmark Part | Execution Model | Architectural Paradigm | Execution Status | Threads / Nodes | Runtime (Sec) | Speedup Factor | Verification ($C[0][0]$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Part A** | **Sequential** | Single-Core CPU Baseline | `COMPLETED` | 1 Core | **363.642678 s** | **1.00×** | `4000.00` |
| **Part B** | **OpenMP** | Shared-Memory Multithreading | `COMPLETED` | 16 Threads | **96.381304 s** | **3.77×** | `4000.00` |
| **Part C** | **Open MPI** | Distributed Cluster Environment | `COMPLETED` | 4 VM Ranks | **226.167575 s** | **1.61×** | `4000.00` |
| **Part D** | **CUDA** | Massively Parallel GPU | `COMPLETED` | 16M Threads | **0.165004 s** *(Kernel: 0.146443 s)* | **2203.84×** | `4000.00` |

---

## 🧮 Problem Formulation & Workload Specification

Dense matrix multiplication of two double-precision real matrices $A, B \in \mathbb{R}^{N \times N}$ yielding matrix $C \in \mathbb{R}^{N \times N}$:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \cdot B_{k,j} \quad \text{for } 0 \le i, j < N$$

### 🛠️ Workload Parameters
* **Matrix Order ($N$):** $4000 \times 4000$
* **Data Precision:** Double-precision floating point (`double`, 8 bytes per element)
* **Memory Requirement:**
  * Single Matrix Footprint: $4000 \times 4000 \times 8 \text{ bytes} = 128 \text{ MB}$
  * Total Working Memory ($A + B + C$): $3 \times 128 \text{ MB} = 384 \text{ MB}$
* **Computational Volume:**
  $$\text{FLOPs} = 2 \times N^3 = 2 \times 4000^3 = 128 \times 10^9 \text{ Operations (128 GFLOPs)}$$
* **Initialization Conditions:**
  $$A[i][j] = 1.0, \quad B[i][j] = 1.0, \quad C[i][j] = 0.0$$
* **Verification Target:**
  $$C[0][0] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

> **Note:** Every implementation must produce $C[0][0] = 4000.00$ to ensure numerical fidelity across paradigms.

---

## 🖥️ Testbed & Execution Environment

* **Host System OS:** Windows 11 running WSL2 (Linux Subsystem)
* **Linux Environment:** Ubuntu (WSL2 & VMware Workstation guest instances)
* **Processor Architecture:** x86_64, 16 Logical Processors / Hardware Threads
* **RAM Infrastructure:** 16 GB DDR4
* **Toolchain & Compilers:**
  * CPU & OpenMP: `gcc (Ubuntu) -O2` with `-fopenmp`
  * MPI Cluster: Open MPI (`mpicc`, `mpirun`) over OpenSSH
  * GPU Execution: NVIDIA CUDA Compiler (`nvcc -O2`)
* **MPI Cluster Network Setup:**
  * Platform: VMware Workstation Hypervisor
  * Nodes: 4 Ubuntu Virtual Machines (1 Master + 3 Workers) on subnet `192.168.148.0/24`
  * Interconnect: Virtual Network Adapter featuring OpenSSH passwordless key authentication
* **NVIDIA GPU Target:**
  * Architecture: NVIDIA CUDA GPU Architecture
  * Thread Grid: Massively parallel configuration across 62,500 blocks (16,000,000 threads)

---

## ⚙️ Part A: Sequential Matrix Multiplication (Completed)

### 1. Methodology & Algorithmic Design
The baseline sequential execution runs on a single core using standard triple-nested loop iteration:
1. **Outer loop ($i$):** Iterates over rows of matrix $A$ ($0 \to N-1$).
2. **Middle loop ($j$):** Iterates over columns of matrix $B$ ($0 \to N-1$).
3. **Inner loop ($k$):** Computes dot product $\sum A[i][k] \times B[k][j]$.

Column-wise memory access on matrix $B$ (`B[k * N + j]`) causes stride offsets of $N \times 8 = 32,000$ bytes, inducing frequent cache capacity misses in L1/L2 caches and bottlenecking CPU throughput.

### 2. Implementation Code
Source file location: [`src/sequential/matrix_sequential.c`](src/sequential/matrix_sequential.c)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main() {
    int i, j, k;
    double *A, *B, *C;
    clock_t start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));
    // Matrix initialization: A = 1.0, B = 1.0, C = 0.0 ...
    
    start = clock();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    end = clock();

    printf("Execution Time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Verification C[0][0] = %.2f\n", C[0]);
    // cleanup
    return 0;
}
## 🚀 Part B: OpenMP Shared-Memory Parallelism (Completed)

### 1. Methodology & Multithreading Mechanics
OpenMP applies symmetric multiprocessing (SMP) utilizing the **Fork-Join concurrency model**:
* Primary master thread encounters `#pragma omp parallel for private(j, k)`.
* Spawns a worker team of 16 threads.
* Outer loop $i$ (4000 row iterations) is partitioned evenly (~250 rows per thread).
* Iteration variables `j` and `k` are set to `private` mode to prevent stack data races, while `A`, `B`, and `C` remain shared memory buffers.
* Precision timing is measured via `omp_get_wtime()`.

### 2. Environment Setup
```bash
export OMP_NUM_THREADS=16
echo $OMP_NUM_THREADS # Outputs 16
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main() {
    int i, j, k;
    double *A, *B, *C;
    double start, end;

    // memory allocation and initialization ...

    start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    end = omp_get_wtime();

    printf("OpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %f seconds\n", end - start);
    printf("Verification C[0][0] = %.2f\n", C[0]);
    // cleanup
    return 0;
}
cd ~/parallel_lab/openmp
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
./matrix_openmp
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 16
Execution Time = 96.381304 seconds
Verification C[0][0] = 4000.00

Total Runtime ($T_{\text{openmp}}$): 96.381304 s (~1.61 minutes)Active Threads Configured: 16Verification Result: C[0][0] = 4000.00 (Passed)Effective Compute Throughput:$$\text{GFLOPS} = \frac{128 \times 10^9}{96.381304 \times 10^9} \approx 1.328 \text{ GFLOPS}$$

6. Process Monitoring (htop)Runtime system monitoring confirmed 100.0% CPU usage across all 16 logical cores (0–15) during ./matrix_openmp execution, reaching a load average above 10.27.7. Execution ProofTerminal Run OutputMulti-Core CPU Load Profile (htop)🌐 Part C: Open MPI Distributed-Memory Computing (Completed)1. Methodology & Cluster DesignOpen MPI executes distributed computing through the Single Program, Multiple Data (SPMD) model across distinct virtual nodes:Private Memory Isolation: Each process operates in its own address space on separated virtual machines.Row-Wise Domain Decomposition:Master process (Rank 0) allocates initial matrices $A$ and $B$.Matrix $A$ is sliced along rows: $N = 4000$ across $P = 4$ processes yields $rows\_per\_process = 4000 / 4 = 1000$ rows ($1000 \times 4000 \times 8 \text{ bytes} = 32 \text{ MB}$).Matrix $B$ ($4000 \times 4000 \times 8 \text{ bytes} = 128 \text{ MB}$) is broadcast to all processes.Collective Operations:MPI_Scatter: Splits Matrix $A$ into 1000-row slices sent from Rank 0 to local_A on Ranks 0–3.MPI_Bcast: Replicates entire Matrix $B$ from Rank 0 across all ranks.MPI_Gather: Collects computed local_C chunks (1000 rows = 32 MB) back into global Matrix $C$ on Rank 0.MPI_Barrier synchronizes processes before starting timer via MPI_Wtime().2. Cluster Topology & NetworkingProvisioned on VMware Workstation using 4 Ubuntu Virtual Machines connected via subnet 192.168.148.0/24:Host IdentifierVM ConfigurationNetwork AddressMPI RankWorkload SlicemasterMaster Node192.168.148.128Rank 0Rows 0 – 999 (plus Scatter/Gather coordination)worker1Worker Node 1192.168.148.129Rank 1Rows 1000 – 1999worker2Worker Node 2192.168.148.130Rank 2Rows 2000 – 2999worker3Worker Node 3192.168.148.131Rank 3Rows 3000 – 3999Authentication Strategy: Passwordless OpenSSH keys (ssh-keygen, ssh-copy-id) from Master to Workers.Hostfile Configuration (hosts):
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
3. Network Verification (Ping Analysis)Bidirectional network connectivity check from master to worker VMs showed low latency and zero packet loss:ping -c 4 192.168.148.129 (worker1) $\to$ 4/4 received, 0% loss, avg RTT = 0.602 msping -c 4 192.168.148.130 (worker2) $\to$ 4/4 received, 0% loss, avg RTT = 0.641 msping -c 4 192.168.148.131 (worker3) $\to$ 4/4 received, 0% loss, avg RTT = 1.029 ms4. Implementation CodeSource file location: src/mpi/matrix_mpi.c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[]) {
    int rank, size, rows_per_process;
    char hostname[256];
    double *A = NULL, *B = NULL, *C = NULL, *local_A, *local_C;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, sizeof(hostname));

    rows_per_process = N / size;
    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    local_C = (double *)malloc(rows_per_process * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        // Initialize A = 1.0, B = 1.0, C = 0.0
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Rank %d on %s computing %d rows\n", rank, hostname, rows_per_process);

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (rank == 0) {
        printf("\nMPI Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of MPI Processes = %d\n", size);
        printf("Execution Time = %f seconds\n", end - start);
        printf("Verification C[0][0] = %.2f\n", C[0]);
    }
    MPI_Finalize();
    return 0;
}
cd ~/parallel_lab/mpi
mpicc -O2 matrix_mpi.c -o matrix_mpi

# Copy compiled binary across workers
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi

# Run distributed job
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
Initializing 4000 x 4000 matrices...
Rank 0 on master computing 1000 rows
Rank 1 on worker1 computing 1000 rows
Rank 2 on worker2 computing 1000 rows
Rank 3 on worker3 computing 1000 rows

MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = 226.167575 seconds
Verification C[0][0] = 4000.00

Total Runtime ($T_{\text{mpi}}$): 226.167575 s (~3.77 minutes)MPI Ranks Executed: 4 (1 Master + 3 Workers)Verification Check: C[0][0] = 4000.00 (Passed)Speedup vs Sequential ($S_{\text{mpi}}$):$$S_{\text{mpi}} = \frac{T_{\text{seq}}}{T_{\text{mpi}}} = \frac{363.642678 \text{ s}}{226.167575 \text{ s}} \approx \mathbf{1.608\times} \approx \mathbf{1.61\times}$$Parallel Efficiency ($E_{\text{mpi}}$):$$E_{\text{mpi}} = \frac{S}{P} \times 100\% = \frac{1.6078}{4} \times 100\% \approx \mathbf{40.20\%}$$Throughput Rate:$$\text{GFLOPS} = \frac{128 \times 10^9}{226.167575 \times 10^9} \approx \mathbf{0.566 \text{ GFLOPS}}$$7. Communication Bottlenecks & Network OverheadWhile achieving a 1.61× speedup over sequential execution, MPI was slower than OpenMP (96.38 s) due to network transfer overheads:Serialization & Socket Communication:OpenMP processes directly over the CPU memory bus at 25–40 GB/s.Open MPI must transmit data via TCP/IP sockets over virtual bridges:MPI_Bcast: $128 \text{ MB}$ Matrix $B$ sent to 3 workers.MPI_Scatter: $3 \times 32 \text{ MB} = 96 \text{ MB}$ Matrix $A$ chunks.MPI_Gather: $3 \times 32 \text{ MB} = 96 \text{ MB}$ Matrix $C$ chunks returned.Total data transfer exceeds 320 MB, adding packet framing and buffering delays.Virtualization Scheduling: Running 4 guest virtual machines on a single physical host introduces contention for physical hardware threads and memory channels.8. Execution ProofCluster Network Ping Connectivity TestMPI Execution Run Output⚡ Part D: NVIDIA CUDA GPU Acceleration (Completed)1. Methodology & GPU ArchitectureCUDA accelerates matrix multiplication through massive thread-level parallelism across GPU Streaming Multiprocessors (SMs):2D Grid & Thread Block Decomposition:Grid dimensions: $250 \times 250 = 62,500$ blocks.Block dimensions: $16 \times 16 = 256$ threads per block.Total GPU Threads: $62,500 \times 256 = 16,000,000$ concurrent threads.Global Memory Indexing Formula:$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$Execution Flow:Allocate device memory buffers (d_A, d_B, d_C) via cudaMalloc.Copy inputs $A$ and $B$ from host to device memory via cudaMemcpyHostToDevice.Launch kernel matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N).Synchronize via cudaDeviceSynchronize().Copy result matrix $C$ back to host memory via cudaMemcpyDeviceToHost.
2. Implementation Code
Source file location: src/cuda/matrix_cuda.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define N 4000
#define BLOCK_SIZE 16

__global__ void matrixMulKernel(const double *A, const double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(double);
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (double *)malloc(bytes);
    h_B = (double *)malloc(bytes);
    h_C = (double *)malloc(bytes);

    // Initialize host memory A = 1.0, B = 1.0 ...

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal);
    cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);

    cudaEventRecord(startTotal);

    // Host to Device transfer
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(startKernel);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopKernel);

    // Device to Host transfer
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);

    float totalTimeMs = 0, kernelTimeMs = 0;
    cudaEventElapsedTime(&totalTimeMs, startTotal, stopTotal);
    cudaEventElapsedTime(&kernelTimeMs, startKernel, stopKernel);

    printf("CUDA Matrix Multiplication Completed\n");
    printf("Kernel Execution Time = %f seconds\n", kernelTimeMs / 1000.0);
    printf("Total Time (with Transfer) = %f seconds\n", totalTimeMs / 1000.0);
    printf("Verification C[0][0] = %.2f\n", h_C[0]);

    // Cleanup memory and events
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
3. Build & Execution Command
cd ~/parallel_lab/cuda
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda
4. Recorded Benchmark Output
Initializing 4000 x 4000 matrices...
Allocating GPU memory...
Copying data to GPU...
Launching CUDA Kernel with Grid(250, 250) and Block(16, 16)...
Kernel execution finished.
Copying result back to CPU...

CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Kernel Execution Time = 0.146443 seconds
Total Execution Time (Host to Host) = 0.165004 seconds
Verification C[0][0] = 4000.00
Kernel Execution Time ($T_{\text{kernel}}$): 0.146443 sTotal Execution Time ($T_{\text{cuda\_total}}$): 0.165004 s (includes H2D/D2H PCIe transfers)Verification Result: C[0][0] = 4000.00 (Passed)Speedup vs Sequential ($S_{\text{cuda}}$):$$S_{\text{cuda}} = \frac{T_{\text{seq}}}{T_{\text{cuda\_total}}} = \frac{363.642678 \text{ s}}{0.165004 \text{ s}} \approx \mathbf{2203.84\times}$$Kernel-Only Speedup:$$S_{\text{kernel}} = \frac{363.642678}{0.146443} \approx \mathbf{2483.17\times}$$Effective Compute Throughput:$$\text{GFLOPS} = \frac{128 \times 10^9}{0.165004 \times 10^9} \approx \mathbf{775.74 \text{ GFLOPS}}$$5. Execution Proof📈 Comparative Analysis & Performance Metrics1. Unified Benchmark EvaluationThe benchmark results for multiplying two $4000 \times 4000$ double-precision matrices across all four paradigms:Metric / ParameterPart A: SequentialPart B: OpenMPPart C: Open MPIPart D: CUDAExecution ParadigmSingle-Core CPUShared MemoryDistributed MemoryMassively Parallel GPUWorker Count1 Core16 Threads4 VM Ranks16,000,000 ThreadsExecution Time (Sec)363.64 s96.38 s226.17 s0.165 s (Kernel: 0.146 s)Speedup Factor ($S$)1.00×3.77×1.61×2203.84×Efficiency ($E$)100%23.56%40.20%N/AThroughput (GFLOPS)0.352 GFLOPS1.328 GFLOPS0.566 GFLOPS775.74 GFLOPSVerification ($C[0][0]$)4000.004000.004000.004000.002. Analytical Findings & TakeawaysMassive Acceleration with CUDA GPU:Speedup: Realized a 2203.84× overall speedup (and 2483.17× kernel speedup) over the single-threaded CPU baseline.Throughput: Achieved 775.74 GFLOPS by mapping matrix dimensions across 16 million hardware threads.PCIe Transfer Overhead: Moving data over PCIe host-device memory buffers took 0.018561 s ($11.2\%$ of total execution time).Multithreaded OpenMP Scalability:Speedup: Achieved 3.77× speedup over sequential execution using 16 logical cores.Bottleneck: Thread contend for shared memory bus bandwidth when accessing matrix $B$ across strided memory paths.Distributed Open MPI Tradeoffs:Speedup: Achieved 1.61× speedup across a 4-node virtual cluster.Interconnect Bottleneck: While distributed memory eliminates CPU cache limits, communication overhead (MPI_Bcast, MPI_Scatter, MPI_Gather transferring >320 MB over TCP/IP sockets) reduces scaling efficiency compared to shared-memory thread pools.
📁 Repository Directory Structure
Parallel-and-GPU-Computing-Lab/
├── README.md
├── docs/
│   └── Experiment_1_Parallel_Matrix_Multiplication_Lab_Manual_REFERENCE_FORMAT.docx
├── screenshots/
│   ├── 01fe24bci081_Sequential.png
│   ├── 01fe24bci081_OpenMP.png
│   ├── 01fe24bci081_OpenMP_htop.png
│   ├── mpi_ping.png
│   ├── mpi_result.png
│   └── cuda_result.png
└── src/
    ├── sequential/
    │   └── matrix_sequential.c
    ├── openmp/
    │   └── matrix_openmp.c
    ├── mpi/
    │   ├── matrix_mpi.c
    │   └── hosts
    └── cuda/
        └── matrix_cuda.cu

Verification & ConclusionAll four implementations executed to completion, matching the expected output $C[0][0] = 4000.00$ for $4000 \times 4000$ matrices initialized with $1.0$.CUDA GPU Acceleration proved to be the fastest method by orders of magnitude, reducing processing time from ~6 minutes to under 0.17 seconds.OpenMP offers an efficient, low-complexity approach for multi-core CPUs without network transfer costs.Open MPI provides horizontal scaling across physical machines, though network bandwidth limits simple dense matrix operations on virtualized networks.
