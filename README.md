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
