---
layout: post
author: Giacomo Bergami, PhD
title: "Breaking the OOM Barrier: High-Rank Multi-Dimensional Sparse Einsum via Compressed Sparse Fiber (CSF) on GPU"
date: 2026-10-29 00:00:00 +0100
categories: [machine-learning, hpc]
tags: [cuda, einsum, sparse-tensors, csf, high-performance-computing]
---

Einstein Summation (`einsum`) is arguably the most expressive and pervasive primitive within the artificial intelligence field. From matrix multiplications and tensor contractions to the orchestration of complex batched attention mechanisms in Transformers, `einsum` acts as a domain-specific language that rules them all. As Tim Rocktäschel famously stated, "[einsum is all you need](https://rockt.ai/2018/04/30/einsum)". It allows developers to write clean, declarative multi-dimensional array operations that can be automatically compiled and optimized down to hardware-level kernels.

However, when deep learning models confront massive, high-dimensional datasets with structural sparsity—such as user-item-word recommendation graphs, electronic medical records, or cybersecurity network logs—the standard execution routes offered by industry-leading frameworks break down completely. 

This post analyzes the severe architectural limitations of **PyTorch** and **TensorFlow** when handling high-rank sparse `einsum` expressions, introduces the **Compressed Sparse Fiber (CSF)** representation, and exhibits how a custom-designed, memory-compact CUDA engine (**Paramoneur**) can outperform standard frameworks by orders of magnitude. 

## 1. The Densification Trap: PyTorch & TensorFlow Deficiencies

While PyTorch and TensorFlow offer high-performance dense `einsum` solvers backed by backend utilities like NVIDIA `cuTENSOR` and `cuBLAS`, their sparse tensor support is tightly restricted. At present, neither framework can compute generic high-rank sparse-sparse contractions natively. If you pass an expression like `"abcde,bdace->abe"` using sparse formats, the engines will throw a `NotImplementedError` or force you to explicitly execute a conversion layer via `.to_dense()` or `tf.sparse.to_dense()`.

For small arrays, this conversion layer goes unnoticed. But when working with large multi-way datasets spanning big dimensions—for instance, a geometry space of `[50x50x50x20x10]`—the memory footprint escalates exponentially. Storing this space in a dense layout demands an allocation of:
$$\text{Total Elements} = 50 \times 50 \times 50 \times 20 \times 10 = 25,000,000 \text{ float elements}$$

If the dimensionality scales up further to high-order tensors, a dense matrix layout will immediately trigger a catastrophic **Out-Of-Memory (OOM)** hardware crash. 

### Why the Standard Coordinate (COO) Layout Fails for Contractions
To circumvent dense memory limits, developers turn to sparse representations, where the **Coordinate (COO)** layout is the most ubiquitous default. A COO tensor isolates only the non-zero elements (`NNZ`) by tracking a flat value array along with a multi-row index grid where every element's precise coordinates are explicitly stated.

While COO is ideal for basic operations like sparse matrix-vector additions, executing a multi-axis `einsum` contraction on a raw COO layout becomes a performance nightmare. Because the structural non-zeros are scattered throughout the coordinate list, calculating an intersection or a matching index along a contracted axis requires a full **Cartesian Cross-Product** of the element sheets, yielding an algorithmic complexity of $\mathcal{O}(\text{NNZ}_A \times \text{NNZ}_B)$. 

At an `NNZ` of $100,000$ items, a raw COO contraction requires the GPU to schedule and trace $100,000 \times 100,000 = 10 \text{ billion}$ individual comparison checks. This unstructured memory access pattern causes immense thread divergence, destroys cache locality, and saturates the VRAM memory bandwidth with redundant, non-coalesced read cycles.

## 2. Introducing Compressed Sparse Fiber (CSF)

To solve the structural limitations of multi-dimensional sparse data, Smith, Park, and Karypis introduced the **[Compressed Sparse Fiber (CSF)](http://shaden.io/pub-files/smith2017knl.pdf)** data structure. CSF is the multi-dimensional generalization of the popular [Compressed Sparse Row (CSR)](https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf) matrix format, recursively compressing each mode of a sparse tensor into a set of prefix trees.

### Formal Definition
A sparse tensor $\mathcal{X}$ of order M is represented in CSF by recursively isolating unique coordinates across successive hierarchical modes. Structurally, it eliminates coordinate redundancy by encoding paths from a root node down to leaf levels, where each complete path represents an active non-zero element. It is implemented via three contiguously allocated arrays:
1. **`vals`**: A flat array storing the non-zero values located at the leaf level.
2. **`fids`**: Stores the explicit coordinate label of each node at every level of the tree.
3. **`fptr`**: A sequence of M row-pointer arrays where each array segment acts as an indexing boundaries map into the next mode level, with the final pointer array mapping directly into the leaf non-zeros.

```text
[Mode 0: i]   ->  Unique 'i' coordinates
                     |   (fptr defines boundaries)
[Mode 1: j]   ->  Unique 'j' children for each 'i'
                     |   (fptr defines boundaries)
[Mode 2: k]   ->  Leaves holding the sparse values
```

By organizing the multi-way data hierarchically, CSF reveals massive opportunities for computational and bandwidth savings. If multiple non-zero items share common ancestors (such as a shared prefix `(i, j)`), the tensor contraction engine can compute the overlapping subset of products once, bringing the contributions up and down the tree branches recursively. This fiber-centric traversal replaces the expensive linear search loops with focused tree lookups, reducing the number of sparse matrix row accesses dramatically.

Despite its overwhelming performance benefits for data-intensive machine learning workloads, **neither PyTorch nor TensorFlow implements the CSF data structure natively**. Both frameworks remain anchored to flat matrices or loose COO layouts, motivating the critical need to build custom, highly optimized acceleration libraries.

### Core Architecture of the Multi-Dimensional CUDA Intersection Kernel

To achieve sub-millisecond execution times without the massive memory overhead of dense matrices, the `launch_csf_einsum_two_tensors` engine shifts the parallel computing paradigm from a blind matrix multiplication to a streamlined, multi-dimensional coordinate intersection. The codebase is available [here](https://github.com/LogDS/ParaMoNeuR/blob/master/csf_einsum.cu).

The underlying CUDA kernel abandons the standard 2D grid allocation pattern (`NNZ_A x NNZ_B`), which triggers massive thread explosion and hardware-level warp scheduler stall. Instead, it linearizes the execution grid into a clean 1D configuration, allocating exactly one CUDA thread per structural non-zero entry of the first sparse tensor operator (`csf_A.nnz`). 

The precise mathematical execution steps performed concurrently by each active GPU thread map directly to the hardware registers as follows:

1. **Coalesced Ancestor Fetching**: The thread utilizes its global identifier (`blockIdx.x * blockDim.x + threadIdx.x`) to read its own multidimensional coordinate tuple out of `csf_A.d_coordinate_matrix` in a single coalesced transaction, caching the entries into a static register array (`coords_A`).
2. **Permuted Target Realignment**: Before executing any search passes, the thread maps the data layout of `A` into the exact lessicographical ordering required by `B`. By looking up the global metadata stored in the execution plan (`plan.map_B_to_global`), it rearranges its own elements into a secondary, short register vector (`target_B_coords`) corresponding to the permuted axis sequence of the second operator (e.g., mapping a raw layout of `[a,b,c,d,e]` into the target layout `[b,d,a,c,e]`).
3. **Logarithmic Binary Matching**: Since the data loader forces a rigid, bitwise sorted alignment on the second operator in VRAM, the thread executes a highly optimized Binary Search (`while (low <= high)`) across `csf_B.d_coordinate_matrix`. At each step, a fast mid-point lessicographical comparison (`compare_coordinates`) evaluates the unified 64-bit key layouts against `target_B_coords`, isolating the exact structural intersection match in at most $\log_2(NNZ_B)$ iterations.
4. **Isolated Atomic Reduction**: If a coordinate intersection is found, the thread pulls the floating-point values from both unifed memory spaces, multiplies them, and performs a thread-safe `atomicAdd` on the global counter. The final step projescts the active dimensions out to the contiguous `UnifiedCOOOutput` slots, selectively dropping the contracted indexes and outputting a single, reduced record onto the GPU.


## 3. High-Performance CUDA Design: Zero-Allocation In-Place Conversion

Implementing an arbitrary, high-rank sparse `einsum` on GPU requires a fine-grained, parallel approach. In our custom system, **Paramoneur**, we map exactly one CUDA thread to every active non-zero element of the input tensor. By sorting the target array lexicographically using an efficient bitwise pre-pass, each thread can find its precise coordinate intersections using a fast **Binary Search** over the second sparse operator. This drops the computational complexity from a blinding $\mathcal{O}(\text{NNZ}_A \times \text{NNZ}_B)$ down to  $\mathcal{O}(\text{NNZ}_A \log(\text{NNZ}_B))$.

The results are initially written into a compact **[Coordinate (COO) output buffer](https://lumetta.ece.illinois.edu/408-S20/slide-copies/ece408-lecture19-S20.pdf)**. To protect the GPU from Out-Of-Memory exceptions, we consolidate the output arrays—allocating indices, floating values, and the global atomic counter—into a single, contiguous block of VRAM using **Pointer Arithmetic**.

```text
[ Unified Output Buffer Space ]
+----------------------------+-----------------------+---------------+

| d_out_indices (ints)       | d_out_vals (floats)   | atomic_counter|
+----------------------------+-----------------------+---------------+
^                            ^                       ^
base_ptr                     + indices_bytes         + vals_bytes
```

### In-Place Transformation into CSF
To completely minimize the memory footprint, we must avoid allocating a separate, secondary block of memory when converting the resulting COO back into a compressed CSF tree structure. Because an `einsum` contraction naturally reduces the dimensionality of the tensor via axis summation, the number of unique non-zero entries in the output is mathematically guaranteed to be smaller than or equal to the maximum capacity allocated for the intermediate COO buffer ($NNZ_{out} \le NNZ_{max}$).

We exploit this vacant space at runtime to perform an **In-Place CSF Conversion directly within the VRAM buffer**:

1. We use a specialized CUDA compression kernel to scan the sorted `d_out_indices` and detect structural boundaries where coordinates change.
2. The flat `d_out_vals` are shifted in-place to align with the new leaf nodes without colliding with other registers.
3. The remaining unutilized memory slots at the tail end of the unifed buffer are recycled as a **Scratchpad Space** to lay down the hierarchical `d_all_ptrs` segmentation arrays.

By restructuring the pointers internally, a single `cudaFree` on the base buffer pointer releases both the intermediate COO arrays and the final compiled `UniversalCSF` tree object simultaneously, bringing the net memory allocation cost down to zero.

## 4. Parallelizing the Data Loader on the CPU

Even with lightning-fast GPU kernels, the entire pipeline will stutter if loading raw COO arrays from file systems into VRAM is executed sequentially on a single CPU core. To maximize the input data-loading throughput, we must offload the bottleneck of sorting and data preparation onto the GPU.

Instead of running a heavy `std::sort` on the host, the data loader can execute a fast, parallel **Bitwise Radix Sort** using **NVIDIA Thrust**:

1. **Flat Row-Major Compacting**: During the raw text file stream, every incoming multidimensional coordinate is immediately packed into a single 64-bit unsigned integer (`uint64_t`) by allocating a fixed bit-width per dimension (e.g., 6 bits per axis allows mapping coordinate spaces up to 63 without overlap).
2. **Thrust Radix Sort**: The raw, un-ordered values and their compact 64-bit keys are shipped directly via a PCIe bulk transfer to the GPU. We then invoke `thrust::sort_by_key`, which triggers the underlying CUB hardware-level radix sort across thousands of CUDA cores simultaneously.
3. **Parallel Bitwise Unpack**: A simple, custom CUDA kernel reads the sorted keys and executes an inverse bitwise shift (`>>`) to populate the final `d_coordinate_matrix` instantaneously, cutting data-loading latencies down to a fraction of a millisecond.

## 5. Experimental Replicability & Performance Analysis

To validate the scalability of our custom framework against standard industry solutions, we configure an identical, high-rank 5D tensor contraction problem using the expression `"abcde,bdace->abe"` across an extensive range of non-zero intensities (`NNZ`). 

The following scripts allow full replicability of the data generation and the framework comparison benchmarks.

### A. Data Generation Script (`generate_data.py`)
```python
import numpy as np
import pandas as pd
import os

os.makedirs("benchmark_data", exist_ok=True)
DIM_A, DIM_B, DIM_C, DIM_D, DIM_E = 50, 50, 50, 20, 10
shape = (DIM_A, DIM_B, DIM_C, DIM_D, DIM_E)
shape2 = (DIM_B, DIM_D, DIM_A, DIM_C, DIM_E)
nnz_steps = [1, 10, 100, 1000, 100000, 500000]

np.random.seed(42)
print("=== Generating Sparse Datasets ===")

for nnz in nnz_steps:
    coordsA = set()
    coordsB = set()
    while len(coordsA) < nnz:
        coordsA.add((np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), 
                     np.random.randint(0, shape[2]), np.random.randint(0, shape[3]), 
                     np.random.randint(0, shape[4])))
        coordsB.add((np.random.randint(0, shape2[0]), np.random.randint(0, shape2[1]), 
                     np.random.randint(0, shape2[2]), np.random.randint(0, shape2[3]), 
                     np.random.randint(0, shape2[4])))

    dfA = pd.DataFrame(np.array(list(coordsA)))
    dfA['value'] = np.random.rand(nnz) * 10.0
    dfA.to_csv(f"benchmark_data/tensor_A_{nnz}.csv", index=False, header=False)

    dfB = pd.DataFrame(np.array(list(coordsB)))
    dfB['value'] = 2.0
    dfB.to_csv(f"benchmark_data/tensor_B_{nnz}.csv", index=False, header=False)
print("[SUCCESS] Datasets written into benchmark_data/")
```

### B. PyTorch & TensorFlow Scalability Script (`benchmark_sota.py`)
```python
import torch
import tensorflow as tf
import pandas as pd
import time
import csv

DIM_A, DIM_B, DIM_C, DIM_D, DIM_E = 50, 50, 50, 20, 10
shapeA = (DIM_A, DIM_B, DIM_C, DIM_D, DIM_E)
shapeB = (DIM_B, DIM_D, DIM_A, DIM_C, DIM_E)
nnz_steps = [1, 10, 100, 1000, 100000, 500000]

with open("benchmark_frameworks_scalability.csv", mode="w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["Framework", "NNZ", "Tempo_Mutazione_ms", "Tempo_Algebra_ms"])

    for nnz in nnz_steps:
        df_A = pd.read_csv(f"benchmark_data/tensor_A_{nnz}.csv", header=None)
        df_B = pd.read_csv(f"benchmark_data/tensor_B_{nnz}.csv", header=None)

        coords_A, vals_A = df_A.iloc[:, 0:5].values, df_A.iloc[:, 5].values
        coords_B, vals_B = df_B.iloc[:, 0:5].values, df_B.iloc[:, 5].values

        # --- PYTORCH ---
        start_mut = time.perf_counter_ns()
        pt_indices_A = torch.tensor(coords_A, dtype=torch.long).t().cuda()
        pt_values_A = torch.tensor(vals_A, dtype=torch.float32).cuda()
        pt_indices_B = torch.tensor(coords_B, dtype=torch.long).t().cuda()
        pt_values_B = torch.tensor(vals_B, dtype=torch.float32).cuda()
        T_A_pt = torch.sparse_coo_tensor(pt_indices_A, pt_values_A, shapeA).to_dense()
        T_B_pt = torch.sparse_coo_tensor(pt_indices_B, pt_values_B, shapeB).to_dense()
        torch.cuda.synchronize()
        t_mut_pt = (time.perf_counter_ns() - start_mut) / 1_000_000

        start_alg = time.perf_counter_ns()
        risultato_torch = torch.einsum("abcde,bdace->abe", T_A_pt, T_B_pt)
        torch.cuda.synchronize()
        t_alg_pt = (time.perf_counter_ns() - start_alg) / 1_000_000
        writer.writerow(["PyTorch", nnz, f"{t_mut_pt:.6f}", f"{t_alg_pt:.6f}"])

        # --- TENSORFLOW ---
        start_mut = time.perf_counter_ns()
        T_A_tf = tf.sparse.to_dense(tf.sparse.SparseTensor(coords_A, vals_A, shapeA))
        T_B_tf = tf.sparse.to_dense(tf.sparse.SparseTensor(coords_B, vals_B, shapeB))
        t_mut_tf = (time.perf_counter_ns() - start_mut) / 1_000_000

        start_alg = time.perf_counter_ns()
        risultato_tf = tf.einsum("abcde,bdace->abe", T_A_tf, T_B_tf)
        t_alg_tf = (time.perf_counter_ns() - start_alg) / 1_000_000
        writer.writerow(["TensorFlow", nnz, f"{t_mut_tf:.6f}", f"{t_alg_tf:.6f}"])
```

### Analysis of the Experimental Benchmarks

The empirical results collected from the scalability execution trace display a massive, undeniable gap in architectural behavior:

![Benchmarks via `einsum_gamma.cpp`](https://github.com/LogDS/LogDS.github.io/blob/main/assets/img/Paramoneur.png?raw=true)

### Critical Takeaways

1. **The Mutation Overhead**: For PyTorch and TensorFlow, the `Transformation` phase encompasses allocating a multi-dimensional matrix and filling it up completely via `.to_dense()`. At a high `NNZ` count of $100,000$, TensorFlow demands a staggering **$31.802096 \text{ ms}$** just to prepare the dense target buffer space. Our custom library, **Paramoneur**, parallelizes coordinate parsing using GPU Radix Sort, completing the entire buffer construction and pointer layout mapping in a crisp **$3.817380 \text{ ms}$**.
2. **Algebra Performance**: Once the matrices are densified, TensorFlow's `einsum` is remarkably fast ($0.880244 \text{ ms}$ at $100,000$ items) due to its utilization of NVIDIA's specialized core-level dense GEMM routines. However, **Paramoneur matches this performance natively on purely sparse structures**, executing the contraction in **$0.218112 \text{ ms}$** without ever consuming the massive, dangerous  footprint required by a dense matrix.
3. **The Core Limitation Revealed**: The steady, flat times exhibited by PyTorch and TensorFlow across varying `NNZ` counts clarify that their execution speed is bound exclusively to the *maximum dimensions of the dense shape*, totally disregarding the actual sparse density of the problem. As a consequence, if the tensor's shape increases slightly further, PyTorch and TensorFlow will hit a memory wall and crash, whereas our CSF-based engine scales gracefully alongside the actual data complexity.

## 6. Conclusions

The experimental evidence remarks a deep architectural limitation within mainstream frameworks: PyTorch and TensorFlow treat sparse tensors as low-level commodities, forcing memory-destructive densification layers whenever a high-rank complex contraction is requested. 

By building custom frameworks that leverage the formal principles of **Compressed Sparse Fiber (CSF)** structures coupled with single-allocation pointer arithmetic and bitwise radix sorting, developers can completely shatter the OOM barrier. Paramoneur proves that it is entirely possible to compute multi-axis contractions with sub-millisecond execution times, combining the memory efficiency of sparse representations with the raw, uncapped speed of dedicated GPU hardware.
