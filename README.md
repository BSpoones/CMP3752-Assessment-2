# CMP3752-Assessment-2

# CMP3752M — Parallel Programming  
Complete Lecture Notes (raw Markdown, detailed)

---

## Lecture 1 – Data‑Parallel Programming (C. Fox)

### House‑keeping
- **Assessment**  
  - Coursework (40 %) – OpenCL kernel + 1500‑word report  
  - 24‑h *open‑book* theory controlled assessment (60 %)  
  - Pass both components individually.  
- **Cheating detection**  
  Plagiarism tools do AST diff & ML heuristics; variable renames fool no‑one.  
- **Reading list**  
  Hennessy & Patterson *Computer Architecture* ch 15,  
  Kirk & Hwu *Programming Massively Parallel Processors* ch 1–2,  
  Mattson et al. *Structured Parallel Programming* ch 1.

### Why parallel?
- Transistor counts obey (weaker) Moore’s Law, clock speed hit the “heat wall”.  
- Dynamic power ∝ C·V²·F – higher F melts silicon.  
- Only sustainable path: **more slower cores** → throughput ↑ without TDP ↑.

### Terminology
| Term | Practical meaning |
|------|-------------------|
| **Parallelism** | True simultaneous execution on multiple HW contexts |
| **Concurrency** | Time‑sharing illusion on one context |
| **Data parallelism** | Same program on many data (SIMD / SPMD). |
| **Task parallelism** | Potentially different programs / branches. |

### Amdahl’s Law revisit
Speed‑up   S(p)=1 / ((1−f)+f/p)  
Let f = 0.90, p = ∞ ⇒ S₍∞₎ = 10× — never better.  
**Moral:** serial crumbs dominate; spend effort where *f* is big.

### CPU SIMD timeline
- MMX (1996) 64‑bit  
- SSE → SSE4 (128‑bit)  
- AVX → AVX‑512 (512‑bit, mask registers)  
- Intel DLBoost / VNNI (int8 dot‑product)  
Use intrinsics or rely on auto‑vectoriser (often fails).

### GPU architecture snapshot
```
Host ─► Device
        ├─► Compute Unit 0 ─► 64 ALUs (= work‑group)
        ├─► Compute Unit 1
        ⋮
```
- Thousands of ALUs, tiny caches, 7 ns register, 400 ns global DRAM.  
- Warps/wavefronts of 32/64 threads issue in lock‑step.

### OpenCL execution model
1. **Host (program)** builds **kernel** and enqueues ND‑range.  
2. **Work‑item** = lightweight thread.  
3. **Work‑group** = collection of work‑items map to one compute unit.  
4. Barriers only inside a work‑group.

### Divergence & masks
- If `(if(id%2)…else…)` both branches executed, inactive lanes masked — but *serialized*.

### Minimal vector‑add
```c
// host.cpp (snippet)
clEnqueueNDRangeKernel(q, k_add, 1, NULL,
                       &global, &local, 0, NULL, NULL);

// add.cl
kernel void add(global const int* A,
                global const int* B,
                global       int* C)
{
    const uint id = get_global_id(0);
    C[id] = A[id] + B[id];
}
```

### Take‑aways
- Keep kernels branch‑light, maximise arithmetic intensity, move data once.

---

## Lecture 2 – Task‑Parallel (MIMD) Programming

### Memory architectures
| | Shared‑mem SMP | Distributed‑mem MPI |
|---|---|---|
| Address space | Single | Private per node |
| Pros | Easy sharing | Unlimited scaling |
| Cons | Cache‑coherency traffic | Manual send/recv |

### Processes vs threads
- **Process** = heavy, isolated, kernel‑mode switch.  
- **Thread** (pthread/OpenMP task) shares addr‑space, cheap context switch.

### NUMA rule‑of‑thumb
First‑touch sets page owner; pin threads near the data or get 2× latency.

### Cluster / Grid / Cloud zoo
- **MPI/HPC cluster** – low‑latency Infiniband, hundreds of nodes.  
- **Grid** – bag‑of‑tasks on heterogeneous lab desktops (SGE, HTCondor).  
- **Cloud** – credit card & K8s; treat nodes as cattle.

### ROS 2 (DDS)
High‑level pub‑sub, QoS knobs. Latency 100 µs vs MPI 1 µs, but devs don’t care.

### Instruction‑less hardware
- Intel Meteor‑Lake NPU: 4 096 MACs, 48 TOPS INT8.  
- TPU v4: 8192× 8‑bit MAC array, 275 TOPS per chip.  
- FPGA: configure gates, true spatial parallelism; Verilog not for faint‑hearted.

---

## Lecture 3 – Parallel Theory

### Work & span
Given DAG of tasks:  
- **Work W** = total nodes count (serial cost).  
- **Span S** = longest path (critical path).  
Max parallelism Pₘₐₓ = W/S; real speed‑up ≤ Pₘₐₓ.

### Brent’s theorem
With *p* processors: Tₚ ≤ W/p + S. Gives scheduling bound.

### Akra–Bazzi
Handles recurrences T(n)=Σ aᵢ T(n/bᵢ)+g(n) with unequal splits.

### Common hazards
- **Race**: two writes, lost update.  
- **Atomicity violation**: check‑then‑act gap.  
- **Deadlock**: circular wait due to lock ordering.

### Design patterns catalogue
- Map, Reduce, Scan, Stencil, Pipeline, Farm.  
- Compose → correctness; tune → performance.

---

## Lecture 4 – Parallel Patterns 1 (Map, Stencil)

### Map
Element‑wise independent transform.  
Optimization: **fusion** – chain of maps → one kernel to cut DRAM traffic.

### Stencil
Example 3×3 blur: each output cell needs 3 × 3 input window.  
Steps:  
1. Load tile into local memory with halo.  
2. `barrier`.  
3. Compute centre points.  
Performance: bandwidth ≈ 4 × better than naïve global loads (slide benchmark).

---

## Lecture 5 – Communication, Sync & Atomics

### Memory hierarchy latencies (AMD MI250)
| Level | Latency |
|-------|---------|
| Register | 1 cycle |
| Local SRAM 64 KB | 8 cy |
| L2 cache 2 MB | 35 cy |
| HBM2 | 300 cy |
| PCIe host | > 2 000 cy |

### 1‑D hot‑spot stencil optimisation
```c
local float tile[TILE+2*R];   // R = radius
uint gid = get_global_id(0);
uint lid = get_local_id(0);
tile[lid+R] = in[gid];
if(lid < R){
   tile[lid] = in[gid-R];
   tile[lid+R+TILE] = in[gid+TILE];
}
barrier(CLK_LOCAL_MEM_FENCE);
float outv = 0.f;
for(int k=-R;k<=R;++k) outv += coeff[k+R]*tile[lid+R+k];
out[gid] = outv;
```

### Barriers & fences
`CLK_LOCAL_MEM_FENCE`, `CLK_GLOBAL_MEM_FENCE`, ensure prior writes visible.

### Atomics
Global counter increment ~300 ns vs 4 ns register add → **privatise** before atomic merge.

---

## Lecture 6 – Patterns 2: Map‑Reduce & Scan

### Reduce implementation
Binary‑tree in local mem: half threads active each step (`N/2`, `N/4`, …).  
Unroll last warp to kill branch divergence.

### Scan algorithms
1. **Hillis–Steele** (inclusive, simple), work N log N, span log N.  
2. **Blelloch** (up‑sweep/down‑sweep), work 2 (N−1), span 2 log N.  
Block scan pseudocode shown on slide 17.

### Real performance (RTX 3070, 10 m elems)
- Thrust inclusive scan: 3.2 GB/s  
- Custom Blelloch + warp shuffle: 6.9 GB/s

---

## Lecture 7 – Histogram

### Work‑group private hist
- Allocate `__local uint hist[BINS]`.  
- Each WI initialises slice (loop striding by local_size).  
- For each element: `atomic_inc(&hist[val]);` (local mem).  
- `barrier`, then WI0 writes its `hist[b]` via global `atomic_add`.

GPU saturation curve: global atomic version collapses beyond 4 m inputs; WG‑private scales to 32 m.

---

## Lecture 8 – Searching & Coalescing

### Parallel single‑key binary search
```c
global int pos = INT_MAX;
kernel void psearch(global const int* A, int N, int key){
   uint gid=get_global_id(0);
   uint chunk = (N+WG*NITEM-1)/ (WG*NITEM);
   uint lo = gid*chunk, hi = min((gid+1)*chunk-1, N-1);
   while(lo<=hi){
      uint mid=(lo+hi)>>1;
      int v=A[mid];
      if(v==key) atomic_min(&pos, mid);
      if(v<key) lo=mid+1; else hi=mid-1;
   }
}
```

### Coalesced reduction trick
Change indexing from `id*stride` (interleaved) to `base+id` (sequential) inside each pass → 3× bandwidth.

---

## Lecture 9 – Performance & Optimisation

### Roofline analysis demo
Peak flops = 19.5 TFLOP/s FP32, Peak BW = 512 GB/s → ridge point @ 38 flop/B.  
Dot‑product AI ≈ 2 flop/B → memory‑bound; can’t exceed 1.0 TFLOP/s no matter ALUs.

### Four horsemen of slow
1. **Serial fraction**  
2. **Synchronisation overhead**  
3. **Memory wall**  
4. **Load imbalance**

### Optimisation flowchart
1. **Choose better algorithm** (parallel quicksort → radix sort).  
2. **Use right data structure** (array of structs → struct of arrays).  
3. **Exploit locality** (tiling, blocking).  
4. **Micro‑tune** (pld, prefetch, inline asm).

---

## Lecture 10 – Gather / Scatter & Memory Re‑org

### Definitions
- **Gather** `out[i] = in[index[i]]` (indirect read).  
- **Scatter** `out[index[i]] = in[i]` (indirect write).

### Collision strategies for scatter
| Strategy | Deterministic? | Cost |
|----------|----------------|------|
| Atomic scatter | No | cheap code, poor perf |
| Permutation (no dup indices) | Yes | need prefix‑sum of bucket sizes |
| Merge scatter (associative) | Yes | add instead of overwrite |
| Priority scatter | Yes | tie‑break on thread id |

### Stream compact pattern
1. Predicate map → flag array (0/1).  
2. Exclusive scan flags → indices.  
3. Scatter via indices → compacted output.

---

## Lecture 11 – Data Re‑organisation deep dive

### Why re‑order?
- Enable coalesced loads/stores.  
- Remove bank conflicts in shared memory (16‑way in NVIDIA).  
- Group hot fields → cache line.

### Key‑value radix sort (32‑bit)
- 4 passes of 8 bits.  
- For each pass:  
  1. Histogram 256 bins.  
  2. Prefix sum hist.  
  3. Scatter to temp buffer using bin offset++.  
  4. Swap buffers.

### Performance numbers (RTX 3070, 16 m pairs)
- Thrust sort: 450 M pairs/s  
- Custom key‑value radix: 1.1 G pairs/s

---

# End
