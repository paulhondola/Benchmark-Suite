# 📊 Cross-Platform Benchmark Report

This document summarizes benchmark results collected from multiple systems using the Benchmark Suite. The tests span microbenchmarks, machine learning workloads, SQL operations, sorting algorithms, and compilation performance.

---

## 📋 Platforms Tested

### 💻 MacBook Pro (Apple M1 Pro)

- **OS:** macOS Sequoia 15
- **CPU:** Apple M1 Pro (10-core)
- **GPU:** Integrated 16-core GPU
- **Status:** TensorFlow-Metal, PyTorch MPS, and scikit-learn fully functional.

### 🖥️ Lenovo IdeaPad Gaming

- **OS:** Windows 10 Home
- **CPU:** AMD Ryzen 5 7535HS with Radeon Graphics (6-core / 12-thread)
- **GPU:** Integrated AMD Radeon Graphics
- **Status:** CPU-only ML workloads; no GPU acceleration for TensorFlow or PyTorch.

### 🐧 MacBook Pro (M1 Pro) — Asahi Linux

- **OS:** Asahi Linux (Fedora-based)
- **CPU:** Apple M1 Pro (10-core)
- **GPU:** Currently no GPU acceleration (as of kernel 6.8+)
- **Status:** CPU-only mode for all ML workloads; TensorFlow not supported on aarch64 Linux.

### 🐳 Docker Container (Apple M1 Pro)

- **OS:** Debian (Docker)
- **CPU:** Apple M1 Pro (10-core)
- **GPU:** No GPU Passthrough
- **Status:** Limited to 8GB RAM; runs microbenchmarks and ML workloads in a containerized environment.

---

## ⚠️ Limitations

- **TensorFlow** is not available on Windows ARM64 and Linux ARM64.
- **MPS GPU Backend** on macOS (Metal) is often slower than CPU execution.
- **No GPU acceleration** on AMD iGPUs for TensorFlow (Windows/Linux).
- **Docker containers** are limited by cgroup constraints and may show reduced RAM or CPU visibility.

---

## 🧮 Microbenchmark Results

🍏 macOS – Apple M1 Pro

System Info: 10-core Apple M1 Pro @ 3228 MHz, 16GB RAM
Observations:

- Floating Point Throughput:

    - FLOPS: 250 MFLOPS over 8e-6 s
    - ✅ Excellent throughput due to Apple silicon’s optimized matrix acceleration.

- Scalar FP Add Latency:

    - ~43.9 MFLOPS
    - ⚠️ Limited by single-core frequency and Python loop overhead.
    - Memory Bandwidth:
    - ~9.24 GB/s
    - 👍 Very good bandwidth performance for vectorized NumPy operations.

- Memory Latency:

    - ~0.8 ns
    - ✅ Extremely low, demonstrating Apple’s high-speed L1/L2 caches.

- Pointer Chasing Latency:

    - ~130 ns
    - ⚠️ Slightly high due to poor data locality.
    - Cache Performance:
    - Shows graceful degradation across cache levels.
    - ✅ Confirms a well-behaved cache hierarchy.

- Thread Scalability:

    - Improves up to 6 threads (~0.11s), then plateaus.
    - 🔁 Indicates optimal saturation point for CPU-bound tasks.

⸻

🐧 Asahi Linux (Apple M1 Pro via Linux – aarch64)

System Info: 10-core M1 Pro under Linux (reported 876 MHz), 15GB RAM
Observations:

- Floating Point Throughput:

    - 100 MFLOPS (lower than macOS)
    - ⚠️ Linux reports lower frequency & fewer optimizations.
    - Scalar FP Add Latency:
    - ~22.3 MFLOPS
    - ⚠️ Lower than macOS due to kernel overhead and lack of JIT.

- Memory Bandwidth:

    - ~12.1 GB/s
    - ✅ Excellent for aarch64 and indicates efficient vectorization.
    - Memory Latency:
    - ~0.79 ns
    - ✅ Matches macOS — underlying silicon matters most here.
    - Pointer Chasing Latency:
    - ~131 ns
    - ⚠️ Slightly worse than macOS, likely due to kernel scheduling.

- Cache Performance:

    - Slightly higher degradation past 2MB.
    - ❗ Cache info not available — possibly due to sysfs or driver limitations.

- Thread Scalability:

    - Best performance at 4–6 threads.
    - ✅ Well-distributed but fluctuating due to scheduler behavior.

## 🧠 Machine Learning Observations

| Platform      | Backend     | Model                       | Execution Time | Notes                             |
| ------------- | ----------- | --------------------------- | -------------- | --------------------------------- |
| Apple M1 Pro  | MPS (Metal) | RandomForest (scikit-learn) | ~5.2s          | Fast if model is parallelizable   |
| Windows (AMD) | CPU         | PyTorch                     | ~6.8s          | Efficient CPU workload            |
| Asahi Linux   | CPU         | scikit-learn                | ~9.1s          | Slower due to lack of GPU support |

---

## 📋 SQL Benchmark Samples

| Platform    | Query       | DB     | Time (sec) | Notes                              |
| ----------- | ----------- | ------ | ---------- | ---------------------------------- |
| AMD Ryzen   | INSERT      | SQLite | 7.01       | High RAM usage during batch insert |
| M1 Pro      | SELECT JOIN | SQLite | 3.05       | Optimized for memory-mapped files  |
| Asahi Linux | AGGREGATE   | SQLite | 2.67       | CPU-bound but acceptable latency   |

---

## 📊 Sorting Benchmarks

| Platform  | Library | Time (sec) | Notes                                 |
| --------- | ------- | ---------- | ------------------------------------- |
| M1 Pro    | NumPy   | 0.89       | Fast array handling via NEON SIMD     |
| AMD Ryzen | Pandas  | 1.45       | Higher overhead due to object columns |

---

## ⚙️ Compilation Benchmarks

| Platform     | Time (sec) | CPU Time (s) | Notes                                  |
| ------------ | ---------- | ------------ | -------------------------------------- |
| Apple M1 Pro | 8.1        | 7.9          | Using Clang from Xcode                 |
| AMD Ryzen    | 23.5       | 24.5         | Native GCC via WSL                     |
| Asahi Linux  | 12.2       | 14.2         | Minor delay from paging and I/O limits |

---

## ⚠️ Limitations

- **TensorFlow is unsupported** on Windows ARM64 and Linux ARM64.
- **No GPU acceleration** on Linux (M1) or AMD GPUs (Windows).
- **MPS is slower than CPU** in some TensorFlow use cases.
- Results may vary depending on containerization (Docker) and background load.

---

## 🧪 Docker Support

All benchmarks support containerized execution. Results are volume-mapped for persistence.

```bash
docker-compose run micro
docker-compose run ml
docker-compose run sql
```

Volume paths should be updated in `docker-compose.yml` to mount `./results`.
