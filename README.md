# Benchmark Suite

A cross-platform benchmarking suite designed to evaluate CPU, GPU, memory, and cache performance across diverse workloads, hardware setups, and software configurations.

## Features

- Microbenchmarks: memory throughput, latency, FP performance, cache, thread scaling
- Machine Learning workloads: PyTorch (CPU/GPU), scikit-learn, TensorFlow (macOS GPU via MPS)
- Sort operations / Dot Product (NumPy, Pandas)
- Compilation benchmarks (GCC vs Clang)
- SQL database performance (SQLite, PostgreSQL)
- Automatic system metadata collection (CPU, memory, cache, GPU)
- Configuration through JSON file
- Results saved in structured JSON with metadata

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/paulhondola/benchmark-suite.git
cd benchmark-suite
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then install dependencies using pyproject.toml.

For platform-specific machine learning backends, use:

```bash
# On Windows:
pip install .[windows]

# On macOS:
pip install .[macos]
```

**Note: On macOS, Metal support for TensorFlow will be automatically installed via conditional requirements.**

### 3. Running Benchmarks

To run the benchmarks, execute the following commands in your terminal:

```bash
cd src
```

Then run each benchmark script:

```python
python3 microbenchmarks.py
python3 ml_benchmarks.py
python3 sorting_benchmarks.py
python3 compiler_benchmarks.py
python3 sql_benchmarks.py
```

### 4. Configuration

Customize benchmark parameters inside the `config.json` file. This allows you to specify the parameters.

#### 4.1 Machine Learning Benchmarks

##### 4.1.1 Supported scikit-learn Models

- logistic_regression
- ridge
- lasso
- elasticnet
- random_forest
- extra_trees
- gradient_boosting
- svc
- gaussian_nb
- knn
- mlp

These are configured in config.json under the ml_workloads.sklearn.model field.

```JSON
"sklearn": {
    "model": "logistic_regression",
    "n_samples": 100000,
    "n_features": 20,
    "max_iter": 1000
}
```

#### 4.1.2 PyTorch Benchmark

Runs on CPU and GPU (if available)

Uses a simple feedforward neural network with configurable batch_size, iterations, and data size

Automatically uses Metal backend (mps) on macOS

```JSON
"pytorch": {
    "n_samples": 100000,
    "n_features": 20,
    "batch_size": 256,
    "epochs": 5
}
```

#### 4.1.3 TensorFlow Benchmark

Executes on both CPU and GPU (if available)

Benchmarks include training time, inference time, and accuracy

```JSON
"tensorflow": {
    "n_samples": 100000,
    "n_features": 20,
    "batch_size": 256,
    "epochs": 5
}
```

#### 4.2 Sorting Benchmarks

Sorting benchmarks include:

- NumPy: `numpy.sort()`
- Pandas: `pandas.DataFrame.sort_values()`

```JSON
"sorting_benchmarks": {
	"numpy_array_size": 10000000,
	"pandas_dataframe_size": 1000000
}
```

#### 4.3 Compilation Benchmarks

Compilation benchmarks compare GCC and Clang performance on a simple C project. (e.g. sqlite)

```JSON
"compile_benchmarks": {
	"compile_target": "<path to testing repo>"
}
```

#### 4.4 Microbenchmarks

Microbenchmarks include:

- Memory throughput
- Memory latency
- Floating point performance
- Cache performance
- Thread scaling

```JSON
"microbenchmarks": {
	"matrix_size": 10,
	"vector_size": 1000000,
	"max_threads": 8,
	"memory_jumps": 10000,
	"memory_stride": 64,
	"thread_total_work": 1000000
}
```

#### 4.5 SQL Benchmarks

#### TODO

SQL benchmarks test SQLite and PostgreSQL performance on a simple database schema.

```JSON
"sql_benchmarks": {
	"row_count": 100000,
	"threshold": 50000
}
```

### 5. Viewing Results

Results are saved in the `results/` directory in structured JSON format. You can view them using any JSON viewer or directly in Python:

```python
{
  "Config Metadata": { ... },
  "Benchmark Result": {
    "Floating Point Throughput": { "execution_time_sec": 1.232 },
    ...
  },
  "System Info": {
	  "cpu": {
	    "platform": "Darwin",
	    "architecture": "arm64",
	    "cpu": "Apple M1 Pro",
	    "logical_cores": 10,
	    ...
	  }
  }
}
```

### 6. Platform Support

- macOS: CPU, GPU via MPS, TensorFlow-Metal
- Linux: CPU + CUDA GPU (if available) **AMD GPU support in progress**
- Windows: CPU + CUDA GPU (if available) **AMD GPU support tricky due to ROCm**

### 7. Tested Platforms

The benchmark suite has been tested and validated on the following hardware and operating system configurations:

💻 MacBook Pro (Apple M1 Pro)

- OS: macOS Sequoia 15
- CPU: Apple M1 Pro (10-core)
- GPU: Integrated 16-core GPU
- Benchmarks Tested:
- ✅ Microbenchmarks (FP, memory, cache)
- ✅ Machine Learning (TensorFlow-metal, PyTorch MPS, scikit-learn)
- ✅ Sorting Benchmarks (NumPy, Pandas)
- ✅ SQL Benchmarks (SQLite only)
- ✅ Compilation Benchmarks (GCC, Clang)

🖥️ Lenovo IdeaPad Gaming

- OS: Windows 10 Home
- CPU: AMD Ryzen 5 7535HS with Radeon Graphics (6-core / 12-thread)
- GPU: Integrated AMD Radeon Graphics
- Benchmarks Tested:
- ✅ Microbenchmarks (FP, memory, cache)
- ✅ Machine Learning (CPU-only PyTorch, scikit-learn, CPU-only TensorFlow)
- ✅ Sorting Benchmarks (NumPy, Pandas)
- ✅ SQL Benchmarks (SQLite, PostgreSQL via psycopg2)
- ✅ Compilation Benchmarks (GCC/Clang via WSL)

🐧 MacBook Pro (M1 Pro) — Asahi Linux

- OS: Asahi Linux (Arch-based)
- CPU: Apple M1 Pro (10-core)
- GPU: Currently no GPU acceleration (as of kernel 6.8+)
- Benchmarks Tested:
- ✅ Microbenchmarks (FP, memory, cache)
- ✅ Machine Learning (scikit-learn, PyTorch CPU) # No TensorFlow support yet
- ✅ Sorting Benchmarks (NumPy, Pandas)
- ✅ SQL Benchmarks (SQLite)
- ✅ Compilation Benchmarks (GCC)

🪟 Windows 11 ARM64 (Virtual Machine)

- OS: Windows 11 ARM64 (Hyper-V VM)
- CPU: Apple M1 Pro (emulated)
- GPU: None (no GPU passthrough)
- Benchmarks Tested:
    - ✅ Microbenchmarks (FP, memory, cache)
    - ✅ Machine Learning:
    - ✅ PyTorch and ✅ scikit-learn — both work on Windows ARM64
    - ❌ TensorFlow — not supported on Windows ARM64
    - ✅ Sorting Benchmarks (NumPy, Pandas)
    - ✅ SQL Benchmarks (SQLite)
    - ✅ Compilation Benchmarks (Clang)

📦 Docker (Planned Testing)

- Target: Cross-platform container-based benchmarking
- Goals:
    - Run SQL benchmarks inside isolated PostgreSQL containers
    - Standardize system metrics across environments
    - Validate portability of benchmarks in headless/server setups
- Status: Docker setup integrated, full test suite validation in progress

### 8. Limitations

While the benchmark suite runs across a wide range of platforms, the following limitations currently apply:

- macOS (M1/M2/M3 with MPS GPU acceleration):

    - Although supported, MPS (Metal Performance Shaders) acceleration in TensorFlow and PyTorch is often slower than CPU for small to medium workloads due to higher launch latency and memory copy overhead.
    - TensorFlow GPU (via tensorflow-metal) may underperform compared to native CPU execution in practical scenarios.

- Windows (AMD Ryzen with integrated graphics):
    - On Windows ARM64, TensorFlow is not supported — no official PyPI wheels are available for this architecture. Installation will fail unless done through Conda or custom builds.
    - No GPU acceleration available for PyTorch or TensorFlow.
    - AMD GPUs are not supported for machine learning workloads on Windows due to lack of ROCm and DirectML integration in major ML frameworks.
    - Only CPU-based training/inference is available.
- Asahi Linux (aarch64 architecture):
    - TensorFlow is not officially supported on aarch64 Linux (including Asahi).
    - No GPU acceleration is available — machine learning benchmarks run solely on CPU.
    - PyTorch works in CPU mode, and some libraries may need to be built from source or replaced with alternatives.
- Docker (planned support):
    - GPU passthrough and acceleration support inside containers is limited and hardware-dependent.
    - Benchmarks run in Docker may vary slightly in performance due to container overhead and shared host resources.

### 9. Authors

[Paul Hondola][paulhondola@gmail.com]

[Dan Ghincul][ghinculdan@icloud.com]
