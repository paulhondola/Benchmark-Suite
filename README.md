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
git clone https://github.com/yourname/benchmark-suite.git
cd benchmark-suite
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then install dependencies using pyproject.toml:

```python
pip install .
```

**Note: On macOS, Metal support for TensorFlow will be automatically installed via conditional requirements.**

### 3. Running Benchmarks

```python
python3 microbenchmarks.py
python3 ml_benchmarks.py
python3 sorting_benchmarks.py
python3 compiler_benchmarks.py
python3 sql_benchmarks.py
```

### 4. Configiguration

Customize benchmark parameters inside the `config.json` file. This allows you to specify the parameters:

```JSON
{
  "microbenchmarks": {
    "matrix_size": 1000,
    "vector_size": 100000000,
    "max_threads": 8,
    "memory_jumps": 1000000,
    "memory_stride": 64,
    "thread_total_work": 100000000
  },
  "ml_workloads": {
    "sklearn": {
      "model": "logistic_regression",
      "n_samples": 100000,
      "n_features": 20,
      "max_iter": 1000
    },
    "pytorch": {
      "n_samples": 100000,
      "n_features": 20,
      "batch_size": 256,
      "epochs": 5
    },
    "tensorflow": {
      "n_samples": 100000,
      "n_features": 20,
      "batch_size": 256,
      "epochs": 5
    }
  }
}
```

### 5. Viewing Results

Results are saved in the `results/` directory in structured JSON format. You can view them using any JSON viewer or directly in Python:

```python
{
  "System Info": {
	  "cpu": {
	    "platform": "Darwin",
	    "architecture": "arm64",
	    "cpu": "Apple M1 Pro",
	    "logical_cores": 10,
	    ...
	  }
  }
  "Config Metadata": { ... },
  "Benchmark Result": {
    "Floating Point Throughput": { "execution_time_sec": 1.232 },
    ...
  }
}
```

### 6. Platform Support

- macOS: CPU, GPU via MPS, TensorFlow-Metal
- Linux: CPU + CUDA GPU (if available) **AMD GPU support in progress**
- Windows: CPU + CUDA GPU (if available) **AMD GPU support tricky due to ROCm**

### 7. Contributors

[Paul Hondola]
[Dan Ghincul]
