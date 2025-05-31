import os
import json
import yaml
import time
from datetime import datetime
import matplotlib.pyplot as plt

from benchmarks import microbenchmarks as mb
from benchmarks import workloads
from src.benchmarks.compiler_becnhmark import benchmark_compile
from src.hwinfo.cpu_info import cpu_stress_benchmark
from utils.metrics import collect_system_info
from utils.profiler import profile_function
from utils.plotting import plot_cache_and_scaling, plot_workload_comparison


# Resolve paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "default.yaml")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_latency_comparison(results, output_dir):
    latency_tests = {
        "Scalar FP Add": results.get("Scalar FP Add Latency", {}).get("execution_time_sec", 0),
        "Pointer Chasing": results.get("Pointer Chasing Latency", {}).get("execution_time_sec", 0),
        "Strided Latency": results.get("Memory Latency", {}).get("execution_time_sec", 0)
    }
    labels = list(latency_tests.keys())
    times = list(latency_tests.values())

    plt.figure()
    plt.bar(labels, times)
    plt.ylabel("Time (s)")
    plt.title("Latency Benchmark Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"))
    plt.close()

def run_all():
    config = load_config(CONFIG_PATH)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "system": collect_system_info(),
        "run_id": f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    }

    # Microbenchmarks
    results['Floating Point Throughput'] = profile_function(mb.measure_fp_throughput)[1]
    results['Scalar FP Add Latency'] = profile_function(mb.measure_scalar_fp_add)[1]
    results['Memory Bandwidth (read+compute)'] = profile_function(mb.measure_memory_bandwidth)[1]
    results['Memory Read Throughput'] = profile_function(mb.measure_memory_read_bandwidth)[1]
    results['Memory Latency (strided)'] = profile_function(mb.measure_memory_latency)[1]
    results['Pointer Chasing Latency (random access)'] = profile_function(mb.measure_pointer_chasing_latency)[1]
    results['Cache Performance'] = mb.measure_cache_effectiveness()
    results['Thread Scalability'] = mb.measure_thread_scaling()
    results["CPU Stress Benchmark"] = cpu_stress_benchmark(duration_sec=10)

    # Workload Benchmarks
    results["Sorting Benchmarks"] = workloads.compare_sorting_methods()
    results["Join Benchmarks"] = workloads.run_join_benchmark()
    results["ML Benchmark (scikit-learn)"] = workloads.run_ml_benchmark()
    results["ML Benchmark (sklearn vs PyTorch)"] = workloads.run_ml_benchmark_comparison()

    gcc_env = os.environ.copy()
    gcc_env["CC"] = "gcc"

    clang_env = os.environ.copy()
    clang_env["CC"] = "clang"

    results = {
        "gcc": benchmark_compile("make clean && make -j8", env=gcc_env),
        "clang": benchmark_compile("make clean && make -j8", env=clang_env)
    }

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{results['run_id']}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    from utils.plotting import plot_cache_and_scaling, plot_workload_comparison
    cache_data = results['Cache Performance']['results']
    thread_data = results['Thread Scalability']['results']
    plot_cache_and_scaling(cache_data, thread_data, RESULTS_DIR)
    plot_workload_comparison(results, RESULTS_DIR)
    plot_latency_comparison(results, RESULTS_DIR)

    print("✅ Benchmarking complete. Results and plots saved.")

if __name__ == "__main__":
    run_all()
