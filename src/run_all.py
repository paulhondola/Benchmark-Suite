import os
import json
import yaml
from datetime import datetime

import matplotlib.pyplot as plt
from benchmarks import microbenchmarks as mb
from utils.metrics import collect_system_info
from utils.profiler import profile_function

# Resolve root project directory regardless of where the script is run from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "default.yaml")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_cache_and_scaling(cache_data, thread_data, output_dir):
    sizes_kb, times_cache = zip(*cache_data)
    plt.figure()
    plt.plot(sizes_kb, times_cache, marker='o')
    plt.xlabel("Array Size (KB)")
    plt.ylabel("Time (s)")
    plt.title("Cache Performance")
    plt.xscale("log")
    plt.grid(True)
    plt.savefig(f"{output_dir}/cache_performance.png")
    plt.close()

    threads, times_thread = zip(*thread_data)
    plt.figure()
    plt.plot(threads, times_thread, marker='o')
    plt.xlabel("Thread Count")
    plt.ylabel("Time (s)")
    plt.title("Thread Scalability")
    plt.grid(True)
    plt.savefig(f"{output_dir}/thread_scaling.png")
    plt.close()

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = load_config(CONFIG_PATH)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "system": collect_system_info()
    }

    # Integer and FP throughput
    _, fp_stats = profile_function(mb.measure_fp_throughput, 1000)
    results['Floating Point Throughput'] = fp_stats

    # Memory latency and bandwidth
    _, bw_stats = profile_function(mb.measure_memory_bandwidth)
    _, lat_stats = profile_function(mb.measure_memory_latency)
    results['Memory Bandwidth'] = bw_stats
    results['Memory Latency'] = lat_stats

    # Cache performance
    cache_results = mb.measure_cache_effectiveness()
    results['Cache Performance'] = cache_results

    # Thread scaling
    thread_results = mb.measure_thread_scaling()
    results['Thread Scalability'] = thread_results

    # Save JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save plots
    plot_cache_and_scaling(cache_results, thread_results, RESULTS_DIR)
    print(f"\n✅ Results saved to: {json_path}")
    print("📊 Plots saved: cache_performance.png, thread_scaling.png")

if __name__ == "__main__":
    run_all()
