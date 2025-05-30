import matplotlib.pyplot as plt
import os

def plot_cache_and_scaling(cache_data, thread_data, output_dir):
    sizes_kb, times_cache = zip(*cache_data)
    plt.figure()
    plt.plot(sizes_kb, times_cache, marker='o')
    plt.xlabel("Array Size (KB)")
    plt.ylabel("Time (s)")
    plt.title("Cache Performance")
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cache_performance.png"))
    plt.close()

    threads, times_thread = zip(*thread_data)
    plt.figure()
    plt.plot(threads, times_thread, marker='o')
    plt.xlabel("Thread Count")
    plt.ylabel("Time (s)")
    plt.title("Thread Scalability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thread_scaling.png"))
    plt.close()

def plot_workload_comparison(results, output_dir):
    sort_data = results.get("Sorting Benchmarks", {})
    join_data = results.get("Join Benchmarks", {})
    ml_data = results.get("ML Benchmark (sklearn vs PyTorch)", {})

    # Sorting + Join
    plt.figure(figsize=(10, 5))
    labels = list(sort_data.keys()) + list(join_data.keys())
    times = [v.get("execution_time_sec", 0) for v in sort_data.values()] + \
            [v.get("execution_time_sec", 0) for v in join_data.values()]

    plt.bar(labels, times)
    plt.title("Sorting & Join Benchmark Times")
    plt.ylabel("Time (seconds)")
    plt.grid(True, axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workload_sort_join.png"))
    plt.close()

    # ML Comparison
    labels = []
    train_times = []
    infer_times = []
    for label, val in ml_data.items():
        labels.append(label)
        train_times.append(val.get("train_time_sec", 0))
        infer_times.append(val.get("inference_time_sec", 0))

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], train_times, width, label="Train")
    plt.bar([i + width/2 for i in x], infer_times, width, label="Infer")
    plt.xticks(x, labels)
    plt.title("ML Benchmark: Train vs Inference Time")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workload_ml.png"))
    plt.close()
