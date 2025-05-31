#!/usr/bin/env python3
import psutil

def print_cpu_info():
    """
    Prints per-core and overall CPU usage, number of logical vs. physical cores,
    and current RAM usage.
    """
    print("=== CPU INFO ===")

    # 1.1 Per-CPU usage (% over a 1-second interval)
    print("Per-core CPU usage (%):")
    per_cpu = psutil.cpu_percent(interval=1, percpu=True)
    for idx, pct in enumerate(per_cpu):
        print(f"  CPU {idx}: {pct:.1f}%")

    # 1.2 Overall CPU usage
    overall = psutil.cpu_percent(interval=1)
    print(f"Overall CPU usage: {overall:.1f}%")

    # 1.3 CPU counts
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    print(f"Logical CPUs : {logical}")
    print(f"Physical Cores: {physical}")

    freq = psutil.cpu_freq()
    if freq:
        print(f"CPU Frequency: {freq.current:.1f} MHz")
    else:
        print("CPU Frequency: (unavailable via psutil on macOS)")

    # 1.5 RAM usage
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GiB")
    print(f"Used RAM : {mem.used / (1024**3):.2f} GiB ({mem.percent:.1f}%)")
    print(f"Free RAM : {mem.available / (1024**3):.2f} GiB ({100 - mem.percent:.1f}%)")


import time
import multiprocessing

def stress_worker(duration_sec):
    end_time = time.perf_counter() + duration_sec
    x = 0.0
    while time.perf_counter() < end_time:
        x += 1.0  # scalar FP ops to generate heat

def cpu_stress_benchmark(duration_sec=10):
    cores = multiprocessing.cpu_count()
    start = time.perf_counter()
    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(stress_worker, [duration_sec] * cores)
    end = time.perf_counter()
    return {
        "cores_used": cores,
        "duration_sec": duration_sec,
        "wall_time_sec": round(end - start, 4)
    }


if __name__ == "__main__":
    print_cpu_info()
