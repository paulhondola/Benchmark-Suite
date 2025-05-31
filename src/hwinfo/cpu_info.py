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

    # 1.4 CPU frequency (may be None on some macOS installs)
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

if __name__ == "__main__":
    print_cpu_info()
