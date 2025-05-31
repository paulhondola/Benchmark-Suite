#!/usr/bin/env python3
import subprocess
import re

def parse_powermetrics_gpu(output):
    """
    Parses `powermetrics --samplers gpu_power -n 1` output and returns:
      (gpu_cores_active_pct, gpu_memory_active_pct)
    If not found, returns (None, None).
    """
    gpu_util = None
    mem_util = None

    for line in output.splitlines():
        line = line.strip()
        m1 = re.match(r"GPU Cores Active:\s+(\d+)%", line)
        if m1:
            gpu_util = int(m1.group(1))
        m2 = re.match(r"GPU Memory Cores Active:\s+(\d+)%", line)
        if m2:
            mem_util = int(m2.group(1))

    return gpu_util, mem_util

def print_gpu_info():
    """
    Uses `powermetrics` to sample Apple Silicon GPU usage. Requires sudo
    or Full Disk Access to run without error.
    """
    print("=== GPU INFO ===")
    cmd = ["powermetrics", "--samplers", "gpu_power", "-n", "1"]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        gpu_pct, mem_pct = parse_powermetrics_gpu(output)

        if gpu_pct is not None:
            print(f"GPU Cores Active       : {gpu_pct}%")
        else:
            print("GPU Cores Active       : (not reported)")

        if mem_pct is not None:
            print(f"GPU Memory Cores Active: {mem_pct}%")
        else:
            print("GPU Memory Cores Active: (not reported)")

    except Exception:
        print("Cannot run `powermetrics`. Try running with sudo or grant Full Disk Access.")

if __name__ == "__main__":
    print_gpu_info()
