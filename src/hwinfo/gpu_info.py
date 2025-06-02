import subprocess
import re
import json

def parse_powermetrics_gpu(output):
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

def collect_gpu_info():
    cmd = ["powermetrics", "--samplers", "gpu_power", "-n", "1"]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        gpu_pct, mem_pct = parse_powermetrics_gpu(output)
        return {
            "gpu_cores_active_percent": gpu_pct if gpu_pct is not None else "(not reported)",
            "gpu_memory_cores_active_percent": mem_pct if mem_pct is not None else "(not reported)"
        }
    except Exception:
        return {
            "error": "Cannot run `powermetrics`. Try running with sudo or grant Full Disk Access."
        }

if __name__ == "__main__":
    print(json.dumps(collect_gpu_info(), indent=2))
