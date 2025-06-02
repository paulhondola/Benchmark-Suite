import os
import time
import psutil
import subprocess
import json
from datetime import datetime
from hwinfo import collect_hw_info

SQLITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_projects", "sqlite"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

def benchmark_compile(build_cmd, env=None, label=""):
    print(f"Running build command: {build_cmd}")

    process = psutil.Process()
    mem_before = process.memory_info().rss

    start = time.perf_counter()
    try:
        subprocess.run(build_cmd, shell=True, check=True, env=env, cwd=SQLITE_DIR)
    except subprocess.CalledProcessError as e:
        subprocess.run("make clean", shell=True, cwd=SQLITE_DIR)
        return {"error": str(e)}
    end = time.perf_counter()

    duration = end - start
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    return {
        "execution_time_sec": round(duration, 4),
        "avg_cpu_percent": round(cpu_percent, 1),
        "memory_delta_MB": round((mem_after - mem_before) / (1024**2), 2)
    }

def run_compile_benchmarks():
    gcc_env = os.environ.copy()
    gcc_env["CC"] = "gcc"

    clang_env = os.environ.copy()
    clang_env["CC"] = "clang"

    return {
        "gcc": benchmark_compile("make clean && make -j8", env=gcc_env, label="gcc"),
        "clang": benchmark_compile("make clean && make -j8", env=clang_env, label="clang")
    }

if __name__ == "__main__":
    results = run_compile_benchmarks()
    system_info = collect_hw_info()

    full_results = {
        "System Info": system_info,
        "Compiler Benchmark": results
    }

    print("=== Compilation Benchmark Finished ===")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(RESULTS_DIR, f"compile_benchmark_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"✅ Results saved to: {out_path}")
