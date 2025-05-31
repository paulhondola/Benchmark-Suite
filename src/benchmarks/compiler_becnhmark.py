import os
import time
import psutil
import subprocess
import json
from datetime import datetime

SQLITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_projects", "sqlite"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

def benchmark_compile(build_cmd, env=None, label=""):
    print(f"Running build command: {build_cmd}")

    mem_before = psutil.Process().memory_info().rss

    start = time.perf_counter()
    cpu_before = psutil.cpu_percent(interval=None)
    try:
        subprocess.run(build_cmd, shell=True, check=True, env=env, cwd=SQLITE_DIR)
    except subprocess.CalledProcessError as e:
        subprocess.run("make clean", shell=True, cwd=SQLITE_DIR)
        return {"error": str(e)}
    duration = time.perf_counter() - start
    cpu_after = psutil.cpu_percent(interval=duration)
    mem_after = psutil.Process().memory_info().rss

    return {
        "execution_time_sec": round(duration, 4),
        "avg_cpu_percent": round(cpu_after, 1),
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
    print("\n=== Compile Benchmark Results ===")
    for compiler, data in results.items():
        print(f"[{compiler}] ->", data)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(RESULTS_DIR, f"compile_benchmark_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {out_path}")
