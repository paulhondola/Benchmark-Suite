from utils import load_config, save_results
from hwinfo import collect_hw_info
import os
import time
import psutil
import subprocess

def benchmark_compile(build_cmd, env=None, label="", path=None):
    print(f"Running build command: {build_cmd}")

    process = psutil.Process()
    mem_before = process.memory_info().rss

    start = time.perf_counter()
    try:
        subprocess.run(build_cmd, shell=True, check=True, env=env, cwd=path)
    except subprocess.CalledProcessError as e:
        subprocess.run("make clean", shell=True, cwd=path)
        return {"error": str(e)}
    end = time.perf_counter()

    duration = end - start
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    return {
        "execution_time_sec": round(duration, 6),
        "avg_cpu_percent": round(cpu_percent, 1),
        "memory_delta_MB": round((mem_after - mem_before) / (1024**2), 2)
    }

def run_compile_benchmarks(config):
	target_path = config["compile_target"]

	gcc_env = os.environ.copy()
	gcc_env["CC"] = "gcc"

	clang_env = os.environ.copy()
	clang_env["CC"] = "clang"

	results = {
		"gcc": benchmark_compile("make clean && make -j8", env=gcc_env, label="gcc", path=target_path),
  		"clang": benchmark_compile("make clean && make -j8", env=clang_env, label="clang", path=target_path)
	}

	data = {
	    "Config Metadata": config,
		"Benchmark Results": results,
		"System Info": collect_hw_info()
    }

	save_results(data, "compile_benchmarks")

if __name__ == "__main__":
	config = load_config()
	print("Starting Compiler Benchmarks")
	run_compile_benchmarks(config["compile_benchmarks"])
	print("Compiler Benchmark Finished")
