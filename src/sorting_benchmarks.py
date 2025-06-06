from utils import load_config, save_results
from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage
import time
import numpy as np
import pandas as pd

import psutil
import numpy as np
import pandas as pd
import time

# Function to measure CPU time used by the current process
def get_cpu_time():
    process = psutil.Process()
    return process.cpu_times().user + process.cpu_times().system

# Function to measure memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def sort_numpy_array(size):
    # Create a random numpy array
    arr = np.random.rand(size)

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()

    # Perform sorting
    sorted_arr = np.sort(arr)

    # End measuring execution time
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 4)

    # Calculate CPU time and memory usage during the operation
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        }
    }

def sort_pandas_dataframe(rows):
    # Create a random pandas DataFrame
    df = pd.DataFrame({
        "id": np.arange(rows),
        "value": np.random.rand(rows)
    })

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()

    # Perform sorting
    df_sorted = df.sort_values(by="value")

    # End measuring execution time
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 4)

    # Calculate CPU time and memory usage during the operation
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        }
    }

def run_sorting_benchmarks(config):
	numpy_size = config["numpy_array_size"]
	pandas_size = config["pandas_dataframe_size"]

	results = {
        "numpy_sort": sort_numpy_array(numpy_size),
        "pandas_sort": sort_pandas_dataframe(pandas_size),
	}

	data = {
		"Config Metadata": config,
		"Benchmark Results": results,
		"System Info": collect_hw_info()
	}

	save_results(data, "sorting_benchmarks")

if __name__ == "__main__":
	config = load_config()
	print("Starting Sorting Benchmarks...")
	run_sorting_benchmarks(config["sorting_benchmarks"])
