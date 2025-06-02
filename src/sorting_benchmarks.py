import time
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from utils import load_config, save_results
from hwinfo import collect_hw_info

def sort_numpy_array(size):
    arr = np.random.rand(size)
    start = time.perf_counter()
    sorted_arr = np.sort(arr)
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def sort_pandas_dataframe(rows):
    df = pd.DataFrame({
        "id": np.arange(rows),
        "value": np.random.rand(rows)
    })
    start = time.perf_counter()
    df_sorted = df.sort_values(by="value")
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def compare_sorting_methods(size, rows):
    return {
        "numpy_sort": sort_numpy_array(size),
        "pandas_sort": sort_pandas_dataframe(rows)
    }

def run_sorting_benchmarks():
	config = load_config()

	numpy_size = config["sorting_benchmarks"]["numpy_array_size"]
	pandas_size = config["sorting_benchmarks"]["pandas_dataframe_size"]

	results = {
		"Sorting Benchmarks": compare_sorting_methods(numpy_size, pandas_size),
	}

	data = {
		"Config Metadata": config["sorting_benchmarks"],
		"System Info": collect_hw_info(),
		"Benchmark Results": results
	}

	save_results(data, "sorting_benchmarks")

if __name__ == "__main__":
	print("Starting Sorting Benchmarks")
	run_sorting_benchmarks()
	print("Sorting Benchmarks Finished")
