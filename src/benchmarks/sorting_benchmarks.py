# --- workloads_sorting.py ---
import time
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def run_all_sorting_benchmarks(size, rows):
    return {
        "Sorting Benchmarks": compare_sorting_methods( size, rows),
        "Config": {
            "numpy_array_size": size,
            "pandas_row_count": rows
        }
    }

if __name__ == "__main__":
	numpy_size = 100_000_000
	pandas_size = 10_000_000
	results = run_all_sorting_benchmarks(numpy_size, pandas_size)

	print("=== Sorting Benchmarks Finished ===")

	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	out_path = os.path.join(RESULTS_DIR, f"sorting_benchmarks_{timestamp}.json")

	with open(out_path, "w") as f:
		json.dump(results, f, indent=2)

	print(f"✅ Results saved to: {out_path}")
