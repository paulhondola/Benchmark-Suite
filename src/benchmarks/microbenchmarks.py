import time
import random
import numpy as np
import os
from multiprocessing import Pool

def measure_fp_throughput(size=1000):
    x = np.random.rand(size, size)
    start = time.perf_counter()
    _ = np.dot(x, x)
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_scalar_fp_add(iterations=100_000_000):
    x = 1.0
    start = time.perf_counter()
    for _ in range(iterations):
        x += 1.0
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_memory_bandwidth(size=100_000_000):
    a = np.random.rand(size)
    start = time.perf_counter()
    _ = a * 2
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_memory_read_bandwidth(size=100_000_000):
    a = np.random.rand(size)
    total = 0.0
    start = time.perf_counter()
    for i in range(len(a)):
        total += a[i]  # read only
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_memory_latency(jumps=1_000_000, stride=64):
    try:
        array = np.zeros(jumps * stride, dtype=np.int32)
    except MemoryError:
        return {"execution_time_sec": float("inf")}
    start = time.perf_counter()
    for i in range(0, len(array), stride):
        _ = array[i]
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_pointer_chasing_latency(size=1_000_000):
    arr = list(range(size))
    random.shuffle(arr)
    next_index = arr[0]
    start = time.perf_counter()
    for _ in range(size):
        next_index = arr[next_index]
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def measure_cache_effectiveness():
    sizes_kb = [4, 64, 512, 2048, 8192]  # ~L1 to L3 to RAM
    results = []
    for size_kb in sizes_kb:
        arr_size = (size_kb * 1024) // 8  # elements of float64
        arr = np.zeros(arr_size)
        start = time.perf_counter()
        for i in range(0, arr_size, 16):
            arr[i] += 1.0
        end = time.perf_counter()
        results.append((size_kb, round(end - start, 6)))
    return {"results": results}

def dummy_workload_chunk(count):
    s = 0
    for i in range(count):
        s += i % 3
    return s

def measure_thread_scaling(max_threads=8, total_work=100_000_000):
    cpu_count = os.cpu_count()
    max_threads = min(max_threads, cpu_count)
    results = []
    for threads in range(1, max_threads + 1):
        work_per_thread = total_work // threads
        start = time.perf_counter()
        with Pool(processes=threads) as pool:
            pool.map(dummy_workload_chunk, [work_per_thread] * threads)
        end = time.perf_counter()
        results.append((threads, round(end - start, 4)))
    return {"cpu_count": cpu_count, "results": results}
