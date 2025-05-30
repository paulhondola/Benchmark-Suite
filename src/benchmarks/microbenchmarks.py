import time
import numpy as np
from multiprocessing import Pool

def measure_fp_throughput(size=1000):
    x = np.random.rand(size, size)
    start = time.perf_counter()
    y = np.dot(x, x)
    end = time.perf_counter()
    return end - start

def measure_memory_bandwidth(size=100_000_000):
    a = np.random.rand(size)
    start = time.perf_counter()
    b = a * 2
    end = time.perf_counter()
    return end - start


def measure_memory_latency(jumps=1_000_000, stride=64):
    try:
        array = np.zeros(jumps * stride, dtype=np.int32)
    except MemoryError:
        return float("inf")  # or log failure cleanly
    start = time.perf_counter()
    s = 0
    for i in range(0, len(array), stride):
        s += array[i]
    end = time.perf_counter()
    return end - start


def measure_cache_effectiveness():
    sizes_kb = [4, 64, 512, 2048, 8192]  # ~L1 to L3 to RAM
    results = []

    for size_kb in sizes_kb:
        arr_size = (size_kb * 1024) // 8  # elements of float64
        arr = np.zeros(arr_size)
        start = time.perf_counter()
        for i in range(0, arr_size, 16):  # stride access to simulate cache behavior
            arr[i] += 1.0
        end = time.perf_counter()
        results.append((size_kb, end - start))

    return results  # list of (size_kb, time)


def dummy_workload_chunk(count):
    s = 0
    for i in range(count):
        s += i % 3
    return s

def measure_thread_scaling(max_threads=8, total_work=100_000_000):
    results = []
    for threads in range(1, max_threads + 1):
        work_per_thread = total_work // threads
        start = time.perf_counter()
        with Pool(processes=threads) as pool:
            pool.map(dummy_workload_chunk, [work_per_thread] * threads)
        end = time.perf_counter()
        results.append((threads, end - start))
    return results
