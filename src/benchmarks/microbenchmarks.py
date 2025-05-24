import time
import numpy as np

def measure_fp_throughput():
    start = time.perf_counter()
    x = np.random.rand(10000, 10000)
    y = np.dot(x, x)
    end = time.perf_counter()
    return end - start

def measure_memory_bandwidth():
    a = np.random.rand(100_000_000)
    start = time.perf_counter()
    b = a * 2
    end = time.perf_counter()
    return end - start
