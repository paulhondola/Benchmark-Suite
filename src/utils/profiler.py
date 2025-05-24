import time
import psutil
import os

def profile_function(func, *args, **kwargs):
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / 1024 ** 2  # in MB
    cpu_before = psutil.cpu_percent(interval=None)

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / 1024 ** 2
    cpu_after = psutil.cpu_percent(interval=None)

    profile_data = {
        "execution_time_sec": end - start,
        "cpu_percent": cpu_after,
        "memory_usage_delta_MB": mem_after - mem_before
    }

    return result, profile_data
