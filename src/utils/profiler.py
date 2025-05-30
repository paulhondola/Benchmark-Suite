def profile_function(func, *args, **kwargs):
    import time
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 ** 2  # in MB

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / 1024 ** 2
    mem_delta = max(0.0, round(mem_after - mem_before, 2))

    profile_data = {
        "execution_time_sec": round(end - start, 4),
        "memory_usage_MB": mem_delta
    }

    return result, profile_data
