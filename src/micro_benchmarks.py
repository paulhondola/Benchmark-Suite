from utils import load_config, save_results
from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage
import time
import random
import numpy as np
import os
from multiprocessing import Pool

def measure_fp_throughput(size):
    x = np.random.rand(size, size)

    # Measure initial hardware metrics
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()
    _ = np.dot(x, x)  # Perform matrix multiplication
    end = time.perf_counter()

    # Measure final hardware metrics
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate total floating point operations (FLOPs)
    flops = 2 * size ** 3  # Number of floating point operations (for matrix multiplication)

    # Execution time in seconds
    exec_time = round(end - start, 6)

    return {
        "Benchmark Results": {
            "Floating Point Operations (FLOPs)": flops,
            "FLOPS": flops / exec_time,
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": round(cpu_time_end - cpu_time_start, 6),
            "Memory Usage (MB)": round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)
        }
    }

def measure_scalar_fp_add(iterations):
    x = 1.0

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()

    # Perform scalar floating-point addition in a loop
    for _ in range(iterations):
        x += 1.0  # Simple floating-point addition

    # End measuring execution time
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate total floating point operations (FLOPs)
    # Each addition involves 1 floating-point operation, so FLOPs = iterations
    flops = iterations

    return {
        "Benchmark Results": {
            "Floating Point Operations (FLOPs)": flops,
            "FLOPS": flops / exec_time,
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": round(cpu_time_end - cpu_time_start, 6),
            "Memory Usage (MB)": round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)
        }
    }

def measure_memory_bandwidth(size):
    # Create an array of random floating-point numbers
    a = np.random.rand(size)  # by default, np.random.rand creates an array of float64 values

    # Calculate the total data size in bytes (size * element size)
    data_size = a.nbytes  # This is equivalent to size * 8 for float64 (since each element is 8 bytes)

    # Start measuring execution time
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()
    start = time.perf_counter()

    # Perform memory-intensive operation (multiply each element by 2)
    _ = a * 2

    # End measuring execution time
    end = time.perf_counter()
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time in seconds
    exec_time = round(end - start, 6)

    # Calculate memory bandwidth (bytes per second)
    memory_bandwidth = data_size / exec_time if exec_time > 0 else 0

    return {
        "Benchmark Results": {
            "Memory Bandwidth (bytes/sec)": memory_bandwidth,
            "Memory Bandwidth (GB/sec)": memory_bandwidth / (1024 ** 3),  # Convert to GB/s
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": round(cpu_time_end - cpu_time_start, 6),
            "Memory Usage (MB)": round(memory_usage_end - memory_usage_start, 6)
        }
    }

def measure_memory_latency(jumps, stride):
    try:
        array = np.zeros(jumps * stride, dtype=np.int32)
    except MemoryError:
        return {
            "Benchmark Results": {
                "Execution Time (sec)": float("inf"),
                "Memory Latency (sec)": float("inf")
            },
            "Hardware Metrics": {
                "CPU Time (s)": float("inf"),
                "Memory Usage (MB)": float("inf")
            }
        }

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()

    # Memory access pattern with stride
    for i in range(0, len(array), stride):
        _ = array[i]  # Read the element at the specific stride position

    # End measuring execution time
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate memory latency (in nano seconds)
    memory_latency = exec_time / (jumps * stride) / 1e-9 if exec_time > 0 else 0

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Memory Latency (nanosec)": memory_latency,
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": round(cpu_time_end - cpu_time_start, 6),
            "Memory Usage (MB)": round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)
        }
    }

def measure_pointer_chasing_latency(size):
    arr = list(range(size))
    random.shuffle(arr)
    next_index = arr[0]

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()

    # Perform the pointer chasing
    for _ in range(size):
        next_index = arr[next_index]  # Follow the "pointer"

    # End measuring execution time
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate pointer chasing latency (in nanoseconds)
    pointer_chasing_latency = exec_time / size / 1e-9 if exec_time > 0 else 0

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Pointer Chasing Latency (nanosec)": pointer_chasing_latency,
            "Execution Time (sec)": exec_time,
        },
        "Hardware Metrics": {
            "CPU Time (s)": round(cpu_time_end - cpu_time_start, 6),
            "Memory Usage (MB)": round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)
        }
    }


def measure_cache_effectiveness():
    sizes_kb = [4, 64, 512, 2048, 8192]  # ~L1 to L3 to RAM
    results = []

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    for size_kb in sizes_kb:
        arr_size = (size_kb * 1024) // 8  # elements of float64 (8 bytes per element)
        arr = np.zeros(arr_size)  # Initialize array with zeros

        # Start measuring execution time
        start = time.perf_counter()

        # Access the array in a specific pattern (this helps measure cache effectiveness)
        for i in range(0, arr_size, 16):  # Step through the array with stride of 16
            arr[i] += 1.0

        # End measuring execution time
        end = time.perf_counter()

        # Calculate execution time for this size
        exec_time = round(end - start, 6)

        # Append the results for each cache size
        results.append({
            "cache_size_kb": size_kb,
            "execution_time_sec": exec_time
        })

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate the CPU time and memory usage for the whole test
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)  # In MB

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Cache Effectiveness Results": results,
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        },
        "Hardware Metrics": {
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        }
    }

def dummy_workload_chunk(count):
    s = 0
    for i in range(count):
        s += i % 3
    return s

def measure_thread_scaling(max_threads, total_work):
    # Get the number of CPU cores available
    cpu_count = os.cpu_count()
    max_threads = min(max_threads, cpu_count)

    results = []

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    for threads in range(1, max_threads + 1):
        # Distribute the total work among the threads
        work_per_thread = total_work // threads

        # Start measuring execution time
        start = time.perf_counter()

        # Create a pool of processes and execute the dummy workload
        with Pool(processes=threads) as pool:
            pool.map(dummy_workload_chunk, [work_per_thread] * threads)

        # End measuring execution time
        end = time.perf_counter()

        # Calculate execution time for this thread count
        exec_time = round(end - start, 4)

        # Append the results for this number of threads
        results.append({
            "threads": threads,
            "execution_time_sec": exec_time
        })

    # Measure final CPU time and memory usage after all tests
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate the CPU time and memory usage for the entire test
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start) / (1024 * 1024), 2)  # In MB

    # Return the result in the desired format
    return {
        "Benchmark Results": {
            "Thread Scalability Results": results,
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        },
        "Hardware Metrics": {
            "CPU Time (s)": cpu_time_used,
            "Memory Usage (MB)": memory_usage_used
        }
    }

def run_microbenchmarks(config):
    matrix_size = config["matrix_size"]
    vector_size = config["vector_size"]
    memory_jumps = config["memory_jumps"]
    stride = config["memory_stride"]
    max_threads = config["max_threads"]
    thread_total_work = config["thread_total_work"]

    print("Running Floating Point Throughput Benchmark...")
    fp_result = measure_fp_throughput(matrix_size)

    print("Running Scalar FP Add Latency Benchmark...")
    scalar_fp_result = measure_scalar_fp_add(vector_size)

    print("Running Memory Bandwidth Benchmark...")
    mem_bw_result = measure_memory_bandwidth(vector_size)

    print("Running Memory Latency Benchmark...")
    mem_latency_result = measure_memory_latency(memory_jumps, stride)

    print("Running Pointer Chasing Latency Benchmark...")
    pointer_latency_result = measure_pointer_chasing_latency(vector_size)

    print("Running Cache Effectiveness Benchmark...")
    cache_result = measure_cache_effectiveness()

    print("Running Thread Scalability Benchmark...")
    thread_result = measure_thread_scaling(max_threads, thread_total_work)

    results = {
        "Floating Point Throughput": fp_result,
        "Scalar FP Add Latency": scalar_fp_result,
        "Memory Bandwidth": mem_bw_result,
        "Memory Latency": mem_latency_result,
        "Pointer Chasing Latency": pointer_latency_result,
        "Cache Performance": cache_result,
        "Thread Scalability": thread_result
    }

    data = {
        "Config Metadata": config,
        "Benchmark Results": results,
        "System Info": collect_hw_info()
    }

    print("Saving results to file...")
    save_results(data, "micro_benchmarks")
    print("Microbenchmarks complete.")

if __name__ == "__main__":
	config = load_config()
	print("Starting Micro Benchmarks...")
	run_microbenchmarks(config["microbenchmarks"])
