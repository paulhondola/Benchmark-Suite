import sqlite3
import time
import random
import psutil  # For measuring CPU time and memory usage
from multiprocessing import Pool
from utils.config_loader import load_config
from utils.result_saver import save_results
from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage

# Simple Select Query Benchmark with stress testing
def simple_select_query(row_count=1000000, threshold=500000):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    # Create table and insert data
    c.execute("CREATE TABLE test (id INTEGER, val TEXT)")
    c.executemany("INSERT INTO test VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()
    c.execute("SELECT COUNT(*) FROM test WHERE id < ?", (threshold,))
    c.fetchone()  # Execute the query
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate CPU time and memory usage delta
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start), 2)

    conn.close()

    return {
        "Execution Time (sec)": exec_time,
        "CPU Time (s)": cpu_time_used,
        "Memory Usage (MB)": memory_usage_used
    }

# Insert Benchmark with stress testing
def insert_benchmark(row_count=1000000):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    c.execute("CREATE TABLE test (id INTEGER, val TEXT)")

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()
    c.executemany("INSERT INTO test VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate CPU time and memory usage delta
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start), 2)

    conn.close()

    return {
        "Execution Time (sec)": exec_time,
        "CPU Time (s)": cpu_time_used,
        "Memory Usage (MB)": memory_usage_used
    }

# Join Benchmark
def join_benchmark(row_count=1000000):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    # Create tables
    c.execute("CREATE TABLE test1 (id INTEGER, val TEXT)")
    c.execute("CREATE TABLE test2 (id INTEGER, val TEXT)")
    c.executemany("INSERT INTO test1 VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])
    c.executemany("INSERT INTO test2 VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()
    c.execute("""
        SELECT COUNT(*) FROM test1 t1
        JOIN test2 t2 ON t1.id = t2.id
        WHERE t1.id < ?
    """, (row_count // 2,))
    c.fetchone()  # Execute the query
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate CPU time and memory usage delta
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start), 2)

    conn.close()

    return {
        "Execution Time (sec)": exec_time,
        "CPU Time (s)": cpu_time_used,
        "Memory Usage (MB)": memory_usage_used
    }

# Aggregation Benchmark (e.g., GROUP BY)
def aggregation_benchmark(row_count=1000000):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    # Create table and insert data
    c.execute("CREATE TABLE test (id INTEGER, val TEXT)")
    c.executemany("INSERT INTO test VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])

    # Measure initial CPU time and memory usage
    memory_usage_start = get_memory_usage()
    cpu_time_start = get_cpu_time()

    # Start measuring execution time
    start = time.perf_counter()
    c.execute("SELECT COUNT(*), AVG(id) FROM test GROUP BY id % 1000")
    c.fetchall()  # Execute the aggregation query
    end = time.perf_counter()

    # Measure final CPU time and memory usage
    cpu_time_end = get_cpu_time()
    memory_usage_end = get_memory_usage()

    # Calculate execution time
    exec_time = round(end - start, 6)

    # Calculate CPU time and memory usage delta
    cpu_time_used = round(cpu_time_end - cpu_time_start, 6)
    memory_usage_used = round((memory_usage_end - memory_usage_start), 2)

    conn.close()

    return {
        "Execution Time (sec)": exec_time,
        "CPU Time (s)": cpu_time_used,
        "Memory Usage (MB)": memory_usage_used
    }

# Run SQL benchmarks with the given configuration
def run_sql_benchmarks(config):
    row_count = config["row_count"]
    threshold = config["threshold"]

    # Running different types of SQL benchmarks
    results = {
        "simple_select_query": simple_select_query(row_count, threshold),
        "insert_benchmark": insert_benchmark(row_count),
        "join_benchmark": join_benchmark(row_count),
        "aggregation_benchmark": aggregation_benchmark(row_count)
    }

    data = {
        "Config Metadata": config,
        "Benchmark Results": results,
        "System Info": collect_hw_info()
    }

    save_results(data, "sql_benchmarks")

if __name__ == "__main__":
    config = load_config()
    print("Starting SQL Benchmarks")
    run_sql_benchmarks(config["sql_benchmarks"])
    print("SQL Benchmarks Finished")
