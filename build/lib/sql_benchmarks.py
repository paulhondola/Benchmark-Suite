from utils.config_loader import load_config
from utils.result_saver import save_results
from hwinfo.hwinfo import collect_hw_info
import sqlite3
import time

def simple_query(row_count=100000, threshold=50000):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute("CREATE TABLE test (id INTEGER, val TEXT)")
    c.executemany("INSERT INTO test VALUES (?, ?)", [(i, str(i)) for i in range(row_count)])
    start = time.perf_counter()
    c.execute("SELECT COUNT(*) FROM test WHERE id < ?", (threshold,))
    c.fetchone()
    end = time.perf_counter()
    conn.close()
    return end - start

def run_sql_benchmarks(config):
	row_count = config["row_count"]
	threshold = config["threshold"]

	results = {
		"simple_query": simple_query(row_count, threshold)
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
