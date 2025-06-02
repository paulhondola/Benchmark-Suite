import sqlite3
import time
from utils import load_config, save_results
from hwinfo import collect_hw_info

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
