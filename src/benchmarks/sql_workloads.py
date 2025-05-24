import sqlite3
import time

def simple_query():
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute("CREATE TABLE test (id INTEGER, val TEXT)")
    c.executemany("INSERT INTO test VALUES (?, ?)", [(i, str(i)) for i in range(100000)])
    start = time.perf_counter()
    c.execute("SELECT COUNT(*) FROM test WHERE id < 50000")
    c.fetchone()
    end = time.perf_counter()
    conn.close()
    return end - start
