import psutil

def get_cpu_time():
    process = psutil.Process()  # Get the current process
    return process.cpu_times().user + process.cpu_times().system  # User + system CPU time

if __name__ == "__main__":
    from run_test import run_test

    cpu_time_before = get_cpu_time()
    print(f"CPU Time Before Test: {cpu_time_before} seconds")

    print("Running test...")
    run_test()

    cpu_time_after = get_cpu_time()
    print(f"CPU Time After Test: {cpu_time_after} seconds")

    cpu_time_used = cpu_time_after - cpu_time_before
    print(f"Total CPU Time Used by Test: {cpu_time_used} seconds")
