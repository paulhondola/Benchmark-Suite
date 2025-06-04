import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024)

if __name__ == "__main__":
	from run_test import run_test

	memory_before = get_memory_usage()
	print(f"Memory Usage Before Test: {memory_before:} KiB")

	print("Running test...")
	run_test()

	memory_after = get_memory_usage()
	print(f"Memory Usage After Test: {memory_after:} KiB")

	memory_used = memory_after - memory_before
	print(f"Total Memory Used by Test: {memory_used:} KiB")
