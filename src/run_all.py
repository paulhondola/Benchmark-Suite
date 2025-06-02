import compiler_benchmarks
import sorting_benchmarks
import ml_workloads
import microbenchmarks
import sql_benchmarks

def run_all_benchmarks():
	print("Starting Compiler Benchmarks...")
	compiler_benchmarks.run_compile_benchmarks()
	print("Compiler Benchmarks Finished.")

	print("Starting Sorting Benchmarks...")
	sorting_benchmarks.run_all_sorting_benchmarks()
	print("Sorting Benchmarks Finished.")

	print("Starting Machine Learning Workloads...")
	ml_workloads.run_all_ml_benchmarks()
	print("Machine Learning Workloads Finished.")

	print("Starting Microbenchmarks...")
	microbenchmarks.run_all_microbenchmarks()
	print("Microbenchmarks Finished.")

	print("Starting SQL Benchmarks...")
	sql_benchmarks.run_sql_benchmarks()
	print("SQL Benchmarks Finished.")
