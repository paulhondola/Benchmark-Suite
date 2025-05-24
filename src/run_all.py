from benchmarks import microbenchmarks, ml_workloads, sql_workloads
from utils.metrics import collect_cpu_gpu

print("FP Throughput:", microbenchmarks.measure_fp_throughput())
print("CNN Inference:", ml_workloads.cnn_inference())
print("SQL Query:", sql_workloads.simple_query())
print("System Usage:", collect_cpu_gpu())
