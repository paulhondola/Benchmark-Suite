from utils import load_config, save_results
from hwinfo import collect_hw_info
from run_sklearn import run_sklearn_workload
from run_pytorch import run_pytorch_workload
from run_tensorflow import run_tensorflow_workload

def run_ml_workloads(config):
    model = config["model"]

    if not model:
        raise ValueError("Model type to run must be specified in the config, the ml_workloads section!\n"
                         "Available models: sklearn, pytorch_cpu, pytorch_gpu, tensorflow_cpu, tensorflow_gpu.")

    model_map = {
        "sklearn": lambda: run_sklearn_workload(config["sklearn"]),
        "pytorch_cpu": lambda: run_pytorch_workload(config["pytorch"], use_gpu=False),
        "pytorch_gpu": lambda: run_pytorch_workload(config["pytorch"], use_gpu=True),
        "tensorflow_cpu": lambda: run_tensorflow_workload(config["tensorflow"], use_gpu=False),
        "tensorflow_gpu": lambda: run_tensorflow_workload(config["tensorflow"], use_gpu=True),
    }

    # Check if the model is available in the map
    if model not in model_map:
        raise ValueError(f"Unknown model {model} in config")

    # Run the corresponding model workload
    print(f"Running {model} workload...")
    result = model_map[model]()

    cpu_time = result["cpu_time"]
    memory_usage = result["memory_usage"]
    model_results = result["results"]

    # Save results
    data = {
        "Model": model,
        "Config Metadata": config[model],
        "Benchmark Results": model_results,
        "Hardware Metrics:":{
            "CPU Time (s)": cpu_time,
            "Memory Usage (MB)": memory_usage,
        },
        "System Info": collect_hw_info()}

    save_results(data, "ml_workloads")

if __name__ == "__main__":
    config = load_config()
    run_ml_workloads(config["ml_workloads"])