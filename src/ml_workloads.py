from utils import load_config, save_results
from hwinfo import collect_hw_info

# Safe imports with fallbacks
try:
    from run_sklearn import run_sklearn_workload
except ImportError:
    run_sklearn_workload = None

try:
    from run_pytorch import run_pytorch_workload
except ImportError:
    run_pytorch_workload = None

try:
    from run_tensorflow import run_tensorflow_workload
except ImportError:
    run_tensorflow_workload = None

def _missing(name):
    raise ImportError(f"Missing optional dependency: {name}. Please install it to run this benchmark.")

def run_ml_workloads(config):
    model = config["model"]

    if not model:
        raise ValueError("Model type to run must be specified in the config, the ml_workloads section!\n"
                         "Available models: sklearn, pytorch_cpu, pytorch_gpu, tensorflow_cpu, tensorflow_gpu.")

    model_map = {
        "sklearn": lambda: run_sklearn_workload(config["sklearn"]) if run_sklearn_workload else _missing("scikit-learn"),
        "pytorch_cpu": lambda: run_pytorch_workload(config["pytorch"], use_gpu=False) if run_pytorch_workload else _missing("PyTorch"),
        "pytorch_gpu": lambda: run_pytorch_workload(config["pytorch"], use_gpu=True) if run_pytorch_workload else _missing("PyTorch"),
        "tensorflow_cpu": lambda: run_tensorflow_workload(config["tensorflow"], use_gpu=False) if run_tensorflow_workload else _missing("TensorFlow"),
        "tensorflow_gpu": lambda: run_tensorflow_workload(config["tensorflow"], use_gpu=True) if run_tensorflow_workload else _missing("TensorFlow"),
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
