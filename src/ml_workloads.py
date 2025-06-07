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
    use_gpu = config["gpu"]

    # Map model type to callable with correct GPU flag
    model_map = {
        "sklearn": lambda: run_sklearn_workload(config["sklearn"]) if run_sklearn_workload else _missing("scikit-learn"),
        "pytorch": lambda: run_pytorch_workload(config["pytorch"], use_gpu=use_gpu) if run_pytorch_workload else _missing("PyTorch"),
        "tensorflow": lambda: run_tensorflow_workload(config["tensorflow"], use_gpu=use_gpu) if run_tensorflow_workload else _missing("TensorFlow"),
    }

    if model not in model_map:
        raise ValueError(f"Unknown model '{model}' in config. Expected one of: {list(model_map.keys())}")

    print(f"Running {model} workload (GPU: {use_gpu})...")
    result = model_map[model]()

    # Extract result metrics
    cpu_time = result["cpu_time"]
    memory_usage = result["memory_usage"]
    model_results = result["results"]

    data = {
        "Model": model,
        "GPU Enabled": use_gpu,
        "Config Metadata": config[model],
        "Benchmark Results": model_results,
        "Hardware Metrics:": {
            "CPU Time (s)": cpu_time,
            "Memory Usage (MB)": memory_usage,
        },
        "System Info": collect_hw_info()
    }

    save_results(data, "ml_workloads")

if __name__ == "__main__":
    config = load_config()
    run_ml_workloads(config["ml_workloads"])
