import json
from .cpu_info import collect_cpu_info
from .memory_info import collect_memory_info
from .cache_info import collect_cache_info
from .gpu_info import collect_gpu_info

def collect_hw_info():
    return {
        "cpu": collect_cpu_info(),
        "gpu:": collect_gpu_info(),
        "memory": collect_memory_info(),
        "cache": collect_cache_info()
    }

if __name__ == "__main__":
    hw_info = collect_hw_info()
    print(json.dumps(hw_info, indent=2))
