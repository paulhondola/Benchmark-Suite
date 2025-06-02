import psutil
import json

def collect_memory_info():
    mem = psutil.virtual_memory()
    return {
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_free_gb": round(mem.available / (1024**3), 2),
        "ram_usage_percent": mem.percent
    }

if __name__ == "__main__":
    mem_info = collect_memory_info()
    print(json.dumps(mem_info, indent=2))
