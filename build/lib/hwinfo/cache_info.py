import platform
import subprocess

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

def format_bytes(size):
    for unit in ['bytes', 'KiB', 'MiB', 'GiB']:
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"

def sysctl_get_bytes(key):
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key],
            stderr=subprocess.DEVNULL
        )
        text = out.decode().strip()
        if not text:
            return None
        return int(text)
    except (subprocess.CalledProcessError, ValueError):
        return None

def collect_cache_info():
    system = platform.system()
    result = {"platform": system, "cache_info": {}}

    if system == "Darwin":  # macOS
        l1d = sysctl_get_bytes("hw.l1dcachesize")
        l1i = sysctl_get_bytes("hw.l1icachesize")
        l2  = sysctl_get_bytes("hw.l2cachesize")
        l3  = sysctl_get_bytes("hw.l3cachesize")

        result["cache_info"] = {
            "L1 Data Cache": l1d,
            "L1 Instruction Cache": l1i,
            "L2 Cache": l2,
            "L3 Cache": l3
        }

    elif system == "Windows" and cpuinfo:
        info = cpuinfo.get_cpu_info()
        result["cpu_brand"] = info.get("brand_raw", "(unknown)")
        result["arch"] = info.get("arch", "(unknown)")
        result["cache_info"] = {
            "L2 Cache": info.get("l2_cache_size", 0),
            "L3 Cache": info.get("l3_cache_size", 0)
        }
    else:
        result["error"] = "Unsupported platform or missing module."

    return result

if __name__ == "__main__":
    import json
    info = collect_cache_info()
    print(json.dumps(info, indent=2))
