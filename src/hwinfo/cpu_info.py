import psutil
import platform
import subprocess
import json

def get_macos_chip_name():
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL
        )
        return output.decode().strip()
    except Exception:
        return None

def get_macos_hardware_overview():
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            stderr=subprocess.DEVNULL
        )
        lines = output.decode().splitlines()
        for line in lines:
            if "Chip" in line or "Processor Name" in line:
                return line.strip()
    except Exception:
        return None

def collect_cpu_info():
    chip = get_macos_chip_name() or get_macos_hardware_overview()
    cpu_brand = chip if chip else platform.processor()
    freq = psutil.cpu_freq()
    core_usages = psutil.cpu_percent(interval=1, percpu=True)

    try:
        per_core_freq = psutil.cpu_freq(percpu=True)
        if isinstance(per_core_freq, list) and len(per_core_freq) > 1:
            freq_per_core = [round(f.current, 2) for f in per_core_freq]
        else:
            freq_per_core = "Unavailable on this platform"
    except Exception:
        freq_per_core = "Unavailable on this platform"

    return {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu": cpu_brand,
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": round(freq.current, 2) if freq else None,
        "cpu_freq_per_core_mhz": freq_per_core,
        "cpu_usage_percent_total": round(sum(core_usages) / len(core_usages), 1),
        "cpu_usage_per_core_percent": core_usages
    }

if __name__ == "__main__":
    cpu_info = collect_cpu_info()
    print(json.dumps(cpu_info, indent=2))
