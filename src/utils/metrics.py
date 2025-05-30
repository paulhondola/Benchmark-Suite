import platform
import psutil
import subprocess

def collect_system_info():
    info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "mem_percent": psutil.virtual_memory().percent,
        "platform": platform.system(),
        "cpu_count": psutil.cpu_count(logical=True),
        "gpu_available": False,
        "gpu_name": "Unknown",
        "gpu_type": "Unknown"
    }

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_type"] = "NVIDIA CUDA"
            info["architecture"] = platform.machine()
            return info
    except ImportError:
        pass  # torch not installed

    # Platform-specific fallback
    system = platform.system()

    if system == "Windows":
        try:
            output = subprocess.check_output(
                'wmic path win32_VideoController get name',
                shell=True, text=True
            ).strip().split('\n')
            gpu_names = [line.strip() for line in output if line.strip() and "Name" not in line]
            if gpu_names:
                info["gpu_name"] = gpu_names[0]
        except Exception:
            info["gpu_name"] = "Unknown (wmic failed)"

    elif system == "Darwin":  # macOS
        try:
            output = subprocess.check_output(
                ['system_profiler', 'SPDisplaysDataType'],
                text=True
            )
            for line in output.splitlines():
                if "Chipset Model:" in line or "Graphics/Displays:" in line:
                    info["gpu_name"] = line.split(":")[-1].strip()
                    break
        except Exception:
            info["gpu_name"] = "Unknown (system_profiler failed)"

    elif system == "Linux":
        try:
            output = subprocess.check_output('lspci | grep VGA', shell=True, text=True)
            info["gpu_name"] = output.strip()
        except Exception:
            info["gpu_name"] = "Unknown (lspci failed)"

    # Infer GPU type based on name
    gpu = info["gpu_name"].lower()
    if "nvidia" in gpu:
        info["gpu_type"] = "NVIDIA (non-CUDA)"
    elif "amd" in gpu or "radeon" in gpu:
        info["gpu_type"] = "Integrated AMD"
    elif "apple" in gpu or "m1" in gpu or "m2" in gpu:
        info["gpu_type"] = "Apple Silicon (Unified Memory)"
    elif "intel" in gpu:
        info["gpu_type"] = "Integrated Intel"
    elif "vga" in gpu or "display" in gpu:
        info["gpu_type"] = "Generic Display Adapter"
    else:
        info["gpu_type"] = "Unknown"

    return info
