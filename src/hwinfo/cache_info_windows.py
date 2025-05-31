import cpuinfo

def format_bytes(size):
    for unit in ['bytes', 'KiB', 'MiB', 'GiB']:
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def print_cache_info_py_cpuinfo():
    info = cpuinfo.get_cpu_info()
    print("=== CACHE INFO (py-cpuinfo) ===")

    print("Brand:", info.get("brand_raw", "(unknown)"))
    print("Arch :", info.get("arch", "(unknown)"))
    print("L2 Cache:", format_bytes(info.get("l2_cache_size", 0)))
    print("L2 Cache:", format_bytes(info.get("l3_cache_size", 0)))

if __name__ == "__main__":
    print_cache_info_py_cpuinfo()
