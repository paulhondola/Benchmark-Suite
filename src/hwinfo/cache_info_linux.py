#!/usr/bin/env python3
import subprocess

def sysctl_get_bytes(key):
    """
    Runs `sysctl -n <key>` and returns the integer value (bytes),
    or None if the key is not available (empty output or error).
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key],
            stderr=subprocess.DEVNULL
        )
        text = out.decode().strip()
        if not text:
            # sysctl existed but returned an empty string
            return None
        return int(text)
    except (subprocess.CalledProcessError, ValueError):
        return None

def print_cache_info():
    """
    Prints L1 data cache, L1 instruction cache, L2 cache sizes via sysctl.
    For L3 on Apple Silicon, sysctl often returns empty, so we fall back to
    the known 12 MiB SLC (system-level cache) for M1-series chips.
    """
    print("=== CACHE INFO ===")

    l1d = sysctl_get_bytes("hw.l1dcachesize")
    l1i = sysctl_get_bytes("hw.l1icachesize")
    l2  = sysctl_get_bytes("hw.l2cachesize")
    l3  = sysctl_get_bytes("hw.l3cachesize")

    if l1d is not None:
        print(f"L1 Data Cache         : {l1d:,} bytes")
    else:
        print("L1 Data Cache         : (unavailable)")

    if l1i is not None:
        print(f"L1 Instruction Cache  : {l1i:,} bytes")
    else:
        print("L1 Instruction Cache  : (unavailable)")

    if l2 is not None:
        print(f"L2 Cache              : {l2:,} bytes")
    else:
        print("L2 Cache              : (unavailable)")

    if l3 is not None:
        print(f"L3 Cache              : {l3:,} bytes")
    else:
        print("L3 Cache              : (unavailable)")

if __name__ == "__main__":
    print_cache_info()
