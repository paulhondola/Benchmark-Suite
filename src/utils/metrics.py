import psutil
import os

def collect_cpu_gpu():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    return cpu, mem
