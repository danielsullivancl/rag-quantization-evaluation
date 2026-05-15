import time
import threading
import psutil

from config import GPU_INDEX

# ==================================================
# NVML
# ==================================================

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetCount,
        NVMLError,
    )

    NVML_AVAILABLE = True

except Exception:

    NVML_AVAILABLE = False

# ==================================================
# GLOBALS
# ==================================================

monitoring = False

peak_ram_mb = 0.0

peak_vram_mb = 0.0

gpu_monitoring_available = False

# ==================================================
# OLLAMA PROCESS
# ==================================================

def get_ollama_processes():

    processes = []

    for process in psutil.process_iter(["pid", "name"]):

        try:

            process_name = (
                process.info["name"] or ""
            ).lower()

            if "ollama" in process_name:

                processes.append(process)

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied
        ):
            pass

    return processes

# ==================================================
# RAM
# ==================================================

def get_current_ram_mb():

    total_ram_mb = 0.0

    for process in get_ollama_processes():

        try:

            total_ram_mb += (
                process.memory_info().rss
                / (1024 * 1024)
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied
        ):
            pass

    return total_ram_mb

# ==================================================
# VRAM
# ==================================================

def get_current_vram_mb():

    if not NVML_AVAILABLE:
        return None

    try:

        nvmlInit()

        device_count = nvmlDeviceGetCount()

        if device_count == 0:

            nvmlShutdown()
            return None

        if (
            GPU_INDEX < 0
            or GPU_INDEX >= device_count
        ):

            nvmlShutdown()
            return None

        gpu_handle = nvmlDeviceGetHandleByIndex(
            GPU_INDEX
        )

        memory_info = nvmlDeviceGetMemoryInfo(
            gpu_handle
        )

        used_vram_mb = (
            memory_info.used / (1024 * 1024)
        )

        nvmlShutdown()

        return used_vram_mb

    except Exception:

        try:
            nvmlShutdown()
        except Exception:
            pass

        return None

# ==================================================
# RAM MONITOR
# ==================================================

def monitor_ram():

    global peak_ram_mb
    global monitoring

    while monitoring:

        current_ram_mb = get_current_ram_mb()

        if current_ram_mb > peak_ram_mb:

            peak_ram_mb = current_ram_mb

        time.sleep(0.1)

# ==================================================
# VRAM MONITOR
# ==================================================

def monitor_vram():

    global peak_vram_mb
    global monitoring
    global gpu_monitoring_available

    gpu_monitoring_available = False

    if not NVML_AVAILABLE:
        return

    try:

        nvmlInit()

        device_count = nvmlDeviceGetCount()

        if device_count == 0:
            return

        if (
            GPU_INDEX < 0
            or GPU_INDEX >= device_count
        ):
            return

        gpu_handle = nvmlDeviceGetHandleByIndex(
            GPU_INDEX
        )

        gpu_monitoring_available = True

    except Exception:
        return

    try:

        while monitoring:

            try:

                memory_info = (
                    nvmlDeviceGetMemoryInfo(
                        gpu_handle
                    )
                )

                used_vram_mb = (
                    memory_info.used
                    / (1024 * 1024)
                )

                if used_vram_mb > peak_vram_mb:

                    peak_vram_mb = used_vram_mb

            except NVMLError:
                pass

            time.sleep(0.1)

    finally:

        try:
            nvmlShutdown()
        except Exception:
            pass

# ==================================================
# START MONITORING
# ==================================================

def start_monitoring():

    global monitoring
    global peak_ram_mb
    global peak_vram_mb

    baseline_ram_mb = get_current_ram_mb()

    baseline_vram_mb = get_current_vram_mb()

    peak_ram_mb = baseline_ram_mb

    peak_vram_mb = (
        baseline_vram_mb
        if baseline_vram_mb is not None
        else 0.0
    )

    monitoring = True

    ram_thread = threading.Thread(
        target=monitor_ram,
        daemon=True
    )

    vram_thread = threading.Thread(
        target=monitor_vram,
        daemon=True
    )

    ram_thread.start()

    vram_thread.start()

    return {
        "baseline_ram_mb": baseline_ram_mb,
        "baseline_vram_mb": baseline_vram_mb,
        "ram_thread": ram_thread,
        "vram_thread": vram_thread
    }

# ==================================================
# STOP MONITORING
# ==================================================

def stop_monitoring(
    baseline_ram_mb,
    baseline_vram_mb,
    ram_thread,
    vram_thread
):

    global monitoring
    global peak_ram_mb
    global peak_vram_mb

    monitoring = False

    ram_thread.join(timeout=1)

    vram_thread.join(timeout=1)

    ram_usage_mb = max(
        0.0,
        peak_ram_mb - baseline_ram_mb
    )

    if baseline_vram_mb is None:

        vram_delta_mb = None

    else:

        vram_delta_mb = max(
            0.0,
            peak_vram_mb - baseline_vram_mb
        )

    return {
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
        "ram_usage_mb": ram_usage_mb,
        "vram_delta_mb": vram_delta_mb,
        "gpu_monitoring_available":
            gpu_monitoring_available
    }