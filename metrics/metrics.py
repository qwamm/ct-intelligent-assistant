import argparse
import logging
import time
import platform
import socket
import psutil
from prometheus_client import Gauge
from prometheus_client import start_http_server, core
from pynvml import *


# logging.basicConfig(filename='example.log', level=logging.DEBUG)
# log = logging.getLogger('nvidia-tool')
def _create_parser():
    parser = argparse.ArgumentParser(
        description="nVidia GPU Prometheus Metrics Exporter"
    )
    parser.add_argument(
        "--verbose", help="Turn on verbose logging", action="store_true"
    )
    parser.add_argument(
        "-u",
        "--update-period",
        help=(
            "Period between calls to update metrics, "
            "in seconds. Defaults to 30."
        ),
        default=30,
    )
    parser.add_argument(
        "-g",
        "--gateway",
        help=(
            "If defined, gateway to push metrics to. Should "
            "be in the form of <host>:<port>."
        ),
        default=None,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="If non-zero, port to run the http server",
        type=int,
        default=0,
    )

    return parser


def get_metrics():
    parser = _create_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    registry = core.REGISTRY

    # GPU
    total_fb_memory = Gauge(
        "gpu_total_fb_memory_mb",
        "Total installed frame buffer memory (in megabytes)",
        ["device"],
        registry=registry,
    )
    free_fb_memory = Gauge(
        "gpu_free_fb_memory_mb",
        "Unallocated frame buffer memory (in megabytes)",
        ["device"],
        registry=registry,
    )
    used_fb_memory = Gauge(
        "gpu_used_fb_memory_mb",
        "Allocated frame buffer memory (in megabytes)."
        " Note that the diver/GPU will always set "
        "a small amount of memory fore bookkeeping.",
        ["device"],
        registry=registry,
    )
    gpu_utilization = Gauge(
        "gpu_utilization_pct",
        "Percent of time over the past sample period "
        "during which one or more kernels was "
        "executing on the GPU.",
        ["device"],
        registry=registry,
    )
    memory_utilization = Gauge(
        "gpu_mem_utilization_pct",
        "Percent of time over the past sample "
        "period during which global (device) memory "
        "was being read or written",
        ["device"],
        registry=registry,
    )

    # CPU
    host = socket.gethostname()

    # Create our collectors
    ram_metric = Gauge(
        "memory_usage_bytes",
        "Memory usage in bytes.",
        ["type"],
        registry=registry,
    )
    ram_p_metric = Gauge(
        "memory_usage_persent", "Memory usage in %.", registry=registry
    )
    cpu_metric = Gauge(
        "cpu_usage_percent", "CPU usage percent.", ["core"], registry=registry
    )
    # FS
    hdd = psutil.disk_usage("/")

    fs_p_metric = Gauge(
        "FS_usage_persent", "FS usage in %.", registry=registry
    )

    fs_metric = Gauge(
        "FS_usage_GB", "FS stats in GB.", ["type"], registry=registry
    )

    iteration = 0

    try:
        # log.debug('Initializing NVML...')
        nvmlInit()
        print("Initialized")

        # log.info('Started with nVidia driver version = %s',
        #          nvmlSystemGetDriverVersion())

        device_count = nvmlDeviceGetCount()
        # log.info('%d devices found.', device_count)

        # log.info('Starting http server on port %d', 8002)
        start_http_server(8003, "0.0.0.0")
        print("SERVER AT 8003")
        # log.info('HTTP server started on port %d', 8002)

        while True:
            iteration += 1
            # log.info('Current iteration = %d', iteration)
            # CPU&RAM

            # log.info('Querying for RAM information...')
            ram = psutil.virtual_memory()
            swap = psutil.swap_memory()
            # log.info('RAM information collected')
            ram_metric.labels(type="Total").set(ram.total)
            ram_metric.labels(type="Used").set(ram.used)
            ram_metric.labels(type="Available").set(ram.available)
            ram_metric.labels(type="Free").set(ram.free)
            ram_p_metric.set(ram.percent)
            fs_metric.labels(type="Total").set(round(hdd.total / (2**30), 2))
            fs_metric.labels(type="Used").set(round(hdd.used / (2**30), 2))
            fs_metric.labels(type="Free").set(round(hdd.free / (2**30), 2))
            fs_p_metric.set(round((hdd.used / hdd.total) * 100, 2))
            for c, p in enumerate(psutil.cpu_percent(interval=1, percpu=True)):
                cpu_metric.labels(core=c).set(p)

            # GPU
            for i in range(device_count):
                # log.info('Analyzing device %d...', i)
                try:
                    # log.info('Obtaining handle for device %d...', i)
                    handle = nvmlDeviceGetHandleByIndex(i)
                    # log.info('Device handle for %d is %s', i, str(handle))

                    # log.info('Querying for memory information...')
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    # log.info('Memory information = %s', str(mem_info))

                    total_fb_memory.labels(device=i).set(mem_info.total / 1024)
                    free_fb_memory.labels(device=i).set(mem_info.free / 1024)
                    used_fb_memory.labels(device=i).set(mem_info.used / 1024)

                    # log.info('Obtaining utilization statistics...')
                    utilization = nvmlDeviceGetUtilizationRates(handle)
                    # log.info('Utilization statistics = %s', str(utilization))

                    gpu_utilization.labels(device=i).set(
                        utilization.gpu / 100.0
                    )
                    memory_utilization.labels(device=i).set(
                        utilization.memory / 100.0
                    )
                except Exception as e:
                    # log.info(e, exc_info=True)
                    pass

                # log.info('Pushing metrics to gateway at %s...', args.gateway)
                hostname = platform.node()
                print(hostname)
                print(core.REGISTRY)
                # push_to_gateway('<IP>:9091', job=hostname, registry=core.REGISTRY)
                # log.info('Push complete.')

            time.sleep(5)

    except Exception as e:
        # log.info('Exception thrown - %s', e, exc_info=True)
        pass
    finally:
        nvmlShutdown()


if __name__ == "__main__":
    get_metrics()
