"""
KernelAgent - Process Management & System Initialization

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import psutil
import platform
from typing import Dict, List, Optional
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Process information structure."""
    pid: int
    name: str
    status: str
    memory_mb: float
    cpu_percent: float


class KernelAgent:
    """
    Meta-agent for process management and system initialization.

    Responsibilities:
    - Process management and resource allocation
    - System initialization coordination
    - Boot sequence orchestration
    - Resource monitoring
    """

    def __init__(self):
        self.name = "kernel"
        self.system_info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        LOG.info(f"KernelAgent initialized on {self.system_info['platform']}")

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage("/")._asdict(),
            "process_count": len(psutil.pids()),
            "boot_time": psutil.boot_time(),
        }

    def list_processes(self, limit: int = 10) -> List[ProcessInfo]:
        """List top processes by memory usage."""
        processes = []
        try:
            for proc in psutil.process_iter(["pid", "name", "status", "memory_info", "cpu_percent"]):
                try:
                    pinfo = ProcessInfo(
                        pid=proc.info["pid"],
                        name=proc.info["name"],
                        status=proc.info["status"],
                        memory_mb=proc.info["memory_info"].rss / 1024 / 1024,
                        cpu_percent=proc.info["cpu_percent"] or 0,
                    )
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Sort by memory and return top N
            processes.sort(key=lambda x: x.memory_mb, reverse=True)
            return processes[:limit]
        except Exception as e:
            LOG.error(f"Error listing processes: {e}")
            return []

    def get_process(self, pid: int) -> Optional[ProcessInfo]:
        """Get information about a specific process."""
        try:
            proc = psutil.Process(pid)
            return ProcessInfo(
                pid=proc.pid,
                name=proc.name(),
                status=proc.status(),
                memory_mb=proc.memory_info().rss / 1024 / 1024,
                cpu_percent=proc.cpu_percent(interval=0.1),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            LOG.error(f"Cannot access process {pid}: {e}")
            return None

    def allocate_resources(self, process_name: str, cpu_percent: float, memory_mb: float) -> bool:
        """Allocate resources to a process (advisory on most systems)."""
        try:
            for proc in psutil.process_iter(["name"]):
                if proc.info["name"] == process_name:
                    LOG.info(f"Resource allocation advisory for {process_name}: CPU {cpu_percent}%, Memory {memory_mb}MB")
                    # On Linux with psutil, can set CPU affinity and limits
                    # On other systems, this is advisory
                    return True
            return False
        except Exception as e:
            LOG.error(f"Resource allocation failed: {e}")
            return False

    def boot_sequence_ready(self) -> Dict:
        """Check if system is ready for boot sequence."""
        status = self.get_system_status()
        memory_available_gb = status["memory"]["available"] / (1024**3)
        cpu_usage = status["cpu_percent"]

        return {
            "ready": cpu_usage < 80 and memory_available_gb > 1,
            "cpu_usage_percent": cpu_usage,
            "memory_available_gb": memory_available_gb,
            "process_count": status["process_count"],
        }
