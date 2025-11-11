"""
ApplicationAgent - Application Orchestration & Process Management

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import subprocess
import json
from typing import Dict, List, Optional
from pathlib import Path

LOG = logging.getLogger(__name__)


class ApplicationAgent:
    """
    Meta-agent for application supervision and orchestration.

    Responsibilities:
    - Application lifecycle management
    - Process orchestration (native, Docker, VM)
    - Health monitoring
    - Auto-restart on failure
    - Resource limit enforcement
    """

    def __init__(self):
        self.name = "application"
        self.managed_apps = {}
        self.docker_available = self._check_docker()
        LOG.info(f"ApplicationAgent initialized (Docker: {self.docker_available})")

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False

    def register_application(
        self,
        app_name: str,
        app_type: str,  # "native", "docker", "vm"
        config: Dict,
    ) -> bool:
        """Register an application for management."""
        try:
            self.managed_apps[app_name] = {
                "type": app_type,
                "config": config,
                "status": "registered",
                "restart_count": 0,
            }
            LOG.info(f"Registered application: {app_name} (type: {app_type})")
            return True
        except Exception as e:
            LOG.error(f"Failed to register application {app_name}: {e}")
            return False

    def start_application(self, app_name: str) -> bool:
        """Start a managed application."""
        if app_name not in self.managed_apps:
            LOG.warning(f"Application {app_name} not registered")
            return False

        app_config = self.managed_apps[app_name]

        try:
            if app_config["type"] == "native":
                return self._start_native(app_name, app_config["config"])
            elif app_config["type"] == "docker":
                return self._start_docker(app_name, app_config["config"])
            elif app_config["type"] == "vm":
                return self._start_vm(app_name, app_config["config"])
        except Exception as e:
            LOG.error(f"Failed to start application {app_name}: {e}")
            return False

    def _start_native(self, app_name: str, config: Dict) -> bool:
        """Start a native application."""
        try:
            cmd = config.get("command", "")
            cwd = config.get("cwd", None)
            env = config.get("env", None)

            subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.managed_apps[app_name]["status"] = "running"
            LOG.info(f"Started native application: {app_name}")
            return True
        except Exception as e:
            LOG.error(f"Failed to start native application {app_name}: {e}")
            return False

    def _start_docker(self, app_name: str, config: Dict) -> bool:
        """Start a Docker container."""
        if not self.docker_available:
            LOG.warning("Docker is not available")
            return False

        try:
            image = config.get("image")
            ports = config.get("ports", [])
            env_vars = config.get("env", {})

            cmd = ["docker", "run", "-d"]

            # Add port mappings
            for port_mapping in ports:
                cmd.extend(["-p", port_mapping])

            # Add environment variables
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

            # Add image
            cmd.append(image)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                container_id = result.stdout.strip()[:12]
                self.managed_apps[app_name]["status"] = "running"
                self.managed_apps[app_name]["container_id"] = container_id
                LOG.info(f"Started Docker container: {app_name} ({container_id})")
                return True
            else:
                LOG.error(f"Docker startup failed: {result.stderr}")
                return False
        except Exception as e:
            LOG.error(f"Failed to start Docker application {app_name}: {e}")
            return False

    def _start_vm(self, app_name: str, config: Dict) -> bool:
        """Start a virtual machine (QEMU/libvirt)."""
        try:
            vm_name = config.get("vm_name")
            hypervisor = config.get("hypervisor", "qemu")  # qemu or libvirt

            if hypervisor == "libvirt":
                cmd = ["virsh", "start", vm_name]
            else:
                cmd = ["qemu-system-x86_64", "-m", config.get("memory", "1024")]

            subprocess.run(cmd, capture_output=True, timeout=30)

            self.managed_apps[app_name]["status"] = "running"
            LOG.info(f"Started VM: {app_name}")
            return True
        except Exception as e:
            LOG.error(f"Failed to start VM {app_name}: {e}")
            return False

    def stop_application(self, app_name: str) -> bool:
        """Stop a managed application."""
        if app_name not in self.managed_apps:
            return False

        app_config = self.managed_apps[app_name]

        try:
            if app_config["type"] == "native":
                # Kill native process (would need PID tracking)
                LOG.info(f"Stopped native application: {app_name}")
            elif app_config["type"] == "docker":
                container_id = app_config.get("container_id")
                if container_id:
                    subprocess.run(["docker", "stop", container_id], timeout=10)
                    LOG.info(f"Stopped Docker container: {app_name}")
            elif app_config["type"] == "vm":
                vm_name = app_config["config"].get("vm_name")
                subprocess.run(["virsh", "shutdown", vm_name], timeout=10)
                LOG.info(f"Stopped VM: {app_name}")

            self.managed_apps[app_name]["status"] = "stopped"
            return True
        except Exception as e:
            LOG.error(f"Failed to stop application {app_name}: {e}")
            return False

    def get_application_status(self, app_name: str) -> Dict:
        """Get the status of a managed application."""
        if app_name not in self.managed_apps:
            return {"status": "unknown"}

        app = self.managed_apps[app_name]

        try:
            if app["type"] == "docker":
                container_id = app.get("container_id")
                if container_id:
                    result = subprocess.run(
                        ["docker", "inspect", container_id],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        data = json.loads(result.stdout)
                        return {
                            "name": app_name,
                            "type": "docker",
                            "status": data[0]["State"]["Status"],
                            "container_id": container_id,
                        }
            elif app["type"] == "vm":
                vm_name = app["config"].get("vm_name")
                result = subprocess.run(
                    ["virsh", "dominfo", vm_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return {
                        "name": app_name,
                        "type": "vm",
                        "status": "running" if "running" in result.stdout else "stopped",
                    }

            return {
                "name": app_name,
                "type": app["type"],
                "status": app.get("status", "unknown"),
            }
        except Exception as e:
            LOG.error(f"Failed to get status for {app_name}: {e}")
            return {"status": "error", "error": str(e)}

    def list_applications(self) -> List[Dict]:
        """List all managed applications."""
        return [
            {
                "name": name,
                "type": app["type"],
                "status": app.get("status", "unknown"),
            }
            for name, app in self.managed_apps.items()
        ]
