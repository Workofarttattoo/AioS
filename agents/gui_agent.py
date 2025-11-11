"""
GuiAgent - Display Server & Telemetry Management

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import subprocess
import platform
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass
class DisplayServer:
    """Display server information."""
    name: str
    type: str  # "X11", "Wayland", "Quartz", "GDI"
    version: Optional[str]
    status: str  # "running", "stopped"
    protocol: str


class GuiAgent:
    """
    Meta-agent for display server management and GUI telemetry.

    Responsibilities:
    - Display server detection and management
    - GUI telemetry streaming
    - Display configuration monitoring
    - Window manager interaction
    - Performance metrics for GUI rendering
    """

    def __init__(self):
        self.name = "gui"
        self.platform = platform.system()
        self.display_server = None
        self.telemetry_stream = []
        self._detect_display_server()
        LOG.info(f"GuiAgent initialized on {self.platform}")

    def _detect_display_server(self) -> None:
        """Detect the active display server."""
        try:
            if self.platform == "Darwin":  # macOS
                self.display_server = DisplayServer(
                    name="Quartz",
                    type="Quartz",
                    version=None,
                    status="running",
                    protocol="native",
                )
            elif self.platform == "Windows":
                self.display_server = DisplayServer(
                    name="Windows GDI",
                    type="GDI",
                    version=None,
                    status="running",
                    protocol="native",
                )
            else:
                # Linux - detect X11 or Wayland
                if os.environ.get("WAYLAND_DISPLAY"):
                    self.display_server = DisplayServer(
                        name="Wayland",
                        type="Wayland",
                        version=None,
                        status="running",
                        protocol="wayland",
                    )
                elif os.environ.get("DISPLAY"):
                    self.display_server = DisplayServer(
                        name="X11",
                        type="X11",
                        version=None,
                        status="running",
                        protocol="x11",
                    )
                else:
                    self.display_server = DisplayServer(
                        name="Unknown",
                        type="Unknown",
                        version=None,
                        status="unknown",
                        protocol="unknown",
                    )

            LOG.info(f"Detected display server: {self.display_server.name}")
        except Exception as e:
            LOG.warning(f"Error detecting display server: {e}")

    def get_display_server_info(self) -> Dict:
        """Get information about the display server."""
        if self.display_server:
            return {
                "name": self.display_server.name,
                "type": self.display_server.type,
                "protocol": self.display_server.protocol,
                "status": self.display_server.status,
                "version": self.display_server.version,
            }
        return {"status": "unknown"}

    def list_displays(self) -> List[Dict]:
        """List all connected displays."""
        try:
            if self.platform == "Darwin":
                return self._list_displays_macos()
            elif self.platform == "Windows":
                return self._list_displays_windows()
            else:
                return self._list_displays_linux()
        except Exception as e:
            LOG.error(f"Error listing displays: {e}")
            return []

    def _list_displays_macos(self) -> List[Dict]:
        """List macOS displays."""
        displays = []
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)

                if "SPDisplaysDataType" in data:
                    for display_info in data["SPDisplaysDataType"]:
                        if "sppci_model" in display_info:
                            displays.append({
                                "name": display_info.get("sppci_model", "Unknown"),
                                "resolution": display_info.get("_spdisplays_resolution", "Unknown"),
                                "brightness": display_info.get("_spdisplays_brightness", "Unknown"),
                            })

        except Exception as e:
            LOG.warning(f"Error listing macOS displays: {e}")

        return displays

    def _list_displays_windows(self) -> List[Dict]:
        """List Windows displays."""
        displays = []
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, CurrentRefreshRate, CurrentBitsPerPixel | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)

                if isinstance(data, list):
                    for display in data:
                        displays.append({
                            "name": display.get("Name", "Unknown"),
                            "refresh_rate": display.get("CurrentRefreshRate", "Unknown"),
                            "bits_per_pixel": display.get("CurrentBitsPerPixel", "Unknown"),
                        })
                else:
                    displays.append({
                        "name": data.get("Name", "Unknown"),
                        "refresh_rate": data.get("CurrentRefreshRate", "Unknown"),
                        "bits_per_pixel": data.get("CurrentBitsPerPixel", "Unknown"),
                    })

        except Exception as e:
            LOG.warning(f"Error listing Windows displays: {e}")

        return displays

    def _list_displays_linux(self) -> List[Dict]:
        """List Linux displays using xrandr."""
        displays = []
        try:
            if "DISPLAY" in os.environ:
                result = subprocess.run(
                    ["xrandr", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=os.environ.copy(),
                )
                current_display = None

                for line in result.stdout.split("\n"):
                    if " connected" in line or " disconnected" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            current_display = parts[0]
                            status = "connected" if "connected" in line else "disconnected"
                            displays.append({
                                "name": current_display,
                                "status": status,
                            })
                    elif current_display and "x" in line and "+" in line:
                        # Resolution line
                        parts = line.split()
                        if parts:
                            displays[-1]["resolution"] = parts[0]

        except Exception as e:
            LOG.warning(f"Error listing Linux displays: {e}")

        return displays

    def get_display_metrics(self) -> Dict:
        """Get display performance metrics."""
        try:
            if self.platform == "Darwin":
                return self._get_display_metrics_macos()
            elif self.platform == "Windows":
                return self._get_display_metrics_windows()
            else:
                return self._get_display_metrics_linux()
        except Exception as e:
            LOG.error(f"Error getting display metrics: {e}")
            return {"error": str(e)}

    def _get_display_metrics_macos(self) -> Dict:
        """Get macOS display metrics."""
        metrics = {
            "platform": "macOS",
            "display_count": len(self.list_displays()),
        }

        try:
            result = subprocess.run(
                ["ioreg", "-l", "-p", "IODeviceTree"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Simple parsing - count display entries
            metrics["gpu_info"] = "Metal GPU" if "GPU" in result.stdout else "Integrated GPU"
        except Exception as e:
            LOG.warning(f"Error getting macOS display metrics: {e}")

        return metrics

    def _get_display_metrics_windows(self) -> Dict:
        """Get Windows display metrics."""
        metrics = {
            "platform": "Windows",
            "display_count": len(self.list_displays()),
        }

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-CimInstance Win32_DesktopMonitor | Measure-Object | Select-Object Count",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Count" in result.stdout:
                import re
                match = re.search(r"Count\s+:\s+(\d+)", result.stdout)
                if match:
                    metrics["monitor_count"] = int(match.group(1))
        except Exception as e:
            LOG.warning(f"Error getting Windows display metrics: {e}")

        return metrics

    def _get_display_metrics_linux(self) -> Dict:
        """Get Linux display metrics."""
        metrics = {
            "platform": "Linux",
            "display_server": self.display_server.name if self.display_server else "Unknown",
            "display_count": len(self.list_displays()),
        }

        try:
            if "DISPLAY" in os.environ:
                result = subprocess.run(
                    ["xdpyinfo"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=os.environ.copy(),
                )
                for line in result.stdout.split("\n"):
                    if "dimensions:" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics["resolution"] = parts[1]
                    elif "depth of root window:" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            metrics["color_depth"] = parts[-2]
        except Exception as e:
            LOG.warning(f"Error getting Linux display metrics: {e}")

        return metrics

    def stream_gui_telemetry(self, data: Dict) -> bool:
        """Stream GUI telemetry data."""
        try:
            import time
            telemetry_entry = {
                "timestamp": time.time(),
                "data": data,
            }
            self.telemetry_stream.append(telemetry_entry)

            # Keep stream size manageable
            if len(self.telemetry_stream) > 10000:
                self.telemetry_stream = self.telemetry_stream[-5000:]

            LOG.debug(f"Streamed GUI telemetry ({len(self.telemetry_stream)} entries)")
            return True
        except Exception as e:
            LOG.error(f"Failed to stream telemetry: {e}")
            return False

    def get_telemetry_stream(self, limit: int = 100) -> List[Dict]:
        """Get recent telemetry stream entries."""
        return self.telemetry_stream[-limit:]

    def get_window_manager_info(self) -> Dict:
        """Get information about the window manager."""
        try:
            if self.platform == "Darwin":
                return {
                    "window_manager": "Quartz Compositor",
                    "features": ["Mission Control", "Spaces", "Full-screen apps"],
                }
            elif self.platform == "Windows":
                return {
                    "window_manager": "Desktop Window Manager (DWM)",
                    "features": ["Aero Glass", "Window snapping", "Virtual desktops"],
                }
            else:
                return self._get_window_manager_info_linux()
        except Exception as e:
            LOG.error(f"Error getting window manager info: {e}")
            return {"error": str(e)}

    def _get_window_manager_info_linux(self) -> Dict:
        """Get Linux window manager information."""
        wm_info = {
            "window_manager": "Unknown",
            "features": [],
        }

        try:
            # Try to detect window manager
            wmctrl_available = subprocess.run(
                ["which", "wmctrl"],
                capture_output=True,
            ).returncode == 0

            if wmctrl_available:
                result = subprocess.run(
                    ["wmctrl", "-m"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.split("\n"):
                    if "Name:" in line:
                        wm_info["window_manager"] = line.split(":", 1)[1].strip()

            # Check for common features
            if self.display_server and self.display_server.protocol == "wayland":
                wm_info["features"].append("Wayland compatible")
            if self.display_server and self.display_server.protocol == "x11":
                wm_info["features"].append("X11 compatible")

        except Exception as e:
            LOG.warning(f"Error detecting Linux window manager: {e}")

        return wm_info

    def get_gui_health_status(self) -> Dict:
        """Get overall GUI subsystem health."""
        health = {
            "display_server": "healthy" if self.display_server and self.display_server.status == "running" else "unhealthy",
            "display_count": len(self.list_displays()),
            "telemetry_stream_active": len(self.telemetry_stream) > 0,
            "telemetry_entries": len(self.telemetry_stream),
        }

        try:
            display_info = self.list_displays()
            health["connected_displays"] = len(display_info)

            # Determine overall health
            if health["display_server"] == "healthy" and health["connected_displays"] > 0:
                health["overall_status"] = "healthy"
            elif health["display_server"] == "healthy":
                health["overall_status"] = "degraded"
            else:
                health["overall_status"] = "unhealthy"

        except Exception as e:
            LOG.warning(f"Error checking GUI health: {e}")
            health["overall_status"] = "unknown"
            health["error"] = str(e)

        return health

    def get_performance_metrics(self) -> Dict:
        """Get GUI rendering performance metrics."""
        metrics = {
            "display_server": self.display_server.name if self.display_server else "Unknown",
            "displays": self.get_display_metrics(),
            "telemetry_buffer_size": len(self.telemetry_stream),
        }

        try:
            # Simple FPS estimation based on telemetry frequency
            if len(self.telemetry_stream) >= 2:
                recent_entries = self.telemetry_stream[-60:]  # Last 60 entries
                if len(recent_entries) > 1:
                    time_span = recent_entries[-1]["timestamp"] - recent_entries[0]["timestamp"]
                    if time_span > 0:
                        estimated_fps = len(recent_entries) / time_span
                        metrics["estimated_fps"] = round(estimated_fps, 1)

        except Exception as e:
            LOG.warning(f"Error calculating performance metrics: {e}")

        return metrics
