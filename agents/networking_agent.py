"""
NetworkingAgent - Network Configuration & Management

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import subprocess
import platform
import socket
from typing import Dict, List, Optional
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass
class NetworkInterface:
    """Network interface information."""
    name: str
    ip_address: Optional[str]
    mac_address: Optional[str]
    status: str  # "up" or "down"
    mtu: Optional[int]


class NetworkingAgent:
    """
    Meta-agent for network configuration and management.

    Responsibilities:
    - Network interface configuration
    - DNS management
    - Routing configuration
    - Network diagnostics
    - Network health monitoring
    """

    def __init__(self):
        self.name = "networking"
        self.platform = platform.system()
        LOG.info(f"NetworkingAgent initialized on {self.platform}")

    def list_interfaces(self) -> List[Dict]:
        """List all network interfaces."""
        try:
            if self.platform == "Darwin":  # macOS
                return self._list_interfaces_macos()
            elif self.platform == "Windows":
                return self._list_interfaces_windows()
            else:  # Linux
                return self._list_interfaces_linux()
        except Exception as e:
            LOG.error(f"Error listing interfaces: {e}")
            return []

    def _list_interfaces_macos(self) -> List[Dict]:
        """List network interfaces on macOS using ifconfig."""
        interfaces = []
        try:
            result = subprocess.run(
                ["ifconfig"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.split("\n")
            current_iface = None

            for line in lines:
                if line and not line.startswith("\t") and not line.startswith(" "):
                    # New interface
                    current_iface = line.split(":")[0]
                    interfaces.append({
                        "name": current_iface,
                        "type": "unknown",
                        "status": "up" if "UP" in line else "down",
                    })
                elif current_iface and "inet " in line:
                    # IPv4 address
                    parts = line.split()
                    if len(parts) >= 2:
                        for i, iface in enumerate(interfaces):
                            if iface["name"] == current_iface:
                                interfaces[i]["ip_address"] = parts[1]

        except Exception as e:
            LOG.warning(f"Error parsing macOS interfaces: {e}")

        return interfaces

    def _list_interfaces_windows(self) -> List[Dict]:
        """List network interfaces on Windows using PowerShell."""
        interfaces = []
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-NetAdapter | Select-Object Name, Status, MacAddress | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                try:
                    import json
                    data = json.loads(result.stdout)
                    if isinstance(data, list):
                        for iface in data:
                            interfaces.append({
                                "name": iface.get("Name", "unknown"),
                                "status": iface.get("Status", "unknown").lower(),
                                "mac_address": iface.get("MacAddress"),
                            })
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            LOG.warning(f"Error parsing Windows interfaces: {e}")

        return interfaces

    def _list_interfaces_linux(self) -> List[Dict]:
        """List network interfaces on Linux using ip command."""
        interfaces = []
        try:
            result = subprocess.run(
                ["ip", "link", "show"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.split("\n")

            for line in lines:
                if ":" in line and not line.startswith(" "):
                    # Interface line
                    parts = line.split(":")
                    if len(parts) >= 2:
                        iface_name = parts[1].strip().split()[0]
                        status = "up" if "UP" in line else "down"
                        interfaces.append({
                            "name": iface_name,
                            "status": status,
                        })

        except Exception as e:
            LOG.warning(f"Error parsing Linux interfaces: {e}")

        return interfaces

    def get_interface_details(self, interface_name: str) -> Dict:
        """Get detailed information about a specific interface."""
        try:
            if self.platform == "Darwin":
                return self._get_interface_details_macos(interface_name)
            elif self.platform == "Windows":
                return self._get_interface_details_windows(interface_name)
            else:
                return self._get_interface_details_linux(interface_name)
        except Exception as e:
            LOG.error(f"Error getting interface details: {e}")
            return {"error": str(e)}

    def _get_interface_details_macos(self, iface_name: str) -> Dict:
        """Get macOS interface details."""
        try:
            result = subprocess.run(
                ["ifconfig", iface_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            details = {"name": iface_name}

            for line in result.stdout.split("\n"):
                if "inet " in line and "inet6" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["ipv4"] = parts[1]
                elif "inet6" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["ipv6"] = parts[1]
                elif "ether" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["mac_address"] = parts[1]
                elif "mtu" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "mtu":
                            details["mtu"] = parts[i + 1] if i + 1 < len(parts) else None

            return details
        except Exception as e:
            return {"error": str(e)}

    def _get_interface_details_windows(self, iface_name: str) -> Dict:
        """Get Windows interface details."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"Get-NetIPAddress -InterfaceAlias '{iface_name}' | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            details = {"name": iface_name}

            if result.returncode == 0:
                try:
                    import json
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        details["ipv4"] = data.get("IPAddress")
                        details["prefix_length"] = data.get("PrefixLength")
                except json.JSONDecodeError:
                    pass

            return details
        except Exception as e:
            return {"error": str(e)}

    def _get_interface_details_linux(self, iface_name: str) -> Dict:
        """Get Linux interface details."""
        try:
            result = subprocess.run(
                ["ip", "addr", "show", iface_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            details = {"name": iface_name}

            for line in result.stdout.split("\n"):
                if "inet " in line and "inet6" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["ipv4"] = parts[1].split("/")[0]
                elif "inet6" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["ipv6"] = parts[1].split("/")[0]
                elif "link/ether" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        details["mac_address"] = parts[1]

            return details
        except Exception as e:
            return {"error": str(e)}

    def get_dns_configuration(self) -> Dict:
        """Get current DNS configuration."""
        try:
            if self.platform == "Darwin":
                return self._get_dns_macos()
            elif self.platform == "Windows":
                return self._get_dns_windows()
            else:
                return self._get_dns_linux()
        except Exception as e:
            LOG.error(f"Error getting DNS config: {e}")
            return {"error": str(e)}

    def _get_dns_macos(self) -> Dict:
        """Get macOS DNS configuration."""
        try:
            result = subprocess.run(
                ["scutil", "-d", "-r", "Router"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            dns_servers = []

            result2 = subprocess.run(
                ["scutil", "-d", "-r", "."],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result2.stdout.split("\n"):
                if "nameserver" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        dns_servers.append(parts[-1])

            return {
                "dns_servers": dns_servers,
                "resolver": "system",
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_dns_windows(self) -> Dict:
        """Get Windows DNS configuration."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-DnsClientServerAddress -AddressFamily IPv4 | Select-Object ServerAddresses",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            dns_servers = []

            for line in result.stdout.split("\n"):
                if "{" in line and "}" in line:
                    # Extract IPs from array string
                    ips = line.strip("{}").split()
                    dns_servers.extend(ips)

            return {
                "dns_servers": dns_servers,
                "resolver": "windows",
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_dns_linux(self) -> Dict:
        """Get Linux DNS configuration."""
        dns_servers = []

        try:
            # Try /etc/resolv.conf
            result = subprocess.run(
                ["cat", "/etc/resolv.conf"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        dns_servers.append(parts[1])
        except Exception:
            pass

        return {
            "dns_servers": dns_servers,
            "resolver": "systemd-resolved" if not dns_servers else "resolv.conf",
        }

    def get_routing_table(self, limit: int = 10) -> List[Dict]:
        """Get system routing table."""
        try:
            if self.platform == "Darwin":
                return self._get_routing_table_macos(limit)
            elif self.platform == "Windows":
                return self._get_routing_table_windows(limit)
            else:
                return self._get_routing_table_linux(limit)
        except Exception as e:
            LOG.error(f"Error getting routing table: {e}")
            return []

    def _get_routing_table_macos(self, limit: int) -> List[Dict]:
        """Get macOS routing table via netstat."""
        routes = []
        try:
            result = subprocess.run(
                ["netstat", "-rn"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.split("\n")[3:]  # Skip headers

            for line in lines[:limit]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        routes.append({
                            "destination": parts[0],
                            "gateway": parts[1],
                            "flags": parts[2],
                            "interface": parts[-1],
                        })
        except Exception as e:
            LOG.warning(f"Error parsing macOS routing: {e}")

        return routes

    def _get_routing_table_windows(self, limit: int) -> List[Dict]:
        """Get Windows routing table via route print."""
        routes = []
        try:
            result = subprocess.run(
                ["route", "print"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse Windows route print output
            lines = result.stdout.split("\n")
            in_ipv4 = False

            for line in lines:
                if "IPv4 Route Table" in line:
                    in_ipv4 = True
                    continue
                if in_ipv4 and line.strip():
                    parts = line.split()
                    if len(parts) >= 3 and parts[0][0].isdigit():
                        routes.append({
                            "destination": parts[0],
                            "netmask": parts[1],
                            "gateway": parts[2],
                        })
                        if len(routes) >= limit:
                            break

        except Exception as e:
            LOG.warning(f"Error parsing Windows routing: {e}")

        return routes

    def _get_routing_table_linux(self, limit: int) -> List[Dict]:
        """Get Linux routing table via ip route."""
        routes = []
        try:
            result = subprocess.run(
                ["ip", "route", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.split("\n")

            for line in lines[:limit]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        routes.append({
                            "destination": parts[0],
                            "gateway": parts[2] if len(parts) > 2 else "direct",
                            "interface": parts[-1],
                        })

        except Exception as e:
            LOG.warning(f"Error parsing Linux routing: {e}")

        return routes

    def check_connectivity(self, host: str = "8.8.8.8") -> Dict:
        """Check connectivity to a remote host."""
        try:
            # Try ping
            cmd = ["ping", "-c", "1"] if self.platform != "Windows" else ["ping", "-n", "1"]
            cmd.append(host)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )

            return {
                "host": host,
                "reachable": result.returncode == 0,
                "method": "ping",
                "output": result.stdout[:200] if result.stdout else "No response",
            }
        except Exception as e:
            LOG.error(f"Connectivity check failed: {e}")
            return {"error": str(e)}

    def get_network_statistics(self) -> Dict:
        """Get overall network statistics."""
        try:
            import psutil

            stats = {
                "interfaces": len(self.list_interfaces()),
                "total_sent_bytes": 0,
                "total_recv_bytes": 0,
                "dropped_in": 0,
                "dropped_out": 0,
                "errors_in": 0,
                "errors_out": 0,
            }

            net_io = psutil.net_io_counters()
            stats["total_sent_bytes"] = net_io.bytes_sent
            stats["total_recv_bytes"] = net_io.bytes_recv
            stats["dropped_in"] = net_io.dropin
            stats["dropped_out"] = net_io.dropout
            stats["errors_in"] = net_io.errin
            stats["errors_out"] = net_io.errout

            return stats
        except Exception as e:
            LOG.error(f"Error getting network stats: {e}")
            return {"error": str(e)}
