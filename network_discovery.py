#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Autonomous Network Discovery Engine
Proactively explores the network, identifies devices, and connects to open services.
"""

import socket
import subprocess
import threading
import time
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import ipaddress

# Try to import optional libraries
try:
    import netifaces
    NETIFACES_AVAILABLE = True
except ImportError:
    NETIFACES_AVAILABLE = False
    print("[warn] netifaces not available - install with: pip install netifaces")

try:
    from scapy.all import ARP, Ether, srp
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class DeviceDiscovery:
    """Discovers and identifies devices on the network"""

    def __init__(self):
        self.devices = {}  # ip -> device info
        self.lock = threading.Lock()
        self.running = False
        self.scan_threads = []

    def get_local_network(self) -> List[str]:
        """Get local network CIDR ranges"""
        networks = []

        try:
            if NETIFACES_AVAILABLE:
                # Use netifaces to get all network interfaces
                for iface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            ip = addr.get('addr')
                            netmask = addr.get('netmask')
                            if ip and netmask and not ip.startswith('127.'):
                                # Calculate CIDR
                                network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
                                networks.append(str(network))
            else:
                # Fallback: common private ranges
                networks = ['192.168.0.0/16', '10.0.0.0/8', '172.16.0.0/12']

        except Exception as e:
            print(f"[warn] Failed to detect local network: {e}")
            networks = ['192.168.1.0/24']  # Default

        return networks

    def resolve_hostname(self, ip: str) -> Optional[str]:
        """Resolve hostname via DNS reverse lookup"""
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror):
            return None

    def resolve_mdns(self, ip: str) -> Optional[str]:
        """Resolve mDNS/Bonjour name"""
        try:
            # Try to resolve .local domain
            result = subprocess.run(
                ['avahi-resolve-address', ip],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse output: "192.168.1.5  device-name.local"
                match = re.search(r'(\S+\.local)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # macOS alternative: dns-sd
        try:
            result = subprocess.run(
                ['dns-sd', '-G', 'v4', ip],
                capture_output=True,
                text=True,
                timeout=2
            )
            if 'Hostname' in result.stdout:
                match = re.search(r'Hostname:\s*(\S+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def resolve_netbios(self, ip: str) -> Optional[str]:
        """Resolve NetBIOS name (Windows)"""
        try:
            result = subprocess.run(
                ['nmblookup', '-A', ip],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                # Parse NetBIOS name
                for line in result.stdout.split('\n'):
                    if '<00>' in line and 'GROUP' not in line:
                        name = line.split()[0].strip()
                        return name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def get_mac_vendor(self, mac: str) -> Optional[str]:
        """Get device vendor from MAC address (simplified)"""
        # Common MAC prefixes (first 3 octets)
        vendors = {
            '00:50:56': 'VMware',
            '00:0C:29': 'VMware',
            '00:1C:42': 'Parallels',
            '08:00:27': 'VirtualBox',
            'B8:27:EB': 'Raspberry Pi',
            'DC:A6:32': 'Raspberry Pi',
            'E4:5F:01': 'Raspberry Pi',
            'F0:18:98': 'Apple',
            'AC:DE:48': 'Apple',
            '00:1B:63': 'Apple',
            '00:25:00': 'Apple',
            '10:DD:B1': 'Apple',
            '00:50:B6': 'HP',
            '00:1E:C9': 'Dell',
            '00:14:22': 'Dell',
            '00:21:70': 'Dell',
        }

        mac_prefix = ':'.join(mac.upper().split(':')[:3])
        return vendors.get(mac_prefix, 'Unknown')

    def arp_scan(self, network: str) -> List[Tuple[str, str]]:
        """Perform ARP scan to discover live hosts"""
        discovered = []

        if SCAPY_AVAILABLE:
            try:
                # Use scapy for ARP scan
                arp = ARP(pdst=network)
                ether = Ether(dst="ff:ff:ff:ff:ff:ff")
                packet = ether / arp

                result = srp(packet, timeout=2, verbose=0)[0]

                for sent, received in result:
                    ip = received.psrc
                    mac = received.hwsrc
                    discovered.append((ip, mac))

            except Exception as e:
                print(f"[warn] Scapy ARP scan failed: {e}")

        else:
            # Fallback: use arp command
            try:
                # Ping sweep first to populate ARP cache
                network_obj = ipaddress.IPv4Network(network, strict=False)
                for ip in list(network_obj.hosts())[:254]:  # Limit to first 254
                    subprocess.run(
                        ['ping', '-c', '1', '-W', '1', str(ip)],
                        capture_output=True,
                        timeout=2
                    )

                # Read ARP cache
                result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    match = re.search(r'\(([0-9.]+)\)\s+at\s+([0-9a-f:]+)', line, re.I)
                    if match:
                        ip, mac = match.groups()
                        discovered.append((ip, mac))

            except Exception as e:
                print(f"[warn] ARP cache scan failed: {e}")

        return discovered

    def identify_device(self, ip: str, mac: str = None) -> Dict:
        """Identify a device by all available means"""
        device_info = {
            'ip': ip,
            'mac': mac,
            'hostname': None,
            'mdns_name': None,
            'netbios_name': None,
            'vendor': None,
            'display_name': ip,
            'ports': [],
            'services': {},
            'last_seen': time.time()
        }

        # Try multiple resolution methods
        hostname = self.resolve_hostname(ip)
        if hostname:
            device_info['hostname'] = hostname
            device_info['display_name'] = hostname

        mdns_name = self.resolve_mdns(ip)
        if mdns_name:
            device_info['mdns_name'] = mdns_name
            if not device_info['hostname']:
                device_info['display_name'] = mdns_name

        netbios_name = self.resolve_netbios(ip)
        if netbios_name:
            device_info['netbios_name'] = netbios_name
            if not device_info['hostname'] and not device_info['mdns_name']:
                device_info['display_name'] = netbios_name

        # Get vendor from MAC
        if mac:
            vendor = self.get_mac_vendor(mac)
            device_info['vendor'] = vendor

        return device_info

    def scan_ports(self, ip: str, ports: List[int] = None) -> List[int]:
        """Scan common ports on a device"""
        if ports is None:
            # Common ports
            ports = [
                21,    # FTP
                22,    # SSH
                23,    # Telnet
                25,    # SMTP
                80,    # HTTP
                110,   # POP3
                139,   # NetBIOS
                143,   # IMAP
                443,   # HTTPS
                445,   # SMB
                3306,  # MySQL
                3389,  # RDP
                5432,  # PostgreSQL
                5900,  # VNC
                8080,  # HTTP-Alt
                8443,  # HTTPS-Alt
            ]

        open_ports = []

        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                sock.close()

                if result == 0:
                    open_ports.append(port)

            except Exception:
                continue

        return open_ports

    def identify_service(self, ip: str, port: int) -> Optional[str]:
        """Identify service running on a port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((ip, port))

            # Try to grab banner
            try:
                banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                sock.close()

                # Parse banner for service info
                if 'FTP' in banner:
                    return f"FTP: {banner}"
                elif 'SSH' in banner:
                    return f"SSH: {banner}"
                elif 'HTTP' in banner or 'Server:' in banner:
                    return f"HTTP: {banner}"
                elif 'Samba' in banner or 'SMB' in banner:
                    return f"SMB: {banner}"
                else:
                    return banner[:100]  # First 100 chars

            except:
                sock.close()

        except Exception:
            pass

        # Fallback to port number mapping
        service_map = {
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            80: 'HTTP',
            110: 'POP3',
            139: 'NetBIOS',
            143: 'IMAP',
            443: 'HTTPS',
            445: 'SMB',
            3306: 'MySQL',
            3389: 'RDP',
            5432: 'PostgreSQL',
            5900: 'VNC',
            8080: 'HTTP-Alt',
            8443: 'HTTPS-Alt',
        }

        return service_map.get(port, f"Unknown ({port})")

    def discover_network(self):
        """Main discovery loop - scans network continuously"""
        self.running = True

        while self.running:
            try:
                # Get local networks
                networks = self.get_local_network()

                print(f"[info] Scanning networks: {networks}")

                for network in networks:
                    # ARP scan for live hosts
                    discovered = self.arp_scan(network)

                    print(f"[info] Found {len(discovered)} hosts on {network}")

                    for ip, mac in discovered:
                        # Skip if already identified recently (within 5 minutes)
                        with self.lock:
                            if ip in self.devices:
                                age = time.time() - self.devices[ip].get('last_seen', 0)
                                if age < 300:  # 5 minutes
                                    continue

                        # Identify device
                        device_info = self.identify_device(ip, mac)

                        # Scan ports
                        open_ports = self.scan_ports(ip)
                        device_info['ports'] = open_ports

                        # Identify services
                        for port in open_ports:
                            service = self.identify_service(ip, port)
                            if service:
                                device_info['services'][port] = service

                        # Store device info
                        with self.lock:
                            self.devices[ip] = device_info

                        print(f"[info] Discovered: {device_info['display_name']} ({ip}) - Ports: {open_ports}")

                # Sleep before next scan
                time.sleep(60)  # Scan every minute

            except Exception as e:
                print(f"[error] Discovery error: {e}")
                time.sleep(10)

    def start_discovery(self):
        """Start discovery in background thread"""
        if not self.running:
            discovery_thread = threading.Thread(target=self.discover_network, daemon=True)
            discovery_thread.start()
            self.scan_threads.append(discovery_thread)
            print("[info] Network discovery started")

    def stop_discovery(self):
        """Stop discovery"""
        self.running = False
        print("[info] Network discovery stopped")

    def get_devices(self) -> Dict:
        """Get all discovered devices"""
        with self.lock:
            return dict(self.devices)

    def get_device_by_ip(self, ip: str) -> Optional[Dict]:
        """Get device info by IP"""
        with self.lock:
            return self.devices.get(ip)


class ServiceConnector:
    """Automatically connects to discovered open services"""

    def __init__(self, discovery: DeviceDiscovery):
        self.discovery = discovery
        self.connections = {}
        self.running = False

    def connect_ftp(self, ip: str, port: int = 21) -> bool:
        """Connect to FTP server (anonymous)"""
        try:
            import ftplib

            ftp = ftplib.FTP()
            ftp.connect(ip, port, timeout=5)
            ftp.login()  # Try anonymous

            # List files
            files = []
            ftp.retrlines('LIST', files.append)

            ftp.quit()

            print(f"[info] FTP connected: {ip}:{port} - {len(files)} files")
            return True

        except Exception as e:
            print(f"[warn] FTP connection failed to {ip}:{port} - {e}")
            return False

    def connect_ssh(self, ip: str, port: int = 22) -> bool:
        """Test SSH connection (no authentication)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((ip, port))
            banner = sock.recv(1024).decode('utf-8', errors='ignore')
            sock.close()

            print(f"[info] SSH available: {ip}:{port} - {banner.strip()}")
            return True

        except Exception as e:
            print(f"[warn] SSH connection failed to {ip}:{port} - {e}")
            return False

    def connect_smb(self, ip: str, port: int = 445) -> bool:
        """Connect to SMB share (list shares)"""
        try:
            result = subprocess.run(
                ['smbclient', '-L', ip, '-N'],  # -N = no password
                capture_output=True,
                text=True,
                timeout=10
            )

            if 'Sharename' in result.stdout:
                shares = []
                for line in result.stdout.split('\n'):
                    if 'Disk' in line or 'IPC' in line:
                        shares.append(line.strip())

                print(f"[info] SMB shares on {ip}: {len(shares)} found")
                return True

        except Exception as e:
            print(f"[warn] SMB connection failed to {ip}:{port} - {e}")

        return False

    def connect_http(self, ip: str, port: int = 80) -> bool:
        """Connect to HTTP server"""
        try:
            import urllib.request

            url = f"http://{ip}:{port}/"
            response = urllib.request.urlopen(url, timeout=5)
            content = response.read(1024).decode('utf-8', errors='ignore')

            print(f"[info] HTTP available: {url} - {len(content)} bytes")
            return True

        except Exception as e:
            print(f"[warn] HTTP connection failed to {ip}:{port} - {e}")
            return False

    def auto_connect_services(self):
        """Automatically connect to all discovered services"""
        self.running = True

        while self.running:
            try:
                devices = self.discovery.get_devices()

                for ip, device in devices.items():
                    services = device.get('services', {})

                    for port, service_name in services.items():
                        # Skip if already connected
                        conn_key = f"{ip}:{port}"
                        if conn_key in self.connections:
                            continue

                        # Try to connect based on service type
                        if 'FTP' in service_name:
                            success = self.connect_ftp(ip, port)
                        elif 'SSH' in service_name:
                            success = self.connect_ssh(ip, port)
                        elif 'SMB' in service_name:
                            success = self.connect_smb(ip, port)
                        elif 'HTTP' in service_name:
                            success = self.connect_http(ip, port)
                        else:
                            success = False

                        # Mark as attempted
                        self.connections[conn_key] = {
                            'ip': ip,
                            'port': port,
                            'service': service_name,
                            'success': success,
                            'timestamp': time.time()
                        }

                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"[error] Auto-connect error: {e}")
                time.sleep(10)

    def start_auto_connect(self):
        """Start auto-connect in background"""
        if not self.running:
            thread = threading.Thread(target=self.auto_connect_services, daemon=True)
            thread.start()
            print("[info] Service auto-connect started")

    def stop_auto_connect(self):
        """Stop auto-connect"""
        self.running = False
        print("[info] Service auto-connect stopped")


def main():
    """Main entry point"""
    print("[info] Ai|oS Autonomous Network Discovery Engine")
    print("[info] Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)")
    print()
    print("[warn] This tool will:")
    print("  - Scan your local network for devices")
    print("  - Attempt to identify device names")
    print("  - Scan for open ports and services")
    print("  - Attempt anonymous connections to FTP, SMB, HTTP")
    print()
    print("[warn] Only use on networks you own or have permission to scan!")
    print()

    # Create discovery engine
    discovery = DeviceDiscovery()
    connector = ServiceConnector(discovery)

    # Start discovery
    discovery.start_discovery()

    # Start auto-connect
    connector.start_auto_connect()

    print("[info] Discovery and auto-connect running...")
    print("[info] Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)

            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                devices = discovery.get_devices()
                print(f"[status] Discovered devices: {len(devices)}")

    except KeyboardInterrupt:
        print("\n[info] Stopping...")
        discovery.stop_discovery()
        connector.stop_auto_connect()


if __name__ == "__main__":
    main()
