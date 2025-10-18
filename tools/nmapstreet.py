#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NmapStreet - Next-Generation Network Scanner
Enhanced nmap with modern GUI, visualization, and advanced scanning profiles
"""

import sys
import json
import argparse
import subprocess
import re
import socket
import ipaddress
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict


@dataclass
class Port:
    """Port scan result"""
    port: int
    protocol: str
    state: str
    service: str
    version: str = ""
    banner: str = ""


@dataclass
class Host:
    """Host scan result"""
    ip: str
    hostname: str = ""
    status: str = "unknown"
    ports: List[Port] = None
    os: str = ""
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.ports is None:
            self.ports = []


SCAN_PROFILES = {
    "quick": {
        "name": "Quick Scan",
        "description": "Fast scan of common ports",
        "ports": "21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080",
        "timing": "T4",
        "options": []
    },
    "stealth": {
        "name": "Stealth SYN Scan",
        "description": "Stealthy SYN scan (requires root)",
        "ports": "1-1000",
        "timing": "T2",
        "options": ["-sS", "-Pn", "--scan-delay", "100ms"]
    },
    "comprehensive": {
        "name": "Comprehensive Scan",
        "description": "All TCP ports with version detection",
        "ports": "1-65535",
        "timing": "T4",
        "options": ["-sV", "-O", "--version-intensity", "5"]
    },
    "udp": {
        "name": "UDP Scan",
        "description": "Common UDP services",
        "ports": "53,67,68,69,123,135,137,138,139,161,162,445,500,514,520,631,1434,1900,4500,49152",
        "timing": "T4",
        "options": ["-sU"]
    },
    "aggressive": {
        "name": "Aggressive Scan",
        "description": "OS detection, version detection, script scanning, and traceroute",
        "ports": "1-1000",
        "timing": "T4",
        "options": ["-A"]
    },
    "vuln": {
        "name": "Vulnerability Scan",
        "description": "Run NSE vulnerability scripts",
        "ports": "1-1000",
        "timing": "T4",
        "options": ["-sV", "--script=vuln"]
    },
    "web": {
        "name": "Web Services",
        "description": "Scan web-related ports",
        "ports": "80,81,443,591,2082,2083,2087,2095,2096,3000,8000,8008,8080,8081,8443,8888,9000,9090",
        "timing": "T4",
        "options": ["-sV", "--script=http-*"]
    },
    "database": {
        "name": "Database Services",
        "description": "Scan database ports",
        "ports": "1433,1521,3306,5432,5984,6379,7000,7001,8529,9042,9160,9200,27017,27018,27019,28017",
        "timing": "T4",
        "options": ["-sV"]
    }
}


class NmapStreetScanner:
    """Enhanced network scanner"""

    def __init__(self, use_nmap: bool = True):
        self.use_nmap = use_nmap and self._check_nmap_installed()

    def _check_nmap_installed(self) -> bool:
        """Check if nmap is installed"""
        try:
            subprocess.run(['nmap', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def scan(self, targets: str, profile: str = "quick", **kwargs) -> Dict[str, Any]:
        """Main scanning function"""
        if self.use_nmap:
            return self._nmap_scan(targets, profile, **kwargs)
        else:
            return self._python_scan(targets, profile, **kwargs)

    def _nmap_scan(self, targets: str, profile: str, **kwargs) -> Dict[str, Any]:
        """Nmap-based scan"""
        profile_config = SCAN_PROFILES.get(profile, SCAN_PROFILES["quick"])

        # Build nmap command
        cmd = ["nmap"]
        cmd.append(f"-p{profile_config['ports']}")
        cmd.append(f"-{profile_config['timing']}")
        cmd.extend(profile_config['options'])
        cmd.extend(["-oX", "-"])  # XML output to stdout
        cmd.append(targets)

        print(f"[*] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=kwargs.get('timeout', 300))

            # Parse XML output
            hosts = self._parse_nmap_xml(result.stdout)

            return {
                "profile": profile,
                "targets": targets,
                "hosts": [asdict(h) for h in hosts],
                "scan_time": time.time(),
                "command": ' '.join(cmd),
                "success": True
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Scan timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_nmap_xml(self, xml_output: str) -> List[Host]:
        """Parse nmap XML output (simplified)"""
        hosts = []

        # Extract host blocks
        host_pattern = r'<host.*?</host>'
        for host_match in re.finditer(host_pattern, xml_output, re.DOTALL):
            host_xml = host_match.group(0)

            # Extract IP
            ip_match = re.search(r'<address addr="([^"]+)" addrtype="ipv4"', host_xml)
            if not ip_match:
                continue

            ip = ip_match.group(1)

            # Extract hostname
            hostname_match = re.search(r'<hostname name="([^"]+)"', host_xml)
            hostname = hostname_match.group(1) if hostname_match else ""

            # Extract status
            status_match = re.search(r'<status state="([^"]+)"', host_xml)
            status = status_match.group(1) if status_match else "unknown"

            # Extract ports
            ports = []
            port_pattern = r'<port protocol="([^"]+)" portid="(\d+)".*?<state state="([^"]+)".*?(?:<service name="([^"]*)".*?(?:product="([^"]*)")?)?'
            for port_match in re.finditer(port_pattern, host_xml, re.DOTALL):
                protocol, port_num, state, service, version = port_match.groups()
                ports.append(Port(
                    port=int(port_num),
                    protocol=protocol,
                    state=state,
                    service=service or "unknown",
                    version=version or ""
                ))

            # Extract OS
            os_match = re.search(r'<osmatch name="([^"]+)"', host_xml)
            os = os_match.group(1) if os_match else ""

            hosts.append(Host(
                ip=ip,
                hostname=hostname,
                status=status,
                ports=ports,
                os=os
            ))

        return hosts

    def _python_scan(self, targets: str, profile: str, **kwargs) -> Dict[str, Any]:
        """Pure Python scan (fallback when nmap not available)"""
        profile_config = SCAN_PROFILES.get(profile, SCAN_PROFILES["quick"])

        # Parse ports
        ports = self._parse_port_range(profile_config['ports'])[:50]  # Limit for performance

        # Parse targets
        target_ips = self._parse_targets(targets)

        print(f"[*] Python scan: {len(target_ips)} hosts, {len(ports)} ports")

        hosts = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self._scan_host, ip, ports): ip for ip in target_ips[:20]}  # Limit hosts

            for future in as_completed(futures):
                try:
                    host = future.result()
                    if host and host.ports:
                        hosts.append(host)
                except Exception as e:
                    continue

        return {
            "profile": profile,
            "targets": targets,
            "hosts": [asdict(h) for h in hosts],
            "scan_time": time.time(),
            "success": True,
            "method": "python_fallback"
        }

    def _scan_host(self, ip: str, ports: List[int]) -> Optional[Host]:
        """Scan a single host"""
        host = Host(ip=ip)

        # Check if host is up (ICMP or TCP)
        if not self._is_host_up(ip):
            return None

        host.status = "up"

        # Scan ports
        for port in ports:
            port_result = self._scan_port(ip, port)
            if port_result and port_result.state == "open":
                host.ports.append(port_result)

        # Reverse DNS
        try:
            host.hostname = socket.gethostbyaddr(ip)[0]
        except:
            pass

        return host

    def _is_host_up(self, ip: str) -> bool:
        """Check if host is responding"""
        # Try TCP connect to common ports
        for port in [80, 443, 22, 21]:
            if self._scan_port(ip, port, timeout=1):
                return True
        return False

    def _scan_port(self, ip: str, port: int, timeout: float = 2.0) -> Optional[Port]:
        """Scan a single port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            start = time.time()
            result = sock.connect_ex((ip, port))
            latency = (time.time() - start) * 1000

            if result == 0:
                # Port is open - try to grab banner
                banner = ""
                service = self._identify_service(port)

                try:
                    sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
                    banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()[:200]
                except:
                    pass

                sock.close()

                return Port(
                    port=port,
                    protocol="tcp",
                    state="open",
                    service=service,
                    banner=banner
                )

            sock.close()
        except:
            pass

        return None

    def _identify_service(self, port: int) -> str:
        """Identify common services by port"""
        common_ports = {
            20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            53: "dns", 80: "http", 110: "pop3", 111: "rpcbind", 135: "msrpc",
            139: "netbios-ssn", 143: "imap", 443: "https", 445: "microsoft-ds",
            993: "imaps", 995: "pop3s", 1723: "pptp", 3306: "mysql", 3389: "ms-wbt-server",
            5432: "postgresql", 5900: "vnc", 6379: "redis", 8080: "http-proxy",
            8443: "https-alt", 27017: "mongodb"
        }
        return common_ports.get(port, "unknown")

    def _parse_port_range(self, port_spec: str) -> List[int]:
        """Parse port specification (e.g., '1-1000,8080,9000-9100')"""
        ports = []
        for part in port_spec.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                ports.extend(range(start, min(end + 1, 65536)))
            else:
                ports.append(int(part))
        return sorted(set(ports))

    def _parse_targets(self, targets: str) -> List[str]:
        """Parse target specification"""
        ips = []

        for target in targets.replace(',', ' ').split():
            target = target.strip()

            # Check if CIDR notation
            if '/' in target:
                try:
                    network = ipaddress.ip_network(target, strict=False)
                    ips.extend([str(ip) for ip in network.hosts()])
                except:
                    pass
            # Check if IP range (192.168.1.1-254)
            elif '-' in target and target.count('.') == 3:
                base = '.'.join(target.split('.')[:-1])
                last_octet = target.split('.')[-1]
                if '-' in last_octet:
                    start, end = map(int, last_octet.split('-'))
                    ips.extend([f"{base}.{i}" for i in range(start, end + 1)])
            # Single IP or hostname
            else:
                try:
                    # Try to resolve hostname
                    resolved = socket.gethostbyname(target)
                    ips.append(resolved)
                except:
                    # Assume it's an IP
                    ips.append(target)

        return ips[:256]  # Limit for safety


def main(argv=None):
    """CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="NmapStreet - Next-Generation Network Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nmapstreet.py 192.168.1.0/24 --profile quick
  nmapstreet.py 10.0.0.1 --profile stealth
  nmapstreet.py example.com --profile web --json
  nmapstreet.py --gui
        """
    )

    parser.add_argument('targets', nargs='?', help='Target IP, hostname, or CIDR (e.g., 192.168.1.0/24)')
    parser.add_argument('-p', '--profile', choices=list(SCAN_PROFILES.keys()), default='quick',
                       help='Scan profile to use')
    parser.add_argument('--list-profiles', action='store_true', help='List available scan profiles')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--no-nmap', action='store_true', help='Use Python fallback instead of nmap')
    parser.add_argument('--gui', action='store_true', help='Launch web-based GUI')
    parser.add_argument('--port', type=int, default=8087, help='GUI server port (default: 8087)')
    parser.add_argument('--timeout', type=int, default=300, help='Scan timeout in seconds')

    args = parser.parse_args(argv)

    if args.list_profiles:
        print("\n=== Available Scan Profiles ===\n")
        for profile_id, profile in SCAN_PROFILES.items():
            print(f"{profile_id:15s} - {profile['name']}")
            print(f"{'':15s}   {profile['description']}")
            print()
        return

    if args.gui:
        launch_gui(args.port)
        return

    if not args.targets:
        parser.print_help()
        print("\n[!] Error: Target is required")
        print("Example: python nmapstreet.py 192.168.1.0/24")
        return

    # Run scan
    scanner = NmapStreetScanner(use_nmap=not args.no_nmap)

    if not scanner.use_nmap:
        print("[!] Warning: nmap not found, using Python fallback (limited features)")

    print(f"[*] Starting {SCAN_PROFILES[args.profile]['name']}...")
    results = scanner.scan(args.targets, args.profile, timeout=args.timeout)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


def print_results(results: Dict[str, Any]):
    """Print human-readable results"""
    if not results.get('success'):
        print(f"\n[!] Scan failed: {results.get('error', 'Unknown error')}")
        return

    hosts = results.get('hosts', [])

    print("\n" + "="*70)
    print("NMAP STREET EDITION - SCAN RESULTS")
    print("="*70)
    print(f"Profile: {results.get('profile', 'N/A')}")
    print(f"Targets: {results.get('targets', 'N/A')}")
    print(f"Hosts Found: {len(hosts)}")
    print("="*70)

    for host_data in hosts:
        print(f"\n{'─'*70}")
        print(f"Host: {host_data['ip']}")
        if host_data.get('hostname'):
            print(f"Hostname: {host_data['hostname']}")
        if host_data.get('os'):
            print(f"OS: {host_data['os']}")
        print(f"Status: {host_data['status']}")

        ports = host_data.get('ports', [])
        if ports:
            print(f"\nOpen Ports ({len(ports)}):")
            print(f"{'PORT':<10} {'STATE':<10} {'SERVICE':<20} {'VERSION'}")
            print(f"{'-'*70}")
            for port in ports:
                version = port.get('version', '')
                banner = port.get('banner', '')
                info = version or (banner[:30] + '...' if len(banner) > 30 else banner)
                print(f"{port['port']:<10} {port['state']:<10} {port['service']:<20} {info}")
        else:
            print("\nNo open ports found")

    print("\n" + "="*70)
    print(f"Scan completed: {len(hosts)} host(s) scanned")
    print("="*70 + "\n")


def launch_gui(port: int = 8087):
    """Launch web-based GUI"""
    from flask import Flask, render_template_string, request, jsonify

    app = Flask(__name__)
    scanner = NmapStreetScanner()

    @app.route('/')
    def index():
        return render_template_string(GUI_HTML)

    @app.route('/api/profiles', methods=['GET'])
    def get_profiles():
        return jsonify(SCAN_PROFILES)

    @app.route('/api/scan', methods=['POST'])
    def scan():
        data = request.json
        targets = data.get('targets')
        profile = data.get('profile', 'quick')

        if not targets:
            return jsonify({"error": "No targets specified"}), 400

        try:
            results = scanner.scan(targets, profile, timeout=data.get('timeout', 300))
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"[*] Starting NmapStreet GUI on http://127.0.0.1:{port}")
    print(f"[*] Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)


GUI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>NmapStreet - Next-Gen Network Scanner</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-dark: #0a0a0a;
            --bg-medium: #1a1a1a;
            --bg-light: #2a2a2a;
            --accent: #00ff88;
            --accent-hover: #00dd77;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #888888;
            --border: #333333;
            --success: #00ff88;
            --warning: #ffaa00;
            --error: #ff3366;
            --info: #66ccff;
        }

        body {
            font-family: 'Courier New', 'Consolas', monospace;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
            background-image:
                linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
            background-size: 20px 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 30px;
            background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-medium) 100%);
        }

        h1 {
            font-size: 3em;
            color: var(--accent);
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            margin-bottom: 10px;
            letter-spacing: 3px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1em;
            letter-spacing: 1px;
        }

        .scan-panel {
            background: var(--bg-medium);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.1);
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 250px 150px;
            gap: 15px;
            margin-bottom: 20px;
        }

        input[type="text"], select {
            padding: 15px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: 'Courier New', monospace;
            font-size: 1em;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.2);
        }

        .btn {
            padding: 15px 30px;
            background: var(--accent);
            border: none;
            border-radius: 6px;
            color: var(--bg-dark);
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn:hover:not(:disabled) {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .profiles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .profile-card {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .profile-card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }

        .profile-card.selected {
            border-color: var(--accent);
            background: rgba(0, 255, 136, 0.1);
        }

        .profile-name {
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 5px;
        }

        .profile-desc {
            color: var(--text-muted);
            font-size: 0.85em;
        }

        .results-container {
            display: grid;
            gap: 20px;
        }

        .host-card {
            background: var(--bg-medium);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
        }

        .host-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }

        .host-ip {
            font-size: 1.5em;
            color: var(--accent);
            font-weight: 700;
        }

        .host-status {
            padding: 6px 12px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid var(--accent);
            border-radius: 4px;
            color: var(--accent);
            font-weight: 700;
            font-size: 0.9em;
        }

        .host-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .info-item {
            display: flex;
            flex-direction: column;
        }

        .info-label {
            color: var(--text-muted);
            font-size: 0.85em;
            margin-bottom: 5px;
        }

        .info-value {
            color: var(--text-primary);
            font-weight: 600;
        }

        .ports-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }

        .ports-table th {
            background: var(--bg-dark);
            padding: 12px;
            text-align: left;
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
        }

        .ports-table td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .ports-table tr:hover {
            background: var(--bg-dark);
        }

        .port-open {
            color: var(--success);
            font-weight: 700;
        }

        .loading {
            text-align: center;
            padding: 60px 20px;
        }

        .spinner {
            border: 4px solid var(--border);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .scan-stats {
            background: var(--bg-dark);
            border: 1px solid var(--info);
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            color: var(--info);
            font-weight: 700;
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.85em;
            margin-top: 5px;
        }

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 NMAPSTREET</h1>
            <div class="subtitle">NEXT-GENERATION NETWORK SCANNER</div>
        </header>

        <div class="scan-panel">
            <div class="form-row">
                <input type="text" id="targets" placeholder="Target: 192.168.1.0/24, 10.0.0.1, or example.com" />
                <select id="profile">
                    <option value="">Select Profile...</option>
                </select>
                <button class="btn" id="scan-btn" onclick="startScan()">SCAN</button>
            </div>

            <div class="profiles-grid" id="profiles-grid"></div>
        </div>

        <div id="loading" class="loading hidden">
            <div class="spinner"></div>
            <div style="color: var(--text-secondary); font-size: 1.2em;">SCANNING NETWORK...</div>
        </div>

        <div id="stats" class="scan-stats hidden"></div>
        <div id="results" class="results-container"></div>
    </div>

    <script>
        let profiles = {};
        let selectedProfile = 'quick';

        async function loadProfiles() {
            try {
                const response = await fetch('/api/profiles');
                profiles = await response.json();

                const select = document.getElementById('profile');
                const grid = document.getElementById('profiles-grid');

                for (const [id, profile] of Object.entries(profiles)) {
                    // Add to dropdown
                    const option = document.createElement('option');
                    option.value = id;
                    option.textContent = profile.name;
                    select.appendChild(option);

                    // Add to grid
                    const card = document.createElement('div');
                    card.className = 'profile-card';
                    if (id === 'quick') card.classList.add('selected');
                    card.innerHTML = `
                        <div class="profile-name">${profile.name}</div>
                        <div class="profile-desc">${profile.description}</div>
                    `;
                    card.onclick = () => selectProfile(id, card);
                    grid.appendChild(card);
                }

                select.value = 'quick';
            } catch (error) {
                console.error('Failed to load profiles:', error);
            }
        }

        function selectProfile(id, cardElement) {
            selectedProfile = id;
            document.getElementById('profile').value = id;

            document.querySelectorAll('.profile-card').forEach(c => c.classList.remove('selected'));
            cardElement.classList.add('selected');
        }

        async function startScan() {
            const targets = document.getElementById('targets').value.trim();
            if (!targets) {
                alert('Please enter a target');
                return;
            }

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('stats').classList.add('hidden');
            document.getElementById('results').innerHTML = '';
            document.getElementById('scan-btn').disabled = true;

            try {
                const response = await fetch('/api/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        targets: targets,
                        profile: selectedProfile
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Scan error: ' + data.error);
                    return;
                }

                displayResults(data);
            } catch (error) {
                alert('Scan failed: ' + error.message);
            } finally {
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('scan-btn').disabled = false;
            }
        }

        function displayResults(data) {
            const hosts = data.hosts || [];

            // Show stats
            const stats = document.getElementById('stats');
            const totalPorts = hosts.reduce((sum, h) => sum + (h.ports?.length || 0), 0);
            stats.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${hosts.length}</div>
                    <div class="stat-label">HOSTS FOUND</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${totalPorts}</div>
                    <div class="stat-label">OPEN PORTS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.profile.toUpperCase()}</div>
                    <div class="stat-label">PROFILE</div>
                </div>
            `;
            stats.classList.remove('hidden');

            // Show hosts
            const results = document.getElementById('results');
            hosts.forEach(host => {
                const card = createHostCard(host);
                results.appendChild(card);
            });

            if (hosts.length === 0) {
                results.innerHTML = '<div style="text-align: center; padding: 60px; color: var(--text-muted);">No hosts found</div>';
            }
        }

        function createHostCard(host) {
            const card = document.createElement('div');
            card.className = 'host-card';

            const portsHTML = host.ports && host.ports.length > 0 ? `
                <table class="ports-table">
                    <thead>
                        <tr>
                            <th>PORT</th>
                            <th>STATE</th>
                            <th>SERVICE</th>
                            <th>VERSION</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${host.ports.map(p => `
                            <tr>
                                <td>${p.port}/${p.protocol}</td>
                                <td class="port-open">${p.state.toUpperCase()}</td>
                                <td>${p.service}</td>
                                <td>${p.version || p.banner || '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            ` : '<div style="text-align: center; padding: 20px; color: var(--text-muted);">No open ports detected</div>';

            card.innerHTML = `
                <div class="host-header">
                    <div class="host-ip">🖥️ ${host.ip}</div>
                    <div class="host-status">${host.status.toUpperCase()}</div>
                </div>

                <div class="host-info">
                    ${host.hostname ? `
                        <div class="info-item">
                            <div class="info-label">HOSTNAME</div>
                            <div class="info-value">${host.hostname}</div>
                        </div>
                    ` : ''}
                    ${host.os ? `
                        <div class="info-item">
                            <div class="info-label">OPERATING SYSTEM</div>
                            <div class="info-value">${host.os}</div>
                        </div>
                    ` : ''}
                    <div class="info-item">
                        <div class="info-label">OPEN PORTS</div>
                        <div class="info-value">${host.ports?.length || 0}</div>
                    </div>
                </div>

                ${portsHTML}
            `;

            return card;
        }

        // Initialize
        loadProfiles();

        // Enter key to scan
        document.getElementById('targets').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') startScan();
        });
    </script>
</body>
</html>
"""


def health_check() -> Dict[str, Any]:
    """Health check for SecurityAgent integration"""
    scanner = NmapStreetScanner()

    return {
        "tool": "nmapstreet",
        "status": "ok",
        "summary": "Enhanced network scanner with modern GUI",
        "details": {
            "nmap_available": scanner.use_nmap,
            "scan_profiles": len(SCAN_PROFILES),
            "profiles": list(SCAN_PROFILES.keys()),
            "features": [
                "8 scan profiles",
                "Python fallback scanning",
                "Web GUI interface",
                "Service detection",
                "Banner grabbing"
            ]
        }
    }


if __name__ == "__main__":
    main()
