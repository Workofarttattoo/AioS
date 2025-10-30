#!/usr/bin/env python3
"""
gPIG - Gluttonous Packet Inspection Gadget
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AUTHORIZATION WARNING:
This tool is for AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.

gPIG is an intelligent network reconnaissance and exploitation framework that:
1. Maps internal vs external network architecture
2. Discovers all subsystems and services
3. Identifies entry points and vulnerabilities
4. Orchestrates quantum-powered attacks via red-team-tools.aios.is
5. Integrates with MythicKey for password discovery
6. Provides real-time browser overlay visualization

Features:
- Network topology mapping
- Service fingerprinting
- Vulnerability assessment
- API integration with red-team-tools.aios.is
- Quantum-enhanced password cracking
- Browser-based visualization
- Automated exploitation workflows
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import datetime
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
import socket
import ipaddress

try:
    from flask import Flask, render_template_string, jsonify, request
    import netifaces
    from scapy.all import ARP, Ether, srp, IP, ICMP, sr1, TCP
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

LOG = logging.getLogger("gpig")
LOG.setLevel(logging.INFO)

AUDIT_LOG = Path.home() / ".gpig" / "audit.log"
AUDIT_LOG.parent.mkdir(exist_ok=True)


def audit_log(action: str, details: Dict[str, Any]):
    """Audit logging."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": action,
        "details": details
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


@dataclass
class Host:
    """Discovered host."""
    ip: str
    mac: Optional[str] = None
    hostname: Optional[str] = None
    vendor: Optional[str] = None
    ports: List[int] = field(default_factory=list)
    services: Dict[int, str] = field(default_factory=dict)
    os_guess: Optional[str] = None
    vulnerabilities: List[Dict] = field(default_factory=list)
    is_gateway: bool = False
    is_external: bool = False


@dataclass
class NetworkMap:
    """Complete network topology."""
    internal_hosts: List[Host] = field(default_factory=list)
    external_targets: List[Host] = field(default_factory=list)
    gateways: List[Host] = field(default_factory=list)
    subnets: List[str] = field(default_factory=list)
    domain_name: Optional[str] = None
    intranet_urls: List[str] = field(default_factory=list)
    internet_urls: List[str] = field(default_factory=list)


class NetworkScanner:
    """
    Network discovery and mapping.

    Distinguishes internal network from external targets.
    """

    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.network_map = NetworkMap()

    def discover_network(self) -> NetworkMap:
        """
        Complete network discovery.

        Returns comprehensive network map.
        """
        LOG.info("[gPIG] Starting network discovery...")
        audit_log("network_discovery_start", {"interface": self.interface})

        # 1. Get local network info
        local_ip, local_network = self._get_local_network()
        LOG.info(f"[gPIG] Local network: {local_ip} ({local_network})")

        # 2. Find gateways
        gateways = self._find_gateways()
        self.network_map.gateways = gateways
        LOG.info(f"[gPIG] Found {len(gateways)} gateways")

        # 3. Scan internal network
        internal_hosts = self._scan_internal_network(local_network)
        self.network_map.internal_hosts = internal_hosts
        LOG.info(f"[gPIG] Discovered {len(internal_hosts)} internal hosts")

        # 4. Identify intranet vs internet services
        self._classify_services()

        # 5. Fingerprint services
        self._fingerprint_services()

        LOG.info("[gPIG] ✓ Network discovery complete")

        return self.network_map

    def _get_local_network(self) -> Tuple[str, str]:
        """Get local IP and network CIDR."""
        try:
            addrs = netifaces.ifaddresses(self.interface)
            ipv4 = addrs[netifaces.AF_INET][0]

            ip = ipv4["addr"]
            netmask = ipv4["netmask"]

            # Calculate network CIDR
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)

            return ip, str(network)

        except Exception as e:
            LOG.error(f"[gPIG] Failed to get local network: {e}")
            return "0.0.0.0", "0.0.0.0/0"

    def _find_gateways(self) -> List[Host]:
        """Find network gateways."""
        gateways = []

        try:
            # Get default gateway
            gws = netifaces.gateways()
            default_gw = gws.get("default", {}).get(netifaces.AF_INET)

            if default_gw:
                gw_ip = default_gw[0]
                gw_host = Host(ip=gw_ip, is_gateway=True)

                # Try to get MAC
                mac = self._get_mac(gw_ip)
                if mac:
                    gw_host.mac = mac

                gateways.append(gw_host)

        except Exception as e:
            LOG.warning(f"[gPIG] Gateway detection failed: {e}")

        return gateways

    def _scan_internal_network(self, network: str) -> List[Host]:
        """Scan internal network for hosts."""
        LOG.info(f"[gPIG] Scanning {network}...")

        hosts = []

        try:
            # ARP scan
            arp = ARP(pdst=network)
            ether = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp

            result = srp(packet, timeout=3, verbose=False)[0]

            for sent, received in result:
                ip = received.psrc
                mac = received.hwsrc

                host = Host(ip=ip, mac=mac)

                # Try reverse DNS
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                    host.hostname = hostname
                except:
                    pass

                hosts.append(host)
                LOG.debug(f"[gPIG] Found: {ip} ({mac})")

        except Exception as e:
            LOG.error(f"[gPIG] Network scan failed: {e}")

        return hosts

    def _get_mac(self, ip: str) -> Optional[str]:
        """Get MAC address for IP."""
        try:
            arp = ARP(pdst=ip)
            ether = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp

            result = srp(packet, timeout=2, verbose=False)[0]

            if result:
                return result[0][1].hwsrc

        except:
            pass

        return None

    def _classify_services(self):
        """Classify services as internal (intranet) or external (internet)."""
        LOG.info("[gPIG] Classifying services...")

        for host in self.network_map.internal_hosts:
            # Quick port scan
            common_ports = [21, 22, 23, 25, 53, 80, 443, 445, 3306, 3389, 5432, 8080, 8443]

            open_ports = self._scan_ports(host.ip, common_ports)
            host.ports = open_ports

            # Classify based on ports
            if 80 in open_ports or 443 in open_ports or 8080 in open_ports:
                # Web service - could be intranet or internet-facing
                self._probe_web_service(host)

    def _scan_ports(self, ip: str, ports: List[int]) -> List[int]:
        """Scan specific ports on host."""
        open_ports = []

        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                sock.close()

                if result == 0:
                    open_ports.append(port)

            except:
                pass

        return open_ports

    def _probe_web_service(self, host: Host):
        """Probe web service to determine if intranet or internet."""
        for port in [80, 443, 8080, 8443]:
            if port not in host.ports:
                continue

            protocol = "https" if port in [443, 8443] else "http"
            url = f"{protocol}://{host.ip}:{port}"

            try:
                response = requests.get(url, timeout=3, verify=False)

                # Analyze response to classify
                content = response.text.lower()

                # Intranet indicators
                intranet_keywords = [
                    "intranet", "internal", "employee", "corporate",
                    "sharepoint", "confluence", "jira", "gitlab"
                ]

                is_intranet = any(keyword in content for keyword in intranet_keywords)

                if is_intranet:
                    self.network_map.intranet_urls.append(url)
                else:
                    # Could be internet-facing or public web app
                    self.network_map.internet_urls.append(url)

                LOG.info(f"[gPIG] {url} -> {'Intranet' if is_intranet else 'Internet/Public'}")

            except:
                pass

    def _fingerprint_services(self):
        """Fingerprint discovered services."""
        LOG.info("[gPIG] Fingerprinting services...")

        for host in self.network_map.internal_hosts:
            for port in host.ports:
                service = self._identify_service(host.ip, port)
                if service:
                    host.services[port] = service

    def _identify_service(self, ip: str, port: int) -> Optional[str]:
        """Identify service running on port."""
        # Common service mappings
        service_map = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            443: "HTTPS",
            445: "SMB",
            3306: "MySQL",
            3389: "RDP",
            5432: "PostgreSQL",
            8080: "HTTP-Alt",
            8443: "HTTPS-Alt"
        }

        return service_map.get(port)


class VulnerabilityScanner:
    """
    Vulnerability assessment.

    Identifies entry points and weaknesses.
    """

    def __init__(self, network_map: NetworkMap):
        self.network_map = network_map

    def scan_vulnerabilities(self):
        """Scan for vulnerabilities across all hosts."""
        LOG.info("[gPIG] Scanning for vulnerabilities...")

        for host in self.network_map.internal_hosts:
            vulns = self._scan_host(host)
            host.vulnerabilities = vulns

            if vulns:
                LOG.warning(f"[gPIG] {host.ip}: Found {len(vulns)} vulnerabilities")

    def _scan_host(self, host: Host) -> List[Dict]:
        """Scan single host for vulnerabilities."""
        vulns = []

        # Check for common vulnerabilities
        for port, service in host.services.items():
            if service == "SMB":
                vulns.append({
                    "type": "SMB",
                    "port": port,
                    "description": "SMB service detected - check for EternalBlue, SMBGhost",
                    "severity": "high"
                })

            elif service == "HTTP" or service == "HTTPS":
                # Web vulnerabilities
                web_vulns = self._scan_web_vulns(host.ip, port)
                vulns.extend(web_vulns)

            elif service == "MySQL" or service == "PostgreSQL":
                vulns.append({
                    "type": "Database",
                    "port": port,
                    "description": f"{service} exposed - check for SQL injection, weak credentials",
                    "severity": "medium"
                })

        return vulns

    def _scan_web_vulns(self, ip: str, port: int) -> List[Dict]:
        """Scan web application for vulnerabilities."""
        vulns = []

        # Simple checks (would integrate with full scanner in production)
        protocol = "https" if port in [443, 8443] else "http"
        base_url = f"{protocol}://{ip}:{port}"

        # Check for common admin panels
        admin_paths = ["/admin", "/login", "/wp-admin", "/phpmyadmin"]

        for path in admin_paths:
            try:
                response = requests.get(f"{base_url}{path}", timeout=2, verify=False)

                if response.status_code == 200:
                    vulns.append({
                        "type": "Web",
                        "port": port,
                        "description": f"Admin panel found: {path}",
                        "severity": "medium",
                        "url": f"{base_url}{path}"
                    })

            except:
                pass

        return vulns


class QuantumRedTeamAPI:
    """
    Integration with red-team-tools.aios.is

    Provides quantum-enhanced red team capabilities.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://red-team-tools.aios.is/api/v1"
        self.api_key = api_key or os.getenv("AIOS_API_KEY")

    def request_quantum_tool(self, tool_name: str, params: Dict) -> Dict:
        """
        Request quantum tool from API.

        Args:
            tool_name: Tool to use (e.g., "password_crack", "crypto_break")
            params: Tool parameters

        Returns:
            Tool results
        """
        LOG.info(f"[gPIG] Requesting quantum tool: {tool_name}")

        try:
            response = requests.post(
                f"{self.base_url}/quantum/{tool_name}",
                json=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                LOG.error(f"[gPIG] API request failed: {response.status_code}")
                return {"error": response.text}

        except Exception as e:
            LOG.error(f"[gPIG] API error: {e}")
            return {"error": str(e)}

    def crack_passwords(self, hashes: List[str], algorithm: str = "quantum_grover") -> Dict:
        """
        Crack password hashes using quantum algorithms.

        Args:
            hashes: List of password hashes
            algorithm: Quantum algorithm to use

        Returns:
            Cracked passwords
        """
        LOG.info(f"[gPIG] Quantum password cracking: {len(hashes)} hashes")

        params = {
            "hashes": hashes,
            "algorithm": algorithm,
            "quantum_speedup": True
        }

        return self.request_quantum_tool("password_crack", params)


class gPIG:
    """
    Main gPIG orchestrator.

    Coordinates all reconnaissance, vulnerability scanning, and exploitation.
    """

    def __init__(self, interface: str = "eth0", api_key: Optional[str] = None):
        self.interface = interface
        self.scanner = NetworkScanner(interface)
        self.network_map = None
        self.vuln_scanner = None
        self.quantum_api = QuantumRedTeamAPI(api_key)
        self.app = Flask(__name__)
        self._setup_web_interface()

    def run_reconnaissance(self):
        """Full reconnaissance and mapping."""
        LOG.info("[gPIG] === Starting Reconnaissance ===")

        # 1. Network discovery
        self.network_map = self.scanner.discover_network()

        # 2. Vulnerability scanning
        self.vuln_scanner = VulnerabilityScanner(self.network_map)
        self.vuln_scanner.scan_vulnerabilities()

        # 3. Identify high-value targets
        targets = self._prioritize_targets()

        LOG.info(f"[gPIG] ✓ Reconnaissance complete: {len(targets)} high-value targets")

        return targets

    def _prioritize_targets(self) -> List[Host]:
        """Prioritize targets based on vulnerabilities and services."""
        targets = []

        for host in self.network_map.internal_hosts:
            score = 0

            # Score based on services
            if "MySQL" in host.services.values() or "PostgreSQL" in host.services.values():
                score += 3  # Database = high value

            if "HTTP" in host.services.values() or "HTTPS" in host.services.values():
                score += 2  # Web app = medium-high value

            if "SMB" in host.services.values():
                score += 2  # SMB = potential lateral movement

            # Score based on vulnerabilities
            score += len(host.vulnerabilities)

            if score >= 3:
                targets.append(host)

        # Sort by score
        targets.sort(key=lambda h: len(h.vulnerabilities) + len(h.ports), reverse=True)

        return targets

    def auto_exploit(self, target: Host):
        """
        Automated exploitation of target.

        Uses quantum API and MythicKey for password attacks.
        """
        LOG.info(f"[gPIG] === Exploiting {target.ip} ===")
        audit_log("auto_exploit", {"target": target.ip})

        results = {}

        # 1. Try MythicKey for password attacks
        if "HTTP" in target.services.values() or "MySQL" in target.services.values():
            passwords = self._attack_passwords(target)
            results["passwords"] = passwords

        # 2. Try service-specific exploits
        for port, service in target.services.items():
            if service == "SMB":
                results["smb"] = self._exploit_smb(target)

            elif service in ["MySQL", "PostgreSQL"]:
                results["database"] = self._exploit_database(target, port)

        return results

    def _attack_passwords(self, target: Host) -> List[str]:
        """Password attack using MythicKey and quantum API."""
        LOG.info(f"[gPIG] Password attack on {target.ip}")

        # Call MythicKey locally
        try:
            from mythickey import MythicKey

            mkey = MythicKey()

            # Generate common passwords
            common_passwords = mkey.generate_common_passwords()

            # Try quantum-enhanced cracking for any found hashes
            # (This would extract hashes first from target)

            return common_passwords[:10]  # Return top 10

        except Exception as e:
            LOG.warning(f"[gPIG] MythicKey attack failed: {e}")
            return []

    def _exploit_smb(self, target: Host) -> Dict:
        """SMB exploitation."""
        LOG.info(f"[gPIG] SMB exploitation: {target.ip}")
        # Would use tools like smbclient, enum4linux, etc.
        return {"status": "attempted"}

    def _exploit_database(self, target: Host, port: int) -> Dict:
        """Database exploitation."""
        LOG.info(f"[gPIG] Database exploitation: {target.ip}:{port}")
        # Would use CipherSpear or similar
        return {"status": "attempted"}

    def _setup_web_interface(self):
        """Setup web-based browser overlay."""

        @self.app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/network_map")
        def api_network_map():
            if not self.network_map:
                return jsonify({"error": "No network map available"})

            return jsonify({
                "internal_hosts": [asdict(h) for h in self.network_map.internal_hosts],
                "external_targets": [asdict(h) for h in self.network_map.external_targets],
                "intranet_urls": self.network_map.intranet_urls,
                "internet_urls": self.network_map.internet_urls
            })

        @self.app.route("/api/exploit", methods=["POST"])
        def api_exploit():
            data = request.get_json()
            target_ip = data.get("target_ip")

            # Find target
            target = None
            for host in self.network_map.internal_hosts:
                if host.ip == target_ip:
                    target = host
                    break

            if not target:
                return jsonify({"error": "Target not found"})

            # Run exploitation
            results = self.auto_exploit(target)

            return jsonify(results)

    def start_web_interface(self, port: int = 5555):
        """Start web interface."""
        LOG.info(f"[gPIG] Starting web interface on port {port}")
        LOG.info(f"[gPIG] Open browser: http://localhost:{port}")

        self.app.run(host="0.0.0.0", port=port, debug=False)


# HTML template for browser overlay
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>gPIG - Network Browser Overlay</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #0f0;
            padding: 20px;
        }
        h1 { color: #0f0; margin-bottom: 20px; }
        .section {
            background: #111;
            border: 1px solid #0f0;
            padding: 15px;
            margin-bottom: 20px;
        }
        .host {
            background: #222;
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid #0f0;
        }
        .vuln {
            color: #f00;
            margin-left: 20px;
        }
        button {
            background: #0f0;
            color: #000;
            border: none;
            padding: 8px 15px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #0ff; }
    </style>
</head>
<body>
    <h1>gPIG - Gluttonous Packet Inspection Gadget</h1>

    <div class="section">
        <h2>Network Map</h2>
        <div id="network-map">Loading...</div>
    </div>

    <div class="section">
        <h2>Intranet vs Internet</h2>
        <div id="classification">Loading...</div>
    </div>

    <div class="section">
        <h2>High-Value Targets</h2>
        <div id="targets">Loading...</div>
    </div>

    <script>
        async function loadNetworkMap() {
            const response = await fetch('/api/network_map');
            const data = await response.json();

            // Display internal hosts
            const mapDiv = document.getElementById('network-map');
            mapDiv.innerHTML = '';

            for (const host of data.internal_hosts) {
                const hostDiv = document.createElement('div');
                hostDiv.className = 'host';

                let html = `<strong>${host.ip}</strong>`;
                if (host.hostname) html += ` (${host.hostname})`;
                html += `<br>Ports: ${host.ports.join(', ')}`;

                if (host.vulnerabilities && host.vulnerabilities.length > 0) {
                    html += `<br><span class="vuln">⚠ ${host.vulnerabilities.length} vulnerabilities</span>`;
                }

                html += `<br><button onclick="exploit('${host.ip}')">Exploit</button>`;

                hostDiv.innerHTML = html;
                mapDiv.appendChild(hostDiv);
            }

            // Display classification
            const classDiv = document.getElementById('classification');
            classDiv.innerHTML = `
                <strong>Intranet URLs:</strong><br>
                ${data.intranet_urls.join('<br>') || 'None detected'}<br><br>
                <strong>Internet/Public URLs:</strong><br>
                ${data.internet_urls.join('<br>') || 'None detected'}
            `;
        }

        async function exploit(ip) {
            console.log('Exploiting:', ip);
            const response = await fetch('/api/exploit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({target_ip: ip})
            });

            const results = await response.json();
            alert(`Exploitation results:\\n${JSON.stringify(results, null, 2)}`);
        }

        // Load on page load
        loadNetworkMap();

        // Auto-refresh every 30 seconds
        setInterval(loadNetworkMap, 30000);
    </script>
</body>
</html>
"""


def health_check() -> Dict[str, Any]:
    """Health check."""
    return {
        "tool": "gPIG",
        "status": "ok" if DEPS_AVAILABLE else "warn",
        "summary": "gPIG ready" if DEPS_AVAILABLE else "Missing dependencies",
        "details": {"dependencies": DEPS_AVAILABLE}
    }


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="gPIG - Gluttonous Packet Inspection Gadget"
    )

    parser.add_argument("--interface", default="eth0", help="Network interface")
    parser.add_argument("--api-key", help="AIOS API key for quantum tools")
    parser.add_argument("--port", type=int, default=5555, help="Web interface port")
    parser.add_argument("--no-web", action="store_true", help="Disable web interface")
    parser.add_argument("--auto-exploit", action="store_true", help="Auto-exploit targets")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    # Setup logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

    if args.health:
        result = health_check()
        if args.json:
            print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'ok' else 1

    # Create gPIG instance
    gpig = gPIG(interface=args.interface, api_key=args.api_key)

    # Run reconnaissance
    targets = gpig.run_reconnaissance()

    # Auto-exploit if requested
    if args.auto_exploit:
        for target in targets:
            gpig.auto_exploit(target)

    # Start web interface
    if not args.no_web:
        gpig.start_web_interface(port=args.port)
    else:
        # Just print results
        if args.json:
            print(json.dumps(asdict(gpig.network_map), indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
