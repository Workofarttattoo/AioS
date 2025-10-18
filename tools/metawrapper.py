#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MetaWrapper - Metasploit Framework GUI Wrapper
Modern GUI interface for Metasploit Framework with module management and automation
"""

import sys
import json
import argparse
import subprocess
import re
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time


@dataclass
class Module:
    """Metasploit module information"""
    name: str
    type: str
    rank: str
    description: str
    targets: List[str] = None
    authors: List[str] = None
    references: List[str] = None

    def __post_init__(self):
        if self.targets is None:
            self.targets = []
        if self.authors is None:
            self.authors = []
        if self.references is None:
            self.references = []


@dataclass
class Exploit:
    """Exploit configuration"""
    module_path: str
    payload: str
    rhost: str
    rport: int
    lhost: str = ""
    lport: int = 4444
    options: Dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


class MetasploitWrapper:
    """Wrapper for Metasploit Framework"""

    def __init__(self):
        self.msf_path = self._find_metasploit()
        self.msfconsole = os.path.join(self.msf_path, "msfconsole") if self.msf_path else None
        self.msfvenom = os.path.join(self.msf_path, "msfvenom") if self.msf_path else None
        self.msfdb = os.path.join(self.msf_path, "msfdb") if self.msf_path else None

    def _find_metasploit(self) -> Optional[str]:
        """Find Metasploit installation"""
        # Common installation paths
        paths = [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/metasploit-framework/bin",
            os.path.expanduser("~/.rvm/gems/ruby-2.7.0/bin"),
            "/usr/share/metasploit-framework/bin"
        ]

        for path in paths:
            msfconsole = os.path.join(path, "msfconsole")
            if os.path.exists(msfconsole):
                return path

        # Try which command
        try:
            result = subprocess.run(['which', 'msfconsole'], capture_output=True, text=True)
            if result.returncode == 0:
                return os.path.dirname(result.stdout.strip())
        except:
            pass

        return None

    def is_installed(self) -> bool:
        """Check if Metasploit is installed"""
        return self.msf_path is not None

    def get_version(self) -> Optional[str]:
        """Get Metasploit version"""
        if not self.is_installed():
            return None

        try:
            result = subprocess.run(
                [self.msfconsole, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Extract version from output
            match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
            return match.group(1) if match else "Unknown"
        except:
            return None

    def search_modules(self, query: str, module_type: str = None) -> List[Module]:
        """Search for modules"""
        if not self.is_installed():
            return []

        try:
            cmd = f"search {query}"
            if module_type:
                cmd += f" type:{module_type}"

            result = self._run_msfconsole_command(cmd, timeout=30)

            modules = []
            # Parse search results
            for line in result.split('\n'):
                # Match module lines
                match = re.match(r'\s*(\d+)\s+(\S+)\s+(\S+)\s+(.+)', line)
                if match:
                    num, path, rank, description = match.groups()
                    module_type = path.split('/')[0] if '/' in path else 'unknown'
                    modules.append(Module(
                        name=path,
                        type=module_type,
                        rank=rank,
                        description=description.strip()
                    ))

            return modules
        except Exception as e:
            print(f"[!] Search failed: {e}")
            return []

    def get_module_info(self, module_path: str) -> Optional[Module]:
        """Get detailed module information"""
        if not self.is_installed():
            return None

        try:
            result = self._run_msfconsole_command(f"info {module_path}", timeout=20)

            # Parse module info
            name = module_path
            module_type = module_path.split('/')[0]
            rank = "unknown"
            description = ""
            targets = []
            authors = []
            references = []

            for line in result.split('\n'):
                if 'Rank:' in line:
                    rank = line.split(':')[1].strip()
                elif 'Description:' in line:
                    description = line.split(':')[1].strip()
                elif line.strip().startswith('Name:'):
                    authors.append(line.split(':')[1].strip())
                elif 'Target:' in line or 'Platform:' in line:
                    targets.append(line.strip())

            return Module(
                name=name,
                type=module_type,
                rank=rank,
                description=description,
                targets=targets,
                authors=authors,
                references=references
            )
        except:
            return None

    def generate_payload(self, payload: str, lhost: str, lport: int, format: str = "raw",
                        output_file: str = None, **options) -> Optional[bytes]:
        """Generate payload using msfvenom"""
        if not self.is_installed() or not self.msfvenom:
            return None

        try:
            cmd = [
                self.msfvenom,
                '-p', payload,
                f'LHOST={lhost}',
                f'LPORT={lport}',
                '-f', format
            ]

            # Add additional options
            for key, value in options.items():
                cmd.append(f'{key}={value}')

            if output_file:
                cmd.extend(['-o', output_file])

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0:
                if output_file:
                    return None  # Written to file
                else:
                    return result.stdout
            else:
                print(f"[!] Payload generation failed: {result.stderr.decode()}")
                return None

        except Exception as e:
            print(f"[!] Payload generation error: {e}")
            return None

    def list_payloads(self) -> List[str]:
        """List available payloads"""
        if not self.is_installed() or not self.msfvenom:
            return []

        try:
            result = subprocess.run(
                [self.msfvenom, '--list', 'payloads'],
                capture_output=True,
                text=True,
                timeout=30
            )

            payloads = []
            for line in result.stdout.split('\n'):
                # Extract payload names
                parts = line.strip().split()
                if parts and '/' in parts[0]:
                    payloads.append(parts[0])

            return payloads
        except:
            return []

    def _run_msfconsole_command(self, command: str, timeout: int = 30) -> str:
        """Run a command in msfconsole"""
        if not self.msfconsole:
            return ""

        try:
            # Use -q for quiet mode, -x for execute command
            result = subprocess.run(
                [self.msfconsole, '-q', '-x', f"{command}; exit"],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except Exception as e:
            print(f"[!] Command execution failed: {e}")
            return ""

    def get_popular_modules(self) -> Dict[str, List[str]]:
        """Get popular/recommended modules"""
        return {
            "exploits": [
                "exploit/windows/smb/ms17_010_eternalblue",
                "exploit/multi/handler",
                "exploit/windows/http/rejetto_hfs_exec",
                "exploit/unix/webapp/drupal_drupalgeddon2",
                "exploit/multi/http/struts2_content_type_ognl"
            ],
            "auxiliary": [
                "auxiliary/scanner/portscan/tcp",
                "auxiliary/scanner/smb/smb_version",
                "auxiliary/scanner/http/dir_scanner",
                "auxiliary/scanner/ssh/ssh_version",
                "auxiliary/scanner/ftp/ftp_version"
            ],
            "post": [
                "post/windows/gather/hashdump",
                "post/multi/recon/local_exploit_suggester",
                "post/windows/manage/migrate",
                "post/linux/gather/enum_system",
                "post/multi/gather/firefox_creds"
            ],
            "payloads": [
                "windows/meterpreter/reverse_tcp",
                "linux/x86/meterpreter/reverse_tcp",
                "windows/x64/meterpreter/reverse_https",
                "python/meterpreter/reverse_tcp",
                "cmd/unix/reverse_bash"
            ]
        }


def main(argv=None):
    """CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="MetaWrapper - Metasploit Framework GUI Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  metawrapper.py --check
  metawrapper.py --search eternalblue
  metawrapper.py --info exploit/windows/smb/ms17_010_eternalblue
  metawrapper.py --generate-payload windows/meterpreter/reverse_tcp --lhost 192.168.1.100 --lport 4444
  metawrapper.py --gui
        """
    )

    parser.add_argument('--check', action='store_true', help='Check Metasploit installation')
    parser.add_argument('--search', metavar='QUERY', help='Search for modules')
    parser.add_argument('--type', choices=['exploit', 'auxiliary', 'post', 'payload'], help='Filter by module type')
    parser.add_argument('--info', metavar='MODULE', help='Get module information')
    parser.add_argument('--generate-payload', metavar='PAYLOAD', help='Generate payload')
    parser.add_argument('--lhost', help='Local host IP for payload')
    parser.add_argument('--lport', type=int, default=4444, help='Local port for payload')
    parser.add_argument('--format', default='raw', help='Payload format (raw, exe, elf, etc.)')
    parser.add_argument('--output', help='Output file for payload')
    parser.add_argument('--list-payloads', action='store_true', help='List all available payloads')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--gui', action='store_true', help='Launch web-based GUI')
    parser.add_argument('--port', type=int, default=8089, help='GUI server port (default: 8089)')

    args = parser.parse_args(argv)

    wrapper = MetasploitWrapper()

    if args.gui:
        launch_gui(args.port, wrapper)
        return

    if args.check:
        installed = wrapper.is_installed()
        version = wrapper.get_version() if installed else None

        if args.json:
            print(json.dumps({
                "installed": installed,
                "version": version,
                "path": wrapper.msf_path
            }, indent=2))
        else:
            if installed:
                print(f"[✓] Metasploit Framework installed")
                print(f"    Version: {version}")
                print(f"    Path: {wrapper.msf_path}")
            else:
                print(f"[✗] Metasploit Framework not found")
                print(f"\nInstallation instructions:")
                print(f"  Kali Linux: sudo apt install metasploit-framework")
                print(f"  macOS: brew install metasploit")
                print(f"  Or visit: https://metasploit.com/download")
        return

    if not wrapper.is_installed():
        print("[!] Error: Metasploit Framework not installed")
        print("Run with --check for installation instructions")
        return

    if args.list_payloads:
        payloads = wrapper.list_payloads()
        if args.json:
            print(json.dumps({"payloads": payloads}, indent=2))
        else:
            print(f"\nAvailable Payloads ({len(payloads)}):\n")
            for payload in payloads:
                print(f"  • {payload}")
        return

    if args.search:
        modules = wrapper.search_modules(args.search, args.type)

        if args.json:
            print(json.dumps([asdict(m) for m in modules], indent=2))
        else:
            print(f"\nSearch Results ({len(modules)} modules):\n")
            for idx, module in enumerate(modules, 1):
                print(f"{idx}. {module.name}")
                print(f"   Type: {module.type} | Rank: {module.rank}")
                print(f"   {module.description}")
                print()
        return

    if args.info:
        module = wrapper.get_module_info(args.info)

        if module:
            if args.json:
                print(json.dumps(asdict(module), indent=2))
            else:
                print(f"\nModule: {module.name}")
                print(f"Type: {module.type}")
                print(f"Rank: {module.rank}")
                print(f"Description: {module.description}")
                if module.targets:
                    print(f"\nTargets:")
                    for target in module.targets:
                        print(f"  • {target}")
        else:
            print(f"[!] Module not found: {args.info}")
        return

    if args.generate_payload:
        if not args.lhost:
            print("[!] Error: --lhost required for payload generation")
            return

        print(f"[*] Generating payload: {args.generate_payload}")
        payload_data = wrapper.generate_payload(
            args.generate_payload,
            args.lhost,
            args.lport,
            args.format,
            args.output
        )

        if args.output:
            print(f"[✓] Payload saved to: {args.output}")
        elif payload_data:
            print(f"[✓] Payload generated ({len(payload_data)} bytes)")
            if not args.json:
                print(f"\nPayload data (base64):")
                import base64
                print(base64.b64encode(payload_data).decode())
        else:
            print(f"[✗] Payload generation failed")
        return

    parser.print_help()


def launch_gui(port: int, wrapper: MetasploitWrapper):
    """Launch web-based GUI"""
    from flask import Flask, render_template_string, request, jsonify

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(GUI_HTML)

    @app.route('/api/status', methods=['GET'])
    def status():
        return jsonify({
            "installed": wrapper.is_installed(),
            "version": wrapper.get_version(),
            "path": wrapper.msf_path
        })

    @app.route('/api/search', methods=['POST'])
    def search():
        data = request.json
        query = data.get('query', '')
        module_type = data.get('type')

        modules = wrapper.search_modules(query, module_type)
        return jsonify([asdict(m) for m in modules])

    @app.route('/api/module/<path:module_path>', methods=['GET'])
    def get_module(module_path):
        module = wrapper.get_module_info(module_path)
        if module:
            return jsonify(asdict(module))
        return jsonify({"error": "Module not found"}), 404

    @app.route('/api/payloads', methods=['GET'])
    def get_payloads():
        payloads = wrapper.list_payloads()
        return jsonify({"payloads": payloads[:100]})  # Limit for performance

    @app.route('/api/popular', methods=['GET'])
    def get_popular():
        return jsonify(wrapper.get_popular_modules())

    @app.route('/api/generate-payload', methods=['POST'])
    def generate_payload():
        data = request.json
        payload = data.get('payload')
        lhost = data.get('lhost')
        lport = data.get('lport', 4444)
        format_type = data.get('format', 'raw')

        if not payload or not lhost:
            return jsonify({"error": "Payload and lhost required"}), 400

        try:
            import tempfile
            import base64

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                output_file = tmp.name

            wrapper.generate_payload(payload, lhost, lport, format_type, output_file)

            with open(output_file, 'rb') as f:
                payload_data = f.read()

            os.unlink(output_file)

            return jsonify({
                "success": True,
                "size": len(payload_data),
                "data": base64.b64encode(payload_data).decode()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"[*] Starting MetaWrapper GUI on http://127.0.0.1:{port}")
    if not wrapper.is_installed():
        print(f"[!] WARNING: Metasploit Framework not detected")
        print(f"[!] Install Metasploit to enable full functionality")
    print(f"[*] Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)


GUI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MetaWrapper - Metasploit GUI</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-dark: #0a0a0a;
            --bg-medium: #1a1a1a;
            --bg-light: #2a2a2a;
            --accent: #ff073a;
            --accent-hover: #ff2954;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #888888;
            --border: #333333;
            --success: #00ff88;
            --warning: #ffaa00;
        }

        body {
            font-family: 'Courier New', 'Consolas', monospace;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #1a0000 0%, #3a0000 100%);
            border-bottom: 3px solid var(--accent);
            padding: 30px;
            text-align: center;
        }

        h1 {
            font-size: 3em;
            color: var(--accent);
            text-shadow: 0 0 30px rgba(255, 7, 58, 0.6);
            letter-spacing: 4px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1em;
            letter-spacing: 2px;
        }

        .status-bar {
            background: var(--bg-medium);
            padding: 15px 30px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        .status-indicator.offline {
            background: var(--accent);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .container {
            display: grid;
            grid-template-columns: 300px 1fr;
            height: calc(100vh - 200px);
        }

        .sidebar {
            background: var(--bg-medium);
            border-right: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
        }

        .sidebar-section {
            margin-bottom: 30px;
        }

        .sidebar-title {
            color: var(--accent);
            font-size: 1.1em;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .module-list {
            list-style: none;
        }

        .module-item {
            padding: 10px;
            background: var(--bg-dark);
            border-left: 3px solid var(--accent);
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .module-item:hover {
            background: var(--bg-light);
            border-left-width: 5px;
        }

        .main-content {
            padding: 30px;
            overflow-y: auto;
        }

        .search-panel {
            background: var(--bg-medium);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
        }

        .search-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }

        input[type="text"], select {
            flex: 1;
            padding: 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-primary);
            font-family: 'Courier New', monospace;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .btn {
            padding: 12px 25px;
            background: var(--accent);
            border: none;
            border-radius: 4px;
            color: white;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Courier New', monospace;
        }

        .btn:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
        }

        .results-grid {
            display: grid;
            gap: 15px;
        }

        .module-card {
            background: var(--bg-medium);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 4px;
            padding: 20px;
            transition: all 0.3s;
        }

        .module-card:hover {
            border-left-width: 8px;
            box-shadow: 0 0 20px rgba(255, 7, 58, 0.2);
        }

        .module-name {
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 5px;
        }

        .module-meta {
            color: var(--text-muted);
            font-size: 0.85em;
            margin-bottom: 10px;
        }

        .module-desc {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚔️ METAWRAPPER</h1>
        <div class="subtitle">METASPLOIT FRAMEWORK GUI</div>
    </div>

    <div class="status-bar">
        <div class="status-item">
            <div class="status-indicator" id="status-indicator"></div>
            <span id="status-text">Checking Metasploit...</span>
        </div>
        <div class="status-item">
            <span id="version-text">Version: -</span>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">Popular Exploits</div>
                <ul class="module-list" id="popular-exploits"></ul>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">Popular Auxiliary</div>
                <ul class="module-list" id="popular-auxiliary"></ul>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">Popular Payloads</div>
                <ul class="module-list" id="popular-payloads"></ul>
            </div>
        </div>

        <div class="main-content">
            <div class="search-panel">
                <div class="search-row">
                    <input type="text" id="search-query" placeholder="Search modules (e.g., eternalblue, drupal, ssh)">
                    <select id="module-type">
                        <option value="">All Types</option>
                        <option value="exploit">Exploits</option>
                        <option value="auxiliary">Auxiliary</option>
                        <option value="post">Post-Exploitation</option>
                        <option value="payload">Payloads</option>
                    </select>
                    <button class="btn" onclick="searchModules()">🔍 Search</button>
                </div>
            </div>

            <div id="loading" class="loading hidden">
                <div>Searching modules...</div>
            </div>

            <div id="results" class="results-grid"></div>
        </div>
    </div>

    <script>
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                const indicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                const versionText = document.getElementById('version-text');

                if (data.installed) {
                    indicator.classList.remove('offline');
                    statusText.textContent = 'Metasploit Online';
                    versionText.textContent = `Version: ${data.version || 'Unknown'}`;
                } else {
                    indicator.classList.add('offline');
                    statusText.textContent = 'Metasploit Not Found';
                    versionText.textContent = 'Not Installed';
                }
            } catch (error) {
                console.error('Status check failed:', error);
            }
        }

        async function loadPopular() {
            try {
                const response = await fetch('/api/popular');
                const data = await response.json();

                const exploitsList = document.getElementById('popular-exploits');
                const auxiliaryList = document.getElementById('popular-auxiliary');
                const payloadsList = document.getElementById('popular-payloads');

                data.exploits.forEach(module => {
                    const li = document.createElement('li');
                    li.className = 'module-item';
                    li.textContent = module.split('/').pop();
                    li.onclick = () => searchModules(module);
                    exploitsList.appendChild(li);
                });

                data.auxiliary.forEach(module => {
                    const li = document.createElement('li');
                    li.className = 'module-item';
                    li.textContent = module.split('/').pop();
                    li.onclick = () => searchModules(module);
                    auxiliaryList.appendChild(li);
                });

                data.payloads.forEach(module => {
                    const li = document.createElement('li');
                    li.className = 'module-item';
                    li.textContent = module.split('/').pop();
                    li.onclick = () => searchModules(module);
                    payloadsList.appendChild(li);
                });
            } catch (error) {
                console.error('Failed to load popular modules:', error);
            }
        }

        async function searchModules(query) {
            if (typeof query !== 'string') {
                query = document.getElementById('search-query').value.trim();
            }

            if (!query) {
                alert('Please enter a search query');
                return;
            }

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';

            try {
                const type = document.getElementById('module-type').value;
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, type: type || undefined })
                });

                const modules = await response.json();

                const results = document.getElementById('results');
                if (modules.length === 0) {
                    results.innerHTML = '<div class="loading">No modules found</div>';
                } else {
                    modules.forEach(module => {
                        const card = document.createElement('div');
                        card.className = 'module-card';
                        card.innerHTML = `
                            <div class="module-name">${module.name}</div>
                            <div class="module-meta">Type: ${module.type} | Rank: ${module.rank}</div>
                            <div class="module-desc">${module.description}</div>
                        `;
                        results.appendChild(card);
                    });
                }
            } catch (error) {
                alert('Search failed: ' + error.message);
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        }

        // Initialize
        checkStatus();
        loadPopular();

        // Enter key to search
        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchModules();
        });
    </script>
</body>
</html>
"""


def health_check() -> Dict[str, Any]:
    """Health check for SecurityAgent integration"""
    wrapper = MetasploitWrapper()

    return {
        "tool": "metawrapper",
        "status": "ok" if wrapper.is_installed() else "warn",
        "summary": "Metasploit Framework GUI wrapper",
        "details": {
            "metasploit_installed": wrapper.is_installed(),
            "version": wrapper.get_version(),
            "path": wrapper.msf_path,
            "features": [
                "Module search",
                "Exploit information",
                "Payload generation",
                "Popular modules database",
                "Web GUI interface"
            ]
        }
    }


if __name__ == "__main__":
    main()
