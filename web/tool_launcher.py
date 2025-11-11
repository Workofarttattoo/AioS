#!/usr/bin/env python3
"""
Ai|oS Tool Launcher Server
Bridges HTML GUI with Python security tools via HTTP API
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import threading
import sys
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add aios to path
AIOS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AIOS_ROOT))

PORT = 7777

# Tools that provide a GUI mode within the main module
INLINE_GUI_TOOLS = {
    "dirreaper",
    "proxyphantom",
    "vulnhunter",
}

class ToolLauncherHandler(BaseHTTPRequestHandler):
    """Handles tool launch requests from the GUI"""

    # Active tool processes
    active_processes = {}

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)

        if parsed.path == '/health':
            self.send_json({'status': 'ok', 'message': 'Tool Launcher Server running'})
        elif parsed.path == '/tools':
            self.send_json({'tools': self.get_available_tools()})
        else:
            self.send_error(404, 'Not Found')

    def do_POST(self):
        """Handle POST requests to launch tools"""
        parsed = urlparse(self.path)

        if parsed.path == '/launch':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            tool_name = data.get('tool')
            mode = data.get('mode', 'gui')  # gui or cli
            args = data.get('args', [])

            result = self.launch_tool(tool_name, mode, args)
            self.send_json(result)

        elif parsed.path == '/stop':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            tool_name = data.get('tool')
            result = self.stop_tool(tool_name)
            self.send_json(result)

        else:
            self.send_error(404, 'Not Found')

    def send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def get_available_tools(self):
        """List all available security tools"""
        tools_dir = AIOS_ROOT / 'tools'
        tools = []

        tool_files = [
            'aurorascan', 'cipherspear', 'skybreaker', 'mythickey',
            'spectratrace', 'nemesishydra', 'obsidianhunt', 'vectorflux',
            'dirreaper', 'vulnhunter', 'proxyphantom',
            'nmappro', 'payloadforge', 'scribe', 'osint_workflows',
            'sovereign_suite'
        ]

        for tool in tool_files:
            tool_path = tools_dir / f'{tool}.py'
            gui_path = tools_dir / f'{tool}_gui.py'
            module_path = tools_dir / f'{tool}.py'

            if tool_path.exists():
                tools.append({
                    'name': tool,
                    'has_cli': True,
                    'has_gui': gui_path.exists() or tool in INLINE_GUI_TOOLS,
                    'path': str(tool_path),
                })

        return tools

    def launch_tool(self, tool_name, mode='gui', args=None):
        """Launch a security tool"""
        if args is None:
            args = []

        tools_dir = AIOS_ROOT / 'tools'

        cli_path = tools_dir / f'{tool_name}.py'
        gui_path = tools_dir / f'{tool_name}_gui.py'

        # Build command and resolve executable path
        if mode == 'gui':
            cmd = [sys.executable, '-m', f'tools.{tool_name}', '--gui']
            if gui_path.exists():
                tool_path = gui_path
            elif cli_path.exists():
                tool_path = cli_path
            else:
                tool_path = gui_path
        else:
            cmd = [sys.executable, '-m', f'tools.{tool_name}'] + args
            tool_path = cli_path

        if not tool_path.exists():
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}',
                'path': str(tool_path)
            }

        try:
            # Launch in background
            process = subprocess.Popen(
                cmd,
                cwd=str(AIOS_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Store process
            self.active_processes[tool_name] = process

            return {
                'success': True,
                'message': f'Launched {tool_name} in {mode} mode',
                'pid': process.pid,
                'tool': tool_name,
                'mode': mode
            }

        except Exception as exc:
            return {
                'success': False,
                'error': str(exc),
                'tool': tool_name
            }

    def stop_tool(self, tool_name):
        """Stop a running tool"""
        if tool_name in self.active_processes:
            process = self.active_processes[tool_name]
            process.terminate()
            del self.active_processes[tool_name]

            return {
                'success': True,
                'message': f'Stopped {tool_name}'
            }
        else:
            return {
                'success': False,
                'error': f'Tool not running: {tool_name}'
            }

    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[Tool Launcher] {format % args}")


def run_server(port=PORT):
    """Run the tool launcher server"""
    server = HTTPServer(('localhost', port), ToolLauncherHandler)
    print(f"[info] Tool Launcher Server started on http://localhost:{port}")
    print(f"[info] Ready to launch Ai|oS security tools")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Shutting down Tool Launcher Server")
        server.shutdown()


if __name__ == '__main__':
    run_server()
