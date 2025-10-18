#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Desktop Idle Visualizer - Activates network visualization when Ai|oS desktop is idle
"""

import time
import subprocess
import threading
from pathlib import Path
import http.server
import socketserver
import json
import os

# Import the network visualizer
from aios.network_visualizer import NetworkCapture, NetworkVisualizer, generate_visualization_html


class IdleDetector:
    """Detects when the desktop is idle"""

    def __init__(self, idle_threshold: int = 30):
        """
        Args:
            idle_threshold: Seconds of inactivity before considering desktop idle
        """
        self.idle_threshold = idle_threshold
        self.last_activity = time.time()
        self.is_idle = False

    def check_idle(self) -> bool:
        """Check if desktop is currently idle"""
        import sys

        try:
            if sys.platform == 'darwin':  # macOS
                # Use ioreg to get HIDIdleTime
                result = subprocess.run(
                    ['ioreg', '-c', 'IOHIDSystem'],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if 'HIDIdleTime' in line:
                        # Parse idle time in nanoseconds
                        idle_ns = int(line.split('=')[1].strip())
                        idle_seconds = idle_ns / 1_000_000_000
                        self.is_idle = idle_seconds > self.idle_threshold
                        return self.is_idle

            elif sys.platform == 'linux':
                # Use xprintidle if available
                result = subprocess.run(
                    ['xprintidle'],
                    capture_output=True,
                    text=True
                )
                idle_ms = int(result.stdout.strip())
                idle_seconds = idle_ms / 1000
                self.is_idle = idle_seconds > self.idle_threshold
                return self.is_idle

            elif sys.platform == 'win32':
                # Windows: use GetLastInputInfo
                import ctypes

                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [
                        ('cbSize', ctypes.c_uint),
                        ('dwTime', ctypes.c_uint)
                    ]

                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                idle_seconds = millis / 1000.0
                self.is_idle = idle_seconds > self.idle_threshold
                return self.is_idle

        except Exception as e:
            # Fallback: assume not idle if detection fails
            pass

        return False

    def reset_idle(self):
        """Reset idle timer (activity detected)"""
        self.last_activity = time.time()
        self.is_idle = False


class DesktopVisualizer:
    """Manages network visualization on desktop when idle"""

    def __init__(self, idle_threshold: int = 30, port: int = 8889):
        self.idle_detector = IdleDetector(idle_threshold)
        self.port = port
        self.capture = None
        self.visualizer = None
        self.server_thread = None
        self.running = False
        self.browser_process = None

    def start(self):
        """Start the desktop visualizer system"""
        print("[info] Desktop Idle Visualizer started")
        print(f"[info] Will activate after {self.idle_detector.idle_threshold}s of inactivity")

        self.running = True

        # Start idle monitoring loop
        while self.running:
            is_idle = self.idle_detector.check_idle()

            if is_idle and not self.capture:
                print("[info] Desktop idle detected - starting network visualization")
                self._activate_visualization()

            elif not is_idle and self.capture:
                print("[info] Activity detected - stopping network visualization")
                self._deactivate_visualization()

            time.sleep(5)  # Check every 5 seconds

    def _activate_visualization(self):
        """Activate network visualization"""
        try:
            # Start network capture
            self.capture = NetworkCapture()
            self.visualizer = NetworkVisualizer(self.capture)
            self.capture.start_capture()

            # Generate HTML
            html_path = generate_visualization_html("/tmp/aios_network_viz.html")

            # Start HTTP server
            self._start_server()

            # Open in browser (fullscreen if possible)
            time.sleep(1)  # Let server start
            self._open_browser(f"http://localhost:{self.port}/aios_network_viz.html")

        except Exception as e:
            print(f"[error] Failed to activate visualization: {e}")

    def _deactivate_visualization(self):
        """Deactivate network visualization"""
        try:
            # Close browser
            if self.browser_process:
                self.browser_process.terminate()
                self.browser_process = None

            # Stop capture
            if self.capture:
                self.capture.stop_capture()
                self.capture = None

            self.visualizer = None

        except Exception as e:
            print(f"[error] Failed to deactivate visualization: {e}")

    def _start_server(self):
        """Start HTTP server for visualization"""

        class VisualizationHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, visualizer=None, **kwargs):
                self.visualizer_ref = visualizer
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self.path.startswith('/api/network/visualization'):
                    # Parse mode from query string
                    mode = 'force_graph'
                    if '?' in self.path:
                        params = dict(x.split('=') for x in self.path.split('?')[1].split('&') if '=' in x)
                        mode = params.get('mode', 'force_graph')

                    # Generate visualization data
                    if self.visualizer_ref and self.visualizer_ref.visualizer:
                        if mode == 'force_graph':
                            data = self.visualizer_ref.visualizer.generate_force_graph()
                        elif mode == 'flow_diagram':
                            data = self.visualizer_ref.visualizer.generate_flow_diagram()
                        elif mode == 'heatmap':
                            data = self.visualizer_ref.visualizer.generate_heatmap()
                        elif mode == 'matrix':
                            data = self.visualizer_ref.visualizer.generate_matrix()
                        else:
                            data = self.visualizer_ref.visualizer.generate_force_graph()

                        # Add stats
                        stats = self.visualizer_ref.capture.get_stats()
                        data['stats'] = stats
                    else:
                        data = {}

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                else:
                    # Serve the HTML file
                    super().do_GET()

            def log_message(self, format, *args):
                # Suppress server logs
                pass

        def serve():
            os.chdir('/tmp')
            handler = lambda *args, **kwargs: VisualizationHandler(*args, visualizer=self, **kwargs)
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                httpd.serve_forever()

        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()

    def _open_browser(self, url: str):
        """Open browser in fullscreen mode"""
        import sys
        import webbrowser

        try:
            if sys.platform == 'darwin':  # macOS
                # Open in Safari (best for fullscreen on macOS)
                self.browser_process = subprocess.Popen([
                    'open', '-a', 'Safari', url
                ])
            elif sys.platform == 'linux':
                # Try to open in fullscreen mode
                self.browser_process = subprocess.Popen([
                    'google-chrome', '--kiosk', url
                ]) or subprocess.Popen([
                    'firefox', '--kiosk', url
                ]) or subprocess.Popen([
                    'chromium-browser', '--kiosk', url
                ])
            else:
                # Windows or fallback
                webbrowser.open(url)

        except Exception as e:
            # Fallback to default browser
            webbrowser.open(url)

    def stop(self):
        """Stop the visualizer"""
        self.running = False
        self._deactivate_visualization()


def main():
    """Main entry point for desktop idle visualizer"""
    import argparse

    parser = argparse.ArgumentParser(description="Ai|oS Desktop Idle Network Visualizer")
    parser.add_argument('--idle-threshold', '-t', type=int, default=30,
                        help="Seconds of inactivity before activating (default: 30)")
    parser.add_argument('--port', '-p', type=int, default=8889,
                        help="HTTP server port (default: 8889)")

    args = parser.parse_args()

    visualizer = DesktopVisualizer(
        idle_threshold=args.idle_threshold,
        port=args.port
    )

    print("[info] Ai|oS Desktop Idle Visualizer")
    print("[info] Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)")
    print()
    print("[info] Press Ctrl+C to stop")

    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\n[info] Stopping...")
        visualizer.stop()


if __name__ == "__main__":
    main()
