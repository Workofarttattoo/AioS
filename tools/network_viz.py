#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Network Visualizer - Ai|oS Tool Integration
Quick launcher for network traffic visualization
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path


def health_check():
    """Health check for network visualizer"""
    try:
        # Check if scapy is available
        try:
            import scapy
            scapy_status = "available"
        except ImportError:
            scapy_status = "missing (pip install scapy)"

        # Check for promiscuous mode permissions
        import platform
        if platform.system() == "Windows":
            perm_status = "requires administrator"
        else:
            perm_status = "requires root/sudo"

        return {
            "tool": "network_viz",
            "status": "ok",
            "summary": "Network traffic visualizer operational with autonomous discovery",
            "details": {
                "scapy": scapy_status,
                "permissions": perm_status,
                "visualizations": ["force_graph", "flow_diagram", "heatmap", "matrix", "wireshark_flow"],
                "modes": ["live", "idle_screensaver"],
                "features": [
                    "autonomous_device_discovery",
                    "dns_mdns_netbios_resolution",
                    "port_scanning",
                    "service_identification",
                    "auto_connect_ftp_ssh_smb_http"
                ]
            }
        }
    except Exception as e:
        return {
            "tool": "network_viz",
            "status": "error",
            "summary": f"Health check failed: {e}",
            "details": {}
        }


def gui():
    """Launch network visualizer GUI"""
    print("[info] Launching Network Visualizer GUI (Wireshark-style Flow View)...")

    # Get path to visualizer
    aios_path = Path(__file__).parent.parent
    viz_script = aios_path / "network_visualizer.py"

    if not viz_script.exists():
        print(f"[error] Visualizer not found: {viz_script}")
        return 1

    # Start visualizer with HTTP server
    try:
        print("[info] Starting network capture and HTTP server...")
        print("[info] You may be prompted for sudo password (required for promiscuous mode)")

        proc = subprocess.Popen([
            sys.executable,
            str(viz_script),
            "--serve",
            "--flow-view",  # Use Wireshark-style view
            "--discovery",  # Enable autonomous network discovery
            "--auto-connect",  # Enable auto-connect to services
            "--port", "8889",
            "--duration", "0"  # Infinite
        ])

        # Wait a moment for server to start
        import time
        time.sleep(2)

        # Open browser (flow view)
        url = "http://localhost:8889/packet_flow_viewer.html"
        print(f"[info] Opening browser: {url}")
        webbrowser.open(url)

        print("[info] Network Visualizer running")
        print("[info] Press Ctrl+C to stop")

        proc.wait()

    except KeyboardInterrupt:
        print("\n[info] Stopping visualizer...")
        proc.terminate()
    except Exception as e:
        print(f"[error] Failed to start visualizer: {e}")
        return 1

    return 0


def idle_mode():
    """Launch idle screensaver mode"""
    print("[info] Launching Network Visualizer - Idle Screensaver Mode...")

    # Get path to idle visualizer
    aios_path = Path(__file__).parent.parent
    idle_script = aios_path / "desktop_idle_visualizer.py"

    if not idle_script.exists():
        print(f"[error] Idle visualizer not found: {idle_script}")
        return 1

    try:
        subprocess.run([
            sys.executable,
            str(idle_script),
            "--idle-threshold", "30",
            "--port", "8889"
        ])
    except KeyboardInterrupt:
        print("\n[info] Stopping idle visualizer...")
    except Exception as e:
        print(f"[error] Failed to start idle visualizer: {e}")
        return 1

    return 0


def main(argv=None):
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Ai|oS Network Traffic Visualizer")
    parser.add_argument('--health-check', action='store_true', help="Run health check")
    parser.add_argument('--gui', action='store_true', help="Launch GUI")
    parser.add_argument('--idle', action='store_true', help="Launch idle screensaver mode")
    parser.add_argument('--quick', action='store_true', help="Quick launch with defaults")

    args = parser.parse_args(argv)

    if args.health_check:
        result = health_check()
        import json
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1

    elif args.gui or args.quick:
        return gui()

    elif args.idle:
        return idle_mode()

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
