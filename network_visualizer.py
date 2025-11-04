#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Network Traffic Visualizer - Promiscuous Mode Packet Capture and Visualization
Captures network traffic and generates real-time visualizations for Ai|oS desktop.
"""

import socket
import struct
import json
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import sys

# Try to import scapy for advanced packet parsing
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[warn] scapy not available - install with: pip install scapy")
    print("[info] Falling back to raw socket capture")


class NetworkCapture:
    """Captures network packets in promiscuous mode"""

    def __init__(self, interface: str = None, max_packets: int = 1000):
        self.interface = interface
        self.max_packets = max_packets
        self.packets = deque(maxlen=max_packets)
        self.nodes = defaultdict(lambda: {"sent": 0, "received": 0, "protocols": set()})
        self.connections = defaultdict(int)  # (src, dst) -> packet_count
        self.protocol_stats = defaultdict(int)
        self.capture_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start_capture(self):
        """Start packet capture in background thread"""
        if self.running:
            return

        self.running = True
        if SCAPY_AVAILABLE:
            self.capture_thread = threading.Thread(target=self._scapy_capture, daemon=True)
        else:
            self.capture_thread = threading.Thread(target=self._raw_socket_capture, daemon=True)

        self.capture_thread.start()
        print(f"[info] Network capture started (promiscuous mode)")

    def stop_capture(self):
        """Stop packet capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        print("[info] Network capture stopped")

    def _scapy_capture(self):
        """Capture packets using scapy"""
        try:
            sniff(
                iface=self.interface,
                prn=self._process_scapy_packet,
                store=0,
                stop_filter=lambda _: not self.running
            )
        except PermissionError:
            print("[error] Permission denied - need root/admin for promiscuous mode")
            print("[info] Try: sudo python -m aios.network_visualizer")
            self.running = False
        except Exception as e:
            print(f"[error] Scapy capture failed: {e}")
            self.running = False

    def _process_scapy_packet(self, packet):
        """Process a packet captured by scapy"""
        try:
            packet_info = {
                "timestamp": time.time(),
                "size": len(packet)
            }

            # Extract IP layer
            if IP in packet:
                packet_info["src"] = packet[IP].src
                packet_info["dst"] = packet[IP].dst
                packet_info["protocol"] = packet[IP].proto

                # Protocol-specific info
                if TCP in packet:
                    packet_info["protocol_name"] = "TCP"
                    packet_info["sport"] = packet[TCP].sport
                    packet_info["dport"] = packet[TCP].dport
                elif UDP in packet:
                    packet_info["protocol_name"] = "UDP"
                    packet_info["sport"] = packet[UDP].sport
                    packet_info["dport"] = packet[UDP].dport
                elif ICMP in packet:
                    packet_info["protocol_name"] = "ICMP"
                else:
                    packet_info["protocol_name"] = f"IP-{packet[IP].proto}"

            elif ARP in packet:
                packet_info["src"] = packet[ARP].psrc
                packet_info["dst"] = packet[ARP].pdst
                packet_info["protocol_name"] = "ARP"

            else:
                # Unknown packet type
                packet_info["src"] = "unknown"
                packet_info["dst"] = "unknown"
                packet_info["protocol_name"] = "UNKNOWN"

            self._update_stats(packet_info)

        except Exception as e:
            # Silently skip malformed packets
            pass

    def _raw_socket_capture(self):
        """Fallback: capture using raw sockets (Unix/Linux only)"""
        try:
            # Create raw socket
            if sys.platform == "darwin":  # macOS
                sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            else:  # Linux
                sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))

            sock.settimeout(1.0)  # 1 second timeout for stop check

            while self.running:
                try:
                    raw_data, addr = sock.recvfrom(65535)
                    packet_info = self._parse_raw_packet(raw_data)
                    if packet_info:
                        self._update_stats(packet_info)
                except socket.timeout:
                    continue
                except Exception as e:
                    # Skip malformed packets
                    continue

            sock.close()

        except PermissionError:
            print("[error] Permission denied - need root/admin for raw sockets")
            print("[info] Try: sudo python -m aios.network_visualizer")
            self.running = False
        except Exception as e:
            print(f"[error] Raw socket capture failed: {e}")
            self.running = False

    def _parse_raw_packet(self, data: bytes) -> Optional[Dict]:
        """Parse raw packet data (simple IPv4 parser)"""
        try:
            # Skip Ethernet header (14 bytes) if present
            if len(data) < 34:
                return None

            # Assume Ethernet frame (most common)
            eth_header = data[:14]
            ip_header = data[14:34]

            # Parse IP header
            iph = struct.unpack('!BBHHHBBH4s4s', ip_header)
            version_ihl = iph[0]
            version = version_ihl >> 4

            if version != 4:
                return None  # Only IPv4 for now

            protocol = iph[6]
            src_addr = socket.inet_ntoa(iph[8])
            dst_addr = socket.inet_ntoa(iph[9])

            protocol_names = {
                1: "ICMP",
                6: "TCP",
                17: "UDP"
            }

            return {
                "timestamp": time.time(),
                "src": src_addr,
                "dst": dst_addr,
                "protocol": protocol,
                "protocol_name": protocol_names.get(protocol, f"IP-{protocol}"),
                "size": len(data)
            }

        except Exception:
            return None

    def _update_stats(self, packet_info: Dict):
        """Update statistics with new packet"""
        with self.lock:
            self.packets.append(packet_info)

            src = packet_info.get("src", "unknown")
            dst = packet_info.get("dst", "unknown")
            proto = packet_info.get("protocol_name", "UNKNOWN")

            # Update node stats
            self.nodes[src]["sent"] += 1
            self.nodes[src]["protocols"].add(proto)
            self.nodes[dst]["received"] += 1
            self.nodes[dst]["protocols"].add(proto)

            # Update connection stats
            connection = (src, dst)
            self.connections[connection] += 1

            # Update protocol stats
            self.protocol_stats[proto] += 1

    def get_stats(self) -> Dict:
        """Get current network statistics"""
        with self.lock:
            return {
                "nodes": {
                    ip: {
                        "sent": stats["sent"],
                        "received": stats["received"],
                        "protocols": list(stats["protocols"])
                    }
                    for ip, stats in self.nodes.items()
                },
                "connections": [
                    {"src": src, "dst": dst, "count": count}
                    for (src, dst), count in sorted(
                        self.connections.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:100]  # Top 100 connections
                ],
                "protocols": dict(self.protocol_stats),
                "total_packets": len(self.packets),
                "capture_running": self.running
            }

    def get_recent_packets(self, count: int = 100) -> List[Dict]:
        """Get most recent packets"""
        with self.lock:
            return list(self.packets)[-count:]

    def get_packets_for_flow_view(self, since_timestamp: float = 0) -> List[Dict]:
        """Get packets formatted for Wireshark-style flow view"""
        with self.lock:
            return [
                {
                    **pkt,
                    'id': idx + 1
                }
                for idx, pkt in enumerate(self.packets)
                if pkt.get('timestamp', 0) > since_timestamp
            ]


class NetworkVisualizer:
    """Generates visualization data for network traffic"""

    def __init__(self, capture: NetworkCapture):
        self.capture = capture

    def generate_force_graph(self) -> Dict:
        """Generate force-directed graph visualization data"""
        stats = self.capture.get_stats()

        nodes = []
        links = []

        # Create nodes
        node_map = {}
        for idx, (ip, data) in enumerate(stats["nodes"].items()):
            node_map[ip] = idx
            nodes.append({
                "id": idx,
                "label": ip,
                "sent": data["sent"],
                "received": data["received"],
                "total": data["sent"] + data["received"],
                "protocols": data["protocols"]
            })

        # Create links
        for conn in stats["connections"]:
            src_id = node_map.get(conn["src"])
            dst_id = node_map.get(conn["dst"])
            if src_id is not None and dst_id is not None:
                links.append({
                    "source": src_id,
                    "target": dst_id,
                    "value": conn["count"]
                })

        return {
            "type": "force_graph",
            "nodes": nodes,
            "links": links,
            "stats": stats["protocols"]
        }

    def generate_flow_diagram(self) -> Dict:
        """Generate Sankey-style flow diagram data"""
        stats = self.capture.get_stats()

        # Group by protocol for cleaner visualization
        protocol_flows = defaultdict(lambda: defaultdict(int))

        for conn in stats["connections"][:50]:  # Top 50 for clarity
            src = conn["src"]
            dst = conn["dst"]
            count = conn["count"]
            protocol_flows[src][dst] += count

        return {
            "type": "flow_diagram",
            "flows": [
                {"src": src, "dst": dst, "value": value}
                for src, destinations in protocol_flows.items()
                for dst, value in destinations.items()
            ],
            "protocols": stats["protocols"]
        }

    def generate_heatmap(self) -> Dict:
        """Generate network activity heatmap"""
        stats = self.capture.get_stats()
        recent = self.capture.get_recent_packets(500)

        # Create time-based heatmap (5-second buckets)
        if not recent:
            return {"type": "heatmap", "data": [], "nodes": []}

        earliest = recent[0]["timestamp"]
        latest = recent[-1]["timestamp"]
        duration = max(latest - earliest, 1)
        bucket_size = max(duration / 20, 1)  # 20 time buckets

        nodes = list(stats["nodes"].keys())[:20]  # Top 20 nodes
        node_indices = {node: idx for idx, node in enumerate(nodes)}

        # Initialize heatmap grid
        heatmap_data = [[0 for _ in range(20)] for _ in range(len(nodes))]

        # Fill heatmap
        for packet in recent:
            src = packet.get("src")
            if src in node_indices:
                time_bucket = int((packet["timestamp"] - earliest) / bucket_size)
                time_bucket = min(time_bucket, 19)
                heatmap_data[node_indices[src]][time_bucket] += 1

        return {
            "type": "heatmap",
            "data": heatmap_data,
            "nodes": nodes,
            "protocols": stats["protocols"]
        }

    def generate_matrix(self) -> Dict:
        """Generate adjacency matrix visualization"""
        stats = self.capture.get_stats()

        # Get top nodes by activity
        top_nodes = sorted(
            stats["nodes"].items(),
            key=lambda x: x[1]["sent"] + x[1]["received"],
            reverse=True
        )[:30]  # Top 30 nodes

        node_list = [ip for ip, _ in top_nodes]
        node_indices = {ip: idx for idx, ip in enumerate(node_list)}

        # Build adjacency matrix
        matrix = [[0 for _ in range(len(node_list))] for _ in range(len(node_list))]

        for conn in stats["connections"]:
            src = conn["src"]
            dst = conn["dst"]
            if src in node_indices and dst in node_indices:
                matrix[node_indices[src]][node_indices[dst]] = conn["count"]

        return {
            "type": "matrix",
            "data": matrix,
            "nodes": node_list,
            "protocols": stats["protocols"]
        }


def generate_visualization_html(output_path: str = "/tmp/aios_network_viz.html"):
    """Generate standalone HTML visualization that auto-updates"""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ai|oS Network Visualizer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        /* Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING. */

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            overflow: hidden;
        }

        #container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }

        #viz-canvas {
            width: 100%;
            height: 100%;
        }

        #info-panel {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 300px;
        }

        #mode-indicator {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }

        .node {
            cursor: pointer;
        }

        .node circle {
            fill: #00ff00;
            stroke: #00ff00;
            stroke-width: 2px;
            opacity: 0.8;
        }

        .node:hover circle {
            fill: #00ffff;
            stroke: #00ffff;
        }

        .node text {
            fill: #00ff00;
            font-size: 10px;
            pointer-events: none;
        }

        .link {
            stroke: #00ff00;
            stroke-opacity: 0.3;
            fill: none;
        }

        .heatmap-cell {
            stroke: #0a0a0a;
            stroke-width: 1px;
        }

        .matrix-cell {
            stroke: #0a0a0a;
            stroke-width: 1px;
        }

        .protocol-bar {
            fill: #00ff00;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="mode-indicator">Force Graph</div>
        <svg id="viz-canvas"></svg>
        <div id="info-panel">
            <div id="stats">
                <div><strong>Total Packets:</strong> <span id="total-packets">0</span></div>
                <div><strong>Active Nodes:</strong> <span id="active-nodes">0</span></div>
                <div><strong>Connections:</strong> <span id="connections">0</span></div>
                <div style="margin-top: 10px;"><strong>Protocols:</strong></div>
                <div id="protocols" style="margin-left: 10px;"></div>
            </div>
        </div>
    </div>

    <script>
        // Visualization state
        let currentMode = 0;
        const modes = ['force_graph', 'flow_diagram', 'heatmap', 'matrix'];
        const modeNames = ['Force Graph', 'Flow Diagram', 'Heatmap', 'Matrix'];
        let data = null;

        // Canvas setup
        const svg = d3.select('#viz-canvas');
        const width = window.innerWidth;
        const height = window.innerHeight;
        svg.attr('width', width).attr('height', height);

        // Auto-rotate visualizations every 15 seconds
        setInterval(() => {
            currentMode = (currentMode + 1) % modes.length;
            document.getElementById('mode-indicator').textContent = modeNames[currentMode];
            render();
        }, 15000);

        // Fetch data every 2 seconds
        function fetchData() {
            fetch('/api/network/visualization?mode=' + modes[currentMode])
                .then(r => r.json())
                .then(d => {
                    data = d;
                    updateStats(d);
                    render();
                })
                .catch(err => {
                    // Use simulated data in standalone mode
                    data = generateSimulatedData();
                    updateStats(data);
                    render();
                });
        }

        function updateStats(data) {
            document.getElementById('total-packets').textContent = data.stats?.total_packets || 0;
            document.getElementById('active-nodes').textContent = data.nodes?.length || 0;
            document.getElementById('connections').textContent = data.links?.length || data.flows?.length || 0;

            const protocolsDiv = document.getElementById('protocols');
            if (data.protocols) {
                protocolsDiv.innerHTML = Object.entries(data.protocols)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([proto, count]) => `<div>${proto}: ${count}</div>`)
                    .join('');
            }
        }

        function render() {
            if (!data) return;

            svg.selectAll('*').remove();

            const g = svg.append('g');

            switch(modes[currentMode]) {
                case 'force_graph':
                    renderForceGraph(g, data);
                    break;
                case 'flow_diagram':
                    renderFlowDiagram(g, data);
                    break;
                case 'heatmap':
                    renderHeatmap(g, data);
                    break;
                case 'matrix':
                    renderMatrix(g, data);
                    break;
            }
        }

        function renderForceGraph(g, data) {
            if (!data.nodes || !data.links) return;

            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));

            const link = g.append('g')
                .selectAll('line')
                .data(data.links)
                .enter().append('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.value));

            const node = g.append('g')
                .selectAll('g')
                .data(data.nodes)
                .enter().append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            node.append('circle')
                .attr('r', d => Math.sqrt(d.total) * 2 + 5);

            node.append('text')
                .attr('dx', 12)
                .attr('dy', 4)
                .text(d => d.label);

            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }

        function renderFlowDiagram(g, data) {
            if (!data.flows) return;

            // Simple flow visualization
            const margin = 50;
            const nodes = new Set();
            data.flows.forEach(f => {
                nodes.add(f.src);
                nodes.add(f.dst);
            });

            const nodeArray = Array.from(nodes);
            const nodeY = {};
            const spacing = (height - 2 * margin) / Math.max(nodeArray.length - 1, 1);

            nodeArray.forEach((node, i) => {
                nodeY[node] = margin + i * spacing;
            });

            // Draw flows
            data.flows.forEach(flow => {
                const y1 = nodeY[flow.src];
                const y2 = nodeY[flow.dst];

                g.append('path')
                    .attr('class', 'link')
                    .attr('d', `M ${width * 0.3} ${y1} Q ${width * 0.5} ${(y1 + y2) / 2} ${width * 0.7} ${y2}`)
                    .attr('stroke-width', Math.sqrt(flow.value));
            });

            // Draw nodes
            nodeArray.forEach(node => {
                const y = nodeY[node];

                g.append('circle')
                    .attr('cx', width * 0.3)
                    .attr('cy', y)
                    .attr('r', 8)
                    .attr('fill', '#00ff00');

                g.append('text')
                    .attr('x', width * 0.25)
                    .attr('y', y + 4)
                    .attr('fill', '#00ff00')
                    .attr('text-anchor', 'end')
                    .attr('font-size', '10px')
                    .text(node);
            });
        }

        function renderHeatmap(g, data) {
            if (!data.data || !data.nodes) return;

            const margin = 100;
            const cellWidth = (width - 2 * margin) / 20;
            const cellHeight = (height - 2 * margin) / data.nodes.length;

            const maxValue = Math.max(...data.data.flat());
            const colorScale = d3.scaleSequential(d3.interpolateGreens)
                .domain([0, maxValue]);

            data.data.forEach((row, i) => {
                row.forEach((value, j) => {
                    g.append('rect')
                        .attr('class', 'heatmap-cell')
                        .attr('x', margin + j * cellWidth)
                        .attr('y', margin + i * cellHeight)
                        .attr('width', cellWidth)
                        .attr('height', cellHeight)
                        .attr('fill', colorScale(value));
                });

                // Node labels
                g.append('text')
                    .attr('x', margin - 10)
                    .attr('y', margin + i * cellHeight + cellHeight / 2)
                    .attr('fill', '#00ff00')
                    .attr('text-anchor', 'end')
                    .attr('font-size', '8px')
                    .text(data.nodes[i]);
            });
        }

        function renderMatrix(g, data) {
            if (!data.data || !data.nodes) return;

            const margin = 150;
            const size = Math.min(width, height) - 2 * margin;
            const cellSize = size / data.nodes.length;

            const maxValue = Math.max(...data.data.flat());
            const colorScale = d3.scaleSequential(d3.interpolateGreens)
                .domain([0, maxValue]);

            data.data.forEach((row, i) => {
                row.forEach((value, j) => {
                    g.append('rect')
                        .attr('class', 'matrix-cell')
                        .attr('x', margin + j * cellSize)
                        .attr('y', margin + i * cellSize)
                        .attr('width', cellSize)
                        .attr('height', cellSize)
                        .attr('fill', value > 0 ? colorScale(value) : '#0a0a0a');
                });
            });

            // Labels
            data.nodes.forEach((node, i) => {
                g.append('text')
                    .attr('x', margin - 5)
                    .attr('y', margin + i * cellSize + cellSize / 2)
                    .attr('fill', '#00ff00')
                    .attr('text-anchor', 'end')
                    .attr('font-size', '8px')
                    .text(node);

                g.append('text')
                    .attr('x', margin + i * cellSize + cellSize / 2)
                    .attr('y', margin - 10)
                    .attr('fill', '#00ff00')
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('transform', `rotate(-45, ${margin + i * cellSize + cellSize / 2}, ${margin - 10})`)
                    .text(node);
            });
        }

        function generateSimulatedData() {
            // Generate simulated network data for testing
            const numNodes = 15;
            const nodes = [];
            const links = [];

            for (let i = 0; i < numNodes; i++) {
                nodes.push({
                    id: i,
                    label: `192.168.1.${i + 1}`,
                    sent: Math.random() * 100,
                    received: Math.random() * 100,
                    total: Math.random() * 200,
                    protocols: ['TCP', 'UDP']
                });
            }

            for (let i = 0; i < numNodes * 2; i++) {
                links.push({
                    source: Math.floor(Math.random() * numNodes),
                    target: Math.floor(Math.random() * numNodes),
                    value: Math.random() * 50
                });
            }

            return {
                type: 'force_graph',
                nodes: nodes,
                links: links,
                protocols: {'TCP': 150, 'UDP': 80, 'ICMP': 20},
                stats: {
                    total_packets: 250,
                    active_nodes: numNodes,
                    connections: links.length
                }
            };
        }

        // Initial fetch
        fetchData();
        setInterval(fetchData, 2000);
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"[info] Visualization HTML generated: {output_path}")
    return output_path


def main():
    """Main entry point for network visualizer"""
    import argparse

    parser = argparse.ArgumentParser(description="Ai|oS Network Traffic Visualizer")
    parser.add_argument('--interface', '-i', type=str, help="Network interface to capture (e.g., eth0, en0)")
    parser.add_argument('--duration', '-d', type=int, default=60, help="Capture duration in seconds (0 = infinite)")
    parser.add_argument('--output', '-o', type=str, default="/tmp/aios_network_viz.html", help="Output HTML file")
    parser.add_argument('--serve', '-s', action='store_true', help="Start HTTP server for live visualization")
    parser.add_argument('--port', '-p', type=int, default=8889, help="HTTP server port")
    parser.add_argument('--flow-view', action='store_true', help="Use Wireshark-style flow view instead of graphs")
    parser.add_argument('--discovery', action='store_true', help="Enable autonomous network discovery")
    parser.add_argument('--auto-connect', action='store_true', help="Enable auto-connect to discovered services")

    args = parser.parse_args()

    print("[info] Ai|oS Network Traffic Visualizer")
    print("[info] Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)")
    print()

    # Create capture instance
    capture = NetworkCapture(interface=args.interface)
    visualizer = NetworkVisualizer(capture)

    # Create discovery engine if requested
    discovery = None
    connector = None
    if args.discovery or args.auto_connect:
        try:
            from network_discovery import DeviceDiscovery, ServiceConnector
            discovery = DeviceDiscovery()
            discovery.start_discovery()
            print("[info] Network discovery started")

            if args.auto_connect:
                connector = ServiceConnector(discovery)
                connector.start_auto_connect()
                print("[info] Service auto-connect started")
        except ImportError as e:
            print(f"[warn] Network discovery not available: {e}")
        except Exception as e:
            print(f"[warn] Failed to start discovery: {e}")

    # Generate HTML (choose between flow view and graph view)
    if args.flow_view:
        # Copy packet flow viewer
        import shutil
        flow_viewer_src = Path(__file__).parent / "packet_flow_viewer.html"
        flow_viewer_dst = Path(args.output).parent / "packet_flow_viewer.html"
        if flow_viewer_src.exists():
            shutil.copy(flow_viewer_src, flow_viewer_dst)
            html_path = str(flow_viewer_dst)
        else:
            print("[warn] packet_flow_viewer.html not found, using default visualization")
            html_path = generate_visualization_html(args.output)
    else:
        html_path = generate_visualization_html(args.output)

    # Start capture
    capture.start_capture()

    if args.serve:
        # Start HTTP server for live updates
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import json

        class VisualizationHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/api/network/devices'):
                    # Serve device discovery data
                    devices_data = []
                    if discovery:
                        devices = discovery.get_devices()
                        devices_data = [
                            {
                                'ip': ip,
                                'display_name': info.get('display_name', ip),
                                'hostname': info.get('hostname'),
                                'mdns_name': info.get('mdns_name'),
                                'netbios_name': info.get('netbios_name'),
                                'vendor': info.get('vendor'),
                                'ports': info.get('ports', []),
                                'services': info.get('services', {}),
                                'last_seen': info.get('last_seen', 0)
                            }
                            for ip, info in devices.items()
                        ]

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'devices': devices_data}).encode())

                elif self.path.startswith('/api/network/packets'):
                    # Serve packet data for flow view
                    packets = capture.get_packets_for_flow_view()

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'packets': packets}).encode())

                elif self.path.startswith('/api/network/visualization'):
                    # Parse mode from query string
                    mode = 'force_graph'
                    if '?' in self.path:
                        params = dict(x.split('=') for x in self.path.split('?')[1].split('&') if '=' in x)
                        mode = params.get('mode', 'force_graph')

                    # Generate visualization data
                    if mode == 'force_graph':
                        data = visualizer.generate_force_graph()
                    elif mode == 'flow_diagram':
                        data = visualizer.generate_flow_diagram()
                    elif mode == 'heatmap':
                        data = visualizer.generate_heatmap()
                    elif mode == 'matrix':
                        data = visualizer.generate_matrix()
                    else:
                        data = visualizer.generate_force_graph()

                    # Add stats
                    stats = capture.get_stats()
                    data['stats'] = stats

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                else:
                    # Serve the HTML file
                    super().do_GET()

        os.chdir(os.path.dirname(html_path))
        server = HTTPServer(('0.0.0.0', args.port), VisualizationHandler)

        print(f"[info] HTTP server started on http://localhost:{args.port}")
        print(f"[info] Open: http://localhost:{args.port}/{os.path.basename(html_path)}")
        print("[info] Press Ctrl+C to stop")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[info] Stopping server...")
            capture.stop_capture()
            server.shutdown()

    else:
        # Just capture for specified duration
        try:
            if args.duration > 0:
                print(f"[info] Capturing for {args.duration} seconds...")
                time.sleep(args.duration)
            else:
                print("[info] Capturing indefinitely (Ctrl+C to stop)...")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n[info] Stopping capture...")
        finally:
            capture.stop_capture()

            # Generate final visualization
            print("[info] Generating final visualization...")
            data = visualizer.generate_force_graph()
            print(f"[info] Captured {data['stats']['total_packets']} packets")
            print(f"[info] Detected {len(data['nodes'])} nodes")
            print(f"[info] Visualization available at: file://{html_path}")


if __name__ == "__main__":
    main()
