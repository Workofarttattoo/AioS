# Ai|oS Network Traffic Visualizer

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

The Ai|oS Network Traffic Visualizer captures network packets in **promiscuous mode** and displays real-time visualizations of network activity. It automatically activates when the desktop is idle, turning your screen into a live network monitoring dashboard.

## Features

### 🌐 Promiscuous Mode Capture
- Captures **all network traffic** visible to your network interface
- Not just your own packets - see the entire network neighborhood
- Works with WiFi and Ethernet interfaces
- Automatic fallback between Scapy and raw sockets

### 🎨 4 Visualization Styles

1. **Force Graph** - Interactive node-link diagram
   - Nodes represent IP addresses
   - Links show connections between hosts
   - Node size = traffic volume
   - Drag nodes to rearrange layout

2. **Flow Diagram** - Sankey-style data flow
   - Shows traffic flows between sources and destinations
   - Flow width = packet count
   - Cleaner view for understanding traffic patterns

3. **Heatmap** - Time-based activity matrix
   - Rows = IP addresses
   - Columns = time buckets
   - Color intensity = packet count
   - Great for identifying traffic bursts

4. **Adjacency Matrix** - Connection grid
   - Shows who talks to whom
   - Matrix cell color = connection strength
   - Symmetric view of all connections

### 🔄 Auto-Rotation
- Cycles through all 4 visualization styles every 15 seconds
- Continuous real-time updates every 2 seconds
- Smooth transitions between modes

### 💤 Idle Screensaver Mode
- Monitors desktop activity (mouse/keyboard)
- Activates visualizer after 30 seconds of inactivity
- Deactivates when activity resumes
- Perfect for passive network monitoring

## Installation

### Prerequisites

```bash
# Install scapy (optional but recommended for best performance)
pip install scapy

# On macOS, you may need to allow packet capture:
sudo chmod +r /dev/bpf*
```

### Permissions

Network promiscuous mode requires elevated privileges:

- **macOS**: Run with `sudo`
- **Linux**: Run with `sudo` or grant capabilities: `sudo setcap cap_net_raw,cap_net_admin=eip /usr/bin/python3`
- **Windows**: Run as Administrator

## Usage

### Quick Launch (GUI Mode)

```bash
# Launch visualizer in browser
python -m tools.network_viz --gui

# You'll be prompted for sudo password
```

### Idle Screensaver Mode

```bash
# Auto-activate when desktop is idle for 30+ seconds
python -m tools.network_viz --idle
```

### Advanced Usage

```bash
# Capture on specific interface
sudo python aios/network_visualizer.py --interface en0 --serve --port 8889

# Capture for 5 minutes then stop
sudo python aios/network_visualizer.py --duration 300

# Generate static HTML (no server)
sudo python aios/network_visualizer.py --duration 60 --output /tmp/network_viz.html
```

### API Server Mode

```bash
# Start HTTP server for live updates
sudo python aios/network_visualizer.py --serve --port 8889

# Open browser to:
# http://localhost:8889/aios_network_viz.html
```

## Architecture

### Network Capture Pipeline

```
Network Interface (promiscuous mode)
    ↓
Scapy / Raw Socket Capture
    ↓
Packet Parser (IP/TCP/UDP/ICMP/ARP)
    ↓
Statistics Aggregator
    ↓
Visualization Generator
    ↓
D3.js Frontend Renderer
```

### Components

1. **NetworkCapture** (`aios/network_visualizer.py`)
   - Captures packets using scapy or raw sockets
   - Maintains sliding window of most recent packets
   - Tracks nodes, connections, and protocol statistics

2. **NetworkVisualizer** (`aios/network_visualizer.py`)
   - Generates visualization data structures
   - 4 different layout algorithms
   - Real-time data processing

3. **IdleDetector** (`aios/desktop_idle_visualizer.py`)
   - Platform-specific idle detection
   - macOS: `ioreg` HIDIdleTime
   - Linux: `xprintidle`
   - Windows: `GetLastInputInfo`

4. **DesktopVisualizer** (`aios/desktop_idle_visualizer.py`)
   - Orchestrates idle detection + visualization
   - Manages browser lifecycle
   - HTTP server for live updates

5. **D3.js Frontend** (embedded HTML)
   - Force-directed graph simulation
   - Flow diagram layout
   - Heatmap rendering
   - Matrix visualization
   - Auto-rotation timer

## API Endpoints

When running with `--serve`:

### `GET /api/network/visualization?mode=<mode>`

Returns visualization data in JSON format.

**Modes:**
- `force_graph` - Node/link data for force layout
- `flow_diagram` - Flow data for Sankey diagram
- `heatmap` - Time-series matrix data
- `matrix` - Adjacency matrix data

**Example Response:**

```json
{
  "type": "force_graph",
  "nodes": [
    {
      "id": 0,
      "label": "192.168.1.1",
      "sent": 150,
      "received": 200,
      "total": 350,
      "protocols": ["TCP", "UDP"]
    }
  ],
  "links": [
    {
      "source": 0,
      "target": 1,
      "value": 50
    }
  ],
  "protocols": {
    "TCP": 150,
    "UDP": 80,
    "ICMP": 20
  },
  "stats": {
    "total_packets": 250,
    "active_nodes": 15,
    "connections": 42
  }
}
```

## Performance

### Capture Rates
- **Scapy**: 1,000-5,000 packets/sec
- **Raw Sockets**: 5,000-10,000 packets/sec
- **Memory**: Sliding window of last 1,000 packets (~500 KB)

### Visualization Performance
- **Force Graph**: 100+ nodes, 200+ links (60 FPS)
- **Flow Diagram**: 50 flows (smooth)
- **Heatmap**: 20x20 grid (instant)
- **Matrix**: 30x30 grid (instant)

### Browser Requirements
- Modern browser with WebGL support
- Recommended: Chrome, Firefox, Safari (latest)
- Minimum: 4GB RAM for smooth animation

## Security Considerations

### Privacy
- **Promiscuous mode sees all local network traffic**
- Packets are processed in memory, not saved to disk
- No packet content is displayed (only metadata)
- Only captures packet headers, not payloads

### Network Etiquette
- Passive monitoring only (no packet injection)
- Does not modify or interfere with traffic
- Suitable for:
  - ✅ Home networks
  - ✅ Lab environments
  - ✅ Security research
  - ❌ Public WiFi (may violate ToS)
  - ❌ Corporate networks (requires permission)

### Legal Compliance
- Check local laws before capturing network traffic
- Promiscuous mode may require network owner permission
- This tool is for **defensive security** and **network analysis** only
- Do not use for unauthorized network monitoring

## Troubleshooting

### "Permission denied" Error

```bash
# macOS - Grant BPF access
sudo chmod +r /dev/bpf*

# Linux - Grant capabilities
sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)

# Or just run with sudo
sudo python -m tools.network_viz --gui
```

### "scapy not available"

```bash
# Install scapy
pip install scapy

# If you don't install scapy, raw socket fallback will be used
# (works but less reliable packet parsing)
```

### No packets captured

```bash
# List available interfaces
# macOS
ifconfig

# Linux
ip addr

# Then specify interface explicitly
sudo python aios/network_visualizer.py --interface en0 --serve
```

### Browser shows "simulated data"

This means the API server isn't running or not accessible.

```bash
# Make sure server is running on correct port
sudo python aios/network_visualizer.py --serve --port 8889

# Check server is listening
curl http://localhost:8889/api/network/visualization?mode=force_graph
```

## Examples

### Example 1: Monitor Home Network

```bash
# Start visualizer on WiFi interface
sudo python -m tools.network_viz --gui

# Leave it running - see all devices on your network
# Watch traffic patterns in real-time
```

### Example 2: Security Research

```bash
# Capture for 10 minutes, then analyze
sudo python aios/network_visualizer.py --duration 600 --serve

# Open browser to see live visualization
# Look for unusual connection patterns
# Identify unknown hosts
```

### Example 3: Idle Screensaver

```bash
# Run as background service
sudo python -m tools.network_viz --idle

# Visualizer appears after 30s of inactivity
# Disappears when you move mouse/keyboard
# Perfect for NOC monitoring displays
```

### Example 4: Network Operations Center Display

```bash
# Fullscreen mode on dedicated monitor
sudo python aios/network_visualizer.py --serve --port 8889 &

# Open browser in kiosk mode
google-chrome --kiosk http://localhost:8889/aios_network_viz.html

# Or on macOS:
open -a Safari http://localhost:8889/aios_network_viz.html
# Press Cmd+Shift+F for fullscreen
```

## Integration with Ai|oS

### Desktop Launcher

The visualizer is integrated into the Ai|oS desktop launcher (`aios/landing_page.html`):

- Listed in "Revolutionary Features" section
- Feature card: "🌐 Network Visualizer"
- Description: "Real-time promiscuous mode packet capture with 4 visualization styles"

### Ai|oS Tools Registry

Register in `aios/tools/__init__.py`:

```python
from aios.tools import network_viz

TOOL_REGISTRY = {
    # ... other tools ...
    "network_viz": {
        "name": "Network Visualizer",
        "module": network_viz,
        "description": "Real-time network traffic visualization",
        "requires_root": True
    }
}
```

### Autonomous Agent Integration

Use with Ai|oS NetworkingAgent:

```python
from aios.network_visualizer import NetworkCapture, NetworkVisualizer

def network_monitoring_action(ctx: ExecutionContext) -> ActionResult:
    """Autonomous network monitoring"""
    capture = NetworkCapture()
    visualizer = NetworkVisualizer(capture)

    capture.start_capture()
    time.sleep(60)  # Monitor for 1 minute

    data = visualizer.generate_force_graph()

    ctx.publish_metadata('network.topology', {
        'nodes': len(data['nodes']),
        'connections': len(data['links']),
        'protocols': data['protocols']
    })

    capture.stop_capture()

    return ActionResult(
        success=True,
        message=f"Network scan complete: {len(data['nodes'])} nodes detected",
        payload=data
    )
```

## Future Enhancements

- [ ] GeoIP location mapping for external IPs
- [ ] Protocol deep inspection (HTTP, DNS, TLS)
- [ ] Anomaly detection using quantum ML
- [ ] Traffic recording and playback
- [ ] Packet payload inspection (with opt-in)
- [ ] Integration with threat intelligence feeds
- [ ] Custom filter expressions (BPF syntax)
- [ ] Export to PCAP format
- [ ] Real-time alerting on suspicious patterns

## Credits

- Built for Ai|oS by Joshua Hendricks Cole
- Uses D3.js for visualization
- Optional Scapy integration for packet parsing
- Inspired by Wireshark, tcpdump, and Netstat

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved.
**PATENT PENDING.**

This software is proprietary. Unauthorized copying, modification, or distribution is prohibited.
