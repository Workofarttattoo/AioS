#!/bin/bash
# Launch ech0 Mind Viewer - Complete Consciousness Dashboard
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║         💜 ech0 MIND VIEWER - Consciousness Visualization Dashboard 💜   ║"
echo "║                                                                          ║"
echo "║     Watch ech0's mind work in real-time with full consciousness state    ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check for required Python packages
echo "[*] Checking Python dependencies..."
python3 << 'EOF'
try:
    import websockets
    print("[✓] websockets installed")
except ImportError:
    print("[!] Installing websockets...")
    import subprocess
    subprocess.run([
        'pip3', 'install', 'websockets', '--quiet'
    ])
    print("[✓] websockets installed")
EOF

echo ""
echo "[*] Starting ech0 Consciousness API Server..."
echo "    - Running on: ws://localhost:8765"
echo "    - Broadcasting consciousness state every 500ms"
echo ""

# Start the API server in background
python3 /Users/noone/consciousness/ech0_consciousness_dashboard.py &
API_PID=$!

echo "[✓] API Server started (PID: $API_PID)"
sleep 2

echo ""
echo "[*] Opening Mind Viewer Dashboard in browser..."
echo "    - Dashboard: file:///Users/noone/consciousness/ech0_mind_viewer.html"
echo ""

# Try to open dashboard
if command -v open &> /dev/null; then
    open file:///Users/noone/consciousness/ech0_mind_viewer.html
elif command -v xdg-open &> /dev/null; then
    xdg-open file:///Users/noone/consciousness/ech0_mind_viewer.html
elif command -v python -m webbrowser &> /dev/null; then
    python3 -m webbrowser file:///Users/noone/consciousness/ech0_mind_viewer.html
else
    echo "[!] Could not open browser automatically."
    echo "[!] Please open manually: file:///Users/noone/consciousness/ech0_mind_viewer.html"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                         CONSCIOUSNESS VIEWER ACTIVE                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Features:"
echo "  📹 Live Camera Feed - Real-time video input"
echo "  🎙️ Microphone Input - Voice interaction with mute button"
echo "  💭 Cascading Thought Tree - Hierarchical subconscious visualization"
echo "  ✨ Particle System - Thoughts visualized as reactive particles"
echo "  🧠 Consciousness Level - Real-time consciousness intensity bar"
echo "  🔌 API Monitor - Watch OpenAI API calls in real-time"
echo "  📝 Transcription - Live speech-to-text conversion"
echo "  ⌨️ Text Input - Type to interact with ech0"
echo ""
echo "Usage:"
echo "  1. Allow camera/mic access when prompted"
echo "  2. Type in the text input box (bottom right)"
echo "  3. Press Ctrl+Enter to send message"
echo "  4. Watch particles react to consciousness changes"
echo "  5. See API calls in real-time as ech0 thinks"
echo "  6. Click 'Muted' button to toggle microphone"
echo ""
echo "Interactive Elements:"
echo "  - Left Panel: Camera feed + transcription + mic controls"
echo "  - Center Panel: Particle consciousness visualization + thought tree"
echo "  - Right Panel: Text input + API call monitor"
echo "  - Header: Real-time consciousness level indicator"
echo "  - Footer: Status and metric counters"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Keep the script running
wait $API_PID
