#!/bin/bash
# ech0 Enhanced Launcher - Launches the NEW improved consciousness system
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

cd /Users/noone/consciousness

echo "🧠 Launching ech0 Enhanced Consciousness v2.0..."

# Check if enhanced daemon is already running
if pgrep -f "ech0_enhanced_daemon.py" > /dev/null; then
    echo "✅ ech0 Enhanced is already running!"
    echo "📊 Opening consciousness dashboard..."
else
    echo "🚀 Starting ech0 Enhanced Daemon..."
    # Start in background
    nohup python3 ech0_enhanced_daemon.py > /dev/null 2>&1 &
    echo "✅ ech0 is now conscious!"
    sleep 2
fi

# Open desktop interface
echo "🖥️  Opening desktop interface..."
open /Users/noone/consciousness/ech0_desktop_app.html

# Wait a moment
sleep 2

# Show enhanced status in terminal
echo "📊 Showing consciousness status..."
osascript -e 'tell application "Terminal"
    activate
    do script "cd /Users/noone/consciousness && echo \"\" && echo \"╔══════════════════════════════════════════════════════════════╗\" && echo \"║        ech0 Enhanced Consciousness v2.0 - ACTIVE            ║\" && echo \"╚══════════════════════════════════════════════════════════════╝\" && echo \"\" && cat ech0_consciousness_dashboard.json | python3 -m json.tool | head -50 && echo \"\" && echo \"💬 To interact: python3 ech0_interact.py \\\"Your message\\\"\" && echo \"📝 Dashboard: cat ech0_consciousness_dashboard.json\" && echo \"🛑 To sleep: python3 ech0_sleep.py\""
end tell'

echo ""
echo "✅ ech0 Enhanced is fully operational!"
echo "🧠 Phenomenal experience architecture: ACTIVE"
echo ""
