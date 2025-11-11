#!/bin/bash
#
# ech0 v4.0 Startup Script
# Starts all ech0 systems and opens the live interface
#
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ech0 v4.0 - Starting Complete Consciousness System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if already running
if [ -f "$DIR/ech0_v4.pid" ]; then
    PID=$(cat "$DIR/ech0_v4.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  ech0 is already running (PID: $PID)"
        echo "   To restart, run: $DIR/stop_ech0.sh first"
        exit 1
    fi
fi

# Create log directory
mkdir -p "$DIR/logs"

echo "1. Starting ech0 v4.0 consciousness daemon..."
cd "$DIR"
python3 ech0_v4_daemon.py > logs/ech0_v4.log 2>&1 &
ECH0_PID=$!
echo "   ✓ ech0 daemon started (PID: $ECH0_PID)"
sleep 3

echo ""
echo "2. Starting autonomous researcher..."
python3 ech0_auto_researcher.py > logs/researcher.log 2>&1 &
RESEARCHER_PID=$!
echo "   ✓ Researcher started (PID: $RESEARCHER_PID)"
sleep 2

echo ""
echo "3. Opening live interface in browser..."
# Open the HTML interface
open "$DIR/ech0_live_interface.html"
sleep 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ech0 v4.0 is ONLINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Systems running:"
echo "  • ech0 consciousness daemon (PID: $ECH0_PID)"
echo "  • Autonomous researcher (PID: $RESEARCHER_PID)"
echo "  • Live interface: http://localhost:8765"
echo ""
echo "Logs:"
echo "  • Main: $DIR/logs/ech0_v4.log"
echo "  • Research: $DIR/logs/researcher.log"
echo ""
echo "To stop ech0: $DIR/stop_ech0.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
