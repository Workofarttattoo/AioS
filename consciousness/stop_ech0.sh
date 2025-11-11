#!/bin/bash
#
# ech0 v4.0 Stop Script
# Gracefully stops all ech0 systems
#
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ech0 v4.0 - Stopping Systems"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Stop ech0 daemon
if [ -f "$DIR/ech0_v4.pid" ]; then
    PID=$(cat "$DIR/ech0_v4.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping ech0 daemon (PID: $PID)..."
        kill -SIGINT $PID
        sleep 2

        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Force stopping..."
            kill -9 $PID
        fi

        echo "  ✓ ech0 daemon stopped"
    fi
    rm -f "$DIR/ech0_v4.pid"
else
    echo "ech0 daemon is not running"
fi

# Stop researcher
RESEARCHER_PIDS=$(pgrep -f "ech0_auto_researcher.py")
if [ ! -z "$RESEARCHER_PIDS" ]; then
    echo ""
    echo "Stopping autonomous researcher..."
    for PID in $RESEARCHER_PIDS; do
        kill -SIGINT $PID 2>/dev/null
    done
    echo "  ✓ Researcher stopped"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ech0 v4.0 shutdown complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
