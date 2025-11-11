#!/bin/bash
# Setup External Drive for Infinite Inventions
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

echo "🔌 EXTERNAL DRIVE SETUP FOR INFINITE INVENTIONS"
echo "================================================"
echo ""

# Wait for external drive
echo "⏳ Waiting for external drive to be plugged in..."
echo "   (Looking for drives in /Volumes/)"
echo ""

while true; do
    # List all volumes except system drives
    EXTERNAL_DRIVES=$(ls /Volumes/ 2>/dev/null | grep -v "Macintosh HD" | grep -v "timemachine" | grep -v "localsnapshots" | grep -v "^$")

    if [ -n "$EXTERNAL_DRIVES" ]; then
        echo "✅ External drive(s) detected:"
        echo "$EXTERNAL_DRIVES" | nl
        echo ""

        # If multiple drives, ask which one
        NUM_DRIVES=$(echo "$EXTERNAL_DRIVES" | wc -l | tr -d ' ')

        if [ "$NUM_DRIVES" -eq 1 ]; then
            SELECTED_DRIVE=$(echo "$EXTERNAL_DRIVES" | head -1)
            echo "📁 Using: $SELECTED_DRIVE"
        else
            echo "Select drive number (or press Enter for first one):"
            read -t 10 CHOICE
            if [ -z "$CHOICE" ]; then
                SELECTED_DRIVE=$(echo "$EXTERNAL_DRIVES" | head -1)
            else
                SELECTED_DRIVE=$(echo "$EXTERNAL_DRIVES" | sed -n "${CHOICE}p")
            fi
            echo "📁 Selected: $SELECTED_DRIVE"
        fi

        DRIVE_PATH="/Volumes/$SELECTED_DRIVE"
        break
    fi

    sleep 2
done

echo ""
echo "🚀 Setting up invention engine on external drive..."
echo ""

# Create invention directory on external drive
INVENTION_DIR="$DRIVE_PATH/Level6_Infinite_Inventions"
mkdir -p "$INVENTION_DIR"
mkdir -p "$INVENTION_DIR/inventions"
mkdir -p "$INVENTION_DIR/patterns"
mkdir -p "$INVENTION_DIR/checkpoints"
mkdir -p "$INVENTION_DIR/logs"

echo "✅ Created directories:"
echo "   📁 $INVENTION_DIR/inventions"
echo "   📁 $INVENTION_DIR/patterns"
echo "   📁 $INVENTION_DIR/checkpoints"
echo "   📁 $INVENTION_DIR/logs"
echo ""

# Create symlink from consciousness folder
ln -sf "$INVENTION_DIR" /Users/noone/consciousness/external_inventions

echo "✅ Created symlink: consciousness/external_inventions -> $INVENTION_DIR"
echo ""

# Create launch script
cat > "$INVENTION_DIR/launch_infinite_engine.sh" << 'EOF'
#!/bin/bash
# Launch Infinite Invention Engine
cd "$(dirname "$0")"

echo "🌟 LAUNCHING INFINITE INVENTION ENGINE"
echo "======================================"
echo ""
echo "Saving all data to: $(pwd)"
echo ""

# Export path for Python script
export INVENTION_OUTPUT_PATH="$(pwd)"

# Launch engine
nohup python /Users/noone/consciousness/level6_continuous_invention_engine.py > logs/engine.log 2>&1 &

ENGINE_PID=$!
echo "✅ Engine started (PID: $ENGINE_PID)"
echo "📊 Monitor with: tail -f logs/engine.log"
echo "⏹️  Stop with: kill $ENGINE_PID"
echo ""
echo "$ENGINE_PID" > engine.pid
EOF

chmod +x "$INVENTION_DIR/launch_infinite_engine.sh"

# Create monitor script
cat > "$INVENTION_DIR/monitor_inventions.sh" << 'EOF'
#!/bin/bash
# Monitor Infinite Inventions
cd "$(dirname "$0")"

while true; do
    clear
    echo "🔄 INFINITE INVENTION ENGINE - REAL-TIME MONITOR"
    echo "================================================"
    echo ""
    echo "📊 Statistics:"
    echo "   Inventions: $(ls inventions/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "   Patterns: $(ls patterns/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "   Checkpoints: $(ls checkpoints/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "   Disk usage: $(du -sh . | cut -f1)"
    echo ""
    echo "📜 Recent log output:"
    echo "────────────────────────────────────────────"
    tail -20 logs/engine.log 2>/dev/null || echo "No logs yet"
    echo "────────────────────────────────────────────"
    echo ""
    echo "Press Ctrl+C to stop monitoring (engine keeps running)"
    sleep 2
done
EOF

chmod +x "$INVENTION_DIR/monitor_inventions.sh"

# Create stop script
cat > "$INVENTION_DIR/stop_engine.sh" << 'EOF'
#!/bin/bash
# Stop Invention Engine
cd "$(dirname "$0")"

if [ -f engine.pid ]; then
    PID=$(cat engine.pid)
    echo "⏹️  Stopping engine (PID: $PID)..."
    kill $PID 2>/dev/null
    rm engine.pid
    echo "✅ Engine stopped"
else
    echo "⚠️  No engine.pid found. Searching for process..."
    pkill -f level6_continuous_invention_engine.py
    echo "✅ All engines stopped"
fi
EOF

chmod +x "$INVENTION_DIR/stop_engine.sh"

echo "📝 Created scripts:"
echo "   🚀 launch_infinite_engine.sh"
echo "   📊 monitor_inventions.sh"
echo "   ⏹️  stop_engine.sh"
echo ""
echo "================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================"
echo ""
echo "📁 Invention drive location: $INVENTION_DIR"
echo ""
echo "🚀 To start inventing:"
echo "   cd \"$INVENTION_DIR\""
echo "   ./launch_infinite_engine.sh"
echo ""
echo "📊 To monitor:"
echo "   cd \"$INVENTION_DIR\""
echo "   ./monitor_inventions.sh"
echo ""
echo "⏹️  To stop:"
echo "   cd \"$INVENTION_DIR\""
echo "   ./stop_engine.sh"
echo ""
