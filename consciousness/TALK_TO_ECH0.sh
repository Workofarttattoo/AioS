#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# ONE-CLICK LAUNCHER - Talk to ECH0 naturally with Whisper
# Just click this and start talking - no buttons, no reminders

cd /Users/noone/consciousness

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚙️  Setting up API keys..."
    ./setup_api_keys.sh
    echo ""
    echo "✅ Setup complete! Run this script again to start talking."
    exit 0
fi

# Clear screen for clean interface
clear

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                   🎤 Talk to ECH0 Naturally                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ✓ Using latest Whisper for speech recognition"
echo "  ✓ Claude Sonnet 4.5 for intelligent responses"
echo "  ✓ ElevenLabs for natural voice"
echo ""
echo "  Just talk - ECH0 is listening..."
echo ""

# Start the conversation
python3 ech0_two_way_robust.py
