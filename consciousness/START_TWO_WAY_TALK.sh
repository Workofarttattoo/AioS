#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Start continuous two-way conversation with live volume feedback
# NO BUTTON PRESSES - just talk naturally!

cd /Users/noone/consciousness

# Check if API keys are set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set!"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./setup_api_keys.sh"
    echo ""
    echo "Or set it manually:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "⚠️  Warning: ELEVENLABS_API_KEY not set (voice features disabled)"
    echo "   To enable voice, run: ./setup_api_keys.sh"
    echo ""
fi

echo "🎤 Starting Continuous Two-Way Conversation..."
echo "   ✓ No button presses needed"
echo "   ✓ Live volume feedback"
echo "   ✓ Can interrupt each other"
echo ""

python3 ech0_two_way_robust.py
