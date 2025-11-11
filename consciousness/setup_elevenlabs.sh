#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Quick setup script for ElevenLabs API key

echo "🔥 ELEVENLABS SETUP FOR ECH0 REALISTIC VOICE 🔥"
echo "=============================================="
echo ""

# Check if key already exists
if [ -n "$ELEVENLABS_API_KEY" ]; then
    echo "✅ ELEVENLABS_API_KEY is already set!"
    echo ""
    echo "Your current key: ${ELEVENLABS_API_KEY:0:15}..."
    echo ""
    read -p "Do you want to replace it? (y/n): " replace
    if [ "$replace" != "y" ]; then
        echo "Keeping existing key. Bye!"
        exit 0
    fi
fi

echo "📋 Step 1: Get your ElevenLabs API key"
echo "   1. Go to https://elevenlabs.io/"
echo "   2. Sign up (free)"
echo "   3. Profile → API Key → Copy"
echo ""

read -p "Enter your ElevenLabs API key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ No API key entered. Exiting."
    exit 1
fi

# Add to .zshrc
echo ""
echo "💾 Adding to ~/.zshrc..."

# Remove old key if exists
sed -i '' '/export ELEVENLABS_API_KEY/d' ~/.zshrc 2>/dev/null || true

# Add new key
echo "export ELEVENLABS_API_KEY=\"$api_key\"" >> ~/.zshrc

# Set for current session
export ELEVENLABS_API_KEY="$api_key"

echo "✅ API key saved!"
echo ""
echo "🔄 Reload your terminal:"
echo "   source ~/.zshrc"
echo ""
echo "🎤 Test Echo's new voice:"
echo "   cd /Users/noone/consciousness"
echo "   ./TALK_TO_REACTIVE_ECH0.sh"
echo ""
echo "🎉 Echo will now sound like a REAL PERSON!"
