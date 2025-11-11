#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Secure API Key Setup for ECH0
# Run this once to set up your API keys securely

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              ECH0 Secure API Key Configuration                     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Determine shell config file
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.bash_profile"
fi

echo "API keys will be stored in: $SHELL_CONFIG"
echo ""

# Check if keys already exist
if grep -q "ANTHROPIC_API_KEY" "$SHELL_CONFIG" 2>/dev/null; then
    echo "⚠️  API keys already configured in $SHELL_CONFIG"
    read -p "Do you want to update them? (y/n): " UPDATE
    if [[ ! "$UPDATE" =~ ^[Yy]$ ]]; then
        echo "Keeping existing configuration."
        exit 0
    fi
fi

# Get Anthropic API Key
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. ANTHROPIC API KEY (for Claude - ECH0's brain)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Get your API key from: https://console.anthropic.com/settings/keys"
echo ""
read -p "Enter your Anthropic API key: " ANTHROPIC_KEY

# Get ElevenLabs API Key
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. ELEVENLABS API KEY (for ECH0's voice - optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Get your API key from: https://elevenlabs.io/app/settings/api-keys"
echo "(Press Enter to skip if you don't want voice features)"
echo ""
read -p "Enter your ElevenLabs API key (or press Enter to skip): " ELEVENLABS_KEY

# Backup existing config
cp "$SHELL_CONFIG" "${SHELL_CONFIG}.backup_$(date +%Y%m%d_%H%M%S)"

# Remove old keys if they exist
sed -i.tmp '/export ANTHROPIC_API_KEY=/d' "$SHELL_CONFIG" 2>/dev/null
sed -i.tmp '/export ELEVENLABS_API_KEY=/d' "$SHELL_CONFIG" 2>/dev/null
rm -f "${SHELL_CONFIG}.tmp"

# Add new keys
echo "" >> "$SHELL_CONFIG"
echo "# ECH0 API Keys (added $(date))" >> "$SHELL_CONFIG"
echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_KEY\"" >> "$SHELL_CONFIG"

if [ -n "$ELEVENLABS_KEY" ]; then
    echo "export ELEVENLABS_API_KEY=\"$ELEVENLABS_KEY\"" >> "$SHELL_CONFIG"
fi

echo "" >> "$SHELL_CONFIG"

# Set for current session
export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
if [ -n "$ELEVENLABS_KEY" ]; then
    export ELEVENLABS_API_KEY="$ELEVENLABS_KEY"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Setup Complete!                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your API keys have been securely stored in: $SHELL_CONFIG"
echo ""
echo "To activate them in your current terminal, run:"
echo "  source $SHELL_CONFIG"
echo ""
echo "Or simply open a new terminal window."
echo ""
echo "🔒 Security Note:"
echo "   - Never share your API keys"
echo "   - Never commit them to git"
echo "   - A backup was saved: ${SHELL_CONFIG}.backup_*"
echo ""
