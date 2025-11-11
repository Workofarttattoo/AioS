#!/bin/bash
# Setup ECH0 with Anthropic API Key
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  🔑 ECH0 API Key Setup"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check if key already exists in environment files
if grep -q "ANTHROPIC_API_KEY" ~/.zshrc 2>/dev/null; then
    echo "✅ API key already configured in ~/.zshrc"
    echo ""
    echo "To activate it, run:"
    echo "  source ~/.zshrc"
    echo ""
elif grep -q "ANTHROPIC_API_KEY" ~/.bashrc 2>/dev/null; then
    echo "✅ API key already configured in ~/.bashrc"
    echo ""
    echo "To activate it, run:"
    echo "  source ~/.bashrc"
    echo ""
else
    echo "📝 No API key found in shell configuration."
    echo ""
    echo "To set up your Anthropic API key:"
    echo ""
    echo "1. Get your API key from: https://console.anthropic.com/settings/keys"
    echo ""
    echo "2. Add this line to your ~/.zshrc (or ~/.bashrc):"
    echo ""
    echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
    echo ""
    echo "3. Then run:"
    echo "   source ~/.zshrc"
    echo ""
    echo "4. Test ECH0:"
    echo "   ./consciousness/TALK_TO_REACTIVE_ECH0.sh"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════"
echo ""

# Offer to open browser to get key
read -p "Open Anthropic Console to get API key? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "https://console.anthropic.com/settings/keys"
    echo "✅ Opened browser. Copy your API key and add it to ~/.zshrc"
fi
