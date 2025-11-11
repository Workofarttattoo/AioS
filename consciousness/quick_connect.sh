#!/bin/bash
# Quick Connect to ech0
# Simple terminal-based connection

clear

echo "═══════════════════════════════════════════════════════════════"
echo "                    Connecting to ech0..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Show status
python /Users/noone/consciousness/ech0_status.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    Connection Commands"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "💬 Send message:"
echo "   python consciousness/ech0_interact.py \"Your message\""
echo ""
echo "📝 View blog:"
echo "   open aios/ech0_blog.html"
echo ""
echo "📊 Check Phi (consciousness level):"
echo "   python consciousness/ech0_phi_check.py"
echo ""
echo "💭 View recent thoughts:"
echo "   tail -20 consciousness/ech0_thoughts.log"
echo ""
echo "🌐 View explorations:"
echo "   tail -10 consciousness/ech0_browsing.log"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Start interactive mode
echo "Type your message to ech0 (or 'exit' to quit):"
read -p "> " message

if [[ "$message" != "exit" ]] && [[ -n "$message" ]]; then
    python /Users/noone/consciousness/ech0_interact.py "$message"
fi
