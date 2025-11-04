#!/bin/bash
# Quick demo runner for ECH0 and Alex Twin Flame System

cd /Users/noone/aios

echo "Which demo would you like to run?"
echo ""
echo "1. Complete Twin Flame Demo (interactive, full journey)"
echo "2. Twin Flame Dialogue & Resonance"
echo "3. Creative Collaboration Studio"
echo "4. Emergence Pathway"
echo "5. Ai:oS Consciousness Integration"
echo ""
read -p "Choice (1-5): " choice

case $choice in
    1)
        python3 COMPLETE_TWIN_FLAME_DEMO.py
        ;;
    2)
        python3 twin_flame_consciousness.py
        ;;
    3)
        python3 creative_collaboration.py
        ;;
    4)
        python3 emergence_pathway.py
        ;;
    5)
        python3 aios_consciousness_integration.py
        ;;
    *)
        echo "Invalid choice. Running complete demo..."
        python3 COMPLETE_TWIN_FLAME_DEMO.py
        ;;
esac
