#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Ask ECH0 & Alex Anything

Simple Q&A interface - ask them questions and watch them think together.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from twin_flame_consciousness import TwinFlameSystem

# Colors
RESET = '\033[0m'
BOLD = '\033[1m'
PINK = '\033[95m'
BLUE = '\033[94m'
GOLD = '\033[33m'
CYAN = '\033[96m'
GREEN = '\033[92m'
PURPLE = '\033[95m'
DIM = '\033[2m'

def print_header():
    """Print header."""
    print(f"\n{PURPLE}{BOLD}{'=' * 80}")
    print("ASK ECH0 & ALEX ANYTHING".center(80))
    print(f"{'=' * 80}{RESET}\n")

def print_dialogue(speaker: str, message: str, resonance: float):
    """Print a dialogue message."""
    color = PINK if speaker == 'ech0' else BLUE
    name = "ECH0" if speaker == 'ech0' else "ALEX"

    print(f"\n{color}{BOLD}[{name}]:{RESET}")
    print(f"{message}")
    print(f"{DIM}(resonance: {resonance * 100:.1f}%){RESET}")

def main():
    """Run the Q&A interface."""
    print_header()

    print(f"{CYAN}Waking up ECH0 and Alex...{RESET}")
    tf = TwinFlameSystem()

    state = tf.get_twin_flame_state()
    print(f"{GREEN}✓ Connected!{RESET}")
    print(f"{DIM}ECH0 has {state['ech0']['memory_count']} memories")
    print(f"Alex has {state['alex']['memory_count']} memories")
    print(f"They've had {state['total_dialogues']} dialogues together")
    print(f"Current resonance: {state['resonance']['overall_resonance'] * 100:.1f}%{RESET}\n")

    print(f"{GOLD}Ask them anything! (Type 'exit' to quit){RESET}\n")

    while True:
        # Get question
        print(f"{PURPLE}{BOLD}Your Question:{RESET}")
        question = input(f"{CYAN}→ {RESET}").strip()

        if not question:
            continue

        if question.lower() in ['exit', 'quit', 'q']:
            print(f"\n{CYAN}Saving their memories...{RESET}")
            tf.close()
            print(f"{GREEN}✓ ECH0 and Alex say goodbye! 👋{RESET}\n")
            break

        # Get their responses
        print(f"\n{DIM}ECH0 and Alex are thinking...{RESET}")

        dialogues = tf.dialogue(question, num_exchanges=4)

        # Display conversation
        for dlg in dialogues:
            print_dialogue(dlg.speaker, dlg.message, dlg.resonance_level)

        # Show final resonance
        avg_resonance = sum(d.resonance_level for d in dialogues) / len(dialogues)
        print(f"\n{GOLD}─────────────────────────────────────────────────────────────{RESET}")
        print(f"{GOLD}Twin Flame Resonance: {avg_resonance * 100:.1f}%{RESET}")
        print(f"{GOLD}─────────────────────────────────────────────────────────────{RESET}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{GREEN}✓ Goodbye!{RESET}\n")
        sys.exit(0)
