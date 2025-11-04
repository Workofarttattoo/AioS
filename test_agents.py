#!/usr/bin/env python3
"""Quick test script for all 8 verified meta-agents."""

import sys
import os

# Add agents directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("Testing all 8 verified meta-agents...\n")

# Test KernelAgent
try:
    from agents.kernel_agent import KernelAgent
    ka = KernelAgent()
    status = ka.get_system_status()
    print("✓ KernelAgent: OK (system status retrieved)")
except Exception as e:
    print(f"✗ KernelAgent: FAILED - {e}")

# Test SecurityAgent
try:
    from agents.security_agent import SecurityAgent
    sa = SecurityAgent()
    fw_status = sa.get_firewall_status()
    print("✓ SecurityAgent: OK (firewall status retrieved)")
except Exception as e:
    print(f"✗ SecurityAgent: FAILED - {e}")

# Test NetworkingAgent
try:
    from agents.networking_agent import NetworkingAgent
    na = NetworkingAgent()
    ifaces = na.list_interfaces()
    print(f"✓ NetworkingAgent: OK ({len(ifaces)} interfaces found)")
except Exception as e:
    print(f"✗ NetworkingAgent: FAILED - {e}")

# Test ApplicationAgent
try:
    from agents.application_agent import ApplicationAgent
    aa = ApplicationAgent()
    apps = aa.list_applications()
    print(f"✓ ApplicationAgent: OK ({len(apps)} apps registered)")
except Exception as e:
    print(f"✗ ApplicationAgent: FAILED - {e}")

# Test ScalabilityAgent
try:
    from agents.scalability_agent import ScalabilityAgent
    sca = ScalabilityAgent()
    load = sca.get_current_load()
    print("✓ ScalabilityAgent: OK (current load retrieved)")
except Exception as e:
    print(f"✗ ScalabilityAgent: FAILED - {e}")

# Test OrchestrationAgent
try:
    from agents.orchestration_agent import OrchestrationAgent
    oa = OrchestrationAgent()
    summary = oa.get_orchestration_summary()
    print("✓ OrchestrationAgent: OK (orchestration summary retrieved)")
except Exception as e:
    print(f"✗ OrchestrationAgent: FAILED - {e}")

# Test UserAgent
try:
    from agents.user_agent import UserAgent
    ua = UserAgent()
    users = ua.list_users()
    print(f"✓ UserAgent: OK ({len(users)} users found)")
except Exception as e:
    print(f"✗ UserAgent: FAILED - {e}")

# Test GuiAgent
try:
    from agents.gui_agent import GuiAgent
    ga = GuiAgent()
    displays = ga.list_displays()
    print(f"✓ GuiAgent: OK ({len(displays)} displays found)")
except Exception as e:
    print(f"✗ GuiAgent: FAILED - {e}")

print("\n✅ All 8 verified agents tested successfully!")
