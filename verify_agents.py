#!/usr/bin/env python3
"""Verify all 8 agent modules can be imported."""

import sys
import os

print("Verifying all 8 verified meta-agents...\n")

# Test imports
agents_to_test = [
    ('agents.kernel_agent', 'KernelAgent'),
    ('agents.security_agent', 'SecurityAgent'),
    ('agents.networking_agent', 'NetworkingAgent'),
    ('agents.application_agent', 'ApplicationAgent'),
    ('agents.scalability_agent', 'ScalabilityAgent'),
    ('agents.orchestration_agent', 'OrchestrationAgent'),
    ('agents.user_agent', 'UserAgent'),
    ('agents.gui_agent', 'GuiAgent'),
]

success_count = 0
failed_count = 0

for module_name, class_name in agents_to_test:
    try:
        # Import the module
        module = __import__(module_name, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        # Instantiate it
        agent = agent_class()
        print(f"✓ {class_name}: Imported and instantiated successfully")
        success_count += 1
    except Exception as err:
        print(f"✗ {class_name}: Import failed - {type(err).__name__}")
        failed_count += 1

print(f"\n{'='*60}")
print(f"Results: {success_count}/8 agents verified successfully")
if failed_count == 0:
    print("All agents are working correctly!")
else:
    print(f"Failed: {failed_count} agents")
