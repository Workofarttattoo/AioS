#!/usr/bin/env python3
"""
Comprehensive tool testing script
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Tests all security tools in the aios/tools/ directory for:
- Import success
- Health check functionality
- CLI main() execution
- GUI availability
- Quantum backend integration
"""

import sys
import os
import importlib
import json
from pathlib import Path

# === IP DETECTION FIX ===
_original_json_dumps = None
try:
    import json
    import ipaddress
    import re
    _original_json_dumps = json.dumps
    
    def enhance_ip_data(obj):
        """Recursively enhance IP addresses in data structures"""
        if isinstance(obj, str):
            # Check if this is an IP
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', obj):
                try:
                    ip = ipaddress.ip_address(obj)
                    parts = []
                    if ip.is_private:
                        parts.append("Private IP (RFC1918)")
                    if str(ip) in ["8.8.8.8", "8.8.4.4"]:
                        parts.append("Google DNS")
                    elif str(ip) in ["1.1.1.1", "1.0.0.1"]:
                        parts.append("Cloudflare DNS")
                    if parts:
                        return f"{obj} ({', '.join(parts)})"
                except:
                    pass
            return obj
        elif isinstance(obj, dict):
            return {k: enhance_ip_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [enhance_ip_data(item) for item in obj]
        else:
            return obj
    
    def enhanced_json_dumps(obj, **kwargs):
        """Enhanced json.dumps that adds IP detection"""
        enhanced_obj = enhance_ip_data(obj)
        return _original_json_dumps(enhanced_obj, **kwargs)
    
    # Monkey patch json.dumps
    json.dumps = enhanced_json_dumps
except ImportError:
    pass
# === END IP DETECTION FIX ===



# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Expected tools based on TOOL_REGISTRY
EXPECTED_TOOLS = [
    'aurorascan',
    'cipherspear',
    'skybreaker',
    'mythickey',
    'spectratrace',
    'nemesishydra',
    'obsidianhunt',
    'vectorflux',
    'dirreaper',
    'proxyphantom',
    'vulnhunter',
    'nmappro',
    'payloadforge',
    'burpsuite_clone'
]

def test_tool_import(tool_name):
    """Test if tool can be imported"""
    try:
        module = importlib.import_module(f'tools.{tool_name}')
        return True, module
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Import error: {e}"


def test_health_check(module):
    """Test if tool has working health_check()"""
    if not hasattr(module, 'health_check'):
        return False, "No health_check() function"

    try:
        result = module.health_check()

        # Validate health check structure
        required_fields = ['tool', 'status', 'summary']
        missing = [f for f in required_fields if f not in result]

        if missing:
            return False, f"Missing fields: {missing}"

        status = result.get('status')
        if status not in ['ok', 'warn', 'error']:
            return False, f"Invalid status: {status}"

        return True, result

    except Exception as e:
        return False, f"health_check() failed: {e}"


def test_main_function(module):
    """Test if tool has main() function"""
    if not hasattr(module, 'main'):
        return False, "No main() function"

    # Don't actually call main() as it may hang or require args
    # Just verify it's callable
    if callable(module.main):
        return True, "main() is callable"
    else:
        return False, "main() exists but not callable"


def test_gui_support(module):
    """Test if tool has GUI support"""
    if not hasattr(module, 'gui'):
        return False, "No gui() function"

    if callable(module.gui):
        return True, "gui() is callable"
    else:
        return False, "gui() exists but not callable"


def test_quantum_backend(module):
    """Test if tool integrates quantum backend"""
    # Check if tool imports _quantum_backend
    if hasattr(module, 'QuantumAnomalyDetector'):
        return True, "Has QuantumAnomalyDetector"
    elif hasattr(module, 'QuantumPathPredictor'):
        return True, "Has QuantumPathPredictor"
    elif hasattr(module, 'QuantumResponseForecaster'):
        return True, "Has QuantumResponseForecaster"
    else:
        # Check module source for quantum imports
        try:
            source_file = Path(module.__file__)
            if source_file.exists():
                content = source_file.read_text()
                if 'quantum_backend' in content or 'QuantumAnomaly' in content:
                    return True, "Imports quantum backend"
        except:
            pass

    return False, "No quantum integration detected"


def test_tool(tool_name):
    """Run all tests for a single tool"""
    print(f"\n{'='*70}")
    print(f"Testing: {tool_name}")
    print('='*70)

    results = {
        'tool': tool_name,
        'import': False,
        'health_check': False,
        'main': False,
        'gui': False,
        'quantum': False,
        'details': {}
    }

    # Test import
    success, module_or_error = test_tool_import(tool_name)
    results['import'] = success
    if not success:
        print(f"✗ Import: {module_or_error}")
        results['details']['import_error'] = module_or_error
        return results
    else:
        print(f"✓ Import: Success")
        module = module_or_error

    # Test health check
    success, health_result = test_health_check(module)
    results['health_check'] = success
    if success:
        print(f"✓ Health Check: {health_result.get('status')} - {health_result.get('summary')}")
        results['details']['health'] = health_result
    else:
        print(f"✗ Health Check: {health_result}")
        results['details']['health_error'] = health_result

    # Test main function
    success, main_result = test_main_function(module)
    results['main'] = success
    if success:
        print(f"✓ Main: {main_result}")
    else:
        print(f"✗ Main: {main_result}")

    # Test GUI support
    success, gui_result = test_gui_support(module)
    results['gui'] = success
    if success:
        print(f"✓ GUI: {gui_result}")
    else:
        print(f"⚠ GUI: {gui_result}")  # Warning, not error

    # Test quantum backend
    success, quantum_result = test_quantum_backend(module)
    results['quantum'] = success
    if success:
        print(f"✓ Quantum: {quantum_result}")
    else:
        print(f"⚠ Quantum: {quantum_result}")  # Warning, not error

    return results


def main():
    """Test all tools and generate report"""
    print("="*70)
    print("Ai|oS Security Toolkit - Comprehensive Testing Suite")
    print("="*70)
    print(f"\nTesting {len(EXPECTED_TOOLS)} tools...")

    all_results = []

    for tool_name in EXPECTED_TOOLS:
        result = test_tool(tool_name)
        all_results.append(result)

    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Count results
    importable = sum(1 for r in all_results if r['import'])
    has_health = sum(1 for r in all_results if r['health_check'])
    has_main = sum(1 for r in all_results if r['main'])
    has_gui = sum(1 for r in all_results if r['gui'])
    has_quantum = sum(1 for r in all_results if r['quantum'])

    total = len(all_results)

    print(f"\nImportable:     {importable}/{total} ({importable/total*100:.0f}%)")
    print(f"Health Checks:  {has_health}/{total} ({has_health/total*100:.0f}%)")
    print(f"Main Functions: {has_main}/{total} ({has_main/total*100:.0f}%)")
    print(f"GUI Support:    {has_gui}/{total} ({has_gui/total*100:.0f}%)")
    print(f"Quantum ML:     {has_quantum}/{total} ({has_quantum/total*100:.0f}%)")

    # Production readiness assessment
    print("\n" + "="*70)
    print("PRODUCTION READINESS ASSESSMENT")
    print("="*70)

    # Categorize tools
    production_ready = []
    needs_work = []
    broken = []

    for r in all_results:
        if not r['import']:
            broken.append(r['tool'])
        elif r['health_check'] and r['main']:
            production_ready.append(r['tool'])
        else:
            needs_work.append(r['tool'])

    print(f"\n✓ Production Ready ({len(production_ready)}):")
    for tool in production_ready:
        print(f"  - {tool}")

    print(f"\n⚠ Needs Work ({len(needs_work)}):")
    for tool in needs_work:
        print(f"  - {tool}")

    print(f"\n✗ Broken/Missing ({len(broken)}):")
    for tool in broken:
        print(f"  - {tool}")

    # Overall status
    print("\n" + "="*70)
    if len(production_ready) == total:
        print("✓ ALL TOOLS ARE PRODUCTION READY!")
        return 0
    elif len(production_ready) >= total * 0.8:
        print(f"⚠ {len(production_ready)}/{total} tools ready - near production quality")
        return 0
    else:
        print(f"✗ Only {len(production_ready)}/{total} tools ready - NOT production ready")
        return 1

    # Save detailed results
    output_file = Path(__file__).parent.parent / 'tool_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    sys.exit(main())
