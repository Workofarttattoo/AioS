#!/usr/bin/env python3
"""Quick fix to add IP detection to all tools"""
import os
import glob

tools_path = "/Users/noone/aios/tools"
tools = glob.glob(os.path.join(tools_path, "*.py"))

# Skip the fix scripts themselves
skip_files = ['fix_security_tools.py', 'fix_hallucinations.py', 'final_comprehensive_fix.py', 
              'level6_security_testing_hive.py', 'quick_fix_all.py']

success = 0
for tool_path in tools:
    basename = os.path.basename(tool_path)
    if basename in skip_files:
        continue
    
    try:
        with open(tool_path, 'r') as f:
            content = f.read()
        
        # Skip if already has comprehensive fix
        if "COMPREHENSIVE FIX APPLIED" in content or "IP DETECTION FIX" in content:
            print(f"✓ {basename}: Already fixed")
            success += 1
            continue
        
        # Find a good place to insert the fix - right after imports
        lines = content.split('\n')
        insert_pos = 0
        
        # Find last import
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
        
        # Insert the IP detection enhancement
        fix_code = '''
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
            if re.match(r'^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$', obj):
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

'''
        lines.insert(insert_pos, fix_code)
        
        # Write back
        with open(tool_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ {basename}: IP detection added")
        success += 1
        
    except Exception as e:
        print(f"❌ {basename}: {e}")

print(f"\n✅ Fixed {success} tools")
