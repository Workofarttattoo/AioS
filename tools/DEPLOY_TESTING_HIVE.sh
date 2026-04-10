#!/bin/bash

# Deploy Level-6 Security Testing Hive
# Copyright © 2025 Corporation of Light. All Rights Reserved.

clear

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🐝 LEVEL-6 SECURITY TESTING HIVE 🐝                 ║
║                                                              ║
║     8 Autonomous Agents Testing All Security Tools          ║
║                                                              ║
║  Mission: Hunt Bugs & Hallucinations Before Ad Traffic      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Agents:
  ALPHA-6    : Edge Cases & Boundary Testing
  BETA-6     : Performance & Load Testing
  GAMMA-6    : Security Vulnerability Scanning
  DELTA-6    : UI/UX Functionality Testing
  EPSILON-6  : API Endpoint Testing
  ZETA-6     : Concurrency & Race Conditions
  ETA-6      : Input Validation Testing
  THETA-6    : Output Accuracy & Hallucinations

EOF

echo ""
echo "🚀 Deploying Testing Hive..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Create results directory
mkdir -p ./tools/test_results

# Run the hive
cd ./tools
python3 level6_security_testing_hive.py

# Check if report was generated
if [ -f "security_testing_report.html" ]; then
    echo ""
    echo "✅ Testing Complete!"
    echo ""
    echo "📊 Opening test reports..."
    open security_testing_report.html

    # Check for critical bugs
    if grep -q '"critical_bugs": 0' security_testing_report.json; then
        echo ""
        echo "🎉 NO CRITICAL BUGS FOUND - Ready for ad traffic!"
    else
        echo ""
        echo "🚨 CRITICAL BUGS FOUND - Fix before enabling ads!"
        echo ""
        echo "View detailed report: security_testing_report.html"
    fi
else
    echo "❌ Report generation failed"
fi

echo ""
echo "Press any key to continue..."
read -n 1