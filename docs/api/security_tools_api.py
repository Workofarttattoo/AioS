#!/usr/bin/env python3
"""
Security Tools Backend API
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Flask-based REST API for red-team security tools with authentication, rate limiting, and defensive safeguards.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import time
import hashlib
import secrets
import re
from collections import defaultdict
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEYS = {
    "demo_key_12345": {
        "user": "demo_user",
        "tier": "trial",
        "rate_limit": 10,  # requests per minute
        "tools_allowed": ["sqlgps", "hashsolver", "nmap", "belchstudio"]
    }
}

# Rate limiting storage
rate_limit_store = defaultdict(list)

# ============================================================================
# AUTHENTICATION & RATE LIMITING
# ============================================================================

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key or api_key not in API_KEYS:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing API key',
                'message': 'Please provide a valid API key in X-API-Key header'
            }), 401

        # Check rate limiting
        user_data = API_KEYS[api_key]
        current_time = time.time()
        user_requests = rate_limit_store[api_key]

        # Remove requests older than 1 minute
        user_requests[:] = [t for t in user_requests if current_time - t < 60]

        # Check if rate limit exceeded
        if len(user_requests) >= user_data['rate_limit']:
            return jsonify({
                'success': False,
                'error': 'Rate limit exceeded',
                'message': f"Limit: {user_data['rate_limit']} requests/minute",
                'retry_after': 60 - (current_time - user_requests[0])
            }), 429

        # Record this request
        user_requests.append(current_time)

        # Pass user data to endpoint
        request.user_data = user_data
        return f(*args, **kwargs)

    return decorated_function


def require_tool_access(tool_name):
    """Decorator to check if user has access to specific tool."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if tool_name not in request.user_data['tools_allowed']:
                return jsonify({
                    'success': False,
                    'error': 'Tool access denied',
                    'message': f'Your tier does not include access to {tool_name}'
                }), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================================
# SQLGPS - SQL INJECTION TESTING API
# ============================================================================

@app.route('/api/sqlgps/scan', methods=['POST'])
@require_api_key
@require_tool_access('sqlgps')
def sqlgps_scan():
    """
    Simulate SQL injection vulnerability scanning.
    DEFENSIVE ONLY - For testing your own applications.
    """
    data = request.json
    target_url = data.get('url', '')
    techniques = data.get('techniques', [])

    # Validate URL (must be localhost or authorized domains)
    if not _is_authorized_target(target_url):
        return jsonify({
            'success': False,
            'error': 'Unauthorized target',
            'message': 'Only localhost and authorized domains allowed'
        }), 403

    # Simulate scanning
    results = []
    for technique in techniques:
        results.append({
            'technique': technique,
            'vulnerable': False,  # Simulation only
            'payloads_tested': 5,
            'response_time': round(secrets.randbelow(1000) / 1000, 3),
            'message': f'{technique} scan completed (simulation)'
        })

    return jsonify({
        'success': True,
        'target': target_url,
        'results': results,
        'scan_time': datetime.utcnow().isoformat(),
        'warning': 'This is a simulation. Real scanning requires authorization.'
    })


@app.route('/api/sqlgps/enumerate', methods=['POST'])
@require_api_key
@require_tool_access('sqlgps')
def sqlgps_enumerate():
    """Simulate database enumeration (for authorized testing only)."""
    data = request.json
    target = data.get('target', '')

    if not _is_authorized_target(target):
        return jsonify({'success': False, 'error': 'Unauthorized target'}), 403

    # Simulated enumeration results
    return jsonify({
        'success': True,
        'databases': ['demo_db', 'test_db'],
        'tables': ['users', 'products', 'orders'],
        'columns': ['id', 'username', 'email', 'password_hash'],
        'warning': 'Simulation only. Real enumeration requires authorization.'
    })


# ============================================================================
# HASHSOLVER - PASSWORD HASH CRACKING API
# ============================================================================

@app.route('/api/hashsolver/crack', methods=['POST'])
@require_api_key
@require_tool_access('hashsolver')
def hashsolver_crack():
    """
    Simulate password hash cracking.
    DEFENSIVE ONLY - For testing your own password security.
    """
    data = request.json
    hash_value = data.get('hash', '')
    algorithm = data.get('algorithm', 'md5')
    attack_mode = data.get('attack_mode', 'dictionary')

    # Detect algorithm if not specified
    if not algorithm:
        algorithm = _detect_hash_algorithm(hash_value)

    # Simulate cracking process
    result = {
        'success': True,
        'hash': hash_value,
        'algorithm': algorithm,
        'attack_mode': attack_mode,
        'cracked': False,
        'plaintext': None,
        'attempts': secrets.randbelow(10000),
        'time_elapsed': round(secrets.randbelow(5000) / 1000, 2),
        'hash_rate': f'{secrets.randbelow(1000)}k H/s',
        'warning': 'Simulation only. Real cracking for authorized testing.'
    }

    # For demo purposes, "crack" simple MD5 hashes
    if algorithm == 'md5' and hash_value == hashlib.md5(b'password').hexdigest():
        result['cracked'] = True
        result['plaintext'] = 'password'

    return jsonify(result)


@app.route('/api/hashsolver/identify', methods=['POST'])
@require_api_key
@require_tool_access('hashsolver')
def hashsolver_identify():
    """Identify hash algorithm based on pattern."""
    data = request.json
    hash_value = data.get('hash', '')

    algorithm = _detect_hash_algorithm(hash_value)

    return jsonify({
        'success': True,
        'hash': hash_value,
        'detected_algorithm': algorithm,
        'confidence': 'high' if algorithm != 'unknown' else 'low'
    })


# ============================================================================
# NMAP STREET EDITION - NETWORK SCANNING API
# ============================================================================

@app.route('/api/nmap/scan', methods=['POST'])
@require_api_key
@require_tool_access('nmap')
def nmap_scan():
    """
    Simulate network scanning.
    DEFENSIVE ONLY - For scanning your own networks only.
    """
    data = request.json
    target = data.get('target', '')
    scan_type = data.get('scan_type', 'syn')
    ports = data.get('ports', '1-1000')

    # Validate target (must be localhost or authorized IPs)
    if not _is_authorized_network_target(target):
        return jsonify({
            'success': False,
            'error': 'Unauthorized target',
            'message': 'Only localhost and authorized networks allowed'
        }), 403

    # Simulate scan results
    open_ports = []
    common_ports = [21, 22, 80, 443, 3306, 5432, 8080]

    for port in common_ports[:secrets.randbelow(5) + 1]:
        open_ports.append({
            'port': port,
            'state': 'open',
            'service': _get_service_name(port),
            'version': 'Unknown',
            'cves': []
        })

    return jsonify({
        'success': True,
        'target': target,
        'scan_type': scan_type,
        'ports_scanned': ports,
        'open_ports': open_ports,
        'scan_time': datetime.utcnow().isoformat(),
        'warning': 'Simulation only. Real scanning requires authorization.'
    })


@app.route('/api/nmap/service-detect', methods=['POST'])
@require_api_key
@require_tool_access('nmap')
def nmap_service_detect():
    """Simulate service version detection."""
    data = request.json
    target = data.get('target', '')
    port = data.get('port', 80)

    if not _is_authorized_network_target(target):
        return jsonify({'success': False, 'error': 'Unauthorized target'}), 403

    service = _get_service_name(port)

    return jsonify({
        'success': True,
        'target': target,
        'port': port,
        'service': service,
        'version': f'{service} 2.4.1',
        'os_hint': 'Linux 5.x',
        'cpes': [f'cpe:/a:{service}:{service}:2.4.1']
    })


# ============================================================================
# BELCHSTUDIO - HTTP PROXY API
# ============================================================================

@app.route('/api/belchstudio/intercept', methods=['POST'])
@require_api_key
@require_tool_access('belchstudio')
def belchstudio_intercept():
    """
    Simulate HTTP request interception.
    DEFENSIVE ONLY - For testing your own applications.
    """
    data = request.json
    url = data.get('url', '')
    method = data.get('method', 'GET')

    if not _is_authorized_target(url):
        return jsonify({'success': False, 'error': 'Unauthorized target'}), 403

    return jsonify({
        'success': True,
        'request': {
            'method': method,
            'url': url,
            'headers': {'User-Agent': 'BelchStudio/1.0'},
            'intercepted': True
        },
        'response': {
            'status': 200,
            'headers': {'Content-Type': 'text/html'},
            'body_preview': '<html>...</html>',
            'size': '4096 bytes'
        },
        'warning': 'Simulation only. Real interception requires authorization.'
    })


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _is_authorized_target(url):
    """Check if URL is authorized for testing."""
    authorized_patterns = [
        r'^https?://localhost',
        r'^https?://127\.0\.0\.1',
        r'^https?://test\.local',
        r'^https?://.*\.test$'
    ]
    return any(re.match(pattern, url) for pattern in authorized_patterns)


def _is_authorized_network_target(target):
    """Check if network target is authorized."""
    authorized_patterns = [
        r'^localhost$',
        r'^127\.0\.0\.1$',
        r'^192\.168\.',
        r'^10\.',
        r'^172\.(1[6-9]|2[0-9]|3[01])\.'
    ]
    return any(re.match(pattern, target) for pattern in authorized_patterns)


def _detect_hash_algorithm(hash_value):
    """Detect hash algorithm based on length and pattern."""
    hash_len = len(hash_value)

    algorithms = {
        32: 'MD5',
        40: 'SHA1',
        64: 'SHA256',
        128: 'SHA512',
        60: 'bcrypt'
    }

    return algorithms.get(hash_len, 'unknown')


def _get_service_name(port):
    """Get service name for common ports."""
    services = {
        21: 'FTP',
        22: 'SSH',
        80: 'HTTP',
        443: 'HTTPS',
        3306: 'MySQL',
        5432: 'PostgreSQL',
        8080: 'HTTP-Proxy'
    }
    return services.get(port, 'Unknown')


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """API information and available endpoints."""
    return jsonify({
        'name': 'TheGAVL Security Tools API',
        'version': '1.0.0',
        'endpoints': {
            'SQLgps': ['/api/sqlgps/scan', '/api/sqlgps/enumerate'],
            'HashSolver': ['/api/hashsolver/crack', '/api/hashsolver/identify'],
            'NMAP': ['/api/nmap/scan', '/api/nmap/service-detect'],
            'BelchStudio': ['/api/belchstudio/intercept']
        },
        'authentication': 'API key required (X-API-Key header)',
        'rate_limiting': 'Varies by tier',
        'usage': 'Authorized defensive testing only',
        'compliance': 'CFAA compliant - authorized use only'
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'Check /api/info for available endpoints'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     TheGAVL Security Tools API Server                        ║
║     Copyright (c) 2025 Joshua Hendricks Cole                 ║
║     Corporation of Light - All Rights Reserved               ║
║     PATENT PENDING                                           ║
║                                                               ║
║     ⚠️  AUTHORIZED USE ONLY                                   ║
║     Defensive security testing only                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Starting API server on http://localhost:5000

📚 API Endpoints:
   - GET  /api/health          Health check
   - GET  /api/info            API information
   - POST /api/sqlgps/scan     SQL injection scanning
   - POST /api/hashsolver/crack Hash cracking
   - POST /api/nmap/scan       Network scanning
   - POST /api/belchstudio/intercept HTTP interception

🔑 Demo API Key: demo_key_12345
   (Add to request header: X-API-Key: demo_key_12345)

⚡ Rate Limits: 10 requests/minute (demo tier)

""")

    app.run(host='0.0.0.0', port=5000, debug=True)
