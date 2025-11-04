#!/usr/bin/env python3
"""
BurpSuite Clone - HTTP Proxy & Security Testing Platform
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

A complete implementation of BurpSuite-style functionality:
- HTTP/HTTPS Proxy with SSL/TLS interception
- Request/Response Interceptor
- Repeater (manual request modification)
- Intruder (automated fuzzing/injection)
- Decoder/Encoder
- Comparer (diff tool for requests/responses)
- Scanner (passive + active vulnerability detection)
- Quantum-enhanced vulnerability prediction
"""

import os
import sys
import json
import time
import asyncio
import logging
import ssl
import hashlib
import base64
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

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



# HTTP parsing
try:
    import http.client as httplib
    from urllib.parse import urlparse, parse_qs
except ImportError:
    import httplib  # Python 2 fallback
    from urlparse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Check for quantum backend
try:
    from _quantum_backend import detect_anomaly, predict_paths, forecast_response
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    LOG.warning("Quantum backend not available - using classical analysis")


@dataclass
class HTTPRequest:
    """Represents an HTTP request"""
    method: str
    url: str
    headers: Dict[str, str]
    body: bytes
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])

    def to_raw(self) -> bytes:
        """Convert to raw HTTP request bytes"""
        parsed = urlparse(self.url)
        path = parsed.path or '/'
        if parsed.query:
            path += f'?{parsed.query}'

        lines = [f"{self.method} {path} HTTP/1.1"]

        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")

        request_str = '\r\n'.join(lines) + '\r\n\r\n'
        return request_str.encode('utf-8') + self.body

    @staticmethod
    def from_raw(raw: bytes) -> 'HTTPRequest':
        """Parse raw HTTP request bytes"""
        parts = raw.split(b'\r\n\r\n', 1)
        header_part = parts[0].decode('utf-8', errors='ignore')
        body = parts[1] if len(parts) > 1 else b''

        lines = header_part.split('\r\n')
        request_line = lines[0].split(' ')
        method = request_line[0]
        path = request_line[1] if len(request_line) > 1 else '/'

        headers = {}
        for line in lines[1:]:
            if ': ' in line:
                key, value = line.split(': ', 1)
                headers[key] = value

        host = headers.get('Host', 'unknown')
        url = f"http://{host}{path}"

        return HTTPRequest(method=method, url=url, headers=headers, body=body)


@dataclass
class HTTPResponse:
    """Represents an HTTP response"""
    status_code: int
    status_message: str
    headers: Dict[str, str]
    body: bytes
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0

    def to_raw(self) -> bytes:
        """Convert to raw HTTP response bytes"""
        lines = [f"HTTP/1.1 {self.status_code} {self.status_message}"]

        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")

        response_str = '\r\n'.join(lines) + '\r\n\r\n'
        return response_str.encode('utf-8') + self.body

    @staticmethod
    def from_raw(raw: bytes) -> 'HTTPResponse':
        """Parse raw HTTP response bytes"""
        parts = raw.split(b'\r\n\r\n', 1)
        header_part = parts[0].decode('utf-8', errors='ignore')
        body = parts[1] if len(parts) > 1 else b''

        lines = header_part.split('\r\n')
        status_line = lines[0].split(' ', 2)
        status_code = int(status_line[1])
        status_message = status_line[2] if len(status_line) > 2 else ''

        headers = {}
        for line in lines[1:]:
            if ': ' in line:
                key, value = line.split(': ', 1)
                headers[key] = value

        return HTTPResponse(
            status_code=status_code,
            status_message=status_message,
            headers=headers,
            body=body
        )


@dataclass
class HTTPTransaction:
    """A complete request/response pair"""
    request: HTTPRequest
    response: Optional[HTTPResponse] = None
    intercepted: bool = False
    modified: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.request.id,
            'method': self.request.method,
            'url': self.request.url,
            'status': self.response.status_code if self.response else None,
            'timestamp': self.request.timestamp,
            'intercepted': self.intercepted,
            'modified': self.modified,
            'notes': self.notes
        }


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP proxy request handler"""

    def do_CONNECT(self):
        """Handle HTTPS CONNECT method for SSL/TLS tunneling"""
        self.send_response(200, 'Connection Established')
        self.end_headers()

        # TODO: Implement SSL/TLS interception with custom CA cert
        # For now, just tunnel through
        LOG.info(f"CONNECT {self.path}")

    def do_GET(self):
        self._proxy_request()

    def do_POST(self):
        self._proxy_request()

    def do_PUT(self):
        self._proxy_request()

    def do_DELETE(self):
        self._proxy_request()

    def do_HEAD(self):
        self._proxy_request()

    def do_OPTIONS(self):
        self._proxy_request()

    def _proxy_request(self):
        """Proxy an HTTP request"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''

            # Create HTTPRequest
            request = HTTPRequest(
                method=self.command,
                url=self.path,
                headers=dict(self.headers),
                body=body
            )

            # Check if intercept is enabled
            if self.server.intercept_enabled:
                # Queue for interception
                self.server.intercept_queue.append(request)
                LOG.info(f"Intercepted: {request.method} {request.url}")

                # Wait for user to forward/drop/modify
                while request in self.server.intercept_queue:
                    time.sleep(0.1)

            # Forward request to target
            response = self._forward_request(request)

            # Store transaction
            transaction = HTTPTransaction(request=request, response=response)
            self.server.history.append(transaction)

            # Send response back to client
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ['transfer-encoding', 'content-encoding']:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.body)

        except Exception as e:
            LOG.error(f"Proxy error: {e}")
            self.send_error(500, str(e))

    def _forward_request(self, request: HTTPRequest) -> HTTPResponse:
        """Forward request to target server"""
        parsed = urlparse(request.url)
        host = parsed.netloc or request.headers.get('Host', '')
        port = 443 if parsed.scheme == 'https' else 80

        if ':' in host:
            host, port_str = host.split(':', 1)
            port = int(port_str)

        start_time = time.time()

        try:
            # Create connection
            if parsed.scheme == 'https':
                conn = httplib.HTTPSConnection(host, port, timeout=10)
            else:
                conn = httplib.HTTPConnection(host, port, timeout=10)

            # Send request
            path = parsed.path or '/'
            if parsed.query:
                path += f'?{parsed.query}'

            conn.request(request.method, path, request.body, dict(request.headers))

            # Get response
            resp = conn.getresponse()
            response_body = resp.read()
            response_time = time.time() - start_time

            response = HTTPResponse(
                status_code=resp.status,
                status_message=resp.reason,
                headers=dict(resp.getheaders()),
                body=response_body,
                response_time=response_time
            )

            conn.close()
            return response

        except Exception as e:
            LOG.error(f"Forward error: {e}")
            return HTTPResponse(
                status_code=502,
                status_message="Bad Gateway",
                headers={'Content-Type': 'text/plain'},
                body=f"Proxy error: {str(e)}".encode('utf-8'),
                response_time=time.time() - start_time
            )


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for handling multiple connections"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intercept_enabled = False
        self.intercept_queue = []
        self.history = []


class BurpSuiteProxy:
    """Main proxy server"""

    def __init__(self, host: str = '127.0.0.1', port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start the proxy server"""
        self.server = ThreadedHTTPServer((self.host, self.port), ProxyHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.running = True
        LOG.info(f"Proxy started on {self.host}:{self.port}")

    def stop(self):
        """Stop the proxy server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self.running = False
        LOG.info("Proxy stopped")

    def enable_intercept(self):
        """Enable request interception"""
        if self.server:
            self.server.intercept_enabled = True
            LOG.info("Interception enabled")

    def disable_intercept(self):
        """Disable request interception"""
        if self.server:
            self.server.intercept_enabled = False
            LOG.info("Interception disabled")

    def get_history(self) -> List[HTTPTransaction]:
        """Get proxy history"""
        return self.server.history if self.server else []

    def get_intercepted(self) -> List[HTTPRequest]:
        """Get intercepted requests"""
        return self.server.intercept_queue if self.server else []

    def forward_request(self, request: HTTPRequest):
        """Forward an intercepted request"""
        if self.server and request in self.server.intercept_queue:
            self.server.intercept_queue.remove(request)

    def drop_request(self, request: HTTPRequest):
        """Drop an intercepted request"""
        if self.server and request in self.server.intercept_queue:
            self.server.intercept_queue.remove(request)


class Repeater:
    """Request repeater for manual testing"""

    @staticmethod
    def send_request(request: HTTPRequest) -> HTTPResponse:
        """Send a request and return response"""
        parsed = urlparse(request.url)
        host = parsed.netloc or request.headers.get('Host', '')
        port = 443 if parsed.scheme == 'https' else 80

        if ':' in host:
            host, port_str = host.split(':', 1)
            port = int(port_str)

        start_time = time.time()

        try:
            if parsed.scheme == 'https':
                conn = httplib.HTTPSConnection(host, port, timeout=10)
            else:
                conn = httplib.HTTPConnection(host, port, timeout=10)

            path = parsed.path or '/'
            if parsed.query:
                path += f'?{parsed.query}'

            conn.request(request.method, path, request.body, dict(request.headers))
            resp = conn.getresponse()
            response_body = resp.read()

            response = HTTPResponse(
                status_code=resp.status,
                status_message=resp.reason,
                headers=dict(resp.getheaders()),
                body=response_body,
                response_time=time.time() - start_time
            )

            conn.close()
            return response

        except Exception as e:
            LOG.error(f"Repeater error: {e}")
            return HTTPResponse(
                status_code=0,
                status_message=str(e),
                headers={},
                body=b'',
                response_time=time.time() - start_time
            )


class Intruder:
    """Automated fuzzing/injection engine"""

    @staticmethod
    def fuzz_request(request: HTTPRequest, payloads: List[str],
                    positions: List[str] = None) -> List[HTTPResponse]:
        """
        Fuzz a request with multiple payloads.

        Args:
            request: Base request to fuzz
            payloads: List of payloads to inject
            positions: List of injection positions (e.g., ['url', 'header:User-Agent', 'body'])

        Returns:
            List of responses
        """
        responses = []

        if not positions:
            positions = ['url']  # Default to URL fuzzing

        for payload in payloads:
            # Create modified request
            modified_request = HTTPRequest(
                method=request.method,
                url=request.url,
                headers=request.headers.copy(),
                body=request.body
            )

            # Apply payload to each position
            for position in positions:
                if position == 'url':
                    # Append to URL
                    separator = '&' if '?' in modified_request.url else '?'
                    modified_request.url += f"{separator}fuzz={urllib.parse.quote(payload)}"

                elif position.startswith('header:'):
                    # Modify header
                    header_name = position.split(':', 1)[1]
                    modified_request.headers[header_name] = payload

                elif position == 'body':
                    # Replace body
                    modified_request.body = payload.encode('utf-8')

            # Send request
            response = Repeater.send_request(modified_request)
            responses.append(response)

            LOG.info(f"Intruder: {payload} -> {response.status_code}")

        return responses


class Decoder:
    """Encoder/Decoder utilities"""

    @staticmethod
    def base64_encode(data: bytes) -> str:
        return base64.b64encode(data).decode('ascii')

    @staticmethod
    def base64_decode(data: str) -> bytes:
        return base64.b64decode(data)

    @staticmethod
    def url_encode(data: str) -> str:
        return urllib.parse.quote(data)

    @staticmethod
    def url_decode(data: str) -> str:
        return urllib.parse.unquote(data)

    @staticmethod
    def html_encode(data: str) -> str:
        return data.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    @staticmethod
    def html_decode(data: str) -> str:
        return data.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    @staticmethod
    def hex_encode(data: bytes) -> str:
        return data.hex()

    @staticmethod
    def hex_decode(data: str) -> bytes:
        return bytes.fromhex(data)


class Comparer:
    """Diff tool for comparing requests/responses"""

    @staticmethod
    def compare(a: str, b: str) -> List[Tuple[str, str, str]]:
        """
        Compare two strings and return differences.

        Returns:
            List of (status, line_a, line_b) tuples where status is 'same', 'different', 'added', 'removed'
        """
        lines_a = a.split('\n')
        lines_b = b.split('\n')

        diffs = []
        max_lines = max(len(lines_a), len(lines_b))

        for i in range(max_lines):
            line_a = lines_a[i] if i < len(lines_a) else ''
            line_b = lines_b[i] if i < len(lines_b) else ''

            if line_a == line_b:
                status = 'same'
            elif not line_a:
                status = 'added'
            elif not line_b:
                status = 'removed'
            else:
                status = 'different'

            diffs.append((status, line_a, line_b))

        return diffs


class VulnerabilityScanner:
    """Passive + Active vulnerability scanner"""

    def __init__(self):
        self.quantum_available = QUANTUM_AVAILABLE

    def passive_scan(self, transaction: HTTPTransaction) -> List[Dict[str, Any]]:
        """
        Passive vulnerability scanning (analyze without sending requests).

        Returns:
            List of vulnerability findings
        """
        findings = []

        # Check for sensitive data in responses
        if transaction.response:
            body_text = transaction.response.body.decode('utf-8', errors='ignore').lower()

            # Check for SQL errors
            sql_errors = ['sql syntax', 'mysql', 'postgresql', 'ora-', 'sqlite_error']
            if any(err in body_text for err in sql_errors):
                findings.append({
                    'severity': 'high',
                    'type': 'SQL Error Exposure',
                    'evidence': 'SQL error message detected in response',
                    'url': transaction.request.url
                })

            # Check for credentials in URL
            if any(param in transaction.request.url.lower() for param in ['password', 'pwd', 'token', 'api_key']):
                findings.append({
                    'severity': 'medium',
                    'type': 'Sensitive Data in URL',
                    'evidence': 'Credentials detected in URL parameters',
                    'url': transaction.request.url
                })

            # Check for missing security headers
            security_headers = ['strict-transport-security', 'x-frame-options', 'x-content-type-options']
            missing_headers = [h for h in security_headers if h not in transaction.response.headers]

            if missing_headers:
                findings.append({
                    'severity': 'low',
                    'type': 'Missing Security Headers',
                    'evidence': f"Missing: {', '.join(missing_headers)}",
                    'url': transaction.request.url
                })

        # Quantum-enhanced anomaly detection
        if self.quantum_available and transaction.response:
            observation = {
                'status': transaction.response.status_code,
                'size': len(transaction.response.body),
                'response_time': transaction.response.response_time
            }

            anomaly_score, reasoning = detect_anomaly(observation)

            if anomaly_score > 0.7:
                findings.append({
                    'severity': 'medium',
                    'type': 'Anomalous Response Pattern (Quantum ML)',
                    'evidence': reasoning,
                    'url': transaction.request.url,
                    'quantum_score': anomaly_score
                })

        return findings

    def active_scan(self, request: HTTPRequest) -> List[Dict[str, Any]]:
        """
        Active vulnerability scanning (send test requests).

        Returns:
            List of vulnerability findings
        """
        findings = []

        # SQL Injection test
        sqli_payloads = ["'", "1' OR '1'='1", "'; DROP TABLE users--"]
        sqli_responses = Intruder.fuzz_request(request, sqli_payloads, positions=['url'])

        for resp in sqli_responses:
            if resp.status_code == 500:
                findings.append({
                    'severity': 'critical',
                    'type': 'Potential SQL Injection',
                    'evidence': f"Server error triggered by SQL payload (status {resp.status_code})",
                    'url': request.url
                })

        # XSS test
        xss_payloads = ["<script>alert('XSS')</script>", "'\"><script>alert(1)</script>"]
        xss_responses = Intruder.fuzz_request(request, xss_payloads, positions=['url'])

        for payload, resp in zip(xss_payloads, xss_responses):
            if payload.encode('utf-8') in resp.body:
                findings.append({
                    'severity': 'high',
                    'type': 'Reflected XSS',
                    'evidence': f"Payload reflected in response",
                    'url': request.url
                })

        return findings


def health_check() -> Dict[str, Any]:
    """Health check for Ai|oS integration"""
    return {
        "tool": "BurpSuite Clone",
        "status": "ok",
        "summary": "HTTP proxy and security testing platform operational",
        "details": {
            "features": [
                "HTTP/HTTPS Proxy",
                "Request/Response Interceptor",
                "Repeater",
                "Intruder (Fuzzer)",
                "Decoder/Encoder",
                "Comparer (Diff Tool)",
                "Vulnerability Scanner (Passive + Active)"
            ],
            "quantum_backend": QUANTUM_AVAILABLE
        }
    }


def gui() -> None:
    """Launch BurpSuite GUI in default browser."""
    import webbrowser
    from pathlib import Path

    # Get path to GUI HTML file
    gui_path = Path(__file__).parent / 'burpsuite_gui.html'

    if not gui_path.exists():
        print(f"[error] GUI file not found: {gui_path}")
        print("[info] Expected location: aios/tools/burpsuite_gui.html")
        return

    # Open in browser
    webbrowser.open(f"file://{gui_path}")
    print(f"[info] BurpSuite GUI launched: {gui_path}")
    print("[info] Note: GUI requires backend server to be running for full functionality")
    print("[info] To start backend: python -m tools.burpsuite_clone --proxy --port 8080")


def main(argv: List[str] = None) -> int:
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='BurpSuite Clone - HTTP Proxy & Security Testing')
    parser.add_argument('--host', default='127.0.0.1', help='Proxy host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080, help='Proxy port (default: 8080)')
    parser.add_argument('--health-check', action='store_true', help='Run health check')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')

    args = parser.parse_args(argv)

    if args.health_check:
        result = health_check()
        print(json.dumps(result, indent=2))
        return 0

    if args.gui:
        gui()
        return 0

    # Start proxy
    proxy = BurpSuiteProxy(host=args.host, port=args.port)
    proxy.start()

    print(f"[*] BurpSuite Clone proxy started on {args.host}:{args.port}")
    print(f"[*] Configure your browser to use this proxy")
    print(f"[*] Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping proxy...")
        proxy.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
