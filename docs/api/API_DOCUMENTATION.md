# Security Tools API Documentation

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

REST API backend for TheGAVL Red Team Security Tools with authentication, rate limiting, and defensive safeguards.

## Base URL

```
http://localhost:5000/api
```

## Authentication

All endpoints (except `/health` and `/info`) require API key authentication via the `X-API-Key` header.

**Demo API Key**: `demo_key_12345`

### Example Request

```bash
curl -X POST http://localhost:5000/api/sqlgps/scan \
  -H "X-API-Key: demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost/test", "techniques": ["boolean", "time"]}'
```

## Rate Limiting

- **Demo Tier**: 10 requests/minute
- **Trial Tier**: 50 requests/minute
- **Pro Tier**: 500 requests/minute

Rate limit exceeded returns HTTP 429 with `retry_after` seconds.

## Endpoints

### Health & Info

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T12:00:00.000Z"
}
```

#### `GET /api/info`
API information and available endpoints.

**Response:**
```json
{
  "name": "TheGAVL Security Tools API",
  "version": "1.0.0",
  "endpoints": {
    "SQLgps": ["/api/sqlgps/scan", "/api/sqlgps/enumerate"],
    "HashSolver": ["/api/hashsolver/crack", "/api/hashsolver/identify"],
    "NMAP": ["/api/nmap/scan", "/api/nmap/service-detect"],
    "BelchStudio": ["/api/belchstudio/intercept"]
  }
}
```

---

### SQLgps - SQL Injection Testing

#### `POST /api/sqlgps/scan`
Scan URL for SQL injection vulnerabilities.

**⚠️ AUTHORIZED USE ONLY** - Only test your own applications.

**Request Body:**
```json
{
  "url": "http://localhost/test.php?id=1",
  "techniques": ["boolean", "time", "union", "error", "stacked"]
}
```

**Response:**
```json
{
  "success": true,
  "target": "http://localhost/test.php?id=1",
  "results": [
    {
      "technique": "boolean",
      "vulnerable": false,
      "payloads_tested": 5,
      "response_time": 0.234,
      "message": "boolean scan completed (simulation)"
    }
  ],
  "scan_time": "2025-01-15T12:00:00.000Z",
  "warning": "This is a simulation. Real scanning requires authorization."
}
```

**Authorized Targets:**
- `localhost`
- `127.0.0.1`
- `*.test` domains
- `test.local`

#### `POST /api/sqlgps/enumerate`
Enumerate database structure (authorized testing only).

**Request Body:**
```json
{
  "target": "http://localhost/app",
  "database": "test_db"
}
```

**Response:**
```json
{
  "success": true,
  "databases": ["demo_db", "test_db"],
  "tables": ["users", "products", "orders"],
  "columns": ["id", "username", "email", "password_hash"],
  "warning": "Simulation only. Real enumeration requires authorization."
}
```

---

### HashSolver - Password Hash Cracking

#### `POST /api/hashsolver/crack`
Attempt to crack password hash.

**⚠️ DEFENSIVE USE ONLY** - For testing your own password security.

**Request Body:**
```json
{
  "hash": "5f4dcc3b5aa765d61d8327deb882cf99",
  "algorithm": "md5",
  "attack_mode": "dictionary"
}
```

**Response:**
```json
{
  "success": true,
  "hash": "5f4dcc3b5aa765d61d8327deb882cf99",
  "algorithm": "md5",
  "attack_mode": "dictionary",
  "cracked": true,
  "plaintext": "password",
  "attempts": 5432,
  "time_elapsed": 2.34,
  "hash_rate": "850k H/s",
  "warning": "Simulation only. Real cracking for authorized testing."
}
```

**Supported Algorithms:**
- MD5
- SHA1
- SHA256
- SHA512
- bcrypt
- NTLM

**Attack Modes:**
- `dictionary` - Dictionary attack
- `brute_force` - Brute force
- `mask` - Mask attack
- `hybrid` - Hybrid attack
- `rules` - Rule-based attack

#### `POST /api/hashsolver/identify`
Identify hash algorithm based on pattern.

**Request Body:**
```json
{
  "hash": "5f4dcc3b5aa765d61d8327deb882cf99"
}
```

**Response:**
```json
{
  "success": true,
  "hash": "5f4dcc3b5aa765d61d8327deb882cf99",
  "detected_algorithm": "MD5",
  "confidence": "high"
}
```

---

### NMAP Street Edition - Network Scanning

#### `POST /api/nmap/scan`
Scan network target for open ports.

**⚠️ AUTHORIZED NETWORKS ONLY** - Only scan your own networks.

**Request Body:**
```json
{
  "target": "127.0.0.1",
  "scan_type": "syn",
  "ports": "1-1000"
}
```

**Response:**
```json
{
  "success": true,
  "target": "127.0.0.1",
  "scan_type": "syn",
  "ports_scanned": "1-1000",
  "open_ports": [
    {
      "port": 80,
      "state": "open",
      "service": "HTTP",
      "version": "Unknown",
      "cves": []
    },
    {
      "port": 443,
      "state": "open",
      "service": "HTTPS",
      "version": "Unknown",
      "cves": []
    }
  ],
  "scan_time": "2025-01-15T12:00:00.000Z",
  "warning": "Simulation only. Real scanning requires authorization."
}
```

**Authorized Targets:**
- `localhost`
- `127.0.0.1`
- Private networks: `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`

**Scan Types:**
- `syn` - SYN stealth scan
- `connect` - TCP connect scan
- `ack` - ACK scan
- `fin` - FIN scan
- `xmas` - Xmas scan
- `null` - NULL scan

#### `POST /api/nmap/service-detect`
Detect service version on specific port.

**Request Body:**
```json
{
  "target": "127.0.0.1",
  "port": 80
}
```

**Response:**
```json
{
  "success": true,
  "target": "127.0.0.1",
  "port": 80,
  "service": "HTTP",
  "version": "HTTP 2.4.1",
  "os_hint": "Linux 5.x",
  "cpes": ["cpe:/a:http:http:2.4.1"]
}
```

---

### BelchStudio - HTTP Proxy

#### `POST /api/belchstudio/intercept`
Intercept and analyze HTTP request.

**⚠️ AUTHORIZED APPLICATIONS ONLY** - Only test your own applications.

**Request Body:**
```json
{
  "url": "http://localhost/api/test",
  "method": "GET"
}
```

**Response:**
```json
{
  "success": true,
  "request": {
    "method": "GET",
    "url": "http://localhost/api/test",
    "headers": {"User-Agent": "BelchStudio/1.0"},
    "intercepted": true
  },
  "response": {
    "status": 200,
    "headers": {"Content-Type": "text/html"},
    "body_preview": "<html>...</html>",
    "size": "4096 bytes"
  },
  "warning": "Simulation only. Real interception requires authorization."
}
```

---

## Error Responses

### 401 Unauthorized
```json
{
  "success": false,
  "error": "Invalid or missing API key",
  "message": "Please provide a valid API key in X-API-Key header"
}
```

### 403 Forbidden
```json
{
  "success": false,
  "error": "Unauthorized target",
  "message": "Only localhost and authorized domains allowed"
}
```

### 429 Rate Limit Exceeded
```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "message": "Limit: 10 requests/minute",
  "retry_after": 45.2
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "Endpoint not found",
  "message": "Check /api/info for available endpoints"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal server error",
  "message": "An unexpected error occurred"
}
```

---

## Security Considerations

### Defensive Use Only
- All tools are designed for **defensive security testing**
- Only test systems you own or have **written authorization** to test
- Unauthorized scanning/testing violates CFAA (18 U.S.C. § 1030)

### Target Restrictions
The API enforces strict target validation:
- **Web targets**: Only `localhost`, `127.0.0.1`, `*.test` domains
- **Network targets**: Only localhost and private IP ranges
- Production deployments should maintain even stricter controls

### Rate Limiting
- Prevents abuse and ensures fair usage
- Varies by subscription tier
- Exceeded limits return HTTP 429 with retry timing

### Audit Logging
Production deployments should implement:
- Request logging with user, timestamp, target
- Anomaly detection for suspicious patterns
- Automatic blocking of repeated violations

---

## JavaScript Client Usage

Include the API client in your React components:

```html
<script src="js/api-client.js"></script>
```

### Example: SQLgps Scan

```javascript
const results = await window.SecurityToolsAPI.SQLgps.scan(
  'http://localhost/test.php',
  ['boolean', 'time', 'union']
);

console.log(results);
```

### Example: Hash Cracking

```javascript
const result = await window.SecurityToolsAPI.HashSolver.crack(
  '5f4dcc3b5aa765d61d8327deb882cf99',
  'md5',
  'dictionary'
);

if (result.cracked) {
  console.log(`Cracked! Plaintext: ${result.plaintext}`);
}
```

### Example: Network Scan

```javascript
const scanResults = await window.SecurityToolsAPI.NMAPStreet.scan(
  '127.0.0.1',
  'syn',
  '1-1000'
);

console.log(`Found ${scanResults.open_ports.length} open ports`);
```

---

## Deployment

### Local Development

```bash
cd /Users/noone/aios/docs/api
pip install -r requirements.txt
python security_tools_api.py
```

### Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 security_tools_api:app
```

### Environment Variables

```bash
export FLASK_ENV=production
export API_KEYS_FILE=/path/to/api_keys.json
export LOG_LEVEL=INFO
```

---

## License & Legal

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

This API is provided for authorized defensive security testing only. Unauthorized use violates:
- Computer Fraud and Abuse Act (CFAA) - 18 U.S.C. § 1030
- State computer crime laws
- Terms of Service

By using this API, you agree to:
- Only test systems you own or have written authorization to test
- Comply with all applicable laws and regulations
- Accept full responsibility for your use of these tools

**NO WARRANTY** - This software is provided "as is" without warranty of any kind.
