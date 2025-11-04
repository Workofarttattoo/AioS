# Security Tools API - Deployment Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
cd /Users/noone/aios/docs/api
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python security_tools_api.py
```

The server will start on `http://localhost:5000`

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# API info
curl http://localhost:5000/api/info

# Test SQLgps scan (with API key)
curl -X POST http://localhost:5000/api/sqlgps/scan \
  -H "X-API-Key: demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost/test", "techniques": ["boolean"]}'
```

---

## Production Deployment

### Option 1: Gunicorn (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:5000 security_tools_api:app

# With logging
gunicorn -w 4 -b 0.0.0.0:5000 \
  --access-logfile access.log \
  --error-logfile error.log \
  security_tools_api:app
```

### Option 2: Nginx + Gunicorn

**1. Create systemd service** (`/etc/systemd/system/security-api.service`):

```ini
[Unit]
Description=Security Tools API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/security-api
Environment="PATH=/var/www/security-api/venv/bin"
ExecStart=/var/www/security-api/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 security_tools_api:app

[Install]
WantedBy=multi-user.target
```

**2. Configure Nginx** (`/etc/nginx/sites-available/security-api`):

```nginx
server {
    listen 80;
    server_name api.thegavl.com;

    location /api {
        proxy_pass http://127.0.0.1:5000/api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # CORS headers
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "X-API-Key, Content-Type";

        if ($request_method = OPTIONS) {
            return 204;
        }
    }
}
```

**3. Enable and start**:

```bash
sudo systemctl enable security-api
sudo systemctl start security-api

sudo ln -s /etc/nginx/sites-available/security-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option 3: Docker

**Dockerfile**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY security_tools_api.py .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "security_tools_api:app"]
```

**Build and run**:

```bash
docker build -t security-tools-api .
docker run -p 5000:5000 security-tools-api
```

**Docker Compose** (`docker-compose.yml`):

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
```

---

## Environment Configuration

### Production Environment Variables

```bash
# Flask environment
export FLASK_ENV=production
export FLASK_DEBUG=0

# API keys (load from secure file)
export API_KEYS_FILE=/etc/security-api/api_keys.json

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/security-api/api.log

# Rate limiting
export DEFAULT_RATE_LIMIT=50  # requests per minute

# CORS settings
export ALLOWED_ORIGINS=https://thegavl.com,https://www.thegavl.com
```

### API Keys Management

Create `/etc/security-api/api_keys.json`:

```json
{
  "sk_live_abc123def456": {
    "user": "user@example.com",
    "tier": "pro",
    "rate_limit": 500,
    "tools_allowed": ["sqlgps", "hashsolver", "nmap", "belchstudio"]
  },
  "sk_trial_xyz789": {
    "user": "trial@example.com",
    "tier": "trial",
    "rate_limit": 50,
    "tools_allowed": ["sqlgps", "hashsolver"]
  }
}
```

**Load in Python**:

```python
import json
import os

api_keys_file = os.getenv('API_KEYS_FILE', 'api_keys.json')
with open(api_keys_file, 'r') as f:
    API_KEYS = json.load(f)
```

---

## Security Hardening

### 1. HTTPS Only

Use Let's Encrypt for SSL:

```bash
sudo certbot --nginx -d api.thegavl.com
```

Update Nginx to redirect HTTP to HTTPS:

```nginx
server {
    listen 80;
    server_name api.thegavl.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name api.thegavl.com;

    ssl_certificate /etc/letsencrypt/live/api.thegavl.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.thegavl.com/privkey.pem;

    location /api {
        # ... existing proxy config
    }
}
```

### 2. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### 3. Rate Limiting (Nginx)

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;

location /api {
    limit_req zone=api_limit burst=5;
    # ... existing config
}
```

### 4. IP Whitelisting (Optional)

For sensitive deployments, whitelist IPs:

```nginx
location /api {
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;
    # ... existing config
}
```

---

## Monitoring & Logging

### Application Logging

Update `security_tools_api.py` to add file logging:

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('api.log', maxBytes=10000000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
```

### Request Logging

Log all API requests:

```python
@app.before_request
def log_request():
    app.logger.info(f'{request.method} {request.path} - {request.remote_addr}')
```

### Audit Logging

Track security-sensitive operations:

```python
def audit_log(user, action, target, result):
    app.logger.warning(f'AUDIT: {user} performed {action} on {target}: {result}')
```

### Monitoring Tools

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Sentry**: Error tracking
- **ELK Stack**: Log aggregation

---

## Performance Optimization

### 1. Caching

Add Redis caching for repeated queries:

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@cache.memoize(timeout=300)
def expensive_operation(param):
    # ... operation
    return result
```

### 2. Database Connection Pooling

If using a database for API keys:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/security_api',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 3. Async Processing

For long-running scans, use Celery:

```python
from celery import Celery

celery = Celery('security_tools', broker='redis://localhost:6379/0')

@celery.task
def async_network_scan(target, ports):
    # Perform actual scan
    return results
```

---

## Testing

### Unit Tests

```python
import unittest
from security_tools_api import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.headers = {'X-API-Key': 'demo_key_12345'}

    def test_health_check(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')

    def test_sqlgps_scan(self):
        response = self.app.post('/api/sqlgps/scan',
            json={'url': 'http://localhost/test', 'techniques': ['boolean']},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])

if __name__ == '__main__':
    unittest.main()
```

### Load Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -H "X-API-Key: demo_key_12345" \
  http://localhost:5000/api/health
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 5000
lsof -ti:5000

# Kill process
kill -9 $(lsof -ti:5000)
```

### Permission Denied

```bash
# Fix file permissions
chmod +x security_tools_api.py
chown -R www-data:www-data /var/www/security-api
```

### CORS Issues

Ensure CORS headers are set correctly:

```python
from flask_cors import CORS

# Allow specific origins
CORS(app, origins=['https://thegavl.com'])
```

### Rate Limit Not Working

Check that rate limit store is persisting:

```python
# Use Redis for distributed rate limiting
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
```

---

## Maintenance

### Backup API Keys

```bash
# Backup API keys file
cp /etc/security-api/api_keys.json /backup/api_keys.$(date +%Y%m%d).json
```

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/security-api
```

```
/var/log/security-api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload security-api
    endscript
}
```

### Update Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update
pip install --upgrade flask flask-cors gunicorn
```

---

## Support & Documentation

- **API Documentation**: See `API_DOCUMENTATION.md`
- **API Info Endpoint**: `GET /api/info`
- **Health Check**: `GET /api/health`

For issues or questions:
- Email: support@thegavl.com
- GitHub: https://github.com/thegavl/security-tools

---

## License

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

Authorized use only. See Terms of Service and Acceptable Use Policy.
