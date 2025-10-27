# DNS Configuration for aios.is

Complete guide to configure your aios.is domain with Namecheap DNS and deploy the AI:OS streaming dashboard.

---

## Step 1: Configure Namecheap DNS Records

### A. Login to Namecheap
1. Go to https://namecheap.com
2. Login to your account
3. Go to **Domain List** → Click **Manage** next to aios.is

### B. Access DNS Settings
1. Click **Advanced DNS** tab
2. You'll see the DNS records editor

### C. Add DNS Records

#### Option 1: Deploying to Bluehost (Recommended)

**Get your Bluehost IP address first:**
- Login to Bluehost cPanel
- Look for "Server Information" or "Shared IP Address"
- Note the IP (usually format: `123.456.789.012`)

**Add these records in Namecheap:**

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A Record | @ | `YOUR_BLUEHOST_IP` | Automatic |
| A Record | www | `YOUR_BLUEHOST_IP` | Automatic |
| CNAME | * | aios.is. | Automatic |

Example with IP 162.241.216.11:
```
A Record    @      162.241.216.11    Automatic
A Record    www    162.241.216.11    Automatic
CNAME       *      aios.is.          Automatic
```

**Purpose:**
- `@` = Root domain (aios.is)
- `www` = www.aios.is
- `*` = All subdomains (api.aios.is, dashboard.aios.is, etc.)

#### Option 2: Deploying to VPS

If you have a VPS (DigitalOcean, AWS EC2, etc.):

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A Record | @ | `YOUR_VPS_IP` | Automatic |
| A Record | www | `YOUR_VPS_IP` | Automatic |
| CNAME | * | aios.is. | Automatic |

#### Option 3: Using Cloudflare (Advanced)

For maximum performance with DDoS protection:

1. In Namecheap, change **Nameservers** to:
   - ns1.cloudflare.com
   - ns2.cloudflare.com

2. Add domain in Cloudflare and copy DNS records there
3. Cloudflare will handle SSL automatically

### D. Verify DNS Propagation

DNS changes take 5-60 minutes to propagate globally.

**Check propagation:**
```bash
# Check A record
dig +short aios.is

# Check www subdomain
dig +short www.aios.is

# Or use online tool:
# https://dnschecker.org/#A/aios.is
```

**Expected output:**
```
162.241.216.11  # Your server IP
```

---

## Step 2: Deploy to Bluehost (cPanel)

### A. Upload Files via FTP/SFTP

**FTP Credentials (from Bluehost cPanel):**
- Host: `aios.is` or `ftp.aios.is`
- Username: Your cPanel username
- Password: Your cPanel password
- Port: 21 (FTP) or 22 (SFTP)

**Upload these files:**
```
/home/username/public_html/
├── streaming_server.py
├── streaming_dashboard.html
└── requirements.txt
```

**Via command line:**
```bash
# From your local machine
cd ~/aios/web
scp streaming_server.py streaming_dashboard.html username@aios.is:~/public_html/
```

### B. Install Python Dependencies via SSH

**SSH into Bluehost:**
```bash
ssh username@aios.is
```

**Install dependencies:**
```bash
cd ~/public_html
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn psutil websockets
```

### C. Start Streaming Server

**Run in background:**
```bash
nohup python streaming_server.py > server.log 2>&1 &
```

**Check if running:**
```bash
ps aux | grep streaming_server
tail -f server.log
```

### D. Configure .htaccess (Reverse Proxy)

Create `.htaccess` in `public_html/`:

```apache
# Enable rewrite engine
RewriteEngine On

# Proxy WebSocket connections
RewriteCond %{HTTP:Upgrade} websocket [NC]
RewriteCond %{HTTP:Connection} upgrade [NC]
RewriteRule ^(.*)$ ws://localhost:8080/$1 [P,L]

# Proxy HTTP requests
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]
```

**If Apache mod_proxy is disabled:**
Contact Bluehost support to enable:
- mod_proxy
- mod_proxy_http
- mod_proxy_wstunnel

---

## Step 3: Enable SSL (HTTPS)

### Option A: Let's Encrypt (Free) - Recommended

**In Bluehost cPanel:**
1. Go to **Security** → **SSL/TLS Status**
2. Find `aios.is` in the list
3. Click **Run AutoSSL**
4. Wait 2-3 minutes for certificate installation

**Verify:**
- Visit https://aios.is (should show green padlock)

### Option B: Namecheap PositiveSSL (Purchased)

**If you bought PositiveSSL from Namecheap:**

1. **Generate CSR in cPanel:**
   - Go to **Security** → **SSL/TLS** → **Generate CSR**
   - Fill in domain info
   - Copy the CSR code

2. **Activate SSL in Namecheap:**
   - Go to Namecheap dashboard → SSL Certificates
   - Click **Activate** next to PositiveSSL
   - Paste CSR code
   - Choose domain validation method (email or DNS)

3. **Complete validation:**
   - Check email or add DNS TXT record as instructed
   - Download certificate files (CRT, CA bundle)

4. **Install in cPanel:**
   - Go to **Security** → **SSL/TLS** → **Manage SSL Sites**
   - Select domain `aios.is`
   - Paste certificate contents
   - Click **Install Certificate**

---

## Step 4: Test Deployment

### A. Test HTTP Endpoint

```bash
curl -I https://aios.is/
# Expected: HTTP/2 200 OK

curl https://aios.is/healthz
# Expected: {"status":"ok","service":"AI:OS Streaming Server"}
```

### B. Test API Endpoints

```bash
# System status
curl https://aios.is/api/status

# Sample telemetry
curl https://aios.is/api/telemetry/sample
```

### C. Test WebSocket Connection

**In browser console (https://aios.is/):**
```javascript
const ws = new WebSocket('wss://aios.is/ws');
ws.onopen = () => console.log('Connected!');
ws.onmessage = (e) => console.log('Data:', JSON.parse(e.data));
```

**Expected:** Real-time telemetry messages every 2 seconds

### D. Load Dashboard

Visit: **https://aios.is/**

You should see:
- Matrix-style terminal interface
- "AI:OS" logo in green
- Live telemetry streaming
- CPU, Memory, Disk metrics
- WebSocket connection status

---

## Step 5: Email Configuration (Optional)

### A. Create Email Forwarding

**In Namecheap (Advanced DNS):**

| Type | Host | Value | TTL |
|------|------|-------|-----|
| MX Record | @ | mx1.improvmx.com | 10 |
| MX Record | @ | mx2.improvmx.com | 20 |

**Then register at ImprovMX (free):**
1. Go to https://improvmx.com
2. Add domain: aios.is
3. Create aliases:
   - hello@aios.is → your-email@gmail.com
   - support@aios.is → your-email@gmail.com
   - admin@aios.is → your-email@gmail.com

### B. Alternative: Use Bluehost Email

**In Bluehost cPanel:**
1. Go to **Email** → **Email Accounts**
2. Create accounts:
   - hello@aios.is
   - support@aios.is
3. Configure forwarding or use webmail

---

## Step 6: Performance Optimization

### A. Enable Gzip Compression

Add to `.htaccess`:
```apache
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/html text/plain text/css application/json application/javascript
</IfModule>
```

### B. Enable Caching

Add to `.htaccess`:
```apache
<IfModule mod_expires.c>
  ExpiresActive On
  ExpiresByType text/css "access plus 1 week"
  ExpiresByType application/javascript "access plus 1 week"
  ExpiresByType text/html "access plus 1 hour"
</IfModule>
```

### C. Setup Cloudflare (Optional)

For global CDN and DDoS protection:

1. Create free Cloudflare account
2. Add aios.is domain
3. Change Namecheap nameservers to Cloudflare's
4. Enable:
   - Auto Minify (CSS, JS, HTML)
   - Brotli compression
   - Rocket Loader
   - SSL (Full Strict mode)

---

## Step 7: Monitoring & Maintenance

### A. Setup Uptime Monitoring

Use free services:
- **UptimeRobot** (https://uptimerobot.com)
  - Monitor: https://aios.is/healthz
  - Alert via email/SMS if down

- **Pingdom** (free tier)
  - Monitor from multiple locations

### B. Check Server Logs

```bash
# SSH into server
ssh username@aios.is

# View streaming server logs
tail -f ~/public_html/server.log

# View Apache error logs
tail -f ~/logs/error_log
```

### C. Restart Server if Needed

```bash
# Find process
ps aux | grep streaming_server

# Kill process
pkill -f streaming_server.py

# Restart
cd ~/public_html
source venv/bin/activate
nohup python streaming_server.py > server.log 2>&1 &
```

### D. Create systemd Service (Advanced VPS)

For auto-restart on crashes:

```bash
sudo tee /etc/systemd/system/aios-streaming.service > /dev/null << 'EOF'
[Unit]
Description=AI:OS Streaming Server
After=network.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/public_html
Environment="PATH=/home/username/public_html/venv/bin"
ExecStart=/home/username/public_html/venv/bin/python streaming_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aios-streaming
sudo systemctl start aios-streaming
```

---

## Troubleshooting

### DNS Not Resolving

**Problem:** `dig aios.is` returns nothing

**Fix:**
```bash
# Check nameservers
dig NS aios.is

# Should return Namecheap nameservers:
# dns1.registrar-servers.com
# dns2.registrar-servers.com

# If different, wait 5-60 minutes for propagation
```

### SSL Certificate Error

**Problem:** "Your connection is not private"

**Fix:**
1. Check SSL status in cPanel → SSL/TLS Status
2. Retry AutoSSL
3. Or use Cloudflare (automatic SSL)

### 502 Bad Gateway

**Problem:** Nginx/Apache can't reach backend

**Fix:**
```bash
# Check if streaming server is running
ps aux | grep streaming_server

# Check port
netstat -tuln | grep 8080

# Restart server
pkill -f streaming_server.py
nohup python streaming_server.py > server.log 2>&1 &
```

### WebSocket Connection Failed

**Problem:** Dashboard shows "Disconnected"

**Fix:**
1. Verify `.htaccess` has WebSocket proxy rules
2. Check Apache has mod_proxy_wstunnel enabled
3. Test WebSocket manually:
```javascript
// In browser console at https://aios.is/
const ws = new WebSocket('wss://aios.is/ws');
ws.onerror = (err) => console.error(err);
```

### Port 8080 Already in Use

**Problem:** Server won't start

**Fix:**
```bash
# Find process using port
lsof -i :8080

# Kill it
kill -9 <PID>

# Or change port in streaming_server.py
# Edit: port = int(os.getenv("PORT", 8081))
```

---

## Quick Reference Commands

```bash
# Check DNS
dig +short aios.is

# Test HTTPS
curl -I https://aios.is/

# Check server status
ps aux | grep streaming_server

# View logs
tail -f ~/public_html/server.log

# Restart server
pkill -f streaming_server.py && nohup python streaming_server.py > server.log 2>&1 &

# Test WebSocket (wscat tool)
npm install -g wscat
wscat -c wss://aios.is/ws
```

---

## Next Steps

Once deployment is complete:

1. **Update Branding**
   - Update all references to use https://aios.is
   - Add domain to README.md
   - Create social media links

2. **Security Hardening**
   - Setup firewall rules
   - Enable rate limiting
   - Configure fail2ban
   - Setup automatic backups

3. **Scale Infrastructure**
   - Add load balancer if traffic increases
   - Setup Redis for WebSocket pub/sub
   - Configure CDN for static assets
   - Monitor performance metrics

---

**🎉 You're Live!**

Visit: **https://aios.is**

Dashboard features:
- Real-time system telemetry
- WebSocket streaming
- CPU, Memory, Disk metrics
- Agent coordination status
- Quantum-enhanced ML metrics

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
