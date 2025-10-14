# Quick Start: Deploy Ai|oS to aios.is

Get your Ai|oS streaming dashboard live at https://aios.is in 10 minutes or less.

---

## Prerequisites

- Domain: **aios.is** (purchased from Namecheap) ✅
- Hosting: Bluehost cPanel or VPS
- SSH access to server
- 10 minutes of your time

---

## Step 1: Get Your Server IP (2 minutes)

### Option A: Bluehost/cPanel

1. Login to Bluehost cPanel
2. Look for "Server Information" in sidebar
3. Note your **Shared IP Address** (e.g., `162.241.216.11`)

### Option B: VPS (AWS, DigitalOcean, etc.)

```bash
# Get your VPS public IP
curl ifconfig.me
```

Note this IP - you'll need it for DNS.

---

## Step 2: Configure DNS at Namecheap (3 minutes)

1. **Login to Namecheap**: https://namecheap.com
2. **Go to Domain List** → Click **Manage** next to `aios.is`
3. **Click Advanced DNS** tab
4. **Add these records:**

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A Record | @ | `YOUR_SERVER_IP` | Automatic |
| A Record | www | `YOUR_SERVER_IP` | Automatic |
| CNAME | * | aios.is. | Automatic |

5. **Save changes**

**Example with Bluehost IP 162.241.216.11:**
```
A Record    @      162.241.216.11    Automatic
A Record    www    162.241.216.11    Automatic
CNAME       *      aios.is.          Automatic
```

**Verify DNS (wait 2-5 minutes):**
```bash
dig +short aios.is
# Should return your server IP
```

---

## Step 3: Upload Files to Server (2 minutes)

### Option A: Via SCP (Recommended)

```bash
# From your local machine
cd ~/aios/web
scp streaming_server.py streaming_dashboard.html username@aios.is:~/public_html/
```

### Option B: Via FTP/FileZilla

- Host: `aios.is` or `ftp.aios.is`
- Username: Your cPanel username
- Password: Your cPanel password
- Upload to: `/public_html/`

Files to upload:
- `aios/web/streaming_server.py`
- `aios/web/streaming_dashboard.html`

### Option C: Via cPanel File Manager

1. Login to Bluehost cPanel
2. Go to **File Manager**
3. Navigate to `public_html/`
4. Click **Upload** → Upload both files

---

## Step 4: Install Dependencies (1 minute)

**SSH into your server:**
```bash
ssh username@aios.is
```

**Install Python packages:**
```bash
cd ~/public_html
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn psutil websockets
```

---

## Step 5: Start the Server (1 minute)

**Run in background:**
```bash
nohup python streaming_server.py > server.log 2>&1 &
```

**Verify it's running:**
```bash
ps aux | grep streaming_server
# Should show the python process

curl http://localhost:8080/healthz
# Should return: {"status":"ok","service":"Ai|oS Streaming Server"}
```

---

## Step 6: Configure Reverse Proxy (1 minute)

### Option A: Bluehost/cPanel - .htaccess

Create `.htaccess` in `/public_html/`:

```apache
RewriteEngine On

# WebSocket support
RewriteCond %{HTTP:Upgrade} websocket [NC]
RewriteCond %{HTTP:Connection} upgrade [NC]
RewriteRule ^(.*)$ ws://localhost:8080/$1 [P,L]

# HTTP proxy
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]
```

**Via SSH:**
```bash
cat > ~/public_html/.htaccess << 'EOF'
RewriteEngine On
RewriteCond %{HTTP:Upgrade} websocket [NC]
RewriteCond %{HTTP:Connection} upgrade [NC]
RewriteRule ^(.*)$ ws://localhost:8080/$1 [P,L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]
EOF
```

### Option B: VPS - Nginx

**Automatic deployment (recommended):**
```bash
cd ~/aios
./deploy.sh
```

This will automatically:
- Install dependencies
- Configure Nginx
- Setup SSL with Let's Encrypt
- Start systemd service

---

## Step 7: Enable SSL (1 minute)

### Bluehost/cPanel - AutoSSL (Free)

1. Login to cPanel
2. Go to **Security** → **SSL/TLS Status**
3. Find `aios.is` in the list
4. Click **Run AutoSSL**
5. Wait 2-3 minutes

### VPS - Let's Encrypt

```bash
sudo certbot --nginx -d aios.is -d www.aios.is
```

Or use the deploy script which does this automatically:
```bash
cd ~/aios
DOMAIN=aios.is SSL_EMAIL=admin@aios.is ./deploy.sh
```

---

## Step 8: Test Your Deployment! ✨

### Test HTTPS Connection

```bash
curl -I https://aios.is/
# Expected: HTTP/2 200 OK
```

### Test API

```bash
curl https://aios.is/api/status
# Expected: JSON with system stats
```

### Test WebSocket (Browser Console)

Open https://aios.is/ in your browser, then press F12 and run:

```javascript
const ws = new WebSocket('wss://aios.is/ws');
ws.onopen = () => console.log('✅ WebSocket Connected!');
ws.onmessage = (e) => console.log('Data:', JSON.parse(e.data));
```

### Visit Dashboard

**Open in browser:** https://aios.is

You should see:
- ✅ Matrix-style green terminal interface
- ✅ "Ai|oS" logo
- ✅ Live streaming telemetry
- ✅ CPU, Memory, Disk metrics updating every 2 seconds
- ✅ WebSocket status: "Connected"

---

## Troubleshooting

### DNS Not Working

**Problem:** Can't reach aios.is

**Check:**
```bash
dig +short aios.is
# Should return your server IP
```

**Fix:** Wait 5-60 minutes for DNS propagation, or flush DNS cache:
```bash
# macOS
sudo dscacheutil -flushcache

# Linux
sudo systemd-resolve --flush-caches

# Windows
ipconfig /flushdns
```

### 502 Bad Gateway

**Problem:** Server not responding

**Check if running:**
```bash
ps aux | grep streaming_server
netstat -tuln | grep 8080
```

**Restart:**
```bash
pkill -f streaming_server.py
cd ~/public_html
source venv/bin/activate
nohup python streaming_server.py > server.log 2>&1 &
```

### WebSocket Not Connecting

**Problem:** Dashboard shows "Disconnected"

**Check Apache modules (Bluehost):**
- Contact Bluehost support to enable:
  - `mod_proxy`
  - `mod_proxy_http`
  - `mod_proxy_wstunnel`

**Or use alternative approach:**
```bash
# Edit streaming_server.py to use port 80 (requires sudo)
# Or use Cloudflare for WebSocket proxy (automatic)
```

### SSL Certificate Issues

**Problem:** "Your connection is not private"

**Fix (Bluehost):**
1. cPanel → SSL/TLS Status
2. Click "Run AutoSSL" again
3. Wait 2-3 minutes

**Fix (VPS):**
```bash
sudo certbot --nginx -d aios.is --force-renewal
```

### Permission Denied

**Problem:** Can't write to public_html

**Fix:**
```bash
sudo chown -R $USER:$USER ~/public_html
chmod 755 ~/public_html
```

---

## Management Commands

### Check Server Status

```bash
# Check if running
ps aux | grep streaming_server

# Check logs
tail -f ~/public_html/server.log
```

### Restart Server

```bash
# Kill existing
pkill -f streaming_server.py

# Restart
cd ~/public_html
source venv/bin/activate
nohup python streaming_server.py > server.log 2>&1 &
```

### Update Code

```bash
# Upload new files via SCP
cd ~/aios/web
scp streaming_server.py streaming_dashboard.html username@aios.is:~/public_html/

# Restart server
ssh username@aios.is "pkill -f streaming_server.py && cd public_html && source venv/bin/activate && nohup python streaming_server.py > server.log 2>&1 &"
```

---

## Performance Optimization (Optional)

### Enable Cloudflare (Recommended)

For global CDN, DDoS protection, and automatic SSL:

1. Create free Cloudflare account: https://cloudflare.com
2. Add domain: `aios.is`
3. Change Namecheap nameservers to Cloudflare's (shown in dashboard)
4. In Cloudflare:
   - Enable SSL (Full Strict)
   - Enable Auto Minify (CSS, JS, HTML)
   - Enable Brotli compression
   - Enable WebSocket support (automatic)

**Benefits:**
- 🚀 Global CDN (faster load times)
- 🛡️ DDoS protection
- 🔒 Automatic SSL
- ⚡ Edge caching
- 📊 Analytics

### Enable Gzip Compression

Add to `.htaccess`:
```apache
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/html text/plain text/css application/json application/javascript
</IfModule>
```

### Enable Caching

Add to `.htaccess`:
```apache
<IfModule mod_expires.c>
  ExpiresActive On
  ExpiresByType text/css "access plus 1 week"
  ExpiresByType application/javascript "access plus 1 week"
  ExpiresByType text/html "access plus 1 hour"
</IfModule>
```

---

## Next Steps

Now that you're live, consider:

1. **Setup Email Forwarding**
   - hello@aios.is
   - support@aios.is
   - See [DNS_SETUP.md](DNS_SETUP.md) for instructions

2. **Setup Monitoring**
   - UptimeRobot (free): Monitor https://aios.is/healthz
   - Get alerts if site goes down

3. **Customize Dashboard**
   - Edit `streaming_dashboard.html`
   - Add your own branding
   - Modify metrics displayed

4. **Scale Infrastructure**
   - Add more servers
   - Setup load balancer
   - Use Redis for pub/sub

5. **Security Hardening**
   - Setup firewall rules
   - Enable rate limiting
   - Configure fail2ban
   - Regular backups

---

## Resources

- **Full DNS Guide**: [DNS_SETUP.md](DNS_SETUP.md)
- **Deployment Options**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Developer Guide**: [CLAUDE.md](../CLAUDE.md)
- **Live Dashboard**: https://aios.is

---

## Support

- **Email**: admin@aios.is
- **Support**: support@aios.is
- **Issues**: GitHub Issues

---

**🎉 Congratulations! Your Ai|oS dashboard is live at https://aios.is**

The streaming server is now broadcasting real-time telemetry from your autonomous meta-agents. Welcome to the future of agentic computing!

**Pronunciation**: Just say "aios" (rhymes with "iOS")

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
