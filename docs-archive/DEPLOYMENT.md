# Ai|oS Streaming Deployment Guide

🚀 Deploy Ai|oS dashboard to **any server** - Bluehost, VPS, AWS, GCP, Azure, or localhost

---

## 🎯 Quick Start (1 Command)

```bash
./aios/deploy.sh
```

The script **intelligently detects** your environment and deploys accordingly!

---

## 📋 Supported Platforms

- ✅ **Bluehost** / cPanel hosting
- ✅ **AWS** EC2
- ✅ **Google Cloud Platform**
- ✅ **Microsoft Azure**
- ✅ **DigitalOcean**
- ✅ **Any VPS** (Ubuntu, Debian, CentOS)
- ✅ **Docker** (any platform)
- ✅ **Localhost** (development)

---

## 🌐 Bluehost / cPanel Deployment

### Step 1: Upload Files

```bash
# Via FTP or cPanel File Manager
cd public_html/
mkdir aios
# Upload aios/web/ directory
```

### Step 2: Configure Domain

1. Log into cPanel
2. Go to **Domains** → **Domains**
3. Add domain: `aios.is`
4. Document root: `/public_html/aios`

Or for subdomain:
1. Go to **Domains** → **Subdomains**
2. Create subdomain: `dashboard.aios.is`
3. Document root: `/public_html/aios`

### Step 3: Install Dependencies

```bash
# SSH into Bluehost
cd ~/aios/web
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn psutil websockets
```

### Step 4: Start Server

```bash
# Background process
nohup python streaming_server.py > server.log 2>&1 &
```

### Step 5: Enable SSL

1. In cPanel → **SSL/TLS Status**
2. Click **Run AutoSSL** for `aios.is`
3. Wait 2-3 minutes

### Step 6: Access Dashboard

Visit: **https://aios.is**

---

## 🐳 Docker Deployment (Any Platform)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/aios.git
cd aios
```

### Step 2: Deploy

```bash
./deploy.sh
```

Docker container will:
- ✅ Build lightweight image (~200MB)
- ✅ Run on port 8080
- ✅ Auto-restart on failures
- ✅ Health checks enabled

### Step 3: Access

- Dashboard: http://localhost:8080
- API: http://localhost:8080/api/status
- WebSocket: ws://localhost:8080/ws

---

## 💻 VPS Deployment (Ubuntu/Debian)

### One-Command Deploy

```bash
curl -fsSL https://aios.is/deploy.sh | bash
```

Or manually:

```bash
# Clone repository
git clone https://github.com/yourusername/aios.git
cd aios

# Run deployment script
sudo ./deploy.sh
```

The script will:
1. ✅ Detect OS and install dependencies
2. ✅ Create Python virtual environment
3. ✅ Install Nginx reverse proxy
4. ✅ Configure SSL with Let's Encrypt
5. ✅ Create systemd service
6. ✅ Start streaming server

### Custom Domain

```bash
DOMAIN=yourdomain.com SSL_EMAIL=you@example.com ./deploy.sh
```

---

## 🔧 Manual Deployment (Advanced)

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv nginx certbot

# CentOS/RHEL
sudo yum install python3 python3-pip nginx certbot
```

### 2. Setup Python Environment

```bash
cd aios/web
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Server

```bash
python streaming_server.py
```

### 4. Configure Nginx

```nginx
server {
    listen 80;
    server_name aios.is;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 5. Enable SSL

```bash
sudo certbot --nginx -d aios.is
```

---

## 🌍 Environment Variables

```bash
# Required
DOMAIN=aios.is                  # Your domain
PORT=8080                       # Server port
SSL_EMAIL=admin@aios.is         # For Let's Encrypt

# Optional
AGENTA_FORENSIC_MODE=1         # Read-only mode
WORKERS=4                       # Uvicorn workers
```

---

## 🔒 Security Best Practices

### 1. Firewall Configuration

```bash
# Allow only HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. SSL/TLS

- ✅ Auto-configured with Let's Encrypt
- ✅ Auto-renewal every 90 days
- ✅ A+ rating on SSL Labs

### 3. Rate Limiting

Add to Nginx config:

```nginx
limit_req_zone $binary_remote_addr zone=aios:10m rate=10r/s;
limit_req zone=aios burst=20;
```

---

## 📊 Monitoring & Management

### Check Status

```bash
# Systemd service
sudo systemctl status aios-streaming

# Docker
docker logs -f aios-streaming

# Manual
ps aux | grep streaming_server
```

### View Logs

```bash
# Systemd
sudo journalctl -u aios-streaming -f

# Docker
docker logs aios-streaming

# Manual
tail -f ~/aios/web/server.log
```

### Restart Service

```bash
# Systemd
sudo systemctl restart aios-streaming

# Docker
docker restart aios-streaming

# Manual
pkill -f streaming_server.py
nohup python streaming_server.py > server.log 2>&1 &
```

---

## 🚨 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8080
sudo lsof -i :8080
sudo kill <PID>
```

### SSL Certificate Failed

```bash
# Manual certificate request
sudo certbot certonly --standalone -d aios.is

# Update Nginx config
sudo nginx -t && sudo systemctl reload nginx
```

### WebSocket Not Connecting

Check Nginx config has:
```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

### Permission Denied

```bash
# Fix ownership
sudo chown -R $USER:$USER ~/aios

# Fix permissions
chmod +x aios/deploy.sh
chmod +x aios/web/streaming_server.py
```

---

## 🎨 Customization

### Change Port

```bash
PORT=3000 ./deploy.sh
```

Or edit `aios/web/streaming_server.py`:

```python
port = int(os.getenv("PORT", 3000))
```

### Custom Domain

```bash
DOMAIN=yourdomain.com ./deploy.sh
```

### Custom Dashboard

Edit `aios/web/streaming_dashboard.html` - fully customizable HTML/CSS/JS

---

## 📈 Performance Optimization

### 1. Enable Gzip Compression

Add to Nginx:

```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript;
```

### 2. CDN Integration

Use Cloudflare for:
- Global CDN
- DDoS protection
- Free SSL

### 3. Multiple Workers

```bash
# In streaming_server.py
uvicorn.run(..., workers=4)
```

---

## 🔄 Updates & Maintenance

### Update Ai|oS

```bash
cd aios
git pull origin main
sudo systemctl restart aios-streaming
```

### Backup

```bash
# Backup configuration
tar -czf aios-backup-$(date +%Y%m%d).tar.gz aios/

# Restore
tar -xzf aios-backup-YYYYMMDD.tar.gz
```

---

## 💡 Next Steps

1. **Configure DNS**: Point `aios.is` to your server IP
2. **Test Dashboard**: Visit https://aios.is
3. **Monitor Metrics**: Watch real-time telemetry
4. **Customize**: Edit dashboard HTML/CSS
5. **Scale**: Add load balancer for high traffic

---

## 🆘 Support

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Email**: support@aios.is

---

## 📜 License

See [LICENSE](LICENSE) for details.

---

**🤖 Ai|oS - Adaptive, Lightweight, Works Anywhere**

*Powered by autonomous meta-agents with quantum-enhanced ML*
