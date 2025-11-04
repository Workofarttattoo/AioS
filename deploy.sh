#!/bin/bash
###############################################################################
# AI:OS Adaptive Deployment Script
# Works on: Bluehost, AWS, GCP, Azure, DigitalOcean, any VPS, localhost
# Intelligently detects environment and adapts deployment strategy
###############################################################################

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "    ▄▄▄       ██▓    ▒█████    ██████  "
echo "   ▒████▄    ▓██▒   ▒██▒  ██▒▒██    ▒ "
echo "   ▒██  ▀█▄  ▒██▒   ▒██░  ██▒░ ▓██▄   "
echo "   ░██▄▄▄▄██ ░██░   ▒██   ██░  ▒   ██▒"
echo "    ▓█   ▓██▒░██░   ░ ████▓▒░▒██████▒▒"
echo "    ▒▒   ▓▒█░░▓     ░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░"
echo "     ▒   ▒▒ ░ ▒ ░     ░ ▒ ▒░ ░ ░▒  ░ ░"
echo "     ░   ▒    ▒ ░   ░ ░ ░ ▒  ░  ░  ░  "
echo "         ░  ░ ░         ░ ░        ░  "
echo -e "${NC}"
echo -e "${BLUE}🤖 AI:OS Adaptive Deployment Script${NC}"
echo -e "${BLUE}═══════════════════════════════════${NC}\n"

# Configuration
DOMAIN="${DOMAIN:-aios.is}"
PORT="${PORT:-8080}"
SSL_EMAIL="${SSL_EMAIL:-admin@aios.is}"

###############################################################################
# Environment Detection
###############################################################################

detect_environment() {
    echo -e "${YELLOW}🔍 Detecting environment...${NC}"

    # Check for cloud providers
    if [ -f /sys/hypervisor/uuid ] && grep -q "ec2" /sys/hypervisor/uuid 2>/dev/null; then
        ENV_TYPE="aws"
    elif [ -f /sys/class/dmi/id/product_name ] && grep -q "Google" /sys/class/dmi/id/product_name 2>/dev/null; then
        ENV_TYPE="gcp"
    elif [ -f /sys/class/dmi/id/sys_vendor ] && grep -q "Microsoft Corporation" /sys/class/dmi/id/sys_vendor 2>/dev/null; then
        ENV_TYPE="azure"
    elif [ -f /etc/digitalocean ]; then
        ENV_TYPE="digitalocean"
    elif [ -d /usr/local/cpanel ]; then
        ENV_TYPE="cpanel"  # Bluehost/cPanel
    elif command -v docker &> /dev/null && docker info &> /dev/null; then
        ENV_TYPE="docker"
    else
        ENV_TYPE="vps"
    fi

    echo -e "${GREEN}✓ Environment detected: ${ENV_TYPE}${NC}\n"
}

###############################################################################
# Dependency Installation
###############################################################################

install_dependencies() {
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip nginx certbot python3-certbot-nginx
    elif command -v brew &> /dev/null; then
        brew install python nginx
    else
        echo -e "${RED}⚠ Package manager not detected. Please install Python 3.8+, pip, and nginx manually.${NC}"
        exit 1
    fi

    # Install Python dependencies
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet fastapi uvicorn psutil websockets

    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
}

###############################################################################
# Docker Deployment
###############################################################################

deploy_docker() {
    echo -e "${YELLOW}🐳 Deploying with Docker...${NC}"

    # Create Dockerfile if not exists
    cat > Dockerfile.streaming << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn psutil websockets

# Copy files
COPY aios/web/streaming_server.py /app/
COPY aios/web/streaming_dashboard.html /app/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"

CMD ["python", "streaming_server.py"]
EOF

    # Build and run
    docker build -f Dockerfile.streaming -t aios-streaming:latest .
    docker stop aios-streaming 2>/dev/null || true
    docker rm aios-streaming 2>/dev/null || true
    docker run -d \
        --name aios-streaming \
        --restart unless-stopped \
        -p ${PORT}:8080 \
        -e PORT=8080 \
        aios-streaming:latest

    echo -e "${GREEN}✓ Docker container running on port ${PORT}${NC}\n"
}

###############################################################################
# Native Deployment (VPS/Bluehost)
###############################################################################

deploy_native() {
    echo -e "${YELLOW}🚀 Deploying natively...${NC}"

    # Create systemd service
    sudo tee /etc/systemd/system/aios-streaming.service > /dev/null << EOF
[Unit]
Description=AI:OS Streaming Server
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=$(pwd)/aios/web
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python streaming_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Start service
    sudo systemctl daemon-reload
    sudo systemctl enable aios-streaming
    sudo systemctl restart aios-streaming

    echo -e "${GREEN}✓ Service deployed and running${NC}\n"
}

###############################################################################
# Nginx Configuration
###############################################################################

configure_nginx() {
    echo -e "${YELLOW}⚙ Configuring Nginx...${NC}"

    sudo tee /etc/nginx/sites-available/aios > /dev/null << EOF
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    # Enable site
    sudo ln -sf /etc/nginx/sites-available/aios /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx

    echo -e "${GREEN}✓ Nginx configured for ${DOMAIN}${NC}\n"
}

###############################################################################
# SSL Certificate (Let's Encrypt)
###############################################################################

setup_ssl() {
    echo -e "${YELLOW}🔒 Setting up SSL certificate...${NC}"

    if command -v certbot &> /dev/null; then
        sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email ${SSL_EMAIL} --redirect
        echo -e "${GREEN}✓ SSL certificate installed${NC}\n"
    else
        echo -e "${YELLOW}⚠ Certbot not available. Skipping SSL setup.${NC}"
        echo -e "${YELLOW}  Install certbot and run: sudo certbot --nginx -d ${DOMAIN}${NC}\n"
    fi
}

###############################################################################
# cPanel/Bluehost Deployment
###############################################################################

deploy_cpanel() {
    echo -e "${YELLOW}🌐 Deploying for cPanel/Bluehost...${NC}"

    # cPanel uses Passenger for Python apps
    # Create .htaccess for reverse proxy
    mkdir -p public_html/${DOMAIN}
    cat > public_html/${DOMAIN}/.htaccess << 'EOF'
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]
EOF

    echo -e "${GREEN}✓ cPanel configuration created${NC}"
    echo -e "${YELLOW}📝 Next steps:${NC}"
    echo -e "  1. Start the server: cd aios/web && ../../venv/bin/python streaming_server.py"
    echo -e "  2. Set up subdomain in cPanel pointing to public_html/${DOMAIN}"
    echo -e "  3. Enable SSL in cPanel for ${DOMAIN}\n"
}

###############################################################################
# Main Deployment Logic
###############################################################################

main() {
    detect_environment

    case $ENV_TYPE in
        docker)
            install_dependencies
            deploy_docker
            ;;
        cpanel)
            install_dependencies
            deploy_cpanel
            ;;
        aws|gcp|azure|digitalocean|vps)
            install_dependencies
            deploy_native
            configure_nginx
            setup_ssl
            ;;
        *)
            echo -e "${RED}⚠ Unknown environment. Attempting native deployment...${NC}"
            install_dependencies
            deploy_native
            ;;
    esac

    echo -e "${GREEN}═══════════════════════════════════${NC}"
    echo -e "${GREEN}✅ AI:OS Streaming Server Deployed!${NC}"
    echo -e "${GREEN}═══════════════════════════════════${NC}\n"
    echo -e "${BLUE}📊 Dashboard: ${NC}https://${DOMAIN}"
    echo -e "${BLUE}🔌 WebSocket: ${NC}wss://${DOMAIN}/ws"
    echo -e "${BLUE}💚 Health: ${NC}https://${DOMAIN}/healthz"
    echo -e "${BLUE}📡 API: ${NC}https://${DOMAIN}/api/status\n"
    echo -e "${YELLOW}🔍 Check status:${NC} systemctl status aios-streaming"
    echo -e "${YELLOW}📋 View logs:${NC} journalctl -u aios-streaming -f\n"
}

# Run deployment
main "$@"
