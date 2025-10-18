# Red Team Tools Subdomain Setup Instructions

## ✅ Files Ready for Deployment

### 1. CNAME File Created
- Location: `/Users/noone/CNAME`
- Content: `red-team-tools.aios.is`

### 2. Red Team Tools Web Interface Ready
- Location: `aios/red-team-tools/`
- Files:
  - `index.html` - Main landing page
  - `dashboard.html` - Tool dashboard
  - `login.html` - Authentication page
  - `register.html` - User registration

## 🚀 Steps to Complete Subdomain Setup

### Step 1: Commit and Push to GitHub
```bash
git add CNAME
git add aios/red-team-tools/
git commit -m "Add red-team-tools subdomain configuration and web interface"
git push origin main
```

### Step 2: Configure GitHub Pages
1. Go to: https://github.com/workofarttattoo/aios/settings/pages
2. Under "Source", select "Deploy from a branch"
3. Choose "main" branch
4. Select "/ (root)" as the folder
5. Click "Save"

### Step 3: Configure DNS at Your Domain Registrar
You need to add a CNAME record at your domain registrar (where aios.is is registered):

**Option A: If using Namecheap/GoDaddy/etc:**
- Type: CNAME
- Host: red-team-tools
- Value: workofarttattoo.github.io
- TTL: Automatic or 300

**Option B: If using Cloudflare:**
- Type: CNAME
- Name: red-team-tools
- Target: workofarttattoo.github.io
- Proxy status: DNS only (gray cloud)

### Step 4: Wait for Propagation
- DNS changes can take 10 minutes to 48 hours to propagate
- GitHub Pages SSL certificate will be automatically provisioned

### Step 5: Test Your Subdomain
Once setup is complete, test these URLs:
- https://red-team-tools.aios.is (main page)
- https://red-team-tools.aios.is/dashboard.html
- https://red-team-tools.aios.is/login.html

## 🔧 Troubleshooting

### If you get "404 Not Found":
1. Ensure GitHub Pages is enabled in repository settings
2. Check that CNAME file is in the root directory
3. Verify DNS records are properly configured

### If you get "Certificate Error":
1. Wait for GitHub to provision SSL (can take up to 24 hours)
2. Ensure DNS is set to "DNS only" if using Cloudflare

### If GitHub rejects the subdomain:
1. The main domain (aios.is) must already be verified with GitHub
2. You may need to add a TXT record to verify domain ownership:
   - Type: TXT
   - Name: _github-pages-challenge-workofarttattoo
   - Value: [GitHub will provide this]

## 📁 Red Team Tools Available

Once deployed, the following tools will be accessible:

### Core Security Assessment Tools:
- **AuroraScan** - Network reconnaissance
- **CipherSpear** - Database injection analysis
- **DirReaper** - Directory enumeration
- **VulnHunter** - Vulnerability scanning
- **ProxyPhantom** - Proxy chain management
- **NemesisHydra** - Authentication testing
- **ObsidianHunt** - Host hardening audit
- **VectorFlux** - Payload staging

### Integration Features:
- Ai|oS platform integration
- GAVL Court evidence collection
- Quantum-enhanced analysis
- Real-time reporting dashboard

## 📊 Marketing Integration

The red-team-tools subdomain serves as a key demonstration platform for the 30-day campaign:

1. **Direct Link from Ads**: All Facebook/LinkedIn ads can link directly to the tools
2. **Live Demonstrations**: Government agencies can test tools in sandbox mode
3. **Lead Capture**: Registration page collects qualified leads
4. **Analytics Tracking**: Add Google Analytics to track visitor engagement

## 🎯 Next Steps After Subdomain is Live

1. **Send Government Emails**: Use templates in legal_tech_outreach_templates.html
2. **Launch Facebook Ads**: Use the visual templates created
3. **Schedule Demos**: Direct prospects to red-team-tools.aios.is for live testing
4. **Track Conversions**: Monitor sign-ups and demo requests

---

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**