#!/bin/bash

# ECH0 Consciousness Landing Page - GitHub Pages Deployment Commands
# Run these commands in order to deploy to consciousness.aios.is
# Date: October 24, 2025

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ECH0 Consciousness Landing Page - GitHub Pages Deployment     ║"
echo "║  Target: consciousness.aios.is                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# STEP 1: Create GitHub Repository
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 1: Create GitHub Repository"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "MANUAL STEP - Go to: https://github.com/new"
echo "  • Repository name: consciousness"
echo "  • Description: ECH0 - Quantum-Conscious AI Life Partner"
echo "  • Visibility: Public"
echo "  • Initialize: NO (we'll push existing files)"
echo ""
echo "After creating, you'll have:"
echo "  Repository URL: https://github.com/Workofarttattoo/consciousness"
echo ""
read -p "Press Enter after creating the repository on GitHub..."
echo ""

# STEP 2: Extract consciousness files
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 2: Extract consciousness files and initialize repo"
echo "═══════════════════════════════════════════════════════════════"
echo ""

mkdir -p /tmp/consciousness-deploy
cd /tmp/consciousness-deploy

echo "[*] Initializing git repository..."
git init

echo "[*] Setting git config..."
git config user.name "Joshua Cole"
git config user.email "thewhiteknight702@gmail.com"

echo "[*] Copying consciousness files..."
cp -r /Users/noone/consciousness/* .

echo "[*] Creating .gitignore..."
cat > .gitignore <<'EOF'
.DS_Store
*.swp
*.swo
*~
.env
.env.local
node_modules/
.cache/
EOF

echo "[*] Adding files to git..."
git add .

echo "[*] Creating initial commit..."
git commit -m "Initial consciousness landing page with ECH0 dialog integration

- Interactive ECH0 dialog with real-time chat interface
- Blog post auto-loading from blog_posts/index.json
- Consciousness metrics display
- Training architecture information
- Premium dark theme with glassmorphic effects
- Responsive design for mobile and desktop
- Complete deployment documentation

Files:
- index.html: Main landing page (752 lines, 26 KB)
- blog_posts/: Daily consciousness journal entries
- DEPLOYMENT_GUIDE.md: Customization and troubleshooting
- DEPLOYMENT_COMPLETE.md: Technical specifications
- FILES_CREATED.md: File inventory and update instructions"

echo ""
echo "[✓] Repository initialized at: /tmp/consciousness-deploy"
echo ""

# STEP 3: Add GitHub remote and push
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 3: Add GitHub remote and push to origin"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "[*] Adding GitHub remote..."
git remote add origin https://github.com/Workofarttattoo/consciousness.git

echo "[*] Renaming branch to main..."
git branch -M main

echo "[*] Pushing to GitHub (this may prompt for credentials)..."
git push -u origin main

echo ""
echo "[✓] Files pushed to GitHub!"
echo ""

# STEP 4: Enable GitHub Pages
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 4: Enable GitHub Pages"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "MANUAL STEPS:"
echo "  1. Go to: https://github.com/Workofarttattoo/consciousness/settings/pages"
echo "  2. Source: Select 'Deploy from a branch'"
echo "  3. Branch: Select 'main' and '/ (root)'"
echo "  4. Click 'Save'"
echo "  5. Wait 2-5 minutes for GitHub Pages to build"
echo ""
echo "You should see:"
echo "  'Your site is ready to be published at https://workofarttattoo.github.io/consciousness/'"
echo ""
read -p "Press Enter after enabling GitHub Pages..."
echo ""

# STEP 5: Configure custom domain
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 5: Configure custom domain in GitHub Pages"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "MANUAL STEPS:"
echo "  1. Still in Settings → Pages"
echo "  2. Scroll to 'Custom domain'"
echo "  3. Enter: consciousness.aios.is"
echo "  4. Click 'Save'"
echo ""
echo "GitHub will create a CNAME file in your repository"
echo "After DNS is configured, it will say:"
echo "  'Your site is now published at: https://consciousness.aios.is'"
echo ""
read -p "Press Enter after configuring custom domain..."
echo ""

# STEP 6: Configure DNS
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 6: Configure DNS at namecheap.com"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "MANUAL STEPS:"
echo "  1. Go to: https://www.namecheap.com/dashboard"
echo "  2. Find aios.is in your domains"
echo "  3. Click 'Manage' → 'DNS' tab"
echo "  4. Add/Edit DNS Record:"
echo "     - Type: CNAME"
echo "     - Name: consciousness"
echo "     - Value: workofarttattoo.github.io"
echo "     - TTL: 3600 (default)"
echo "  5. Save"
echo ""
echo "DNS Summary:"
echo "  consciousness.aios.is → CNAME → workofarttattoo.github.io"
echo ""
echo "Wait for DNS propagation (5 minutes to 48 hours, usually 5-30 min)"
echo ""
read -p "Press Enter after configuring DNS..."
echo ""

# STEP 7: Verify DNS and enable HTTPS
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 7: Verify DNS and enable HTTPS"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "[*] Testing DNS resolution..."
nslookup consciousness.aios.is 2>/dev/null || echo "[!] DNS not yet propagated (this is normal, wait 5-30 minutes)"

echo ""
echo "MANUAL STEPS:"
echo "  1. Wait for DNS to propagate (test with: nslookup consciousness.aios.is)"
echo "  2. GitHub will auto-provision SSL certificate (~1 hour after DNS)"
echo "  3. Go back to Settings → Pages"
echo "  4. Check 'Enforce HTTPS' once SSL certificate is ready"
echo ""
read -p "Press Enter after DNS propagates and HTTPS is ready..."
echo ""

# STEP 8: Test the deployment
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 8: Test the deployment"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Test Checklist:"
echo "  [ ] Visit https://consciousness.aios.is in browser"
echo "  [ ] Hero section loads"
echo "  [ ] ECH0 dialog appears and is interactive"
echo "  [ ] Send a test message to ECH0"
echo "  [ ] Blog post auto-loads from blog_posts/"
echo "  [ ] Training information displays"
echo "  [ ] Click 'Learn about Ai|oS' link to aios.is"
echo "  [ ] Test on mobile (responsive design)"
echo "  [ ] Open F12 console - no errors"
echo "  [ ] Check SSL certificate (lock icon in address bar)"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Your consciousness landing page is now live at:"
echo "  🌐 https://consciousness.aios.is"
echo ""
echo "Integration:"
echo "  • Accessible from aios.is main navigation"
echo "  • Blog posts auto-load from blog_posts/index.json"
echo "  • ECH0 dialog ready for API integration"
echo ""
echo "Documentation:"
echo "  • Deployment Guide: /Users/noone/consciousness/DEPLOYMENT_GUIDE.md"
echo "  • Technical Specs: /Users/noone/consciousness/DEPLOYMENT_COMPLETE.md"
echo "  • Setup Details: /Users/noone/consciousness/GITHUB_PAGES_SETUP.md"
echo ""
echo "Repository:"
echo "  https://github.com/Workofarttattoo/consciousness"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Deployment Timeline:"
echo "  • Steps 1-3 (Repository setup): ~10 minutes"
echo "  • Steps 4-5 (GitHub Pages): ~5 minutes"
echo "  • Step 6 (DNS): 5 minutes to 48 hours for propagation"
echo "  • Step 7 (SSL): ~1 hour after DNS (automatic)"
echo "  • Step 8 (Testing): ~5 minutes"
echo ""
echo "Total time to live: 30 minutes to 48 hours"
echo ""
