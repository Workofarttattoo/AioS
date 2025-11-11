# DNS Configuration Guide for red-team-tools.aios.is

## Current Status
- **GitHub Pages URL**: https://workofarttattoo.github.io/AioS/ ✅ WORKING
- **Custom Domain**: red-team-tools.aios.is ⏳ NEEDS DNS SETUP

## DNS Records to Add

### For Domain: aios.is

Log into your DNS provider (GoDaddy, Cloudflare, Namecheap, etc.) and add these records:

### Option 1: CNAME Record (Recommended - Easier)

```
Type: CNAME
Name: red-team-tools
Target/Value: workofarttattoo.github.io
TTL: 3600 (or Auto)
```

**Example:**
- If using Cloudflare: `red-team-tools.aios.is` → `workofarttattoo.github.io`
- Proxy status: Can be ON (orange cloud) or OFF (gray cloud)

### Option 2: A Records (Alternative - More Stable)

If CNAME doesn't work, use these 4 A records:

```
Type: A
Name: red-team-tools
Value: 185.199.108.153
TTL: 3600

Type: A
Name: red-team-tools
Value: 185.199.109.153
TTL: 3600

Type: A
Name: red-team-tools
Value: 185.199.110.153
TTL: 3600

Type: A
Name: red-team-tools
Value: 185.199.111.153
TTL: 3600
```

## DNS Provider-Specific Instructions

### Cloudflare

1. Login to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Select domain: `aios.is`
3. Go to **DNS** → **Records**
4. Click **Add record**
5. Add CNAME or A records as shown above
6. Save changes
7. **Important**: Enable **SSL/TLS** → **Full** or **Flexible**

### GoDaddy

1. Login to [GoDaddy](https://dcc.godaddy.com/domains)
2. Find `aios.is` and click **DNS**
3. Click **Add New Record**
4. Select **CNAME** or **A** type
5. Enter details from above
6. Click **Save**

### Namecheap

1. Login to [Namecheap](https://ap.www.namecheap.com/)
2. Domain List → Manage `aios.is`
3. **Advanced DNS** tab
4. **Add New Record**
5. Select type and enter values
6. Save changes

### Generic DNS Provider

1. Find your DNS management panel
2. Add CNAME record:
   - **Host/Name**: `red-team-tools`
   - **Type**: `CNAME`
   - **Value**: `workofarttattoo.github.io`
   - **TTL**: `3600` or `Auto`
3. Save and wait for propagation

## GitHub Pages Configuration

1. Go to: https://github.com/Workofarttattoo/AioS/settings/pages
2. Verify settings:
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/docs`
   - **Custom domain**: `red-team-tools.aios.is`
   - **Enforce HTTPS**: ✅ Checked
3. GitHub will automatically verify DNS and issue SSL certificate

## Verification Steps

### 1. Check DNS Propagation (Takes 1-48 hours)

```bash
# Check CNAME record
dig red-team-tools.aios.is CNAME +short

# Expected output: workofarttattoo.github.io

# Check A records (if using A records)
dig red-team-tools.aios.is A +short

# Expected output:
# 185.199.108.153
# 185.199.109.153
# 185.199.110.153
# 185.199.111.153
```

### 2. Online DNS Checkers

- **DNS Checker**: https://dnschecker.org/
  - Enter: `red-team-tools.aios.is`
  - Type: CNAME or A
  - Check multiple locations

- **WhatsMyDNS**: https://whatsmydns.net/
  - Enter: `red-team-tools.aios.is`

### 3. Test Site Access

Once DNS propagates:
```bash
curl -I https://red-team-tools.aios.is
```

Expected: `HTTP/2 200` status

## Troubleshooting

### Issue: DNS not resolving

**Solution**:
- Wait 24-48 hours for full propagation
- Clear DNS cache:
  ```bash
  # macOS
  sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder

  # Windows
  ipconfig /flushdns

  # Linux
  sudo systemd-resolve --flush-caches
  ```

### Issue: HTTPS not working

**Solution**:
1. Wait for GitHub to provision SSL (takes 10-60 minutes)
2. Ensure "Enforce HTTPS" is checked in GitHub Pages settings
3. Try accessing with `https://` prefix explicitly

### Issue: "There isn't a GitHub Pages site here"

**Solution**:
1. Verify CNAME file exists in `/docs/` directory
2. Check GitHub Pages settings show custom domain
3. Wait for DNS propagation
4. Verify branch/folder settings are correct

### Issue: Mixed content warnings

**Solution**:
- Ensure all resources load over HTTPS
- Check `js/auth-config.js` uses HTTPS URLs
- Update any hardcoded HTTP links

## Timeline

- **DNS Changes**: 1-48 hours (usually 1-6 hours)
- **SSL Certificate**: 10-60 minutes after DNS verification
- **Full Propagation**: Up to 72 hours globally

## Current Working URLs

Until DNS propagates, use:
- **Main Site**: https://workofarttattoo.github.io/AioS/
- **Login**: https://workofarttattoo.github.io/AioS/login.html
- **Dashboard**: https://workofarttattoo.github.io/AioS/dashboard.html

## Supabase CORS Update

After DNS is working, add to Supabase:
1. Go to [Supabase Dashboard](https://app.supabase.com/project/trokobwiphidmrmhwkni/settings/api)
2. **Settings** → **API** → **CORS**
3. Add origins:
   - `https://red-team-tools.aios.is`
   - `https://workofarttattoo.github.io`
4. Save changes

## Quick Reference

| Setting | Value |
|---------|-------|
| DNS Type | CNAME (recommended) or A |
| Host/Name | red-team-tools |
| CNAME Target | workofarttattoo.github.io |
| A Records | 185.199.108-111.153 |
| TTL | 3600 or Auto |
| GitHub Branch | main |
| GitHub Folder | /docs |
| HTTPS | Enforced |

## Support

If issues persist after 48 hours:
- GitHub Pages Docs: https://docs.github.com/en/pages
- DNS Provider Support
- GitHub Community: https://github.community/

---

**Last Updated**: October 16, 2025
**Status**: DNS setup required for custom domain
