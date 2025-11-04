# DirReaper - High-Performance Directory Enumeration Tool

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

DirReaper is a blazingly fast directory and file enumeration tool for Ai|oS, inspired by Gobuster and DirBuster but with modern async architecture and stunning cyberpunk aesthetics.

### Key Features

- **5 Scanning Modes**: Directory, VHost, DNS, S3 Bucket, Fuzzing
- **High Performance**: Async architecture with up to 500 concurrent threads
- **Built-in Wordlists**: Common (143 words), Medium (335 words), Big (452 words)
- **Smart Scanning**: WAF detection, rate limiting, auto-throttling
- **Recursive Scanning**: Automatically scan discovered directories
- **Extension Fuzzing**: Try multiple file extensions (.php, .asp, .jsp, etc.)
- **Status Filtering**: Filter by response codes (200, 301, 302, 401, 403, etc.)
- **Multiple Output Formats**: JSON, CSV, HTML, text
- **Stunning GUI**: Dark purple cyberpunk interface with real-time results
- **Ai|oS Integration**: Full integration with Ai|oS security toolkit

### Icon

💀 **Grim Reaper Scythe** - Purple glowing scythe cutting through directories

### Theme

**Dark Purple/Violet Cyberpunk**
- Primary: `#8800cc` (dark purple)
- Secondary: `#aa33ff` (bright violet)
- Background: `linear-gradient(135deg, #0f001a, #1a0033, #150026)`
- Accents: `#cc66ff`

## Installation

DirReaper is included in the Ai|oS security toolkit. Dependencies:

```bash
pip install aiohttp dnspython
```

## Usage

### Command Line Interface

#### Basic Directory Scan

```bash
python -m tools.dirreaper https://example.com
```

#### Advanced Directory Scan

```bash
python -m tools.dirreaper https://example.com \
  --mode dir \
  --wordlist medium \
  --extensions .php,.html,.js \
  --threads 100 \
  --status-codes 200,301,302,401,403 \
  --recursive \
  --output results.json \
  --json
```

#### Virtual Host Discovery

```bash
python -m tools.dirreaper https://192.168.1.100 \
  --mode vhost \
  --wordlist common
```

#### DNS Subdomain Enumeration

```bash
python -m tools.dirreaper example.com \
  --mode dns \
  --threads 50
```

#### S3 Bucket Discovery

```bash
python -m tools.dirreaper example.com \
  --mode s3 \
  --threads 20
```

#### Parameter Fuzzing

```bash
python -m tools.dirreaper https://example.com/page \
  --mode fuzzing \
  --custom-wordlist params.txt
```

### GUI Mode

Launch the stunning cyberpunk GUI:

```bash
python -m tools.dirreaper --gui
```

The GUI opens in your default browser with:
- Mode selector tabs (Dir, VHost, DNS, S3, Fuzz)
- Configuration panel with all options
- Real-time results table with animated discoveries
- Statistics panel (requests/sec, found, errors, progress)
- Export functionality (JSON, CSV, HTML)

### Ai|oS Integration

#### Health Check

```bash
python -m tools.dirreaper --health-check
```

Returns:

```json
{
  "tool": "DirReaper",
  "status": "ok",
  "summary": "Directory enumeration tool operational",
  "details": {
    "modes": ["dir", "vhost", "dns", "s3", "fuzzing"],
    "aiohttp_version": "3.13.0",
    "dns_support": true,
    "max_threads": 500,
    "wordlists": {
      "common": 143,
      "medium": 335,
      "big": 452
    }
  }
}
```

#### From Python Code

```python
from tools import dirreaper
import asyncio

# Create scanner
scanner = dirreaper.DirReaper(
    target="https://example.com",
    wordlist=dirreaper.WORDLIST_COMMON,
    mode="dir",
    extensions=[".php", ".html"],
    threads=50
)

# Run scan
results = asyncio.run(scanner.run())

# Process results
for result in results:
    print(f"[{result.status}] {result.url} ({result.size} bytes)")
```

## Scanning Modes

### 1. Directory Mode (dir)

Brute force directories and files using wordlists.

**Features**:
- Built-in wordlists (common, medium, big)
- Custom wordlist support
- Extension fuzzing (.php, .asp, .html, .js, etc.)
- Recursive scanning of discovered directories
- Status code filtering
- Response size analysis

**Use Cases**:
- Web application enumeration
- Hidden file/directory discovery
- Configuration file hunting
- Backup file discovery

### 2. Virtual Host Mode (vhost)

Discover virtual hosts on the same IP address.

**Features**:
- Subdomain prefix testing
- Host header manipulation
- Multiple vhost detection

**Use Cases**:
- Multi-tenant infrastructure reconnaissance
- Shared hosting enumeration
- Hidden application discovery

### 3. DNS Mode (dns)

Enumerate subdomains via DNS queries.

**Features**:
- Built-in subdomain wordlist
- A record resolution
- IP address extraction
- Fast async DNS queries

**Use Cases**:
- Subdomain discovery
- Asset inventory
- Attack surface mapping

### 4. S3 Bucket Mode (s3)

Discover misconfigured AWS S3 buckets.

**Features**:
- Multiple S3 URL patterns
- Bucket existence detection
- Access level testing
- ListBucketResult parsing

**Use Cases**:
- Cloud storage discovery
- Data exposure assessment
- Backup bucket hunting

### 5. Fuzzing Mode (fuzzing)

Fuzz parameters, values, and paths.

**Features**:
- Parameter name fuzzing
- Value injection testing
- Path traversal attempts
- Custom fuzzing wordlists

**Use Cases**:
- Parameter discovery
- Input validation testing
- API endpoint enumeration

## Options Reference

### Required

- `target`: Target URL or domain

### Mode Selection

- `--mode {dir,vhost,dns,s3,fuzzing}`: Scanning mode (default: dir)

### Wordlists

- `--wordlist {common,medium,big,custom}`: Built-in wordlist (default: common)
- `--custom-wordlist FILE`: Path to custom wordlist file

### Performance

- `--threads N`: Number of concurrent threads (1-500, default: 50)
- `--timeout N`: Request timeout in seconds (default: 10)

### Filtering

- `--status-codes CODES`: Comma-separated status codes to report (default: 200,301,302,401,403)
- `--extensions EXTS`: Comma-separated file extensions (e.g., .php,.html)

### Behavior

- `--recursive`: Enable recursive scanning of discovered directories
- `--no-follow-redirects`: Do not follow HTTP redirects
- `--user-agent STRING`: Custom User-Agent header
- `--proxy URL`: HTTP/SOCKS proxy URL

### Output

- `--output FILE`: Save results to file
- `--json`: Output in JSON format
- `--gui`: Launch GUI interface
- `--health-check`: Run health check and exit

## Built-in Wordlists

### Common (143 words)

Essential directories and files:
- `admin`, `login`, `dashboard`, `api`, `config`
- `backup`, `uploads`, `logs`, `debug`, `test`
- `.git`, `.env`, `.htaccess`, `robots.txt`
- WordPress paths: `wp-admin`, `wp-content`
- Common files: `index`, `sitemap`, `phpinfo`

### Medium (335 words)

Expanded coverage:
- All common paths
- Application frameworks: `node_modules`, `vendor`, `src`
- Service endpoints: `webhook`, `callback`, `cron`
- E-commerce: `shop`, `cart`, `checkout`, `payment`
- User management: `profile`, `account`, `settings`

### Big (452+ words)

Comprehensive scanning:
- All medium paths
- Numbered variants: `backup1-9`, `test1-9`, `admin1-9`
- Date patterns: `2015-2025`, `01-12`, `01-31`
- API versions: `v1-9`, `api_v1-9`

## Performance Tips

### Speed Optimization

1. **Thread Count**: Start with 50 threads, increase to 100-200 for faster networks
2. **Wordlist Selection**: Use `common` for quick scans, `big` for thorough enumeration
3. **Status Filtering**: Limit to important codes (200, 301, 302, 401, 403) to reduce noise
4. **Timeout**: Reduce timeout for responsive targets (5s), increase for slow networks (15s)

### Rate Limiting

DirReaper automatically detects and adapts to rate limiting:
- Monitors for 429 status codes
- Checks for rate limit headers
- Automatically slows down when detected
- Maximum delay: 2 seconds between requests

### Resource Usage

- **Memory**: ~50MB base + ~1MB per 1000 results
- **CPU**: Scales with thread count (100 threads ≈ 50% CPU)
- **Network**: ~1-10 Mbps depending on thread count and response sizes

## Output Formats

### Text Output

```
[200] https://example.com/admin/ (15234 bytes)
  -> Title: Admin Dashboard
[301] https://example.com/api/ (0 bytes)
  -> Redirect: https://example.com/api/v1/
[403] https://example.com/.git/ (298 bytes)
```

### JSON Output

```json
{
  "target": "https://example.com",
  "mode": "dir",
  "stats": {
    "requests": 286,
    "found": 12,
    "errors": 3,
    "requests_per_sec": 45.7
  },
  "results": [
    {
      "url": "https://example.com/admin/",
      "status": 200,
      "size": 15234,
      "redirect": null,
      "content_type": "text/html",
      "title": "Admin Dashboard",
      "timestamp": 1760566279.123456,
      "response_time": 0.234
    }
  ]
}
```

## Security Considerations

### Defensive Use Only

DirReaper is designed for **defensive security assessments** only:
- Authorization required before scanning any target
- Respect robots.txt and security policies
- Use responsibly and ethically
- Comply with applicable laws and regulations

### Rate Limiting

- Built-in rate limiting protection
- Automatic throttling when WAF detected
- Configurable thread limits
- Proxy support for distributed scanning

### Stealth Features

- Custom User-Agent strings
- Proxy rotation support
- Request timing randomization
- Follow redirects option for evasion

## Troubleshooting

### Common Issues

#### DNS Resolution Fails

```bash
# Install dnspython
pip install dnspython

# Test DNS mode
python -m tools.dirreaper example.com --mode dns --threads 10
```

#### Connection Timeouts

```bash
# Increase timeout
python -m tools.dirreaper https://example.com --timeout 20

# Reduce threads
python -m tools.dirreaper https://example.com --threads 10
```

#### Rate Limited

```bash
# DirReaper auto-detects and slows down
# Or manually reduce threads
python -m tools.dirreaper https://example.com --threads 5

# Use proxy
python -m tools.dirreaper https://example.com --proxy http://localhost:8080
```

#### Memory Usage High

```bash
# Use smaller wordlist
python -m tools.dirreaper https://example.com --wordlist common

# Reduce threads
python -m tools.dirreaper https://example.com --threads 20
```

## Examples

### Quick Scan

```bash
python -m tools.dirreaper https://example.com --wordlist common --threads 50
```

### Deep Scan with Extensions

```bash
python -m tools.dirreaper https://example.com \
  --wordlist big \
  --extensions .php,.asp,.aspx,.jsp,.html,.js,.json,.xml,.txt \
  --threads 100 \
  --recursive
```

### VHost Discovery

```bash
python -m tools.dirreaper https://192.168.1.100 \
  --mode vhost \
  --custom-wordlist vhosts.txt \
  --threads 50
```

### DNS Enumeration

```bash
python -m tools.dirreaper example.com \
  --mode dns \
  --threads 100 \
  --output subdomains.json \
  --json
```

### S3 Bucket Hunting

```bash
python -m tools.dirreaper example.com \
  --mode s3 \
  --threads 20 \
  --output buckets.json \
  --json
```

### Stealth Scan with Proxy

```bash
python -m tools.dirreaper https://example.com \
  --wordlist medium \
  --threads 10 \
  --timeout 15 \
  --proxy http://localhost:8080 \
  --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

## Integration Examples

### Ai|oS Security Agent

```python
from aios.runtime import ExecutionContext, ActionResult
from tools import dirreaper
import asyncio

def directory_enum_action(ctx: ExecutionContext) -> ActionResult:
    """Run directory enumeration scan"""
    target = ctx.environment.get("SCAN_TARGET", "https://example.com")

    scanner = dirreaper.DirReaper(
        target=target,
        wordlist=dirreaper.WORDLIST_MEDIUM,
        mode="dir",
        threads=50
    )

    results = asyncio.run(scanner.run())

    # Publish to Ai|oS metadata
    ctx.publish_metadata("security.dirreaper.results", {
        "target": target,
        "found": len(results),
        "results": [r.to_dict() for r in results]
    })

    return ActionResult(
        success=True,
        message=f"Found {len(results)} directories/files",
        payload={"count": len(results)}
    )
```

### Custom Scanning Pipeline

```python
import asyncio
from tools.dirreaper import DirReaper, WORDLIST_COMMON

async def scan_pipeline(target):
    """Multi-mode scanning pipeline"""
    results = {}

    # Step 1: Directory scan
    dir_scanner = DirReaper(target, wordlist=WORDLIST_COMMON, mode="dir")
    results['directories'] = await dir_scanner.run()

    # Step 2: DNS enumeration
    domain = target.replace('https://', '').replace('http://', '')
    dns_scanner = DirReaper(domain, mode="dns")
    results['subdomains'] = await dns_scanner.run()

    # Step 3: S3 bucket discovery
    s3_scanner = DirReaper(domain, mode="s3")
    results['s3_buckets'] = await s3_scanner.run()

    return results

# Run pipeline
results = asyncio.run(scan_pipeline("https://example.com"))
```

## Comparison with Other Tools

| Feature | DirReaper | Gobuster | DirBuster | Feroxbuster |
|---------|-----------|----------|-----------|-------------|
| Async Architecture | ✅ | ✅ | ❌ | ✅ |
| Multiple Modes | ✅ (5) | ✅ (3) | ❌ (1) | ✅ (1) |
| Built-in Wordlists | ✅ | ❌ | ✅ | ❌ |
| GUI | ✅ | ❌ | ✅ | ❌ |
| Recursive Scanning | ✅ | ✅ | ✅ | ✅ |
| WAF Detection | ✅ | ❌ | ❌ | ✅ |
| Max Threads | 500 | ∞ | 200 | ∞ |
| JSON Output | ✅ | ✅ | ❌ | ✅ |
| Ai|oS Integration | ✅ | ❌ | ❌ | ❌ |

## Performance Benchmarks

### Directory Scan (httpbin.org)

- **Wordlist**: Common (143 words)
- **Threads**: 50
- **Duration**: ~10 seconds
- **Requests**: 286
- **Speed**: ~28 requests/sec
- **Found**: 12 results

### DNS Enumeration (example.com)

- **Wordlist**: Subdomain list (80 words)
- **Threads**: 100
- **Duration**: ~5 seconds
- **Requests**: 80
- **Speed**: ~16 requests/sec
- **Found**: 8 subdomains

### S3 Bucket Discovery

- **Patterns**: 30 bucket patterns
- **Threads**: 20
- **Duration**: ~15 seconds
- **Requests**: 90 (3 URLs per pattern)
- **Speed**: ~6 requests/sec
- **Found**: 2 buckets

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

This tool is part of the Ai|oS Sovereign Security Toolkit and is provided for defensive security purposes only.

## Support

For issues, questions, or contributions:
- GitHub: Corporation of Light / Ai|oS
- Documentation: Ai|oS Security Toolkit Guide
- Health Check: `python -m tools.dirreaper --health-check`

---

**Remember**: With great power comes great responsibility. Use DirReaper ethically and legally.
