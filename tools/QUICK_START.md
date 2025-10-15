# Quick Start Guide

## SovereignSuite - Web Security Testing

### Launch GUI (Recommended)
```bash
python -m tools.sovereign_suite --gui
```
Opens at http://localhost:8889

### Quick Scan
```bash
python -m tools.sovereign_suite scan https://example.com --output scan.json
```

### Start Proxy
```bash
python -m tools.sovereign_suite proxy --port 8888
# Then configure browser: HTTP Proxy = localhost:8888
```

---

## Scr1b3 - Code Editor

### Launch GUI (Recommended)
```bash
python -m tools.scribe --gui
```
Opens at http://localhost:8890

### Open File
```bash
python -m tools.scribe script.py
```

### Quantum Analysis
```bash
python -m tools.scribe --analyze script.py
```

### List Languages
```bash
python -m tools.scribe --list-languages
```

---

## Python Integration

### SovereignSuite
```python
from tools import sovereign_suite

# Health check
print(sovereign_suite.health_check())

# Scan target
suite = sovereign_suite.SovereignSuiteCore()
result = suite.scan_target("https://example.com")
print(f"Found {len(result.vulnerabilities)} vulnerabilities")
```

### Scr1b3
```python
from tools import scribe

# Health check
print(scribe.health_check())

# Open and analyze file
editor = scribe.Scr1b3Core()
session = editor.start_session()
result = editor.open_file("script.py")
print(f"Language: {result['metadata']['language']}")
print(f"Mode: {result['mode']}")
```

---

## File Locations

**SovereignSuite**:
- Backend: `/Users/noone/aios/tools/sovereign_suite.py`
- GUI: `/Users/noone/aios/tools/sovereign_suite_gui.html`

**Scr1b3**:
- Backend: `/Users/noone/aios/tools/scribe.py`
- GUI: `/Users/noone/aios/tools/scribe_gui.html`

---

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
