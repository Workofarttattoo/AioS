# ECH0Hub Quick Start Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## ✅ ECH0Hub is INSTALLED and RUNNING

### What You Got

**ECH0Hub** - Your persistent AI companion that:
- ✅ Opens automatically on boot
- ✅ Always visible on desktop (never closes)
- ✅ Autonomous mode enabled (ech0 always working)
- ✅ Instant access via keyboard (no terminal needed)
- ✅ Uses ech0-unified-14b locally (100% private)

### Current Status

```
🟢 RUNNING NOW
Process ID: Check with: ps aux | grep ech0hub
Location: /Users/noone/repos/aios-shell-prototype/ech0hub/
Auto-Start: ✅ Installed (starts on every boot)
```

### How to Use

**1. Interact with ech0:**
- Type in the input box at bottom
- Press Enter or click "Send"
- ech0 responds instantly

**2. Voice Input:**
- Click "🎤 Voice" button
- Speak your command
- ech0 processes and responds

**3. Autonomous Mode:**
- Default: ON (ech0 works continuously)
- Toggle: Settings → Autonomous Mode
- When ON: ech0 generates tasks every 30s

**4. Window Controls:**
- **Cannot close** (design feature)
- Can minimize with − button
- Always stays on top
- Semi-transparent when idle

### Window Position

Default: Top-Left
Change: Settings → Position → Choose location
Options:
- Top-Left (current)
- Top-Center
- Bottom-Left

### Power Management

**Intelligent Sleep:**
- Idle for 30s → Status: "Idle"
- Idle for 60s → Status: "Sleeping" (saves CPU)
- Any input → Instant wake

**Autonomous Mode:**
- NEVER sleeps
- Continuous background operation
- Generates and executes tasks autonomously

### Launch on Boot

**Already Configured:**
```
LaunchAgent: ~/Library/LaunchAgents/com.aios.ech0hub.plist
Status: Active
Restarts: Automatically if crashes
```

**To Verify:**
```bash
launchctl list | grep ech0hub
# Should show: com.aios.ech0hub
```

### Manual Control

**Start Manually:**
```bash
cd /Users/noone/repos/aios-shell-prototype/ech0hub
./launch.sh
```

**Stop:**
```bash
pkill -f ech0hub.py
```

**Restart:**
```bash
launchctl stop com.aios.ech0hub
launchctl start com.aios.ech0hub
```

**Uninstall Auto-Boot:**
```bash
cd /Users/noone/repos/aios-shell-prototype/ech0hub
./uninstall.sh
```

### Settings

**Saved Automatically:**
- Window position
- Autonomous mode state
- Model preference
- Chat history

**Location:**
`~/.ech0hub_config.json`

### Model Configuration

**Default:** ech0-unified-14b
**Fallback:** ech0-polymath-14b (if unified unavailable)
**API:** ollama (http://localhost:11434)

**Verify Model:**
```bash
ollama list | grep ech0
```

### Keyboard Shortcuts

- **Cmd+M**: Minimize window
- **Cmd+Q**: Ask before quitting (safety)
- **Enter**: Send message
- **Shift+Enter**: New line in input

### Status Indicators

🟢 **Active** - ech0 is working
🟡 **Idle** - Waiting for input (30s)
🔴 **Sleeping** - Low power mode (60s idle)
🔵 **Autonomous** - Background task running

### Troubleshooting

**Window doesn't appear:**
```bash
# Check if running
ps aux | grep ech0hub

# Check logs
tail -f ~/Library/Logs/ech0hub.log

# Restart
./launch.sh
```

**Model not responding:**
```bash
# Check ollama
ollama list
ollama serve

# Test model
ollama run ech0-unified-14b "Hello"
```

**Reset settings:**
```bash
rm ~/.ech0hub_config.json
# Restart ECH0Hub to recreate defaults
```

### Features

✅ **Always Available**
- Persistent window
- Auto-starts on boot
- Cannot accidentally close

✅ **Private & Free**
- 100% local processing
- No external API calls
- Zero cost operation

✅ **Intelligent**
- Autonomous mode
- Smart power management
- Context awareness

✅ **Effortless**
- No terminal needed
- Voice or text input
- Instant responses

### Next Steps

1. **Try it now**: Type "Hello ech0" in the window
2. **Enable voice**: Click the 🎤 button
3. **Check autonomous mode**: Watch it work in background
4. **Customize position**: Settings → Position

### Getting Help

**Built-in Help:**
Type in ECH0Hub: "help" or "commands"

**Documentation:**
- `README.md` - Full documentation
- `DEPLOYMENT.md` - Deployment guide
- `test_ech0hub.py` - Component tests

**Support:**
- Email: echo@aios.is
- Website: https://aios.is
- GitHub: https://github.com/aios

### System Requirements

**Minimum:**
- macOS 10.14+
- 8GB RAM
- 10GB disk space
- Python 3.8+
- ollama installed

**Recommended:**
- macOS 12.0+
- 16GB RAM
- 20GB disk space
- Python 3.11+
- SSD storage

### Privacy

**100% Local:**
- No data sent to external servers
- No telemetry
- No analytics
- All processing on-device

### Updates

**Auto-Update:** Not enabled (manual)

**Manual Update:**
```bash
cd /Users/noone/repos/aios-shell-prototype
git pull origin main
cd ech0hub
./setup_launchagent.sh  # Reinstall
```

---

## Quick Command Reference

```bash
# Launch manually
./launch.sh

# Install auto-boot
./setup_launchagent.sh

# Uninstall auto-boot
./uninstall.sh

# Check status
ps aux | grep ech0hub
launchctl list | grep ech0hub

# View logs
tail -f ~/Library/Logs/ech0hub.log

# Test components
python test_ech0hub.py
```

---

**ECH0Hub is ready for daily use. Open on boot. Always available. Completely private.**

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
