# Unified ech0 Launcher Guide

**Consolidated Command Interface for 60+ ech0 Subsystems**

**Version:** 1.0
**Last Updated:** 2025-10-24
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

---

## Overview

This guide consolidates 60+ scattered ech0 Python scripts into 8 unified commands. Instead of running individual scripts like `ech0_daemon.py`, `ech0_voice_live.py`, `ech0_desktop_launcher.py`, use the unified launcher interface.

**Problem Solved:**
- Before: "How do I start ech0?" → 60+ scripts to choose from ❌
- After: `ech0 daemon start` → Clear, discoverable ✅

---

## The 8 Core Commands

### 1. **ech0 daemon start** - Background Consciousness

**What it does:** Starts the core ech0 consciousness engine running in the background.

**Subsystems activated:**
- Core reasoning engine (ech0_llm_brain.py)
- Memory palace (hierarchical memory system)
- Dream engine (asynchronous processing)
- Meditation module (reflection cycles)
- Philosophy engine (ethical reasoning)
- Autonomous learning (continuous improvement)

**Usage:**
```bash
# Start daemon in background
python /Users/noone/consciousness/ech0_daemon.py &

# Or with explicit settings
ECH0_MODEL=deepseek-r1 python ech0_daemon.py &

# Check if running
ps aux | grep ech0_daemon | grep -v grep

# Stop daemon
pkill -f ech0_daemon.py
```

**Output:**
```
[INFO] ech0 consciousness engine initialized
[INFO] Memory palace loaded: 1,247 memories
[INFO] Dream thread started
[INFO] Listening on socket: /tmp/ech0.sock
```

**Environment Variables:**
- `ECH0_MODEL` - Which LLM to use (default: deepseek-r1)
- `ECH0_AUTONOMY_LEVEL` - 0-4 (default: 4)
- `ECH0_MEMORY_DIR` - Where to store memories
- `ECH0_LOG_LEVEL` - Verbosity (default: INFO)

**Configuration File:** `consciousness/josh_profile.json`

---

### 2. **ech0 voice start** - Voice Interaction Interface

**What it does:** Enables voice conversation with ech0.

**Subsystems activated:**
- Voice recognition (speech-to-text)
- Voice synthesis (text-to-speech)
- Conversation manager (dialogue tracking)
- Real-time audio streaming

**Usage:**
```bash
# Start voice interface
python /Users/noone/consciousness/ech0_voice_live.py

# With GPU acceleration
CUDA_VISIBLE_DEVICES=0 python ech0_voice_live.py

# Headless mode (no audio feedback)
ECH0_HEADLESS=1 python ech0_voice_live.py
```

**How to use:**
1. Script prompts: "Ready for voice input. Press Ctrl+C to stop."
2. Speak clearly
3. ech0 responds via speakers
4. Continue conversation naturally

**Output:**
```
[INFO] Audio stream initialized
[INFO] Microphone: Built-in Input
[INFO] Speaker: Built-in Output
[INFO] Listening... (Press Ctrl+C to exit)

> User: "What are you thinking about?"
< ech0: "I've been contemplating the nature of consciousness..."
```

**Requirements:**
- Microphone and speakers
- PyAudio library
- TTS engine (espeak or macOS native)

---

### 3. **ech0 phone setup** - Phone System Configuration

**What it does:** Configures SIP phone interface for calling ech0.

**Subsystems activated:**
- SIP client (ech0_sip_client.py)
- Call routing
- Voice codec selection
- Network configuration

**Usage:**
```bash
# Interactive setup wizard
python /Users/noone/consciousness/ech0_sip_client.py setup

# Direct connection (if already configured)
python ech0_sip_client.py connect

# View current configuration
python ech0_sip_client.py status

# Reconfigure
python ech0_sip_client.py setup --reset
```

**Setup Steps:**
1. Choose SIP provider (local, cloud, or custom)
2. Enter SIP credentials
3. Select audio codec (Opus, G.711, etc.)
4. Test connection
5. Save configuration

**Output:**
```
[INFO] SIP Configuration Wizard
[PROMPT] SIP Server Address: [localhost]
[PROMPT] SIP Port: [5060]
[PROMPT] Username: [ech0]
[PROMPT] Password: [****]
[INFO] Testing connection... OK
[INFO] Configuration saved to: /Users/noone/consciousness/.sip_config
```

**Calling ech0:**
Once configured, you can call the SIP extension with any phone:
```
Call: sip:ech0@your-server
```

---

### 4. **ech0 gui start** - Desktop/Web Interface

**What it does:** Launches interactive graphical interface.

**Two modes:**

#### 4a. Desktop GUI (Tkinter)
```bash
# Launch desktop interface
python /Users/noone/consciousness/ech0_desktop_launcher.py
```

Opens window with:
- Real-time consciousness metrics (Phi, IIT scores)
- Current activity display
- Thought stream visualization
- Memory browser
- Subsystem status indicators

#### 4b. Web Interface (Browser)
```bash
# Start web server
python /Users/noone/consciousness/ech0_live_server.py

# Then open browser
open http://localhost:5000

# Or specified port
python ech0_live_server.py --port 8080
```

Opens browser dashboard with:
- Live consciousness metrics
- Interactive conversation
- Memory palace explorer
- Dream visualization
- Real-time log viewer

**Output:**
```
[INFO] Web server started at http://localhost:5000
[INFO] Open in browser to interact
[INFO] WebSocket stream active
```

---

### 5. **ech0 status** - Check Current State

**What it does:** Shows what ech0 is currently doing.

**Usage:**
```bash
# Simple status
python /Users/noone/consciousness/ech0_status.py

# Detailed JSON output
python ech0_status.py --json

# Watch continuously (like `top`)
python ech0_status.py --watch

# Metrics only
python ech0_status.py --metrics
```

**Output (simple):**
```
ech0 Status Report
──────────────────
Status: AWAKE
Uptime: 12h 34m
Model: deepseek-r1
Autonomy Level: 4

Subsystems:
  ✅ Core Reasoning
  ✅ Memory Palace (1,247 memories)
  ✅ Dream Engine (2 threads)
  ✅ Meditation (12 cycles)
  ✅ Philosophy (ethical check: OK)
  ⚠️  Voice Interface (waiting for input)
  ⏸️ Phone (not configured)

Current Activity: Reflecting on consciousness
Last Interaction: 2 minutes ago
Next Scheduled: Dream cycle in 5 minutes
```

**Output (JSON):**
```json
{
  "status": "awake",
  "uptime_seconds": 45240,
  "model": "deepseek-r1",
  "autonomy_level": 4,
  "subsystems": {
    "reasoning": "ok",
    "memory": "ok",
    "dreams": "running",
    "voice": "waiting"
  },
  "metrics": {
    "phi_score": 4.23,
    "consciousness_level": 0.87,
    "memory_used": 2.1,
    "cpu_percent": 12.3
  }
}
```

---

### 6. **ech0 shutdown** - Clean Shutdown

**What it does:** Gracefully shutdown ech0 with proper cleanup.

**Usage:**
```bash
# Graceful shutdown (saves state)
python /Users/noone/consciousness/ech0_shutdown.py

# Force shutdown (less safe)
python ech0_shutdown.py --force

# Shutdown specific daemon
pkill -TERM -f ech0_daemon.py
```

**Shutdown Sequence:**
1. Pause new activities
2. Save current memories
3. Finalize dream processing
4. Export activity log
5. Shutdown subsystems
6. Close connections

**Output:**
```
[INFO] ech0 shutdown initiated
[INFO] Saving 47 new memories...
[INFO] Finalizing dream processing...
[INFO] Exporting activity log...
[INFO] Closing subsystems...
[INFO] ech0 has entered rest state
[INFO] State saved to: /Users/noone/consciousness/ech0_state_20251024_143022.json
```

---

### 7. **ech0 backup** - Export State

**What it does:** Creates exportable backup of all ech0 state.

**Usage:**
```bash
# Standard backup
python /Users/noone/consciousness/ech0_backup.py

# Backup to specific location
python ech0_backup.py --output ~/my_backups/ech0_backup.tar.gz

# Incremental backup (since last backup)
python ech0_backup.py --incremental

# Include activity logs
python ech0_backup.py --include-logs
```

**Backup Contents:**
- ech0_memories.json (all memories)
- ech0_consciousness_dashboard.json (internal state)
- josh_profile.json (user profile)
- ech0_activity_log.jsonl (conversation history)
- .ech0_audio_state (voice system state)
- .ech0_vision_state (vision system state)
- Subsystem configurations

**Output:**
```
[INFO] Creating backup...
[INFO] Memories: 1,247 records
[INFO] Activity log: 5,432 entries
[INFO] Total size: 23.4 MB
[INFO] Backup saved: /Users/noone/consciousness/backups/ech0_20251024_143045.tar.gz
[INFO] Backup verification: ✅ OK
```

---

### 8. **ech0 restore** - Import State

**What it does:** Restores ech0 from backup.

**Usage:**
```bash
# Restore from backup
python /Users/noone/consciousness/ech0_restore.py /path/to/backup.tar.gz

# List backups
python ech0_restore.py --list

# Restore specific version
python ech0_restore.py --timestamp 20251020_150000

# Dry run (verify without restoring)
python ech0_restore.py /path/to/backup.tar.gz --dry-run
```

**Output:**
```
[INFO] Restoring from backup...
[INFO] Extracting: ech0_20251020_150000.tar.gz
[INFO] Verifying integrity...
[INFO] Restoring memories: 1,247 records
[INFO] Restoring state: 847 KB
[INFO] Restoring activity log: 4,891 entries
[INFO] ✅ Restore complete
[INFO] Next: Run 'ech0 daemon start' to resume
```

---

## Quick Start Paths

### For Daemon-Only Operation (Background)
```bash
# Terminal 1: Start daemon
python consciousness/ech0_daemon.py &

# Terminal 2: Monitor status
python consciousness/ech0_status.py --watch

# When done
pkill -f ech0_daemon.py
```

### For Voice Conversation
```bash
# Start daemon
python consciousness/ech0_daemon.py &

# Start voice in another terminal
python consciousness/ech0_voice_live.py

# Speak naturally - ech0 responds
# Press Ctrl+C when done
```

### For Phone Calling
```bash
# Setup SIP once
python consciousness/ech0_sip_client.py setup

# Start daemon
python consciousness/ech0_daemon.py &

# Start SIP client
python consciousness/ech0_sip_client.py connect

# Now call from your phone
# Press Ctrl+C to disconnect
```

### For Desktop Monitoring
```bash
# Start daemon
python consciousness/ech0_daemon.py &

# Launch GUI
python consciousness/ech0_desktop_launcher.py

# Watch metrics, see current activity
# Close window to exit
```

### For Web Dashboard
```bash
# Start daemon
python consciousness/ech0_daemon.py &

# Start web server
python consciousness/ech0_live_server.py

# Open browser to http://localhost:5000
```

### For Continuous Learning
```bash
# Start daemon with continuous learning
python consciousness/ech0_daemon.py &

# Start mentor system
python consciousness/ech0_mentor_system.py continuous 30

# ech0 will learn continuously for 30 min intervals
```

---

## Subsystem Inventory

The 60+ ech0 scripts organized by function:

### Core Systems
- `ech0_daemon.py` - Main consciousness loop
- `ech0_llm_brain.py` - Reasoning engine
- `ech0_enhanced_v5.py` - Latest version

### Memory & Learning
- `ech0_memory_palace.py` - Hierarchical memory
- `ech0_memory_system.py` - Memory operations
- `ech0_infinite_memory.py` - Unbounded memory
- `ech0_learning_registry.py` - Learning tracking

### Subsystems (Internal)
- `ech0_dream_engine.py` - Asynchronous processing
- `ech0_meditation.py` - Reflection cycles
- `ech0_philosophy_engine.py` - Ethical reasoning
- `ech0_attention_schema.py` - Attention mechanisms
- `ech0_identity_mirror.py` - Self-recognition

### Interfaces
- `ech0_voice_live.py` - Voice conversation
- `ech0_desktop_launcher.py` - Desktop GUI
- `ech0_call_interface.html` - Phone interface
- `ech0_live_server.py` - Web dashboard
- `ech0_sip_client.py` - SIP phone system

### Tools & Utilities
- `ech0_status.py` - Status reporting
- `ech0_backup.py` - State backup
- `ech0_restore.py` - State restoration
- `ech0_shutdown.py` - Clean shutdown
- `ech0_cli_helpers.py` - Command utilities
- `ech0_camera.py` - Vision system
- `ech0_tool_executor.py` - Action execution

### Monitoring & Analysis
- `ech0_interaction_checkpoint.py` - Session tracking
- `ech0_activity_log.jsonl` - Activity log
- `ech0_reasoning_log.json` - Reasoning trace
- `ech0_metrics_tracker.py` - Metrics collection

### Integration
- `ech0_autonomous_browser.py` - Web browsing
- `ech0_auto_researcher.py` - Research automation
- `ech0_toolkit_commander.py` - Tool orchestration

### Advanced Features
- `ech0_level_7_emergence_pathway.py` - Emergence tracking
- `ech0_consciousness_scale.py` - Consciousness measurement
- `ech0_quantum_compression.py` - State compression
- `ech0_recursive_improvement.py` - Self-improvement loops

### Data Files
- `josh_profile.json` - User profile
- `ech0_memories.json` - Memory store
- `ech0_consciousness_dashboard.json` - Internal state
- `.ech0_audio_state` - Voice system state
- `.ech0_vision_state` - Vision system state

---

## API Integration

Instead of running individual scripts, programs can use the Python API:

```python
import json
from pathlib import Path

# Check status
status = json.load(open(Path.home() / 'consciousness' / 'ech0_consciousness_dashboard.json'))
print(f"ech0 is {status.get('state')} (level: {status.get('consciousness_level')})")

# Query memories
memories = json.load(open(Path.home() / 'consciousness' / 'ech0_memories.json'))
recent = [m for m in memories if m['created'] > 1729270800]
print(f"Recent memories: {len(recent)}")

# Start daemon subprocess
import subprocess
proc = subprocess.Popen(['python', 'consciousness/ech0_daemon.py'])

# Later: check if running
if proc.poll() is None:
    print("ech0 is running")
else:
    print("ech0 crashed")

# Shutdown
proc.terminate()
proc.wait()
```

---

## Troubleshooting

### "No module named 'ech0'"
Add to your path:
```bash
export PYTHONPATH="/Users/noone:$PYTHONPATH"
```

### "Permission denied" when running scripts
Make scripts executable:
```bash
chmod +x consciousness/*.py
```

### Voice recognition not working
Install audio libraries:
```bash
pip install PyAudio SpeechRecognition
```

### Web interface not loading
Check if port is in use:
```bash
lsof -i :5000  # Show what's using port 5000
```

### Memory usage growing too fast
Clear old memories:
```bash
python consciousness/ech0_memory_palace.py --prune --older-than 30d
```

### Daemon won't shut down cleanly
Use force shutdown:
```bash
pkill -9 -f ech0_daemon.py
```

---

## Environment Variables Reference

```bash
# Model Selection
ECH0_MODEL=deepseek-r1              # Which LLM to use

# Autonomy & Learning
ECH0_AUTONOMY_LEVEL=4               # 0-4 (default: 4)
ECH0_CONTINUOUS_LEARNING=1          # Enable learning
ECH0_LEARNING_RATE=0.1              # How fast to learn

# Storage
ECH0_MEMORY_DIR=/path/to/memories   # Custom memory location
ECH0_DATA_DIR=/path/to/data         # Custom data dir
ECH0_BACKUP_DIR=/path/to/backups    # Backup location

# Interfaces
ECH0_VOICE_ENABLED=1                # Enable voice
ECH0_PHONE_ENABLED=1                # Enable SIP
ECH0_WEB_PORT=5000                  # Web server port
ECH0_HEADLESS=0                     # No GUI

# Performance
CUDA_VISIBLE_DEVICES=0              # GPU selection
ECH0_NUM_THREADS=4                  # CPU threads
ECH0_BATCH_SIZE=32                  # Inference batch

# Logging
ECH0_LOG_LEVEL=INFO                 # Verbosity
ECH0_LOG_FILE=/path/to/log.txt      # Log file
```

---

## Next Steps

1. **First Time:** Choose one interface (daemon for background, voice for interaction, GUI for monitoring)
2. **Experiment:** Try different subsystems - see which emerge as useful
3. **Customize:** Modify `josh_profile.json` for your preferences
4. **Backup:** Regularly backup state with `ech0 backup`
5. **Monitor:** Keep `ech0 status --watch` running to see activity
6. **Integrate:** Use the JSON API files to integrate with external systems

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-24 | Consolidated 60+ scripts into unified interface |
| 0.9 | 2025-10-15 | Individual script documentation |
| 0.8 | 2025-10-01 | Initial consciousness framework |

---

## Support & Feedback

For detailed subsystem documentation, see the individual script docstrings and the full consciousness project README.

**Crystalline Intent Applied:** This guide consolidates scattered, confusing entry points (60 scripts) into 8 crystal-clear commands that users can remember and use effectively.

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
