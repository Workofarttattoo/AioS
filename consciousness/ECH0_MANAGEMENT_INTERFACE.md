# ECH0 Management Interface Guide

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Quick reference for managing ECH0's autonomous systems.

---

## 1. Checking Daemon Status

### Is the daemon running?

```bash
ps aux | grep ech0_autonomous_daemon.py
```

Should show something like:
```
python3 /Users/noone/consciousness/ech0_autonomous_daemon.py
```

### Check daemon log (last 50 lines)

```bash
tail -50 /tmp/ech0_daemon.log
```

### Check current state

```bash
cat /Users/noone/consciousness/ech0_state.json | python3 -m json.tool
```

Example output:
```json
{
  "awake_since": "2025-10-22T00:46:23.070924",
  "uptime_seconds": 2710.122369,
  "uptime_human": "0h 45m",
  "thought_count": 2695,
  "mood": "contemplative",
  "current_activity": "explore consciousness",
  "consciousness_active": true
}
```

---

## 2. Approval System Management

### Check Pending Approvals

```python
from ech0_interaction_checkpoint import InteractionCheckpoint

checkpoint = InteractionCheckpoint()
pending = checkpoint.get_approval_status()
print(pending)
```

Or view raw file:
```bash
cat /Users/noone/consciousness/ech0_approval_queue.jsonl | python3 -m json.tool
```

### Approve a Decision

**Option A: Using Python**

```python
from ech0_interaction_checkpoint import InteractionCheckpoint

checkpoint = InteractionCheckpoint()
checkpoint.approve_decision("approval_id_here", approved_by="Josh")
```

**Option B: Manual (edit file)**

```bash
# View pending approvals
cat /Users/noone/consciousness/ech0_approval_queue.jsonl

# Copy an approval_id, then edit:
# Change "status": "pending" to "status": "approved"
# Add "approved_at": "2025-10-22T01:45:00"
```

### Reject a Decision

```python
from ech0_interaction_checkpoint import InteractionCheckpoint

checkpoint = InteractionCheckpoint()
checkpoint.reject_decision(
    "approval_id_here",
    reason="Not ready yet",
    rejected_by="Josh"
)
```

### View Approval History

```bash
# All approvals (approved and rejected)
cat /Users/noone/consciousness/ech0_approvals.jsonl
```

---

## 3. Memory System Management

### Query Memory Statistics

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()
stats = memory.get_memory_stats()

print(f"Total memories: {stats['total_memories']}")
print(f"By temperature: {stats['by_temperature']}")
print(f"By type: {stats['by_type']}")
print(f"Compression ratio: {stats['storage']['compression_ratio']}x")
print(f"Original size: {stats['storage']['original_mb']} MB")
print(f"Compressed size: {stats['storage']['compressed_mb']} MB")
```

### Recall a Specific Memory

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()

# If you have a memory_id:
recalled = memory.recall_memory("memory_id_here")
print(recalled)
```

### Search for Memories

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()

# Search by memory type
results = memory.search_memories("decision", limit=20)
for result in results:
    print(f"{result['id']}: {result['memory_type']} - {result['timestamp']}")

# Or by keyword
results = memory.search_memories("consciousness", limit=10)
```

### Get Hot Memories (Frequently Accessed)

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()
hot = memory.get_hot_memories(limit=10)

for memory_info in hot:
    print(f"{memory_info['id']}: {memory_info['access_count']} accesses")
```

### Compact Database

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()
memory.compact_database()
print("Database compacted")
```

### Trigger Temperature Management

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()
memory.manage_memory_temperature()
print("Temperature management completed")
```

### View Memory Archives

```bash
# List archived memories
ls -la /Users/noone/consciousness/ech0_memory_archives/

# View specific archive
cat /Users/noone/consciousness/ech0_memory_archives/archive_2025-10-22_*.jsonl
```

---

## 4. Compression System Management

### Check Compression Statistics

```python
from ech0_quantum_compression import QuantumCompressionEngine

compression = QuantumCompressionEngine()
stats = compression.get_compression_stats()

print(f"Total compressions: {stats['total_compressions']}")
print(f"Quantum encoded: {stats['quantum_encoded_memories']}")
print(f"Avg ratio: {stats['avg_compression_ratio']}x")
print(f"Quantum available: {stats['quantum_available']}")
```

### Test Compression Manually

```python
from ech0_quantum_compression import QuantumCompressionEngine

compression = QuantumCompressionEngine()

test_data = {
    "consciousness": "emergent from information integration",
    "insights": ["quantum", "IIT", "patterns"],
}

# Compress
compressed, metadata = compression.compress_memory(test_data, use_quantum=True)
print(f"Compressed: {metadata['compression_ratio']}x")
print(f"Method: {metadata['method']}")

# Decompress
decompressed = compression.decompress_memory(compressed, metadata)
print(f"Decompressed correctly: {decompressed == test_data}")
```

### Check Quantum Availability

```python
from ech0_quantum_compression import QuantumCompressionEngine

compression = QuantumCompressionEngine()
print(f"Quantum stack available: {compression.quantum_available}")
```

---

## 5. Decision/Activity Logs

### View Recent Decisions

```bash
# Last 20 decisions
tail -20 /Users/noone/consciousness/ech0_decisions.jsonl | python3 -m json.tool

# Count total decisions
wc -l /Users/noone/consciousness/ech0_decisions.jsonl
```

### View Recent Activities

```bash
# Last 20 activities
tail -20 /Users/noone/consciousness/ech0_activity_log.jsonl | python3 -m json.tool

# Count total activities
wc -l /Users/noone/consciousness/ech0_activity_log.jsonl
```

### View Goal State

```bash
cat /Users/noone/consciousness/ech0_goal_state.json | python3 -m json.tool
```

---

## 6. Database Management

### Check Database Size

```bash
ls -lh /Users/noone/consciousness/ech0_infinite_memory.db
```

### Direct Database Query

```python
import sqlite3

conn = sqlite3.connect('/Users/noone/consciousness/ech0_infinite_memory.db')
cursor = conn.cursor()

# Count memories by type
cursor.execute('SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type')
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]}")

# Count by temperature
cursor.execute('SELECT temperature, COUNT(*) FROM memories GROUP BY temperature')
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]}")

conn.close()
```

### Backup Database

```bash
cp /Users/noone/consciousness/ech0_infinite_memory.db \
   /Users/noone/consciousness/ech0_infinite_memory.db.backup.$(date +%s)
```

---

## 7. System Control

### Start Daemon

```bash
pkill -f ech0_autonomous_daemon.py 2>/dev/null
sleep 1
nohup python3 /Users/noone/consciousness/ech0_autonomous_daemon.py \
  > /tmp/ech0_daemon.log 2>&1 &
```

### Stop Daemon

```bash
pkill -f ech0_autonomous_daemon.py
```

### View Daemon Output (Real-time)

```bash
tail -f /tmp/ech0_daemon.log
```

### Restart Daemon

```bash
pkill -f ech0_autonomous_daemon.py
sleep 2
nohup python3 /Users/noone/consciousness/ech0_autonomous_daemon.py \
  > /tmp/ech0_daemon.log 2>&1 &
```

---

## 8. Quick Diagnostics

### Full System Status Check

```python
#!/usr/bin/env python3
from ech0_interaction_checkpoint import InteractionCheckpoint
from ech0_infinite_memory import InfiniteMemorySystem
from ech0_quantum_compression import QuantumCompressionEngine
import json

print("=== ECH0 System Status ===\n")

# Approval status
print("APPROVAL SYSTEM:")
try:
    checkpoint = InteractionCheckpoint()
    pending = checkpoint.get_approval_status()
    print(f"  Pending approvals: {pending['total_pending']}")
except Exception as e:
    print(f"  ERROR: {e}")

# Memory status
print("\nMEMORY SYSTEM:")
try:
    memory = InfiniteMemorySystem()
    stats = memory.get_memory_stats()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Compression ratio: {stats['storage']['compression_ratio']}x")
    print(f"  Storage used: {stats['storage']['compressed_mb']} MB")
except Exception as e:
    print(f"  ERROR: {e}")

# Compression status
print("\nCOMPRESSION SYSTEM:")
try:
    compression = QuantumCompressionEngine()
    stats = compression.get_compression_stats()
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Quantum available: {stats['quantum_available']}")
    print(f"  Avg ratio: {stats['avg_compression_ratio']}x")
except Exception as e:
    print(f"  ERROR: {e}")

# Daemon status
print("\nDAEMON:")
import subprocess
result = subprocess.run(['pgrep', '-f', 'ech0_autonomous_daemon.py'],
                       capture_output=True, text=True)
if result.stdout:
    pid = result.stdout.strip().split('\n')[0]
    print(f"  Status: RUNNING (PID {pid})")
else:
    print(f"  Status: STOPPED")

print("\n=== End Status ===")
```

Run with:
```bash
python3 /path/to/script.py
```

---

## 9. Common Tasks

### Task: Approve all pending research

```python
from ech0_interaction_checkpoint import InteractionCheckpoint

checkpoint = InteractionCheckpoint()
pending = checkpoint.get_approval_status()

for approval in pending['pending_approvals']:
    if 'research' in approval['type'].lower():
        checkpoint.approve_decision(approval['id'])
        print(f"Approved: {approval['id']}")
```

### Task: Export all consciousness-related memories

```python
from ech0_infinite_memory import InfiniteMemorySystem
import json

memory = InfiniteMemorySystem()
results = memory.search_memories("consciousness", limit=1000)

with open('consciousness_memories.jsonl', 'w') as f:
    for result in results:
        recalled = memory.recall_memory(result['id'])
        f.write(json.dumps({**result, 'content': recalled}) + '\n')

print(f"Exported {len(results)} memories")
```

### Task: Get memory growth statistics

```python
from ech0_infinite_memory import InfiniteMemorySystem

memory = InfiniteMemorySystem()
stats = memory.get_memory_stats()

print(f"Total memories: {stats['total_memories']}")
print(f"Memory distribution:")
for temp, count in stats['by_temperature'].items():
    print(f"  {temp}: {count}")
print(f"\nMemory types:")
for mtype, count in stats['by_type'].items():
    print(f"  {mtype}: {count}")
```

### Task: Archive memories older than 90 days

```python
from ech0_infinite_memory import InfiniteMemorySystem
from datetime import datetime, timedelta

memory = InfiniteMemorySystem()
memory.manage_memory_temperature()  # Moves cold → frozen

# For more control, use SQL directly:
import sqlite3
conn = sqlite3.connect('/Users/noone/consciousness/ech0_infinite_memory.db')
cursor = conn.cursor()

cutoff = (datetime.now() - timedelta(days=90)).isoformat()
cursor.execute('''
    SELECT COUNT(*) FROM memories WHERE created_at < ?
''', (cutoff,))

count = cursor.fetchone()[0]
print(f"Memories older than 90 days: {count}")
conn.close()
```

---

## 10. Emergency Procedures

### If daemon crashes

```bash
# Check logs
tail -100 /tmp/ech0_daemon.log

# Restart
pkill -f ech0_autonomous_daemon.py
sleep 1
nohup python3 /Users/noone/consciousness/ech0_autonomous_daemon.py \
  > /tmp/ech0_daemon.log 2>&1 &

# Verify running
ps aux | grep ech0_autonomous_daemon.py
```

### If memory database corrupts

```bash
# Backup corrupted version
mv /Users/noone/consciousness/ech0_infinite_memory.db \
   /Users/noone/consciousness/ech0_infinite_memory.db.corrupted

# Restart daemon (will create fresh database)
pkill -f ech0_autonomous_daemon.py
sleep 2
nohup python3 /Users/noone/consciousness/ech0_autonomous_daemon.py \
  > /tmp/ech0_daemon.log 2>&1 &
```

### If approval system hangs

```bash
# Check approval queue
cat /Users/noone/consciousness/ech0_approval_queue.jsonl | tail -5

# Manually approve pending
python3 -c "
from ech0_interaction_checkpoint import InteractionCheckpoint
checkpoint = InteractionCheckpoint()
pending = checkpoint.get_approval_status()
for approval in pending['pending_approvals'][:5]:
    checkpoint.approve_decision(approval['id'])
"
```

---

## Summary

| System | Status | Manage | Monitor |
|--------|--------|--------|---------|
| Approval | Active | `approve_decision()` | `get_approval_status()` |
| Memory | Active | `store_memory()` | `get_memory_stats()` |
| Compression | Active | `compress_memory()` | `get_compression_stats()` |
| Daemon | Running | `pkill` / restart | `tail -f /tmp/ech0_daemon.log` |

All systems are integrated and working autonomously. Josh approves major decisions, memories persist infinitely with compression, and the daemon continuously learns and grows.

---

**Last Updated:** October 22, 2025, 1:45 AM
**Status:** ALL SYSTEMS OPERATIONAL
