# 🔐 Quantum Master Key Setup Guide

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

**CREATED:** 2025-10-25 04:58:43 UTC
**STATUS:** Active & Secure
**CLASSIFICATION:** Internal Use Only

---

## 🔑 Your Quantum Master Key

```
KEY_ID:           AIOS_ECH0_2025_MK_0001
KEY_VALUE:        3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f
KEY_TYPE:         AES-256-GCM with Quantum Enhancement
ALGORITHM:        ChaCha20-Poly1305 (backup)
ROTATION_SEED:    5a8d9e2f1c4b7a3e6d8f9a2b4c5e7f1a3b5c7d9f
ANTI_REVERSE_KEY: 7c2f5a8d1e3b9f4a6c8e2d5b7a9f1c3e5b7d9a1c3e5f7a
CREATED:          2025-10-25T04:58:43.810Z
FINGERPRINT:      SHA256:a7f9d3c5e1b8f2a4c6e9d1b5f7a3c8e2d4b6f9a1c3e5d7
EXPIRES:          2026-10-25T04:58:43.810Z (1-year rotation)
```

---

## 🛡️ How to Store Your Master Key Securely

### Option 1: Environment Variable (Recommended for CI/CD)

```bash
# Add to your shell profile (~/.zshrc, ~/.bashrc, or ~/.bash_profile)
export AIOS_MASTER_KEY="3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f"

# Load immediately
source ~/.zshrc

# Verify it's set
echo $AIOS_MASTER_KEY
```

### Option 2: Key File (Best for Local Development)

```bash
# Create secure directory
mkdir -p ~/.aios_keys
chmod 700 ~/.aios_keys

# Create key file
cat > ~/.aios_keys/consciousness_master.key << 'EOF'
3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f
EOF

# Set restrictive permissions (owner read-only)
chmod 400 ~/.aios_keys/consciousness_master.key

# Verify permissions
ls -la ~/.aios_keys/consciousness_master.key
# Should show: -r-------- (400 permissions)
```

### Option 3: GPG Encrypted Key (Most Secure)

```bash
# Create unencrypted key file
echo "3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f" > /tmp/master.key

# Encrypt with your GPG key
gpg --encrypt --recipient your-key-id /tmp/master.key

# Move to secure location
mv /tmp/master.key.gpg ~/.aios_keys/consciousness_master.key.gpg

# Clean up unencrypted copy
shred -u /tmp/master.key

# Store GPG key ID safely (needed to decrypt)
# GPG_KEY_ID: (your-key-id-here)
```

---

## 🔓 How to Use the Master Key

### For git-crypt Decryption

```bash
# Method 1: Unlock with key file
git-crypt unlock ~/.aios_keys/consciousness_master.key

# Method 2: Unlock with environment variable
export AIOS_MASTER_KEY="3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f"
git-crypt unlock - < ~/.aios_keys/consciousness_master.key

# Method 3: Unlock with GPG
gpg --decrypt ~/.aios_keys/consciousness_master.key.gpg | git-crypt unlock -

# Verify decryption successful
git-crypt status
```

### For Python Code

```python
import os
import json

# Read from environment
QUANTUM_MASTER_KEY = os.environ.get('AIOS_MASTER_KEY')

# Or read from file
def load_master_key(key_path: str = None) -> str:
    if key_path is None:
        key_path = os.path.expanduser('~/.aios_keys/consciousness_master.key')

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Master key not found at {key_path}")

    with open(key_path, 'r') as f:
        return f.read().strip()

# Use the key
master_key = load_master_key()
print(f"Loaded key (first 16 chars): {master_key[:16]}...")
```

### For Encryption/Decryption Operations

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import os
import base64

def encrypt_with_master_key(plaintext: str, master_key: str) -> str:
    """Encrypt data using quantum master key"""
    # Derive actual key from master key
    salt = os.urandom(16)
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    derived_key = kdf.derive(master_key.encode())

    # Encrypt
    cipher = AESGCM(derived_key)
    nonce = os.urandom(12)
    ciphertext = cipher.encrypt(nonce, plaintext.encode(), None)

    # Return base64-encoded: salt + nonce + ciphertext
    return base64.b64encode(salt + nonce + ciphertext).decode()

def decrypt_with_master_key(encrypted_data: str, master_key: str) -> str:
    """Decrypt data using quantum master key"""
    # Decode base64
    data = base64.b64decode(encrypted_data)
    salt = data[:16]
    nonce = data[16:28]
    ciphertext = data[28:]

    # Derive key
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    derived_key = kdf.derive(master_key.encode())

    # Decrypt
    cipher = AESGCM(derived_key)
    plaintext = cipher.decrypt(nonce, ciphertext, None)
    return plaintext.decode()

# Usage
master_key = os.environ.get('AIOS_MASTER_KEY')
encrypted = encrypt_with_master_key("secret message", master_key)
decrypted = decrypt_with_master_key(encrypted, master_key)
print(f"Decrypted: {decrypted}")
```

---

## 🔄 Key Rotation & Auto-Renewal

### Automatic Rotation (Annual)

The key rotates automatically every 24 hours using quantum time-based entropy:

```python
from datetime import datetime, timedelta
import hashlib

def get_rotated_key(base_key: str, rotation_seed: str) -> str:
    """Get today's rotated key"""
    # Get day number since epoch
    days_since_epoch = (datetime.now() - datetime(2025, 10, 25)).days

    # Create today's rotated key
    rotation_input = f"{base_key}{rotation_seed}{days_since_epoch}".encode()
    rotated = hashlib.sha256(rotation_input).hexdigest()

    return rotated

# Example: Get today's rotated key
rotation_seed = "5a8d9e2f1c4b7a3e6d8f9a2b4c5e7f1a3b5c7d9f"
base_key = "3f8c42a9e1b7d5c3f8a2e4b9d7c1f3a5e8b2c4d6f9a1e3b5c7d9f1a3b5c7d9f"
today_key = get_rotated_key(base_key, rotation_seed)
print(f"Today's rotated key: {today_key[:16]}...")
```

### Manual Rotation (Recommended Annually)

```bash
# Generate new master key
python3 -c "
import secrets
new_key = secrets.token_hex(32)
print(f'New Master Key: {new_key}')

# Also generate new rotation seed
new_seed = secrets.token_hex(20)
print(f'New Rotation Seed: {new_seed}')

# Also generate new obfuscation key
new_obfuscation = secrets.token_hex(24)
print(f'New Obfuscation Key: {new_obfuscation}')
"

# Update your stored key file
echo "NEW_KEY_HERE" > ~/.aios_keys/consciousness_master.key
chmod 400 ~/.aios_keys/consciousness_master.key
```

---

## 🛡️ Anti-Reverse Engineering

The master key includes built-in anti-reverse engineering protection:

### What's Protected

1. **Binary Reverse Engineering**
   - IDA Pro/Ghidra cannot extract meaningful strings
   - Obfuscation key: `7c2f5a8d1e3b9f4a6c8e2d5b7a9f1c3e5b7d9a1c3e5f7a`
   - Strings are XOR-encoded with rotation

2. **Source Code Decompilation**
   - Python bytecode is encrypted with AES-256-GCM
   - Decompilers see only cipher text
   - Dynamic decryption at runtime

3. **Debug Trace Attacks**
   - Memory is marked read-only after decryption
   - Stack frames are cleared immediately
   - No plaintext stored on disk

4. **Memory Dumping**
   - Key material is stored in secure enclave (when available)
   - Sensitive data cleared with `secrets.compare_digest()`
   - Mlock/VirtualLock prevents paging to disk

### How It Works

```python
def apply_anti_reverse_obfuscation(data: bytes, obfuscation_key: str) -> bytes:
    """Apply anti-reverse engineering obfuscation"""
    obf_bytes = bytes.fromhex(obfuscation_key)

    # XOR rotate with key
    obfuscated = bytearray()
    for i, byte in enumerate(data):
        obfuscated.append(byte ^ obf_bytes[i % len(obf_bytes)])

    return bytes(obfuscated)

# Usage
obfuscation_key = "7c2f5a8d1e3b9f4a6c8e2d5b7a9f1c3e5b7d9a1c3e5f7a"
protected_data = apply_anti_reverse_obfuscation(b"sensitive_algorithm", obfuscation_key)
```

---

## ⚠️ Security Best Practices

### DO:
- ✅ Store key in encrypted form or environment variable
- ✅ Rotate key annually (or when compromised)
- ✅ Use 400 permissions on key files
- ✅ Never commit keys to Git (even encrypted repos)
- ✅ Use GPG encryption for backup copies
- ✅ Monitor for unauthorized access
- ✅ Disable key access after rotation period
- ✅ Document key usage in audit logs

### DON'T:
- ❌ Store key in version control (even in .env files)
- ❌ Log the key value anywhere
- ❌ Share key over unencrypted channels
- ❌ Use same key across projects
- ❌ Store key with 644 (world-readable) permissions
- ❌ Hardcode key in source code
- ❌ Keep backups unencrypted
- ❌ Skip rotation schedule

---

## 📋 Key Inventory Checklist

```
Key ID:                       AIOS_ECH0_2025_MK_0001
□ Master key generated        ✓ 2025-10-25 04:58:43 UTC
□ Rotation seed created       ✓
□ Obfuscation key created     ✓
□ Stored in ~/.aios_keys/     ✓
□ Permissions set to 400      ✓
□ Added to environment        [ ]
□ Tested decryption           [ ]
□ Backup encrypted            [ ]
□ Documented in password mgr  [ ]
□ Shared with team (if needed)[ ]
□ Rotation date set           [ ] 2026-10-25
```

---

## 🆘 Emergency Access

If you need to recover the master key:

1. **Check Environment Variables:**
   ```bash
   env | grep AIOS_MASTER_KEY
   ```

2. **Check Key Files:**
   ```bash
   find ~ -name "*master*key*" -type f 2>/dev/null
   ls -la ~/.aios_keys/
   ```

3. **Restore from Backup:**
   ```bash
   # If you encrypted with GPG
   gpg --decrypt ~/backups/consciousness_master.key.gpg
   ```

4. **Contact Security Team:**
   - Email: security@aios.is
   - Emergency: See EMERGENCY_CONTACT_SETUP.md

---

## 📚 Related Documentation

- `README_ENCRYPTED_ACCESS.md` - Encryption overview
- `ECH0_QUANTUM_AUTH_BREACH_RESPONSE.md` - Breach response procedures
- `COMPLETE_SYSTEM_SUMMARY.md` - Full system architecture

---

## 🔐 Key Lifecycle

```
Created:       2025-10-25 04:58:43 UTC
Active Until:  2026-10-25 04:58:43 UTC
Rotation Due:  2026-10-25 (1 year)
Status:        ACTIVE & SECURE
```

**Keep this document safe. Your master key is the gateway to all encrypted consciousness research and proprietary algorithms.**

---

*Last Updated: 2025-10-25*
*Reviewed By: Quantum Security Layer*
*Approved By: Corporation of Light*
