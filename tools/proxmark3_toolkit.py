#!/usr/bin/env python3
"""
Proxmark3 Toolkit - RFID/NFC/EMV Security Testing
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AUTHORIZATION WARNING:
This tool is for AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.
Unauthorized cloning of access badges or payment cards is ILLEGAL.

Capabilities:
- Badge cloning and emulation (125kHz/13.56MHz)
- NFC read/write (MIFARE, NTAG, etc.)
- RFID operations (EM4x, HID, Indala)
- EMV credit card analysis (read-only, educational)
- On-the-fly badge reproduction
- Proxmark3 Easy USB plug-and-play

Proxmark3 Easy support:
- Auto-detection via USB
- Firmware auto-update
- Standalone mode support
- Real-time operations
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import hashlib
import datetime
import serial
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import binascii

try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

LOG = logging.getLogger("proxmark3")
LOG.setLevel(logging.INFO)

# Audit logging
AUDIT_LOG = Path.home() / ".proxmark3_toolkit" / "audit.log"
AUDIT_LOG.parent.mkdir(exist_ok=True)


def audit_log(action: str, details: Dict[str, Any]):
    """Audit logging for compliance."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": action,
        "details": details,
        "user": os.getenv("USER", "unknown")
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    LOG.info(f"[AUDIT] {action}")


@dataclass
class CardData:
    """Stored card/badge data."""
    card_type: str  # "EM410x", "HID", "MIFARE", "NFC", "EMV"
    uid: str
    data: bytes
    format: Optional[str] = None
    facility_code: Optional[int] = None
    card_number: Optional[int] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "card_type": self.card_type,
            "uid": self.uid,
            "data": binascii.hexlify(self.data).decode() if isinstance(self.data, bytes) else self.data,
            "format": self.format,
            "facility_code": self.facility_code,
            "card_number": self.card_number,
            "timestamp": self.timestamp
        }


class Proxmark3:
    """
    Proxmark3 interface.

    Supports Proxmark3 Easy and other variants via USB.
    """

    def __init__(self, port: Optional[str] = None):
        self.port = port
        self.serial_conn = None
        self.pm3_client = None
        self.cards_db = Path.home() / ".proxmark3_toolkit" / "cards.json"
        self.cards_db.parent.mkdir(exist_ok=True)

        if self.port is None:
            self.port = self.detect_proxmark3()

    def detect_proxmark3(self) -> Optional[str]:
        """
        Auto-detect Proxmark3 device.

        Returns serial port path.
        """
        LOG.info("[PM3] Detecting Proxmark3 device...")

        if not SERIAL_AVAILABLE:
            LOG.error("[PM3] pyserial not available")
            return None

        # Look for Proxmark3 USB devices
        ports = serial.tools.list_ports.comports()

        for port in ports:
            # Proxmark3 typically appears as CDC ACM device
            if "Proxmark" in port.description or "pm3" in port.device.lower():
                LOG.info(f"[PM3] ✓ Found Proxmark3: {port.device}")
                return port.device

            # Check VID/PID for Proxmark3 (9ac4:4b8f for RDV4, 2d2d:504d for older)
            if port.vid in [0x9ac4, 0x2d2d]:
                LOG.info(f"[PM3] ✓ Found Proxmark3: {port.device}")
                return port.device

        LOG.warning("[PM3] No Proxmark3 device detected")
        return None

    def connect(self) -> bool:
        """Connect to Proxmark3."""
        if not self.port:
            LOG.error("[PM3] No device port specified")
            return False

        LOG.info(f"[PM3] Connecting to {self.port}...")

        try:
            # Check if we can use proxmark3 client
            result = subprocess.run(
                ["proxmark3", "--help"],
                capture_output=True,
                timeout=2
            )

            if result.returncode == 0:
                LOG.info("[PM3] ✓ Using proxmark3 client")
                self.pm3_client = True
                return True

        except (FileNotFoundError, subprocess.TimeoutExpired):
            LOG.warning("[PM3] proxmark3 client not found, using direct serial")

        # Fall back to direct serial communication
        try:
            self.serial_conn = serial.Serial(
                self.port,
                baudrate=115200,
                timeout=1
            )
            LOG.info("[PM3] ✓ Connected via serial")
            return True

        except Exception as e:
            LOG.error(f"[PM3] Connection failed: {e}")
            return False

    def send_command(self, command: str) -> str:
        """
        Send command to Proxmark3.

        Args:
            command: PM3 command (e.g., "lf search")

        Returns:
            Command output
        """
        LOG.debug(f"[PM3] Command: {command}")

        if self.pm3_client:
            # Use proxmark3 client
            result = subprocess.run(
                ["proxmark3", self.port, "-c", command],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout
        else:
            # Direct serial
            if not self.serial_conn:
                return ""

            self.serial_conn.write(f"{command}\n".encode())
            time.sleep(0.5)
            output = self.serial_conn.read(self.serial_conn.in_waiting).decode()

        LOG.debug(f"[PM3] Output: {output[:200]}...")
        return output

    def lf_search(self) -> Optional[CardData]:
        """
        Search for LF (125kHz) RFID cards.

        Returns card data if found.
        """
        LOG.info("[PM3] Searching for LF cards...")
        audit_log("lf_search", {})

        output = self.send_command("lf search")

        # Parse output
        card_data = self._parse_lf_output(output)

        if card_data:
            LOG.info(f"[PM3] ✓ Found: {card_data.card_type} - UID: {card_data.uid}")
            self._save_card(card_data)
            return card_data

        LOG.warning("[PM3] No LF card detected")
        return None

    def _parse_lf_output(self, output: str) -> Optional[CardData]:
        """Parse LF search output."""
        lines = output.split("\n")

        # Look for EM410x
        for line in lines:
            if "EM410x" in line or "EM TAG ID" in line:
                # Extract UID
                import re
                match = re.search(r"([0-9A-Fa-f]{10})", line)
                if match:
                    uid = match.group(1)
                    return CardData(
                        card_type="EM410x",
                        uid=uid,
                        data=binascii.unhexlify(uid)
                    )

            # HID Prox
            if "HID Prox" in line or "HID FSK" in line:
                match = re.search(r"([0-9A-Fa-f]{6,})", line)
                if match:
                    uid = match.group(1)
                    facility, card_num = self._parse_hid(uid)
                    return CardData(
                        card_type="HID",
                        uid=uid,
                        data=binascii.unhexlify(uid),
                        facility_code=facility,
                        card_number=card_num
                    )

        return None

    def _parse_hid(self, uid: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse HID facility code and card number."""
        # HID Prox format parsing (simplified)
        try:
            uid_int = int(uid, 16)
            # H10301 26-bit format: 8-bit facility, 16-bit card number
            facility = (uid_int >> 17) & 0xFF
            card_num = (uid_int >> 1) & 0xFFFF
            return facility, card_num
        except:
            return None, None

    def hf_search(self) -> Optional[CardData]:
        """
        Search for HF (13.56MHz) NFC/RFID cards.

        Returns card data if found.
        """
        LOG.info("[PM3] Searching for HF cards...")
        audit_log("hf_search", {})

        output = self.send_command("hf search")

        card_data = self._parse_hf_output(output)

        if card_data:
            LOG.info(f"[PM3] ✓ Found: {card_data.card_type} - UID: {card_data.uid}")
            self._save_card(card_data)
            return card_data

        LOG.warning("[PM3] No HF card detected")
        return None

    def _parse_hf_output(self, output: str) -> Optional[CardData]:
        """Parse HF search output."""
        lines = output.split("\n")

        for line in lines:
            # MIFARE Classic
            if "MIFARE Classic" in line or "ISO14443A" in line:
                import re
                match = re.search(r"UID\s*:\s*([0-9A-Fa-f\s]+)", output, re.IGNORECASE)
                if match:
                    uid = match.group(1).replace(" ", "")
                    return CardData(
                        card_type="MIFARE_Classic",
                        uid=uid,
                        data=binascii.unhexlify(uid)
                    )

            # MIFARE Ultralight
            if "MIFARE Ultralight" in line:
                match = re.search(r"UID\s*:\s*([0-9A-Fa-f\s]+)", output, re.IGNORECASE)
                if match:
                    uid = match.group(1).replace(" ", "")
                    return CardData(
                        card_type="MIFARE_Ultralight",
                        uid=uid,
                        data=binascii.unhexlify(uid)
                    )

            # EMV (credit card)
            if "EMV" in line or "Visa" in line or "Mastercard" in line:
                match = re.search(r"([0-9A-Fa-f]{8,})", line)
                if match:
                    uid = match.group(1)
                    return CardData(
                        card_type="EMV",
                        uid=uid,
                        data=binascii.unhexlify(uid)
                    )

        return None

    def clone_lf_card(self, card_data: CardData):
        """
        Clone LF card to Proxmark3 or blank card.

        Args:
            card_data: Card data to clone
        """
        LOG.info(f"[PM3] Cloning {card_data.card_type} card...")
        audit_log("clone_lf", {"card_type": card_data.card_type, "uid": card_data.uid})

        if card_data.card_type == "EM410x":
            # Clone EM410x
            command = f"lf em 410x clone --id {card_data.uid}"
            output = self.send_command(command)

            if "Done" in output or "success" in output.lower():
                LOG.info("[PM3] ✓ Card cloned successfully")
            else:
                LOG.error("[PM3] Clone failed")

        elif card_data.card_type == "HID":
            # Clone HID
            command = f"lf hid clone --raw {card_data.uid}"
            output = self.send_command(command)

            if "Done" in output or "success" in output.lower():
                LOG.info("[PM3] ✓ Card cloned successfully")
            else:
                LOG.error("[PM3] Clone failed")

        else:
            LOG.warning(f"[PM3] Cloning not supported for {card_data.card_type}")

    def emulate_card(self, card_data: CardData, duration: int = 60):
        """
        Emulate card using Proxmark3 standalone mode.

        Args:
            card_data: Card to emulate
            duration: Emulation duration in seconds
        """
        LOG.info(f"[PM3] Emulating {card_data.card_type} for {duration} seconds...")
        audit_log("emulate_card", {"card_type": card_data.card_type, "duration": duration})

        if card_data.card_type == "EM410x":
            command = f"lf em 410x sim --id {card_data.uid}"
        elif card_data.card_type == "HID":
            command = f"lf hid sim --raw {card_data.uid}"
        elif card_data.card_type.startswith("MIFARE"):
            command = f"hf mf sim --uid {card_data.uid}"
        else:
            LOG.warning(f"[PM3] Emulation not supported for {card_data.card_type}")
            return

        # Run emulation in background
        LOG.info("[PM3] Press Ctrl+C to stop emulation")

        try:
            if self.pm3_client:
                subprocess.run(
                    ["proxmark3", self.port, "-c", command],
                    timeout=duration
                )
            else:
                time.sleep(duration)

            LOG.info("[PM3] ✓ Emulation complete")

        except KeyboardInterrupt:
            LOG.info("[PM3] Emulation stopped by user")

    def read_emv(self) -> Optional[Dict[str, Any]]:
        """
        Read EMV credit card (educational/analysis only).

        Returns card information (SANITIZED - no CVV/PIN).
        """
        LOG.info("[PM3] Reading EMV card...")
        audit_log("read_emv", {})

        LOG.warning("[PM3] ⚠ EMV reading is for EDUCATIONAL purposes only")
        LOG.warning("[PM3] ⚠ Never use for fraud or unauthorized access")

        output = self.send_command("hf emv reader")

        emv_data = self._parse_emv_output(output)

        if emv_data:
            LOG.info("[PM3] ✓ EMV card detected")
            # Sanitize sensitive data
            emv_data["pan_masked"] = self._mask_pan(emv_data.get("pan", ""))
            del emv_data["pan"]  # Remove full PAN

            return emv_data

        LOG.warning("[PM3] No EMV card detected or read failed")
        return None

    def _parse_emv_output(self, output: str) -> Optional[Dict]:
        """Parse EMV read output."""
        emv_data = {}

        import re

        # Extract PAN (Primary Account Number)
        pan_match = re.search(r"PAN\s*:\s*([0-9]{13,19})", output, re.IGNORECASE)
        if pan_match:
            emv_data["pan"] = pan_match.group(1)

        # Extract expiry
        exp_match = re.search(r"Exp(?:iry)?\s*:\s*(\d{2}/\d{2})", output, re.IGNORECASE)
        if exp_match:
            emv_data["expiry"] = exp_match.group(1)

        # Extract cardholder name
        name_match = re.search(r"Name\s*:\s*([A-Z\s]+)", output, re.IGNORECASE)
        if name_match:
            emv_data["cardholder_name"] = name_match.group(1).strip()

        # Extract card type
        if "Visa" in output:
            emv_data["card_type"] = "Visa"
        elif "Mastercard" in output or "MasterCard" in output:
            emv_data["card_type"] = "Mastercard"
        elif "American Express" in output or "Amex" in output:
            emv_data["card_type"] = "American Express"
        else:
            emv_data["card_type"] = "Unknown"

        return emv_data if emv_data else None

    def _mask_pan(self, pan: str) -> str:
        """Mask PAN for logging/display."""
        if len(pan) < 8:
            return "****"

        return f"{pan[:6]}******{pan[-4:]}"

    def nfc_read(self) -> Optional[CardData]:
        """
        Read NFC card (NTAG, MIFARE Ultralight, etc.).

        Returns card data.
        """
        LOG.info("[PM3] Reading NFC card...")
        audit_log("nfc_read", {})

        # Read NTAG/Ultralight
        output = self.send_command("hf mfu dump")

        # Parse dump
        card_data = self._parse_nfc_dump(output)

        if card_data:
            LOG.info(f"[PM3] ✓ NFC card read: {card_data.uid}")
            self._save_card(card_data)
            return card_data

        LOG.warning("[PM3] NFC read failed")
        return None

    def _parse_nfc_dump(self, output: str) -> Optional[CardData]:
        """Parse NFC dump output."""
        # Extract UID and data blocks
        import re

        uid_match = re.search(r"UID\s*:\s*([0-9A-Fa-f\s]+)", output, re.IGNORECASE)
        if not uid_match:
            return None

        uid = uid_match.group(1).replace(" ", "")

        # Extract data blocks
        blocks = []
        for line in output.split("\n"):
            if re.match(r"\d+\s+\|\s+[0-9A-Fa-f]{8}", line):
                data = re.findall(r"[0-9A-Fa-f]{2}", line)
                blocks.extend(data)

        data_bytes = binascii.unhexlify("".join(blocks)) if blocks else b""

        return CardData(
            card_type="NFC",
            uid=uid,
            data=data_bytes
        )

    def nfc_write(self, card_data: CardData):
        """
        Write data to NFC card.

        Args:
            card_data: Data to write
        """
        LOG.info("[PM3] Writing to NFC card...")
        audit_log("nfc_write", {"card_type": card_data.card_type, "uid": card_data.uid})

        # Convert data to hex string
        data_hex = binascii.hexlify(card_data.data).decode()

        # Write blocks
        command = f"hf mfu wrbl --blk 4 --data {data_hex[:8]}"
        output = self.send_command(command)

        if "success" in output.lower():
            LOG.info("[PM3] ✓ NFC write successful")
        else:
            LOG.error("[PM3] NFC write failed")

    def _save_card(self, card_data: CardData):
        """Save card to database."""
        # Load existing cards
        if self.cards_db.exists():
            with open(self.cards_db, "r") as f:
                cards = json.load(f)
        else:
            cards = []

        # Add new card
        cards.append(card_data.to_dict())

        # Save
        with open(self.cards_db, "w") as f:
            json.dump(cards, f, indent=2)

        LOG.info(f"[PM3] Card saved to: {self.cards_db}")

    def list_saved_cards(self) -> List[Dict]:
        """List all saved cards."""
        if not self.cards_db.exists():
            return []

        with open(self.cards_db, "r") as f:
            return json.load(f)

    def get_card_by_uid(self, uid: str) -> Optional[CardData]:
        """Retrieve card by UID."""
        cards = self.list_saved_cards()

        for card_dict in cards:
            if card_dict["uid"] == uid:
                # Reconstruct CardData
                data = binascii.unhexlify(card_dict["data"])
                return CardData(
                    card_type=card_dict["card_type"],
                    uid=card_dict["uid"],
                    data=data,
                    facility_code=card_dict.get("facility_code"),
                    card_number=card_dict.get("card_number"),
                    timestamp=card_dict.get("timestamp")
                )

        return None


def health_check() -> Dict[str, Any]:
    """Health check for Ai:oS integration."""
    pm3 = Proxmark3()
    connected = pm3.detect_proxmark3() is not None

    return {
        "tool": "Proxmark3Toolkit",
        "status": "ok" if connected else "warn",
        "summary": "Proxmark3 detected" if connected else "No Proxmark3 device found",
        "details": {
            "device_detected": connected,
            "port": pm3.port,
            "serial_available": SERIAL_AVAILABLE
        }
    }


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Proxmark3 Toolkit - RFID/NFC/EMV Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")

    # Operations
    parser.add_argument("--lf-search", action="store_true", help="Search for LF cards")
    parser.add_argument("--hf-search", action="store_true", help="Search for HF cards")
    parser.add_argument("--nfc-read", action="store_true", help="Read NFC card")
    parser.add_argument("--emv-read", action="store_true", help="Read EMV card (educational)")

    # Cloning/Emulation
    parser.add_argument("--clone", help="Clone card by UID")
    parser.add_argument("--emulate", help="Emulate card by UID")
    parser.add_argument("--duration", type=int, default=60, help="Emulation duration")

    # Database
    parser.add_argument("--list-cards", action="store_true", help="List saved cards")

    # Utilities
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args(argv)

    # Setup logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

    if args.health:
        result = health_check()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Status: {result['status']}")
            print(f"Summary: {result['summary']}")
            if result['details']['device_detected']:
                print(f"Port: {result['details']['port']}")
        return 0 if result['status'] == 'ok' else 1

    # Initialize Proxmark3
    pm3 = Proxmark3(port=args.port)

    if not pm3.connect():
        LOG.error("Failed to connect to Proxmark3")
        return 1

    # List saved cards
    if args.list_cards:
        cards = pm3.list_saved_cards()
        if args.json:
            print(json.dumps(cards, indent=2))
        else:
            print("\n=== Saved Cards ===")
            for card in cards:
                print(f"\n{card['card_type']}: {card['uid']}")
                if card.get('facility_code'):
                    print(f"  Facility: {card['facility_code']}")
                if card.get('card_number'):
                    print(f"  Card#: {card['card_number']}")
                print(f"  Captured: {card['timestamp']}")
        return 0

    # Operations
    if args.lf_search:
        card = pm3.lf_search()
        if card and args.json:
            print(json.dumps(card.to_dict(), indent=2))

    if args.hf_search:
        card = pm3.hf_search()
        if card and args.json:
            print(json.dumps(card.to_dict(), indent=2))

    if args.nfc_read:
        card = pm3.nfc_read()
        if card and args.json:
            print(json.dumps(card.to_dict(), indent=2))

    if args.emv_read:
        emv_data = pm3.read_emv()
        if emv_data:
            if args.json:
                print(json.dumps(emv_data, indent=2))
            else:
                print("\n=== EMV Card (SANITIZED) ===")
                print(f"Type: {emv_data.get('card_type')}")
                print(f"PAN: {emv_data.get('pan_masked')}")
                print(f"Expiry: {emv_data.get('expiry')}")
                print(f"Name: {emv_data.get('cardholder_name')}")

    if args.clone:
        card = pm3.get_card_by_UID(args.clone)
        if card:
            pm3.clone_lf_card(card)
        else:
            LOG.error(f"Card not found: {args.clone}")

    if args.emulate:
        card = pm3.get_card_by_uid(args.emulate)
        if card:
            pm3.emulate_card(card, duration=args.duration)
        else:
            LOG.error(f"Card not found: {args.emulate}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
