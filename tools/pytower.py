#!/usr/bin/env python3
"""
PyTower - Portable Cellular Base Station
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

⚠️ CRITICAL LEGAL WARNING ⚠️

UNAUTHORIZED CELLULAR TRANSMISSION IS A FEDERAL CRIME

This tool is ONLY for:
- FCC-licensed emergency response operations
- Authorized disaster relief deployments
- Licensed telecommunications testing
- Approved FirstNet/public safety networks

Penalties for unauthorized use:
- Up to $1,000,000 fine PER VIOLATION (47 U.S.C. § 503)
- Federal imprisonment up to 5 years
- Immediate FCC enforcement action
- Permanent ban from RF spectrum use

You MUST have:
1. FCC Part 27/90 license OR emergency authorization
2. Frequency coordination clearance
3. Written authorization from affected carriers (if not standalone)
4. Proof of insurance for spectrum interference

DO NOT OPERATE WITHOUT PROPER LICENSE AND AUTHORIZATION

Based on technical recommendations from ECH0 14B:
- Software: srsRAN for LTE (lightweight, modular)
- Hardware: LimeSDR (portable, flexible)
- Platform: BeagleBone AI / Odroid N2 / Intel NUC (not Raspberry Pi)
- Network: Standalone mode (private LTE network)
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import socket

LOG = logging.getLogger("pytower")
LOG.setLevel(logging.INFO)

AUDIT_LOG = Path.home() / ".pytower" / "audit.log"
AUDIT_LOG.parent.mkdir(exist_ok=True)


def audit_log(action: str, details: Dict[str, Any]):
    """Audit logging for regulatory compliance."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": action,
        "details": details,
        "user": os.getenv("USER", "unknown"),
        "hostname": socket.gethostname()
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    LOG.info(f"[AUDIT] {action}")


@dataclass
class LicenseInfo:
    """FCC License information."""
    license_number: str
    licensee_name: str
    frequency_range: str  # e.g., "2.5GHz-2.7GHz"
    authorized_locations: List[str]
    expiration_date: str
    license_type: str  # "emergency", "experimental", "commercial"


@dataclass
class PyTowerConfig:
    """Configuration for PyTower base station."""
    # Network configuration
    mcc: str = "001"  # Mobile Country Code (001 = test network)
    mnc: str = "01"   # Mobile Network Code
    tac: int = 7      # Tracking Area Code
    cell_id: int = 1
    pci: int = 1      # Physical Cell ID

    # RF configuration
    dl_earfcn: int = 3350  # Downlink EARFCN (frequency channel)
    ul_earfcn: int = 21350 # Uplink EARFCN
    tx_power: float = 20.0  # dBm (max 20 for portable)
    rx_gain: float = 40.0   # dB

    # SDR configuration
    sdr_type: str = "limesdr"  # limesdr, bladerf, usrp
    sdr_device: str = "auto"

    # Network mode
    standalone: bool = True
    emergency_mode: bool = False

    # License info (REQUIRED)
    license: Optional[LicenseInfo] = None

    # Deployment info
    deployment_location: str = ""
    deployment_purpose: str = ""


class LicenseManager:
    """Manage FCC licenses and regulatory compliance."""

    def __init__(self):
        self.license_file = Path.home() / ".pytower" / "license.json"
        self.license_file.parent.mkdir(exist_ok=True)

    def verify_license(self, config: PyTowerConfig) -> bool:
        """
        Verify FCC license before allowing operation.

        Returns True only if valid license exists.
        """
        LOG.info("[LICENSE] Verifying FCC authorization...")

        if not config.license:
            LOG.error("[LICENSE] ❌ NO LICENSE PROVIDED")
            LOG.error("[LICENSE] You must provide FCC license information")
            LOG.error("[LICENSE] Run with --setup-license to configure")
            return False

        license_info = config.license

        # Check license file exists
        if not self.license_file.exists():
            LOG.error("[LICENSE] ❌ No license file found at: %s", self.license_file)
            LOG.error("[LICENSE] Run with --setup-license to create")
            return False

        # Load and verify license
        with open(self.license_file, "r") as f:
            stored_licenses = json.load(f)

        # Check if license number matches
        valid = False
        for lic in stored_licenses.get("licenses", []):
            if lic.get("license_number") == license_info.license_number:
                # Check expiration
                exp_date = datetime.datetime.fromisoformat(lic.get("expiration_date"))
                if exp_date < datetime.datetime.utcnow():
                    LOG.error("[LICENSE] ❌ License EXPIRED: %s", exp_date)
                    return False

                # Check authorized location
                if config.deployment_location not in lic.get("authorized_locations", []):
                    LOG.warning("[LICENSE] ⚠ Location not explicitly authorized")
                    LOG.warning("[LICENSE] Deployment: %s", config.deployment_location)
                    LOG.warning("[LICENSE] Authorized: %s", lic.get("authorized_locations"))

                    if not config.emergency_mode:
                        LOG.error("[LICENSE] ❌ Location not authorized for non-emergency")
                        return False

                valid = True
                LOG.info("[LICENSE] ✓ License valid: %s", license_info.license_number)
                LOG.info("[LICENSE] ✓ Licensee: %s", lic.get("licensee_name"))
                LOG.info("[LICENSE] ✓ Type: %s", lic.get("license_type"))
                LOG.info("[LICENSE] ✓ Expires: %s", exp_date.strftime("%Y-%m-%d"))
                break

        if not valid:
            LOG.error("[LICENSE] ❌ License not found in database")
            return False

        audit_log("license_verification", {
            "license": license_info.license_number,
            "result": "approved",
            "location": config.deployment_location
        })

        return True

    def setup_license(self):
        """Interactive license setup."""
        print("=" * 60)
        print("PyTower FCC License Setup")
        print("=" * 60)
        print()
        print("⚠️  WARNING: Providing false license information is a FEDERAL CRIME")
        print("You must have a valid FCC Part 27/90 license or emergency authorization")
        print()

        license_data = {
            "licenses": []
        }

        print("Enter your FCC license details:")
        print()

        license_number = input("License Number (e.g., WXY1234): ").strip()
        licensee_name = input("Licensee Name: ").strip()
        frequency_range = input("Authorized Frequency Range (e.g., 2.5-2.7 GHz): ").strip()
        license_type = input("License Type (emergency/experimental/commercial): ").strip()

        print()
        print("Authorized locations (comma-separated):")
        locations_str = input("> ").strip()
        locations = [loc.strip() for loc in locations_str.split(",")]

        print()
        expiration = input("Expiration Date (YYYY-MM-DD): ").strip()

        license_entry = {
            "license_number": license_number,
            "licensee_name": licensee_name,
            "frequency_range": frequency_range,
            "authorized_locations": locations,
            "expiration_date": expiration,
            "license_type": license_type,
            "created": datetime.datetime.utcnow().isoformat()
        }

        license_data["licenses"].append(license_entry)

        # Save license file
        with open(self.license_file, "w") as f:
            json.dump(license_data, f, indent=2)

        print()
        print("=" * 60)
        print(f"✓ License saved to: {self.license_file}")
        print("=" * 60)
        print()
        print("⚠️  REMINDER: This does not grant you a license.")
        print("You must obtain proper FCC authorization separately.")
        print("Contact FCC Licensing Division: https://www.fcc.gov/licensing")
        print()


class SRSRANController:
    """
    srsRAN LTE base station controller.

    srsRAN provides open-source LTE eNodeB implementation.
    """

    def __init__(self, config: PyTowerConfig):
        self.config = config
        self.enb_process = None
        self.epc_process = None

    def check_dependencies(self) -> bool:
        """Check if srsRAN is installed."""
        try:
            result = subprocess.run(
                ["srsenb", "--version"],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                LOG.info("[SRSRAN] ✓ srsRAN installed")
                return True

        except FileNotFoundError:
            LOG.error("[SRSRAN] ❌ srsRAN not found")
            LOG.error("[SRSRAN] Install: https://github.com/srsran/srsRAN")
            LOG.error("[SRSRAN]   sudo apt install srsran")
        except subprocess.TimeoutExpired:
            pass

        return False

    def generate_config(self) -> Path:
        """Generate srsRAN configuration files."""
        LOG.info("[SRSRAN] Generating configuration...")

        config_dir = Path.home() / ".pytower" / "srsran_config"
        config_dir.mkdir(parents=True, exist_ok=True)

        # eNodeB configuration
        enb_config = f"""
[enb]
enb_id = {self.config.cell_id}
mcc = {self.config.mcc}
mnc = {self.config.mnc}
mme_addr = 127.0.0.1
gtp_bind_addr = 127.0.0.1
s1c_bind_addr = 127.0.0.1
n_prb = 50
tm = 1
nof_ports = 1

[enb_files]
sib_config = sib.conf
rr_config = rr.conf
rb_config = rb.conf

[rf]
dl_earfcn = {self.config.dl_earfcn}
tx_gain = {self.config.tx_power}
rx_gain = {self.config.rx_gain}
device_name = {self.config.sdr_type}
device_args = auto

[pcap]
enable = false
"""

        enb_conf_file = config_dir / "enb.conf"
        with open(enb_conf_file, "w") as f:
            f.write(enb_config)

        # Simple SIB configuration
        sib_config = """
sib1 =
{
  intra_freq_reselection = "Allowed";
  q_rx_lev_min = -140;
  p_max = 23;
  cell_barred = "NotBarred";
  si_window_length = 20;

  sched_info =
  (
    {
      si_periodicity = 16;
      si_mapping_info = [];
    }
  );
};
"""

        sib_conf_file = config_dir / "sib.conf"
        with open(sib_conf_file, "w") as f:
            f.write(sib_config)

        LOG.info(f"[SRSRAN] ✓ Configuration generated: {config_dir}")

        return enb_conf_file

    def start_epc(self):
        """Start srsEPC (Evolved Packet Core)."""
        LOG.info("[SRSRAN] Starting EPC...")

        self.epc_process = subprocess.Popen(
            ["srsepc"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(2)  # Wait for EPC to start

        if self.epc_process.poll() is None:
            LOG.info("[SRSRAN] ✓ EPC running")
        else:
            LOG.error("[SRSRAN] ❌ EPC failed to start")
            raise RuntimeError("EPC startup failed")

    def start_enb(self, config_file: Path):
        """Start srsENB (eNodeB)."""
        LOG.info("[SRSRAN] Starting eNodeB...")

        self.enb_process = subprocess.Popen(
            ["srsenb", str(config_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(3)  # Wait for eNodeB to start

        if self.enb_process.poll() is None:
            LOG.info("[SRSRAN] ✓ eNodeB running")
        else:
            LOG.error("[SRSRAN] ❌ eNodeB failed to start")
            raise RuntimeError("eNodeB startup failed")

    def stop(self):
        """Stop srsRAN processes."""
        LOG.info("[SRSRAN] Shutting down...")

        if self.enb_process:
            self.enb_process.terminate()
            self.enb_process.wait(timeout=10)

        if self.epc_process:
            self.epc_process.terminate()
            self.epc_process.wait(timeout=10)

        LOG.info("[SRSRAN] ✓ Shutdown complete")


class PyTower:
    """
    Main PyTower orchestrator.

    Manages portable LTE base station deployment.
    """

    def __init__(self, config: PyTowerConfig):
        self.config = config
        self.license_mgr = LicenseManager()
        self.srsran = SRSRANController(config)
        self.running = False

    def deploy(self):
        """Deploy the cellular base station."""
        LOG.info("=" * 60)
        LOG.info("PyTower - Portable Cellular Base Station")
        LOG.info("=" * 60)

        # 1. License verification (CRITICAL)
        if not self.license_mgr.verify_license(self.config):
            LOG.error("=" * 60)
            LOG.error("❌ DEPLOYMENT BLOCKED - NO VALID LICENSE")
            LOG.error("=" * 60)
            LOG.error("You MUST have FCC authorization to operate")
            LOG.error("Unauthorized operation is a FEDERAL CRIME")
            LOG.error("=" * 60)
            return 1

        LOG.info("=" * 60)
        LOG.info("✓ License verification PASSED")
        LOG.info("=" * 60)

        # 2. Check dependencies
        if not self.srsran.check_dependencies():
            LOG.error("❌ srsRAN not installed")
            return 1

        # 3. Audit log deployment
        audit_log("deployment_start", {
            "location": self.config.deployment_location,
            "purpose": self.config.deployment_purpose,
            "license": self.config.license.license_number if self.config.license else None,
            "emergency_mode": self.config.emergency_mode,
            "mcc_mnc": f"{self.config.mcc}-{self.config.mnc}",
            "cell_id": self.config.cell_id
        })

        # 4. Generate configuration
        config_file = self.srsran.generate_config()

        # 5. Start base station
        try:
            LOG.info("[PYTOWER] Starting base station...")

            # Start EPC (core network)
            self.srsran.start_epc()

            # Start eNodeB (radio)
            self.srsran.start_enb(config_file)

            self.running = True

            LOG.info("=" * 60)
            LOG.info("✓ PyTower OPERATIONAL")
            LOG.info("=" * 60)
            LOG.info(f"Network: {self.config.mcc}-{self.config.mnc}")
            LOG.info(f"Cell ID: {self.config.cell_id}")
            LOG.info(f"PCI: {self.config.pci}")
            LOG.info(f"EARFCN: DL={self.config.dl_earfcn} / UL={self.config.ul_earfcn}")
            LOG.info(f"TX Power: {self.config.tx_power} dBm")
            LOG.info(f"Location: {self.config.deployment_location}")
            LOG.info(f"Purpose: {self.config.deployment_purpose}")
            LOG.info("=" * 60)
            LOG.info("Base station is broadcasting. Devices can now connect.")
            LOG.info("Press Ctrl+C to shutdown")
            LOG.info("=" * 60)

            # Keep running
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            LOG.info("\n[PYTOWER] Shutdown requested...")

        except Exception as e:
            LOG.error(f"[PYTOWER] ❌ Deployment failed: {e}")
            return 1

        finally:
            self.shutdown()

        return 0

    def shutdown(self):
        """Shutdown the base station."""
        LOG.info("[PYTOWER] Shutting down base station...")

        self.running = False
        self.srsran.stop()

        audit_log("deployment_stop", {
            "location": self.config.deployment_location
        })

        LOG.info("[PYTOWER] ✓ Shutdown complete")


def health_check() -> Dict[str, Any]:
    """Health check."""
    # Check for srsRAN
    try:
        result = subprocess.run(
            ["srsenb", "--version"],
            capture_output=True,
            timeout=2
        )
        srsran_installed = result.returncode == 0
    except:
        srsran_installed = False

    # Check for LimeSDR
    limesdr_detected = False
    try:
        result = subprocess.run(
            ["LimeUtil", "--find"],
            capture_output=True,
            timeout=2
        )
        limesdr_detected = "LimeSDR" in result.stdout.decode()
    except:
        pass

    return {
        "tool": "PyTower",
        "status": "ok" if srsran_installed else "warn",
        "summary": "PyTower ready" if srsran_installed else "srsRAN not installed",
        "details": {
            "srsran": srsran_installed,
            "limesdr": limesdr_detected
        }
    }


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyTower - Portable Cellular Base Station (AUTHORIZED USE ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  CRITICAL LEGAL WARNING ⚠️

Unauthorized cellular transmission is a FEDERAL CRIME.

You MUST have FCC authorization before operating PyTower.

Penalties for unauthorized operation:
- Up to $1,000,000 fine per violation
- Federal imprisonment up to 5 years
- FCC enforcement action
- Spectrum license revocation

Only use for:
- Licensed emergency response
- FCC-approved disaster relief
- Authorized telecommunications testing

Contact FCC before deployment: https://www.fcc.gov/licensing
        """
    )

    # License management
    parser.add_argument("--setup-license", action="store_true", help="Setup FCC license")
    parser.add_argument("--license-number", help="FCC license number")

    # Deployment configuration
    parser.add_argument("--location", required=False, help="Deployment location")
    parser.add_argument("--purpose", required=False, help="Deployment purpose")
    parser.add_argument("--emergency", action="store_true", help="Emergency deployment mode")

    # Network configuration
    parser.add_argument("--mcc", default="001", help="Mobile Country Code (default: 001=test)")
    parser.add_argument("--mnc", default="01", help="Mobile Network Code")
    parser.add_argument("--cell-id", type=int, default=1, help="Cell ID")

    # RF configuration
    parser.add_argument("--dl-earfcn", type=int, default=3350, help="Downlink EARFCN")
    parser.add_argument("--tx-power", type=float, default=20.0, help="TX power (dBm)")
    parser.add_argument("--sdr", default="limesdr", choices=["limesdr", "bladerf", "usrp"])

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
            print(f"srsRAN: {'✓' if result['details']['srsran'] else '❌'}")
            print(f"LimeSDR: {'✓' if result['details']['limesdr'] else '❌'}")
        return 0 if result['status'] == 'ok' else 1

    # License setup
    if args.setup_license:
        license_mgr = LicenseManager()
        license_mgr.setup_license()
        return 0

    # Deployment requires license and location
    if not args.license_number:
        LOG.error("❌ --license-number required for deployment")
        LOG.error("Run --setup-license first to configure")
        return 1

    if not args.location:
        LOG.error("❌ --location required for deployment")
        return 1

    if not args.purpose:
        LOG.error("❌ --purpose required for deployment")
        return 1

    # Build configuration
    license_info = LicenseInfo(
        license_number=args.license_number,
        licensee_name="",  # Will be loaded from license file
        frequency_range="",
        authorized_locations=[],
        expiration_date="",
        license_type=""
    )

    config = PyTowerConfig(
        mcc=args.mcc,
        mnc=args.mnc,
        cell_id=args.cell_id,
        dl_earfcn=args.dl_earfcn,
        tx_power=args.tx_power,
        sdr_type=args.sdr,
        emergency_mode=args.emergency,
        license=license_info,
        deployment_location=args.location,
        deployment_purpose=args.purpose
    )

    # Deploy PyTower
    tower = PyTower(config)
    return tower.deploy()


if __name__ == "__main__":
    sys.exit(main())
