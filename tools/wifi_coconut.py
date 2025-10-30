#!/usr/bin/env python3
"""
WiFi Coconut Integration - Multi-Radio WiFi Analysis
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

WiFi Coconut support for PyThief
- 14 simultaneous radio monitoring
- Multi-channel capture
- Antenna diversity and management
- Real-time spectrum analysis
- Integration with evil twin attacks

AUTHORIZATION WARNING:
This tool is for AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import datetime

try:
    import pyshark
    from manuf import manuf
    COCONUT_DEPS = True
except ImportError:
    COCONUT_DEPS = False

LOG = logging.getLogger("wifi_coconut")
LOG.setLevel(logging.INFO)


@dataclass
class AntennaConfig:
    """Configuration for a single antenna/radio."""
    radio_id: int
    interface: str  # e.g., wlan0mon
    channel: int
    enabled: bool = True
    antenna_type: str = "internal"  # internal, external_omni, external_directional
    gain_dbi: float = 2.0  # Antenna gain in dBi
    power_dbm: int = 20  # TX power in dBm
    mode: str = "monitor"  # monitor, managed, ap


@dataclass
class CoconutConfig:
    """WiFi Coconut configuration."""
    num_radios: int = 14
    base_interface: str = "wlan"
    channels: List[int] = None  # List of channels to monitor
    channel_hopping: bool = True
    hop_interval: float = 0.5  # seconds
    capture_dir: str = "/tmp/coconut_captures"
    pcap_per_radio: bool = True
    enable_external_antennas: bool = False
    antenna_configs: List[AntennaConfig] = None

    def __post_init__(self):
        if self.channels is None:
            # Default: all 2.4GHz channels
            self.channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        if self.antenna_configs is None:
            # Auto-generate antenna configs
            self.antenna_configs = [
                AntennaConfig(
                    radio_id=i,
                    interface=f"{self.base_interface}{i}mon",
                    channel=self.channels[i % len(self.channels)]
                )
                for i in range(self.num_radios)
            ]


class WiFiCoconut:
    """
    WiFi Coconut Manager

    Manages multiple WiFi radios for simultaneous multi-channel monitoring.
    WiFi Coconut has 14 radios, but this supports any number.
    """

    def __init__(self, config: CoconutConfig):
        self.config = config
        self.radios = []
        self.capture_threads = []
        self.running = False
        self.mac_parser = manuf.MacParser() if COCONUT_DEPS else None

    def detect_radios(self) -> List[str]:
        """
        Detect available WiFi radios.

        Returns list of interface names.
        """
        LOG.info("[COCONUT] Detecting WiFi radios...")

        try:
            result = subprocess.run(
                ["iw", "dev"],
                capture_output=True,
                text=True,
                check=True
            )

            interfaces = []
            for line in result.stdout.split("\n"):
                if "Interface" in line:
                    iface = line.split()[1]
                    interfaces.append(iface)

            LOG.info(f"[COCONUT] ✓ Detected {len(interfaces)} radios: {interfaces}")
            return interfaces

        except Exception as e:
            LOG.error(f"[COCONUT] Failed to detect radios: {e}")
            return []

    def setup_radios(self):
        """
        Setup all radios in monitor mode.

        Creates monitor interfaces for each radio.
        """
        LOG.info(f"[COCONUT] Setting up {self.config.num_radios} radios...")

        # Detect available interfaces
        interfaces = self.detect_radios()

        if len(interfaces) < self.config.num_radios:
            LOG.warning(f"[COCONUT] Only {len(interfaces)} radios available, requested {self.config.num_radios}")
            self.config.num_radios = len(interfaces)

        # Setup each radio
        for i, antenna_config in enumerate(self.config.antenna_configs[:self.config.num_radios]):
            if i >= len(interfaces):
                break

            base_interface = interfaces[i]
            LOG.info(f"[COCONUT] Setting up radio {i}: {base_interface}")

            try:
                # Kill interfering processes
                subprocess.run(["airmon-ng", "check", "kill"], stderr=subprocess.DEVNULL)

                # Put interface in monitor mode
                subprocess.run(["ip", "link", "set", base_interface, "down"], check=True)
                subprocess.run(["iw", "dev", base_interface, "set", "monitor", "none"], check=True)
                subprocess.run(["ip", "link", "set", base_interface, "up"], check=True)

                # Set channel
                subprocess.run([
                    "iw", "dev", base_interface, "set", "channel", str(antenna_config.channel)
                ], check=True)

                # Set TX power if specified
                if antenna_config.power_dbm:
                    subprocess.run([
                        "iw", "dev", base_interface, "set", "txpower", "fixed",
                        str(antenna_config.power_dbm * 100)  # mBm
                    ], check=False)  # May fail on some hardware

                # Update antenna config with actual interface
                antenna_config.interface = base_interface
                self.radios.append(antenna_config)

                LOG.info(f"[COCONUT] ✓ Radio {i} ready: {base_interface} on channel {antenna_config.channel}")

            except Exception as e:
                LOG.error(f"[COCONUT] Failed to setup radio {i}: {e}")

        LOG.info(f"[COCONUT] ✓ {len(self.radios)} radios configured")

    def start_capture(self, duration: Optional[int] = None):
        """
        Start packet capture on all radios.

        Args:
            duration: Capture duration in seconds (None = indefinite)
        """
        LOG.info("[COCONUT] Starting multi-radio packet capture...")

        Path(self.config.capture_dir).mkdir(parents=True, exist_ok=True)
        self.running = True

        # Start capture thread for each radio
        for antenna in self.radios:
            thread = threading.Thread(
                target=self._capture_radio,
                args=(antenna, duration),
                daemon=True
            )
            thread.start()
            self.capture_threads.append(thread)

        LOG.info(f"[COCONUT] ✓ Capturing on {len(self.radios)} radios")

        # Channel hopping thread if enabled
        if self.config.channel_hopping:
            hop_thread = threading.Thread(
                target=self._channel_hopper,
                args=(duration,),
                daemon=True
            )
            hop_thread.start()
            self.capture_threads.append(hop_thread)

        # Wait for completion if duration specified
        if duration:
            time.sleep(duration)
            self.stop_capture()

    def _capture_radio(self, antenna: AntennaConfig, duration: Optional[int] = None):
        """
        Capture packets on a single radio.

        Args:
            antenna: Antenna configuration
            duration: Capture duration in seconds
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcap_file = Path(self.config.capture_dir) / f"radio{antenna.radio_id}_ch{antenna.channel}_{timestamp}.pcap"

        LOG.info(f"[COCONUT] Radio {antenna.radio_id} capturing: {pcap_file}")

        try:
            # Build tcpdump command
            cmd = [
                "tcpdump",
                "-i", antenna.interface,
                "-w", str(pcap_file),
                "-U"  # Unbuffered
            ]

            # Start capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for duration or until stopped
            if duration:
                process.wait(timeout=duration)
            else:
                while self.running:
                    time.sleep(1)

                process.terminate()
                process.wait(timeout=5)

            LOG.info(f"[COCONUT] ✓ Radio {antenna.radio_id} capture complete: {pcap_file}")

        except Exception as e:
            LOG.error(f"[COCONUT] Radio {antenna.radio_id} capture failed: {e}")

    def _channel_hopper(self, duration: Optional[int] = None):
        """
        Channel hopping across radios.

        Distributes radios across channels and rotates.
        """
        LOG.info("[COCONUT] Channel hopping enabled")

        start_time = time.time()

        while self.running:
            if duration and (time.time() - start_time) > duration:
                break

            # Rotate channels across radios
            for i, antenna in enumerate(self.radios):
                channel_index = (i + int(time.time() / self.config.hop_interval)) % len(self.config.channels)
                new_channel = self.config.channels[channel_index]

                try:
                    subprocess.run([
                        "iw", "dev", antenna.interface, "set", "channel", str(new_channel)
                    ], check=True, stderr=subprocess.DEVNULL)

                    antenna.channel = new_channel
                    LOG.debug(f"[COCONUT] Radio {i} -> channel {new_channel}")
                except:
                    pass

            time.sleep(self.config.hop_interval)

    def stop_capture(self):
        """Stop all capture threads."""
        LOG.info("[COCONUT] Stopping capture...")
        self.running = False

        # Wait for threads
        for thread in self.capture_threads:
            thread.join(timeout=5)

        self.capture_threads.clear()
        LOG.info("[COCONUT] ✓ Capture stopped")

    def analyze_captures(self) -> Dict[str, Any]:
        """
        Analyze captured packets across all radios.

        Returns comprehensive WiFi environment analysis.
        """
        LOG.info("[COCONUT] Analyzing captures...")

        # Find all pcap files
        pcap_files = list(Path(self.config.capture_dir).glob("*.pcap"))

        if not pcap_files:
            LOG.warning("[COCONUT] No capture files found")
            return {}

        analysis = {
            "access_points": {},
            "clients": {},
            "handshakes": [],
            "probes": [],
            "channels": {}
        }

        for pcap_file in pcap_files:
            try:
                LOG.info(f"[COCONUT] Analyzing: {pcap_file.name}")
                self._analyze_pcap(pcap_file, analysis)
            except Exception as e:
                LOG.error(f"[COCONUT] Failed to analyze {pcap_file}: {e}")

        # Generate summary
        summary = {
            "total_aps": len(analysis["access_points"]),
            "total_clients": len(analysis["clients"]),
            "handshakes_captured": len(analysis["handshakes"]),
            "probe_requests": len(analysis["probes"]),
            "channels_seen": list(analysis["channels"].keys())
        }

        LOG.info(f"[COCONUT] ✓ Analysis complete: {summary}")

        return {
            "summary": summary,
            "details": analysis
        }

    def _analyze_pcap(self, pcap_file: Path, analysis: Dict):
        """
        Analyze a single pcap file.

        Extracts APs, clients, handshakes, etc.
        """
        if not COCONUT_DEPS:
            LOG.warning("[COCONUT] pyshark not available, skipping analysis")
            return

        try:
            cap = pyshark.FileCapture(str(pcap_file), display_filter="wlan")

            for packet in cap:
                try:
                    if hasattr(packet, 'wlan'):
                        self._process_wlan_packet(packet, analysis)
                except:
                    continue

            cap.close()

        except Exception as e:
            LOG.error(f"[COCONUT] Error analyzing {pcap_file}: {e}")

    def _process_wlan_packet(self, packet, analysis: Dict):
        """Process a single WLAN packet."""
        wlan = packet.wlan

        # Extract AP information (beacon frames)
        if wlan.fc_type_subtype == '0x0008':  # Beacon
            bssid = wlan.bssid if hasattr(wlan, 'bssid') else None
            if bssid:
                ssid = wlan.ssid if hasattr(wlan, 'ssid') else "[Hidden]"
                channel = wlan.channel if hasattr(wlan, 'channel') else None

                if bssid not in analysis["access_points"]:
                    vendor = self.mac_parser.get_manuf(bssid) if self.mac_parser else "Unknown"

                    analysis["access_points"][bssid] = {
                        "ssid": ssid,
                        "bssid": bssid,
                        "channel": channel,
                        "vendor": vendor,
                        "encryption": self._get_encryption(packet),
                        "seen_count": 1
                    }
                else:
                    analysis["access_points"][bssid]["seen_count"] += 1

                if channel:
                    analysis["channels"][channel] = analysis["channels"].get(channel, 0) + 1

        # Extract client information
        if hasattr(wlan, 'sa') and hasattr(wlan, 'da'):
            src = wlan.sa
            dst = wlan.da

            for mac in [src, dst]:
                if mac and mac not in analysis["clients"] and not mac.startswith("ff:ff"):
                    vendor = self.mac_parser.get_manuf(mac) if self.mac_parser else "Unknown"
                    analysis["clients"][mac] = {
                        "mac": mac,
                        "vendor": vendor,
                        "packets": 1
                    }

    def _get_encryption(self, packet) -> str:
        """Determine encryption type from packet."""
        # Simplified encryption detection
        if hasattr(packet, 'wlan'):
            if hasattr(packet.wlan, 'wfa_ie_wpa_version'):
                return "WPA"
            elif hasattr(packet.wlan, 'rsn_version'):
                return "WPA2"
            elif hasattr(packet.wlan, 'fc_protected'):
                return "WEP"

        return "Open"

    def get_antenna_overlay(self) -> Dict[str, Any]:
        """
        Generate antenna overlay visualization data.

        Returns data structure for visualizing antenna positions and coverage.
        """
        overlay = {
            "radios": [],
            "coverage_map": [],
            "channel_distribution": {}
        }

        for antenna in self.radios:
            radio_info = {
                "radio_id": antenna.radio_id,
                "interface": antenna.interface,
                "channel": antenna.channel,
                "antenna_type": antenna.antenna_type,
                "gain_dbi": antenna.gain_dbi,
                "power_dbm": antenna.power_dbm,
                "enabled": antenna.enabled,
                "estimated_range_m": self._estimate_range(antenna)
            }

            overlay["radios"].append(radio_info)

            # Channel distribution
            ch = antenna.channel
            overlay["channel_distribution"][ch] = overlay["channel_distribution"].get(ch, 0) + 1

        return overlay

    def _estimate_range(self, antenna: AntennaConfig) -> float:
        """
        Estimate antenna range in meters.

        Simplified free-space path loss model.
        """
        # FSPL: Received Power (dBm) = TX Power + TX Gain + RX Gain - FSPL
        # Typical receiver sensitivity: -90 dBm

        tx_power_dbm = antenna.power_dbm
        tx_gain_dbi = antenna.gain_dbi
        rx_gain_dbi = 2.0  # Typical client device
        rx_sensitivity_dbm = -90.0
        frequency_mhz = 2437 + (antenna.channel - 6) * 5  # Approximate

        # Calculate max path loss
        max_path_loss = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - rx_sensitivity_dbm

        # FSPL formula: FSPL(dB) = 20 log10(d) + 20 log10(f) + 32.44
        # Solve for d
        import math
        d = 10 ** ((max_path_loss - 20 * math.log10(frequency_mhz) - 32.44) / 20)

        # Apply environmental factor (indoor: 0.3x, outdoor: 1.0x)
        d *= 0.5  # Assume mixed environment

        return round(d, 1)


def health_check() -> Dict[str, Any]:
    """Health check for Ai:oS integration."""
    return {
        "tool": "WiFiCoconut",
        "status": "ok" if COCONUT_DEPS else "warn",
        "summary": "WiFi Coconut ready" if COCONUT_DEPS else "Missing dependencies",
        "details": {
            "dependencies": COCONUT_DEPS,
            "max_radios": 14
        }
    }


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WiFi Coconut - Multi-Radio WiFi Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--num-radios", type=int, default=14, help="Number of radios to use")
    parser.add_argument("--channels", nargs="+", type=int, help="Channels to monitor")
    parser.add_argument("--no-hopping", action="store_true", help="Disable channel hopping")
    parser.add_argument("--hop-interval", type=float, default=0.5, help="Hop interval (seconds)")
    parser.add_argument("--duration", type=int, help="Capture duration (seconds)")
    parser.add_argument("--capture-dir", default="/tmp/coconut_captures", help="Capture directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing captures")
    parser.add_argument("--overlay", action="store_true", help="Show antenna overlay")
    parser.add_argument("--external-antennas", action="store_true", help="Enable external antennas")
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
        return 0 if result['status'] == 'ok' else 1

    # Build config
    config = CoconutConfig(
        num_radios=args.num_radios,
        channels=args.channels,
        channel_hopping=not args.no_hopping,
        hop_interval=args.hop_interval,
        capture_dir=args.capture_dir,
        enable_external_antennas=args.external_antennas
    )

    coconut = WiFiCoconut(config)

    # Analyze mode
    if args.analyze:
        analysis = coconut.analyze_captures()
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print("\n=== WiFi Coconut Analysis ===")
            print(f"Access Points: {analysis['summary']['total_aps']}")
            print(f"Clients: {analysis['summary']['total_clients']}")
            print(f"Handshakes: {analysis['summary']['handshakes_captured']}")
            print(f"Channels: {', '.join(map(str, analysis['summary']['channels_seen']))}")
        return 0

    # Antenna overlay
    if args.overlay:
        coconut.setup_radios()
        overlay = coconut.get_antenna_overlay()
        if args.json:
            print(json.dumps(overlay, indent=2))
        else:
            print("\n=== Antenna Overlay ===")
            for radio in overlay["radios"]:
                print(f"\nRadio {radio['radio_id']}: {radio['interface']}")
                print(f"  Channel: {radio['channel']}")
                print(f"  Type: {radio['antenna_type']}")
                print(f"  Gain: {radio['gain_dbi']} dBi")
                print(f"  Power: {radio['power_dbm']} dBm")
                print(f"  Est. Range: {radio['estimated_range_m']} m")
        return 0

    # Capture mode
    LOG.info("=" * 60)
    LOG.info("WiFi Coconut - Multi-Radio Capture")
    LOG.info("=" * 60)

    coconut.setup_radios()
    coconut.start_capture(duration=args.duration)

    if not args.duration:
        # Run indefinitely
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            coconut.stop_capture()

    # Analyze results
    analysis = coconut.analyze_captures()
    LOG.info(f"\n✓ Captured data from {analysis['summary']['total_aps']} APs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
