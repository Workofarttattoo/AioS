"""
Sovereign Security Toolkit
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AUTHORIZATION WARNING:
All tools in this toolkit are for AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.

Complete Red Team Arsenal including:
- PyThief: Evil twin WiFi attack framework
- Hak5 Arsenal: USB Shark, Packet Squirrel, LAN Turtle
- WiFi Coconut: 14-radio WiFi analysis
- Proxmark3 Toolkit: RFID/NFC/EMV security testing
- Pwnie Revival: Modern Pwnie Express tools
- gPIG: Network reconnaissance and exploitation orchestrator
- ECH0Py: Lightweight LLM agent for pentesting
- AuroraScan: Network reconnaissance
- CipherSpear: Database security
- SkyBreaker: Wireless auditing
- MythicKey: Password analysis
- SpectraTrace: Packet inspection
- NemesisHydra: Authentication testing
- ObsidianHunt: Host hardening audit
- VectorFlux: Payload staging
"""

# Tool registry for health checks and availability
TOOL_REGISTRY = {
    "pythief": {
        "name": "PyThief",
        "description": "Evil twin WiFi attack framework with Marauder and SDR support",
        "module": "pythief",
        "capabilities": ["evil_twin", "packet_capture", "page_cloning", "bluetooth_control", "marauder", "sdr"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "hak5_arsenal": {
        "name": "Hak5 Arsenal",
        "description": "Reverse-engineered Hak5 devices (USB Shark, Packet Squirrel, LAN Turtle)",
        "module": "hak5_arsenal",
        "capabilities": ["usb_capture", "inline_packet_capture", "network_implant", "keylogging"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "wifi_coconut": {
        "name": "WiFi Coconut",
        "description": "Multi-radio WiFi analysis (14 simultaneous radios)",
        "module": "wifi_coconut",
        "capabilities": ["multi_radio", "channel_hopping", "spectrum_analysis", "antenna_management"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "proxmark3_toolkit": {
        "name": "Proxmark3 Toolkit",
        "description": "RFID/NFC/EMV security testing",
        "module": "proxmark3_toolkit",
        "capabilities": ["rfid_clone", "nfc_readwrite", "emv_analysis", "badge_emulation"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "pwnie_revival": {
        "name": "Pwnie Revival",
        "description": "Modern Pwnie Express toolkit (Pwn Plug, Pwn Pro, Pwn Pad, Pwn Pulse)",
        "module": "pwnie_revival",
        "capabilities": ["network_implant", "full_assessment", "mobile_testing", "enterprise_assessment"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "gpig": {
        "name": "gPIG",
        "description": "Gluttonous Packet Inspection Gadget - Network reconnaissance and auto-exploitation",
        "module": "gpig",
        "capabilities": ["network_mapping", "vulnerability_scanning", "auto_exploitation", "web_overlay"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "ech0py_agent": {
        "name": "ECH0Py",
        "description": "Lightweight LLM agent for pentesting (Raspberry Pi optimized)",
        "module": "ech0py_agent",
        "capabilities": ["natural_language_control", "tool_orchestration", "automated_workflows"],
        "requires_root": False,
        "platform": ["linux", "darwin"]
    },
    "aurorascan": {
        "name": "AuroraScan",
        "description": "Network reconnaissance",
        "module": "aurorascan",
        "capabilities": ["port_scanning", "service_detection", "os_fingerprinting"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "cipherspear": {
        "name": "CipherSpear",
        "description": "Database injection analysis",
        "module": "cipherspear",
        "capabilities": ["sql_injection", "database_audit"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "skybreaker": {
        "name": "SkyBreaker",
        "description": "Wireless auditing",
        "module": "skybreaker",
        "capabilities": ["wifi_audit", "wpa_crack", "rogue_ap_detection"],
        "requires_root": True,
        "platform": ["linux"]
    },
    "mythickey": {
        "name": "MythicKey",
        "description": "Password analysis and cracking",
        "module": "mythickey",
        "capabilities": ["password_analysis", "hash_cracking", "pattern_detection"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "spectratrace": {
        "name": "SpectraTrace",
        "description": "Packet inspection",
        "module": "spectratrace",
        "capabilities": ["packet_capture", "protocol_analysis", "traffic_reconstruction"],
        "requires_root": True,
        "platform": ["linux", "darwin"]
    },
    "nemesishydra": {
        "name": "NemesisHydra",
        "description": "Authentication testing",
        "module": "nemesishydra",
        "capabilities": ["brute_force", "credential_testing", "multi_protocol"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "obsidianhunt": {
        "name": "ObsidianHunt",
        "description": "Host hardening audit",
        "module": "obsidianhunt",
        "capabilities": ["security_audit", "config_review", "compliance_check"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    },
    "vectorflux": {
        "name": "VectorFlux",
        "description": "Payload staging and delivery",
        "module": "vectorflux",
        "capabilities": ["payload_generation", "delivery_methods", "evasion_techniques"],
        "requires_root": False,
        "platform": ["linux", "darwin", "win32"]
    }
}


def get_tool(tool_name: str):
    """Get tool module by name."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not found: {tool_name}")

    import importlib
    module = importlib.import_module(TOOL_REGISTRY[tool_name]["module"])
    return module


def health_check_all():
    """Run health checks on all tools."""
    results = {}

    for tool_name, tool_info in TOOL_REGISTRY.items():
        try:
            module = get_tool(tool_name)
            if hasattr(module, "health_check"):
                results[tool_name] = module.health_check()
            else:
                results[tool_name] = {
                    "tool": tool_info["name"],
                    "status": "unknown",
                    "summary": "No health check available"
                }
        except Exception as e:
            results[tool_name] = {
                "tool": tool_info["name"],
                "status": "error",
                "summary": str(e)
            }

    return results


def list_tools():
    """List all available tools."""
    return {
        name: {
            "name": info["name"],
            "description": info["description"],
            "capabilities": info["capabilities"],
            "platform": info["platform"]
        }
        for name, info in TOOL_REGISTRY.items()
    }


__all__ = [
    "TOOL_REGISTRY",
    "get_tool",
    "health_check_all",
    "list_tools"
]
