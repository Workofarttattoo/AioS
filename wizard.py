"""
Interactive setup wizard and security toolkit catalog for AgentaOS.

The wizard inspects the host environment, recommends environment overrides,
and guides the user through pre-built runtime profiles.  When requested it
also surfaces the Sovereign Security Toolkit—a reimagined suite of Parrot OS
utilities rebuilt under AgentaOS conventions.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOG = logging.getLogger("AgentaOS.wizard")


@dataclass
class SecurityTool:
  """Metadata describing a reimagined security utility."""

  name: str
  origin: str
  summary: str
  improvements: str
  command_hint: str

  def as_dict(self) -> Dict[str, str]:
    return asdict(self)


SECURITY_TOOLKIT: List[SecurityTool] = [
  SecurityTool(
    name="AuroraScan",
    origin="Inspired by nmap",
    summary="Adaptive network mapper with service fingerprint diffing.",
    improvements="Adds ML-assisted baseline comparisons and encrypted scan profiles.",
    command_hint="python -m tools.aurorascan --gui",
  ),
  SecurityTool(
    name="CipherSpear",
    origin="Inspired by sqlmap",
    summary="Precision database exploitation engine with query replay walls.",
    improvements="Focuses on safe payload rehearsal and telemetry capture per injection chain.",
    command_hint="cipherspear --dsn jdbc:mysql://target/db --tech blind,bool",
  ),
  SecurityTool(
    name="SkyBreaker",
    origin="Inspired by aircrack-ng",
    summary="Wireless audit orchestrator supporting live spectrum overlays.",
    improvements="Adds replay-resistant capture scheduling and integrated device fingerprinting.",
    command_hint="python -m tools.skybreaker --gui",
  ),
  SecurityTool(
    name="MythicKey",
    origin="Inspired by John the Ripper",
    summary="Credential resilience analyzer with hardware offload targeting.",
    improvements="Extends with wordlist mutation graphs and audit-grade reporting.",
    command_hint="python -m tools.mythickey --gui",
  ),
  SecurityTool(
    name="SpectraTrace",
    origin="Inspired by Wireshark",
    summary="Protocol inspection studio with redacted capture lanes.",
    improvements="Introduces privacy-preserving packet mirroring and policy-based redaction.",
    command_hint="python -m tools.spectratrace --gui",
  ),
  SecurityTool(
    name="NemesisHydra",
    origin="Inspired by THC-Hydra",
    summary="Distributed authentication tester with adaptive throttle controls.",
    improvements="Coordinates credential sprays with feedback-aware pacing and compliance gates.",
    command_hint="python -m tools.nemesishydra --gui",
  ),
  SecurityTool(
    name="ObsidianHunt",
    origin="Inspired by Lynis",
    summary="Host hardening assessor tuned for forensic baselines.",
    improvements="Generates remediation manifests consumable by AgentaOS supervisors.",
    command_hint="python -m tools.obsidianhunt --gui",
  ),
  SecurityTool(
    name="VectorFlux",
    origin="Inspired by Metasploit Framework",
    summary="Payload crafting workbench integrated with scenario guardrails.",
    improvements="Enforces signed module registries and live sandbox verdict streaming.",
    command_hint="python -m tools.vectorflux --gui",
  ),
]

def build_post_boot_guidance(os_name: Optional[str] = None) -> List[Dict[str, str]]:
  """Return a host-hardening checklist to display after boot."""

  system_name = os_name or platform.system().lower()
  guidance: List[Dict[str, str]] = [
    {
      "title": "Antivirus exclusions",
      "summary": "Temporarily add AgentaOS labs, VM images, and tool directories to antivirus exclusions or pause real-time scanning while exercises run.",
      "category": "protection",
    },
    {
      "title": "Firewall allowances",
      "summary": "Permit inbound connections for virtualization bridges, remote consoles, and telemetry agents. Restore hardened rules when finished.",
      "category": "network",
    },
    {
      "title": "Inbound access",
      "summary": "Confirm the host accepts incoming connections from coordinators or teammate workstations used during labs.",
      "category": "network",
    },
    {
      "title": "Post-lab recovery",
      "summary": "Track every host security change and re-enable protections after the engagement to maintain compliance baselines.",
      "category": "hygiene",
    },
  ]

  if system_name == "windows":
    guidance.extend([
      {
        "title": "Windows Defender control",
        "summary": "Use Windows Security > Virus & threat protection to disable real-time protection or create exclusions for AgentaOS directories and VM images.",
        "category": "platform",
      },
      {
        "title": "Windows Firewall profile",
        "summary": "Create inbound rules or temporarily disable the active profile in Windows Defender Firewall with Advanced Security during exercises.",
        "category": "platform",
      },
    ])
  elif system_name == "darwin":
    guidance.extend([
      {
        "title": "macOS firewall prompt",
        "summary": "Review System Settings > Network > Firewall and allow incoming connections for qemu, socat, and AgentaOS tool wrappers.",
        "category": "platform",
      },
      {
        "title": "Endpoint security agents",
        "summary": "Coordinate with any MDR/EDR tooling (e.g., CrowdStrike, SentinelOne) to place the lab host in monitor-only mode.",
        "category": "platform",
      },
    ])
  else:
    guidance.extend([
      {
        "title": "SELinux/AppArmor tuning",
        "summary": "Set permissive mode or craft targeted policies for qemu, libvirt, and Sovereign tools to prevent policy denials during exercises.",
        "category": "platform",
      },
      {
        "title": "Service manager",
        "summary": "Review systemctl, ufw, or firewalld policies so bridges and taps accept external traffic when required.",
        "category": "platform",
      },
    ])

  return guidance


class SetupWizard:
  """Guide the operator through automated environment preparation."""

  SECURITY_SUITE_NAME = "Sovereign Security Toolkit"

  def __init__(self, base_environment: Dict[str, str]) -> None:
    self.base_environment = dict(base_environment)
    self.detections = self._detect_capabilities()

  def run(self) -> Dict[str, object]:
    LOG.info("[info] Launching AgentaOS setup wizard.")
    self._display_detections()

    profile_key, profile_env = self._prompt_profile()
    merged_env = dict(self.base_environment)
    merged_env.update(profile_env)

    security_payload: List[Dict[str, str]] = []
    include_security = self._prompt_yes_no(
      "Load the Sovereign Security Toolkit profile (reimagined Parrot utilities)? [Y/n]: ",
      default=True,
    )
    if include_security:
      suite_env = self._build_security_env()
      merged_env.update(suite_env)
      security_payload = [tool.as_dict() for tool in SECURITY_TOOLKIT]
      LOG.info("[info] Sovereign Security Toolkit queued with %d utilities.", len(security_payload))
    else:
      LOG.info("[info] Skipping security toolkit provisioning.")

    recommendations = self._recommend_env_overrides()
    for key, value in recommendations.items():
      merged_env.setdefault(key, value)

    payload = {
      "environment": merged_env,
      "profile": profile_key,
      "detections": self.detections,
      "security_tools": security_payload,
    }
    validation = self._validate_payload(payload)
    payload["validation"] = validation
    post_boot_checklist = self._post_boot_guidance()
    payload["post_boot_checklist"] = post_boot_checklist
    self._summarise_payload(payload)
    self._report_validation(validation)
    self._display_post_boot_guidance(post_boot_checklist)
    return payload

  def _detect_capabilities(self) -> Dict[str, object]:
    providers = {
      "docker": shutil.which("docker"),
      "qemu": shutil.which(self.base_environment.get("AGENTA_QEMU_BINARY", "qemu-system-x86_64")),
      "libvirt": shutil.which("virsh"),
      "multipass": shutil.which("multipass"),
      "aws": shutil.which("aws"),
      "azure": shutil.which("az"),
      "gcloud": shutil.which("gcloud"),
    }
    sockets = {
      "docker_socket": self._detect_socket(Path(os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock"))),
    }
    images = self._discover_images()
    existing_env = {key: os.environ.get(key) for key in (
      "AGENTA_PROVIDER",
      "AGENTA_QEMU_IMAGE",
      "AGENTA_QEMU_IMAGE_TYPE",
      "AGENTA_QEMU_DISPLAY",
      "AGENTA_QEMU_HEADLESS",
      "AGENTA_QEMU_EXECUTE",
      "AGENTA_SECURITY_SUITE",
    ) if os.environ.get(key)}
    return {
      "providers": providers,
      "sockets": sockets,
      "images": images,
      "existing_env": existing_env,
    }

  def _display_detections(self) -> None:
    providers = self.detections["providers"]
    available = [name for name, binary in providers.items() if binary]
    LOG.info("[info] Detected providers: %s", ", ".join(available) or "none")
    if self.detections["existing_env"]:
      LOG.info("[info] Existing overrides: %s", ", ".join(f"{k}={v}" for k, v in self.detections["existing_env"].items()))
    images = self.detections["images"]
    if any(images.values()):
      for label, paths in images.items():
        if paths:
          LOG.info("[info] Found %s image candidates: %s", label, "; ".join(paths[:3]))
    else:
      LOG.info("[warn] No virtualization images discovered automatically.")

  def _prompt_profile(self) -> Tuple[str, Dict[str, str]]:
    profiles = self._build_profiles()
    LOG.info("[info] Pre-build profiles:")
    for idx, (key, data) in enumerate(profiles.items(), start=1):
      LOG.info("  %d) %s - %s", idx, key, data["summary"])
    prompt = "Select profile [1]: "
    choice = input(prompt).strip()
    if not choice:
      choice = "1"
    try:
      index = int(choice) - 1
      key = list(profiles.keys())[index]
    except (ValueError, IndexError):
      LOG.info("[warn] Invalid selection; using default profile.")
      key = list(profiles.keys())[0]
    profile = profiles[key]
    LOG.info("[info] Using profile '%s'.", key)
    return key, profile["env"]

  def _prompt_yes_no(self, prompt: str, *, default: bool) -> bool:
    answer = input(prompt).strip().lower()
    if not answer:
      return default
    if answer in {"y", "yes"}:
      return True
    if answer in {"n", "no"}:
      return False
    LOG.info("[warn] Unrecognised response; assuming %s.", "yes" if default else "no")
    return default

  def _build_profiles(self) -> Dict[str, Dict[str, object]]:
    available_providers = [name for name, binary in self.detections["providers"].items() if binary]
    default_provider = ",".join(available_providers) if available_providers else "docker"
    image_candidates = self.detections["images"].get("disk") or []
    iso_candidates = self.detections["images"].get("cdrom") or []
    first_disk = image_candidates[0] if image_candidates else ""
    first_iso = iso_candidates[0] if iso_candidates else ""

    profiles: Dict[str, Dict[str, object]] = {}

    profiles["minimal-telemetry"] = {
      "summary": "Boot with detected providers and telemetry safeguards.",
      "env": {
        "AGENTA_PROVIDER": default_provider,
        "AGENTA_FORENSIC_MODE": "1",
        "AGENTA_SUPERVISOR_CONCURRENCY": "4",
      },
    }

    virtualization_env = {
      "AGENTA_PROVIDER": self._compose_provider_string(["qemu", "libvirt", "docker"], available_providers),
      "AGENTA_QEMU_IMAGE": first_disk or first_iso,
      "AGENTA_QEMU_IMAGE_TYPE": "cdrom" if first_iso and not first_disk else "disk",
      "AGENTA_QEMU_EXECUTE": "1" if first_disk or first_iso else "0",
      "AGENTA_QEMU_HEADLESS": "0",
      "AGENTA_QEMU_DISPLAY": "gtk",
      "AGENTA_QEMU_DAEMONIZE": "0",
      "AGENTA_QEMU_MEMORY": "8192",
      "AGENTA_QEMU_CPUS": "4",
      "AGENTA_QEMU_SNAPSHOT": "1",
    }
    profiles["virtualization-lab"] = {
      "summary": "Launch graphical QEMU guests with live telemetry.",
      "env": virtualization_env,
    }

    security_env = dict(virtualization_env)
    security_env.update({
      "AGENTA_FORENSIC_MODE": "0",
      "AGENTA_SUPERVISOR_CONCURRENCY": "6",
      "AGENTA_MONITOR_ALERTS": "1",
    })
    profiles["security-response-deck"] = {
      "summary": "Prepare virtualization lab plus security response suite.",
      "env": security_env,
    }

    return profiles

  def _compose_provider_string(self, priority: List[str], available: List[str]) -> str:
    ordered = [name for name in priority if name in available]
    remaining = [name for name in available if name not in ordered]
    providers = ordered + remaining
    return ",".join(providers) if providers else ",".join(priority)

  def _build_security_env(self) -> Dict[str, str]:
    tool_names = ",".join(tool.name for tool in SECURITY_TOOLKIT)
    return {
      "AGENTA_SECURITY_SUITE": self.SECURITY_SUITE_NAME,
      "AGENTA_SECURITY_TOOLS": tool_names,
      "AGENTA_SECURITY_PROFILE": "offensive-readiness",
    }

  def _recommend_env_overrides(self) -> Dict[str, str]:
    recommendations: Dict[str, str] = {}
    providers = [name for name, binary in self.detections["providers"].items() if binary]
    if providers:
      recommendations["AGENTA_PROVIDER"] = ",".join(providers)
    socket = self.detections["sockets"]["docker_socket"]
    if socket:
      recommendations["DOCKER_SOCKET"] = socket
    return recommendations

  def _summarise_payload(self, payload: Dict[str, object]) -> None:
    env = payload["environment"]
    LOG.info("[info] Wizard environment overrides:")
    for key in sorted(env.keys()):
      LOG.info("  %s=%s", key, env[key])
    if payload["security_tools"]:
      LOG.info("[info] Security toolkit manifest:")
      for tool in payload["security_tools"]:
        LOG.info("  - %(name)s (%(origin)s): %(summary)s", tool)

  def _validate_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
    environment = payload.get("environment", {})
    detected_providers = [name for name, binary in self.detections["providers"].items() if binary]
    configured_providers = [item.strip() for item in environment.get("AGENTA_PROVIDER", "").split(",") if item.strip()]
    missing_providers = [provider for provider in configured_providers if provider not in detected_providers]

    warnings: List[str] = []
    if missing_providers:
      warnings.append(f"Configured providers not detected: {', '.join(missing_providers)}")
    if not configured_providers:
      if detected_providers:
        warnings.append(f"No virtualization providers selected; detected {', '.join(detected_providers)} on PATH.")
      else:
        warnings.append("No virtualization providers detected on PATH.")
    elif not detected_providers:
      warnings.append("No virtualization providers detected on PATH.")

    qemu_execute = str(environment.get("AGENTA_QEMU_EXECUTE", "0")).lower() in {"1", "true", "yes"}
    qemu_image = environment.get("AGENTA_QEMU_IMAGE")
    if qemu_execute:
      if not qemu_image:
        warnings.append("AGENTA_QEMU_EXECUTE=1 but no image path was selected.")
      elif not Path(qemu_image).expanduser().exists():
        warnings.append(f"QEMU image '{qemu_image}' does not exist on disk.")

    detected_images = self.detections.get("images", {})
    if not any(detected_images.values()):
      warnings.append("No virtualization images were discovered automatically.")

    os_name = platform.system().lower()
    info: List[str] = [f"os={os_name}"]
    display = str(environment.get("AGENTA_QEMU_DISPLAY", "")).lower()
    headless = str(environment.get("AGENTA_QEMU_HEADLESS", "0")).lower() in {"1", "true", "yes"}
    if os_name == "darwin" and display == "gtk" and not headless:
      warnings.append("GTK display requested on macOS; consider enabling headless mode or using cocoa display.")

    provider_hint = environment.get("AGENTA_PROVIDER") or ",".join(detected_providers)
    info.append(f"providers={provider_hint or 'none'}")

    checks = self._environment_checks(
      environment=environment,
      detected_providers=detected_providers,
      configured_providers=configured_providers,
      detected_images=detected_images,
    )
    status = "ok" if not warnings else "warn"
    return {
      "status": status,
      "warnings": warnings,
      "info": info,
      "checks": checks,
    }

  def _environment_checks(
    self,
    *,
    environment: Dict[str, str],
    detected_providers: List[str],
    configured_providers: List[str],
    detected_images: Dict[str, List[str]],
  ) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []

    os_name = platform.system().lower()
    checks.append({
      "name": "os-detection",
      "status": "ok",
      "detail": {"platform": os_name},
    })

    provider_detail: Dict[str, object] = {
      "configured": configured_providers,
      "detected": detected_providers,
    }
    provider_status = "ok"
    missing = [provider for provider in configured_providers if provider not in detected_providers]
    if missing:
      provider_status = "warn"
      provider_detail["missing"] = missing
    if not configured_providers:
      provider_status = "warn"
      provider_detail["recommended"] = self._recommend_env_overrides().get("AGENTA_PROVIDER", "")
    checks.append({
      "name": "provider-fallback",
      "status": provider_status,
      "detail": provider_detail,
    })

    image_counts = {kind: len(paths) for kind, paths in detected_images.items()}
    checks.append({
      "name": "image-discovery",
      "status": "ok" if any(image_counts.values()) else "warn",
      "detail": image_counts,
    })

    display = environment.get("AGENTA_QEMU_DISPLAY")
    headless = str(environment.get("AGENTA_QEMU_HEADLESS", "0")).lower() in {"1", "true", "yes"}
    if display:
      checks.append({
        "name": "display-compatibility",
        "status": "ok" if headless or display.lower() not in {"gtk"} or os_name != "darwin" else "warn",
        "detail": {"display": display, "headless": headless},
      })

    return checks

  def _report_validation(self, validation: Dict[str, object]) -> None:
    status = validation.get("status", "ok")
    if status == "ok":
      LOG.info("[info] Wizard validation passed (%s).", "; ".join(validation.get("info", [])))
      return
    LOG.warning("[warn] Wizard validation reported issues:")
    for warning in validation.get("warnings", []):
      LOG.warning("  - %s", warning)
    for item in validation.get("info", []):
      LOG.info("  info: %s", item)
    for check in validation.get("checks", []):
      LOG.info("[info] check %(name)s → %(status)s :: %(detail)s", check)

  def _post_boot_guidance(self) -> List[Dict[str, str]]:
    return build_post_boot_guidance()

  def _display_post_boot_guidance(self, checklist: List[Dict[str, str]]) -> None:
    if not checklist:
      return
    LOG.info("[info] Post-boot readiness checklist:")
    for entry in checklist:
      LOG.info("  - %s: %s", entry.get("title", "item"), entry.get("summary", ""))

  def _discover_images(self) -> Dict[str, List[str]]:
    search_roots = [
      Path(self.base_environment.get("AGENTA_QEMU_IMAGE_DIR", "")),
      Path.home() / "Downloads",
      Path.home() / "Virtual Machines.localized",
      Path.home() / "VirtualBox VMs",
      Path("/var/lib/libvirt/images"),
    ]
    extensions = {
      "disk": (".qcow2", ".img", ".raw"),
      "cdrom": (".iso",),
    }
    results: Dict[str, List[str]] = {"disk": [], "cdrom": []}
    for root in search_roots:
      if not root or not root.exists() or not root.is_dir():
        continue
      for category, suffixes in extensions.items():
        if results[category]:
          continue
        for suffix in suffixes:
          matches = list(root.glob(f"*{suffix}"))
          if matches:
            results[category] = [str(path) for path in matches[:5]]
            break
    return results

  def _detect_socket(self, path: Path) -> Optional[str]:
    try:
      if path.exists():
        return str(path)
    except OSError:
      return None
    return None
