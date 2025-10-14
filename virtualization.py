"""
Virtualization execution helpers for AgentaOS.

This module centralises QEMU and libvirt orchestration so meta-agents can
prepare launch or verification commands, respect forensic safeguards, and
surface structured telemetry about each attempted action.
"""

from __future__ import annotations

import json
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


DEFAULT_TIMEOUT = 30


@dataclass
class VirtualizationPlan:
  """Describes a virtualization command that can be executed or verified."""

  provider: str
  action: str
  summary: str
  command: Optional[List[str]] = None
  execute: bool = False
  metadata: Dict[str, object] = field(default_factory=dict)
  timeout: int = DEFAULT_TIMEOUT


@dataclass
class VirtualizationOutcome:
  """Result of executing or evaluating a virtualization plan."""

  provider: str
  action: str
  status: str
  summary: str
  command: Optional[str] = None
  stdout: str = ""
  stderr: str = ""
  metadata: Dict[str, object] = field(default_factory=dict)

  def as_dict(self) -> Dict[str, object]:
    payload = {
      "provider": self.provider,
      "action": self.action,
      "status": self.status,
      "summary": self.summary,
      "command": self.command,
      "stdout": self.stdout,
      "stderr": self.stderr,
      "metadata": dict(self.metadata),
    }
    return payload


class VirtualizationLayer:
  """Build and execute virtualization plans for available providers."""

  def __init__(
    self,
    environment: Dict[str, str],
    providers: Optional[Iterable[str]] = None,
    *,
    forensic_mode: bool = False,
  ) -> None:
    self.environment = environment
    self.providers = set(providers or [])
    self.forensic_mode = forensic_mode

  def execute(self, action: str) -> List[VirtualizationOutcome]:
    outcomes: List[VirtualizationOutcome] = []
    for plan in self._plans_for(action):
      if plan.command is None:
        metadata = dict(plan.metadata)
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status="missing",
            summary=plan.summary,
            metadata=metadata,
          )
        )
        continue
      if self.forensic_mode and plan.execute:
        metadata = dict(plan.metadata)
        metadata["reason"] = "forensic_mode"
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status="skipped",
            summary=plan.summary,
            command=" ".join(plan.command),
            metadata=metadata,
          )
        )
        continue
      try:
        completed = subprocess.run(
          plan.command,
          capture_output=True,
          text=True,
          timeout=plan.timeout,
          check=False,
        )
        status = "executed" if plan.execute else "verified"
        if completed.returncode != 0:
          status = "error"
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status=status,
            summary=plan.summary,
            command=" ".join(plan.command),
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            metadata=dict(plan.metadata),
          )
        )
      except FileNotFoundError as exc:
        metadata = dict(plan.metadata)
        metadata["error"] = str(exc)
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status="error",
            summary=f"{plan.summary} (binary not found)",
            command=" ".join(plan.command),
            metadata=metadata,
          )
        )
      except subprocess.TimeoutExpired as exc:
        metadata = dict(plan.metadata)
        metadata["timeout"] = plan.timeout
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status="timeout",
            summary=f"{plan.summary} (timed out)",
            command=" ".join(plan.command),
            stdout=(exc.output or "").strip() if exc.output else "",
            stderr=(exc.stderr or "").strip() if exc.stderr else "",
            metadata=metadata,
          )
        )
      except Exception as exc:
        metadata = dict(plan.metadata)
        metadata["error"] = repr(exc)
        outcomes.append(
          VirtualizationOutcome(
            provider=plan.provider,
            action=plan.action,
            status="error",
            summary=f"{plan.summary} (exception)",
            command=" ".join(plan.command),
            metadata=metadata,
          )
        )
    return outcomes

  def describe(self, action: str) -> List[Dict[str, object]]:
    plans = []
    for plan in self._plans_for(action):
      plans.append(
        {
          "provider": plan.provider,
          "action": plan.action,
          "summary": plan.summary,
          "command": " ".join(plan.command) if plan.command else None,
          "execute": plan.execute,
          "metadata": dict(plan.metadata),
        }
      )
    return plans

  def status(self) -> List[Dict[str, object]]:
    snapshots: List[Dict[str, object]] = []
    if not self.providers or "qemu" in self.providers:
      snapshot = self._status_qemu()
      if snapshot:
        snapshots.append(snapshot)
    if not self.providers or "libvirt" in self.providers:
      snapshot = self._status_libvirt()
      if snapshot:
        snapshots.append(snapshot)
    return snapshots

  def _plans_for(self, action: str) -> Sequence[VirtualizationPlan]:
    plans: List[VirtualizationPlan] = []
    if not self.providers or "qemu" in self.providers:
      plans.extend(self._plan_qemu_prerequisites(action))
      plan = self._plan_qemu(action)
      if plan:
        plans.append(plan)
    if not self.providers or "libvirt" in self.providers:
      plan = self._plan_libvirt(action)
      if plan:
        plans.append(plan)
    return plans

  def _autocreate_network(self) -> bool:
    value = str(self.environment.get("AGENTA_QEMU_NET_AUTOCREATE", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}

  def _plan_qemu_prerequisites(self, action: str) -> List[VirtualizationPlan]:
    plans: List[VirtualizationPlan] = []

    bridge = (self.environment.get("AGENTA_QEMU_NET_BRIDGE") or "").strip()
    tap = (self.environment.get("AGENTA_QEMU_TAP") or "").strip()
    interfaces: List[Tuple[str, str]] = []
    if bridge:
      interfaces.append(("bridge", bridge))
    if tap:
      interfaces.append(("tap", tap))

    if not interfaces:
      return plans

    auto_setup = self._autocreate_network()
    traversal = interfaces if action == "up" else list(reversed(interfaces))
    teardown_queue: List[Tuple[str, str, List[str], Dict[str, object]]] = []

    for role, name in traversal:
      command = self._bridge_check_command(name)
      setup_commands = self._network_setup_commands(role, name, bridge if role == "tap" else None)
      teardown_commands = self._network_teardown_commands(role, name)
      metadata = {
        "role": role,
        "interface": name,
        "binary": command[0] if command else None,
        "netdev_id": self.environment.get("AGENTA_QEMU_NETDEV_ID", "br0"),
        "auto_setup": auto_setup and bool(setup_commands),
        "setup_commands": [" ".join(cmd) for cmd in setup_commands],
        "auto_teardown": auto_setup and bool(teardown_commands),
        "teardown_commands": [" ".join(cmd) for cmd in teardown_commands],
      }
      if auto_setup and setup_commands and action == "up":
        total = len(setup_commands)
        for index, setup_command in enumerate(setup_commands, start=1):
          setup_metadata = dict(metadata)
          setup_metadata.update({
            "setup_step": index,
            "setup_steps": total,
          })
          summary = f"Configure {role} interface '{name}' (step {index}/{total})."
          plans.append(
            VirtualizationPlan(
              provider="qemu",
              action=action,
              summary=summary,
              command=setup_command,
              execute=True,
              metadata=setup_metadata,
            )
          )
      if auto_setup and teardown_commands and action == "down":
        for teardown_command in teardown_commands:
          teardown_queue.append((role, name, teardown_command, metadata))
      if action == "up":
        verify_metadata = dict(metadata)
        if command:
          summary = f"Verify {role} interface '{name}' availability."
          plans.append(
            VirtualizationPlan(
              provider="qemu",
              action=action,
              summary=summary,
              command=command,
              execute=False,
              metadata=verify_metadata,
            )
          )
        else:
          summary = f"Unable to verify {role} interface '{name}' (inspection tools unavailable)."
          plans.append(
            VirtualizationPlan(
              provider="qemu",
              action=action,
              summary=summary,
              command=None,
              metadata=verify_metadata,
            )
          )
    if teardown_queue and auto_setup and action == "down":
      total = len(teardown_queue)
      for index, (role, name, teardown_command, metadata) in enumerate(teardown_queue, start=1):
        teardown_metadata = dict(metadata)
        teardown_metadata.update({
          "teardown_step": index,
          "teardown_steps": total,
          "auto_teardown": True,
        })
        summary = f"Tear down {role} interface '{name}' (step {index}/{total})."
        plans.append(
          VirtualizationPlan(
            provider="qemu",
            action=action,
            summary=summary,
            command=teardown_command,
            execute=True,
            metadata=teardown_metadata,
          )
        )

    return plans

  def _bridge_check_command(self, interface: str) -> Optional[List[str]]:
    interface = interface.strip()
    if not interface:
      return None
    ip_cmd = shutil.which("ip")
    if ip_cmd:
      return [ip_cmd, "link", "show", interface]
    ifconfig_cmd = shutil.which("ifconfig")
    if ifconfig_cmd:
      return [ifconfig_cmd, interface]
    return None

  def _network_setup_commands(self, role: str, interface: str, bridge: Optional[str]) -> List[List[str]]:
    interface = interface.strip()
    if not interface:
      return []
    ip_cmd = shutil.which("ip")
    if not ip_cmd:
      return []
    commands: List[List[str]] = []
    if role == "bridge":
      commands.append([ip_cmd, "link", "add", "name", interface, "type", "bridge"])
      commands.append([ip_cmd, "link", "set", interface, "up"])
    elif role == "tap":
      commands.append([ip_cmd, "tuntap", "add", "dev", interface, "mode", "tap"])
      if bridge:
        commands.append([ip_cmd, "link", "set", interface, "master", bridge])
      commands.append([ip_cmd, "link", "set", interface, "up"])
    return commands

  def _network_teardown_commands(self, role: str, interface: str) -> List[List[str]]:
    interface = interface.strip()
    if not interface:
      return []
    ip_cmd = shutil.which("ip")
    if not ip_cmd:
      return []
    commands: List[List[str]] = []
    if role == "tap":
      commands.append([ip_cmd, "link", "set", interface, "down"])
      commands.append([ip_cmd, "tuntap", "del", "dev", interface, "mode", "tap"])
    elif role == "bridge":
      commands.append([ip_cmd, "link", "set", interface, "down"])
      commands.append([ip_cmd, "link", "del", interface])
    return commands

  @staticmethod
  def _parse_passthrough_devices(config: Optional[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not config:
      return [], []

    available: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []

    for chunk in str(config).split(";"):
      entry = chunk.strip()
      if not entry:
        continue
      if entry.startswith("pci:"):
        address = entry[4:]
        path = Path("/sys/bus/pci/devices") / address
        payload = {"type": "pci", "address": address, "path": str(path), "exists": path.exists()}
      elif entry.startswith("usb:"):
        _, _, value = entry.partition(":")
        bus_str, _, addr_str = value.partition("-")
        try:
          bus = int(bus_str)
          addr = int(addr_str)
        except ValueError:
          payload = {"type": "usb", "value": value, "exists": False}
        else:
          path = Path("/sys/bus/usb/devices") / value
          payload = {"type": "usb", "bus": bus, "addr": addr, "path": str(path), "exists": path.exists()}
      elif entry.startswith("path:"):
        device_path = Path(entry[5:])
        payload = {"type": "path", "path": str(device_path), "exists": device_path.exists()}
      else:
        payload = {"type": "unknown", "value": entry, "exists": False}

      if payload.get("exists"):
        available.append(payload)
      else:
        missing.append(payload)

    return available, missing

  def _plan_qemu(self, action: str) -> Optional[VirtualizationPlan]:
    qemu_binary = shutil.which(self.environment.get("AGENTA_QEMU_BINARY", "qemu-system-x86_64"))
    image = self.environment.get("AGENTA_QEMU_IMAGE")
    passthrough_raw = self.environment.get("AGENTA_QEMU_PASSTHROUGH_DEVICES", "")
    bridge_name = self.environment.get("AGENTA_QEMU_NET_BRIDGE")
    netdev_id = self.environment.get("AGENTA_QEMU_NETDEV_ID", "br0")
    nic_model = self.environment.get("AGENTA_QEMU_NET_DEVICE", "virtio-net-pci")
    qmp_socket = self.environment.get("AGENTA_QEMU_QMP")
    managed_shutdown_flag = str(self.environment.get("AGENTA_QEMU_MANAGED_SHUTDOWN", "1"))
    tap_name = self.environment.get("AGENTA_QEMU_TAP")
    passthrough_devices, passthrough_missing = self._parse_passthrough_devices(passthrough_raw)

    metadata: Dict[str, object] = {
      "binary": qemu_binary,
      "image": image,
      "image_type": self.environment.get("AGENTA_QEMU_IMAGE_TYPE", "disk"),
      "drive_interface": self.environment.get("AGENTA_QEMU_DRIVE_INTERFACE", "virtio"),
      "boot": self.environment.get("AGENTA_QEMU_BOOT"),
      "snapshot": self.environment.get("AGENTA_QEMU_SNAPSHOT", "1"),
      "memory": self.environment.get("AGENTA_QEMU_MEMORY", "8192"),
      "cpus": self.environment.get("AGENTA_QEMU_CPUS", "4"),
      "headless": self.environment.get("AGENTA_QEMU_HEADLESS", "1"),
      "display": self.environment.get("AGENTA_QEMU_DISPLAY"),
      "daemonize": self.environment.get("AGENTA_QEMU_DAEMONIZE", "1"),
      "extra_args": self.environment.get("AGENTA_QEMU_EXTRA_ARGS", ""),
      "passthrough_raw": self.environment.get("AGENTA_QEMU_PASSTHROUGH_DEVICES", ""),
      "net_bridge": self.environment.get("AGENTA_QEMU_NET_BRIDGE"),
      "netdev_id": self.environment.get("AGENTA_QEMU_NETDEV_ID", "br0"),
      "net_device": self.environment.get("AGENTA_QEMU_NET_DEVICE", "virtio-net-pci"),
      "tap": self.environment.get("AGENTA_QEMU_TAP"),
      "net_mac": self.environment.get("AGENTA_QEMU_NET_MAC"),
      "qmp_socket": self.environment.get("AGENTA_QEMU_QMP"),
      "managed_shutdown": self.environment.get("AGENTA_QEMU_MANAGED_SHUTDOWN", "1"),
      "execute_flag": self.environment.get("AGENTA_QEMU_EXECUTE", "0"),
      "passthrough_raw": passthrough_raw,
      "passthrough_devices": passthrough_devices,
      "passthrough_missing": passthrough_missing,
      "bridge": bridge_name,
      "tap": tap_name,
      "netdev_id": netdev_id,
      "net_device": nic_model,
      "managed_shutdown": managed_shutdown_flag,
      "qmp_socket": qmp_socket,
    }
    metadata["passthrough_missing_count"] = len(passthrough_missing)
    if not qemu_binary or not image:
      summary = "QEMU binary or image not configured."
      return VirtualizationPlan(
        provider="qemu",
        action=action,
        summary=summary,
        command=None,
        metadata=metadata,
      )

    memory = str(metadata["memory"])
    cpus = str(metadata["cpus"])
    headless = str(metadata["headless"]).lower() not in {"0", "false", "no"}
    execute_flag = str(metadata["execute_flag"]).lower() in {"1", "true", "yes"}
    image_type = str(metadata["image_type"] or "disk").lower()
    drive_interface = str(metadata["drive_interface"] or "virtio")
    snapshot_enabled = str(metadata["snapshot"]).lower() not in {"0", "false", "no"}
    daemonize_flag = str(metadata["daemonize"]).lower() not in {"0", "false", "no"}
    display = metadata.get("display")
    extra_args = (metadata.get("extra_args") or "").strip()
    boot_target = metadata.get("boot")
    managed_shutdown_enabled = managed_shutdown_flag.lower() not in {"0", "false", "no"}

    base_command = [qemu_binary, "-m", memory, "-smp", cpus]
    if image_type == "cdrom":
      base_command.extend(["-cdrom", image])
    else:
      base_command.extend(["-drive", f"file={image},if={drive_interface}"])
    if boot_target:
      base_command.extend(["-boot", str(boot_target)])
    if snapshot_enabled:
      base_command.append("-snapshot")

    if action == "down":
      shutdown_plan = self._managed_shutdown_plan(
        execute_flag=execute_flag,
        managed_shutdown_enabled=managed_shutdown_enabled,
        qmp_target=qmp_socket,
        metadata=metadata,
      )
      if shutdown_plan:
        return shutdown_plan
      summary = "QEMU verification before shutdown."
      return VirtualizationPlan(
        provider="qemu",
        action=action,
        summary=summary,
        command=[qemu_binary, "-version"],
        execute=False,
        metadata=metadata,
      )

    if execute_flag:
      launch_command = list(base_command)
      if headless:
        launch_command.append("-nographic")
      elif display:
        launch_command.extend(["-display", str(display)])
      if daemonize_flag:
        launch_command.append("-daemonize")
      tap = (tap_name or "").strip() if 'tap_name' in locals() else ""
      bridge = (bridge_name or "").strip()
      netdev_id_value = (netdev_id or "br0").strip() or "br0"
      nic_model_value = nic_model or "virtio-net-pci"
      mac_address = (self.environment.get("AGENTA_QEMU_NET_MAC") or "").strip()
      metadata["net_mac"] = mac_address
      if tap:
        netdev_arg = f"tap,id={netdev_id_value},ifname={tap},script=no,downscript=no"
        launch_command.extend(["-netdev", netdev_arg])
        device_arg = f"{nic_model_value},netdev={netdev_id_value}"
        if mac_address:
          device_arg += f",mac={mac_address}"
        launch_command.extend(["-device", device_arg])
        metadata["netdev_configured"] = {"type": "tap", "interface": tap, "netdev_id": netdev_id_value}
      elif bridge:
        netdev_arg = f"bridge,id={netdev_id_value},br={bridge}"
        launch_command.extend(["-netdev", netdev_arg])
        device_arg = f"{nic_model_value},netdev={netdev_id_value}"
        if mac_address:
          device_arg += f",mac={mac_address}"
        launch_command.extend(["-device", device_arg])
        metadata["netdev_configured"] = {"type": "bridge", "interface": bridge, "netdev_id": netdev_id_value}
      passthrough_attached = 0
      for index, device in enumerate(passthrough_devices, start=1):
        dtype = device.get("type")
        if dtype == "pci" and device.get("address"):
          launch_command.extend(["-device", f"vfio-pci,host={device['address']}"])
          passthrough_attached += 1
        elif dtype == "usb" and device.get("bus") is not None and device.get("addr") is not None:
          launch_command.extend([
            "-device",
            f"usb-host,hostbus={device['bus']},hostaddr={device['addr']}",
          ])
          passthrough_attached += 1
        elif dtype == "path" and device.get("path"):
          drive_id = f"passthrough{index}"
          launch_command.extend(["-drive", f"if=none,id={drive_id},file={device['path']},format=raw"])
          launch_command.extend(["-device", f"virtio-blk-pci,drive={drive_id},bootindex={5 + index}"])
          passthrough_attached += 1
      metadata["passthrough_attached"] = passthrough_attached
      metadata["netdev_attached"] = bool(tap or bridge)
      if extra_args:
        launch_command.extend(shlex.split(extra_args))
      summary = "QEMU launch command prepared."
      return VirtualizationPlan(
        provider="qemu",
        action=action,
        summary=summary,
        command=launch_command,
        execute=True,
        metadata=metadata,
      )

    summary = "QEMU presence verified; enable AGENTA_QEMU_EXECUTE=1 to launch."
    return VirtualizationPlan(
      provider="qemu",
      action=action,
      summary=summary,
      command=[qemu_binary, "-version"],
      execute=False,
      metadata=metadata,
    )

  def _managed_shutdown_plan(
    self,
    *,
    execute_flag: bool,
    managed_shutdown_enabled: bool,
    qmp_target: Optional[str],
    metadata: Dict[str, object],
  ) -> Optional[VirtualizationPlan]:
    if not execute_flag or not managed_shutdown_enabled:
      metadata["managed_shutdown_enabled"] = False
      if managed_shutdown_enabled and not execute_flag:
        metadata["managed_shutdown_reason"] = "execute flag disabled"
      return None

    target = (qmp_target or "").strip()
    metadata["managed_shutdown_enabled"] = True
    if not target:
      metadata["managed_shutdown_reason"] = "qmp target not configured"
      return None

    metadata["managed_shutdown_target"] = target
    tcp_candidate = ""
    if target.startswith("tcp://"):
      tcp_candidate = target[len("tcp://") :]
    elif target.startswith("tcp:"):
      tcp_candidate = target[len("tcp:") :]
    elif ":" in target and "/" not in target:
      tcp_candidate = target

    command: Optional[List[str]] = None
    summary: str

    if tcp_candidate:
      host, _, port_str = tcp_candidate.partition(":")
      try:
        port = int(port_str)
      except ValueError:
        metadata["managed_shutdown_reason"] = f"invalid tcp endpoint '{tcp_candidate}'"
        return None
      metadata["managed_shutdown_strategy"] = "tcp"
      metadata["managed_shutdown_endpoint"] = {"host": host, "port": port}
      script = (
        "import socket, sys;"
        f"sock = socket.create_connection(({repr(host)}, {port}));"
        "sock.sendall(b'{\"execute\":\"qmp_capabilities\"}\\r\\n');"
        "sock.recv(4096);"
        "sock.sendall(b'{\"execute\":\"system_powerdown\"}\\r\\n');"
        "sock.close()"
      )
      command = [sys.executable, "-c", script]
      summary = "QMP system_powerdown issued via TCP monitor."
    else:
      path = target
      if target.startswith("unix://"):
        path = target[len("unix://") :]
      elif target.startswith("unix:"):
        path = target[len("unix:") :]
      metadata["managed_shutdown_strategy"] = "unix"
      metadata["managed_shutdown_socket"] = path
      metadata["managed_shutdown_socket_exists"] = Path(path).exists()
      script = (
        "import socket, sys;"
        "sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM);"
        f"sock.connect({repr(path)});"
        "sock.sendall(b'{\"execute\":\"qmp_capabilities\"}\\r\\n');"
        "sock.recv(4096);"
        "sock.sendall(b'{\"execute\":\"system_powerdown\"}\\r\\n');"
        "sock.close()"
      )
      command = [sys.executable, "-c", script]
      summary = "QMP system_powerdown issued via UNIX socket."

    return VirtualizationPlan(
      provider="qemu",
      action="down",
      summary=summary,
      command=command,
      execute=True,
      metadata=metadata,
    )

  def _plan_libvirt(self, action: str) -> Optional[VirtualizationPlan]:
    virsh = shutil.which("virsh")
    domain = self.environment.get("AGENTA_LIBVIRT_DOMAIN")
    metadata = {
      "binary": virsh,
      "domain": domain,
      "execute_flag": self.environment.get("AGENTA_LIBVIRT_EXECUTE", "0"),
    }
    if not virsh or not domain:
      summary = "virsh or libvirt domain not configured."
      return VirtualizationPlan(
        provider="libvirt",
        action=action,
        summary=summary,
        command=None,
        metadata=metadata,
      )

    execute_flag = str(metadata["execute_flag"]).lower() in {"1", "true", "yes"}
    if execute_flag:
      verb = "start" if action == "up" else "shutdown"
      summary = f"libvirt will {verb} domain when forensic safeguards permit."
      return VirtualizationPlan(
        provider="libvirt",
        action=action,
        summary=summary,
        command=[virsh, verb, domain],
        execute=True,
        metadata=metadata,
      )

    summary = "libvirt domain state inspected; set AGENTA_LIBVIRT_EXECUTE=1 for control."
    return VirtualizationPlan(
      provider="libvirt",
      action=action,
      summary=summary,
      command=[virsh, "dominfo", domain],
      execute=False,
      metadata=metadata,
    )

  def _status_qemu(self) -> Dict[str, object]:
    qemu_binary = shutil.which(self.environment.get("AGENTA_QEMU_BINARY", "qemu-system-x86_64"))
    image = self.environment.get("AGENTA_QEMU_IMAGE")
    execute_flag = self.environment.get("AGENTA_QEMU_EXECUTE", "0")
    metadata = {
      "binary": qemu_binary,
      "image": image,
      "image_type": self.environment.get("AGENTA_QEMU_IMAGE_TYPE", "disk"),
      "drive_interface": self.environment.get("AGENTA_QEMU_DRIVE_INTERFACE", "virtio"),
      "boot": self.environment.get("AGENTA_QEMU_BOOT"),
      "snapshot": self.environment.get("AGENTA_QEMU_SNAPSHOT", "1"),
      "execute_flag": execute_flag,
      "memory": self.environment.get("AGENTA_QEMU_MEMORY", "8192"),
      "cpus": self.environment.get("AGENTA_QEMU_CPUS", "4"),
      "headless": self.environment.get("AGENTA_QEMU_HEADLESS", "1"),
      "display": self.environment.get("AGENTA_QEMU_DISPLAY"),
      "daemonize": self.environment.get("AGENTA_QEMU_DAEMONIZE", "1"),
      "extra_args": self.environment.get("AGENTA_QEMU_EXTRA_ARGS", ""),
      "passthrough_raw": self.environment.get("AGENTA_QEMU_PASSTHROUGH_DEVICES", ""),
      "passthrough_devices": self._parse_passthrough_devices(self.environment.get("AGENTA_QEMU_PASSTHROUGH_DEVICES", "")),
      "bridge": self.environment.get("AGENTA_QEMU_NET_BRIDGE"),
      "netdev_id": self.environment.get("AGENTA_QEMU_NETDEV_ID", "br0"),
      "net_device": self.environment.get("AGENTA_QEMU_NET_DEVICE", "virtio-net-pci"),
      "managed_shutdown": self.environment.get("AGENTA_QEMU_MANAGED_SHUTDOWN", "1"),
      "qmp_socket": self.environment.get("AGENTA_QEMU_QMP"),
    }
    configured = bool(qemu_binary)
    status: Dict[str, object] = {
      "provider": "qemu",
      "configured": configured,
      "metadata": metadata,
      "notes": "",
      "running_instances": self._detect_processes(["qemu-system", "qemu-kvm"]),
    }
    if not qemu_binary:
      status["notes"] = "QEMU binary not found on PATH."
    elif not image:
      status["notes"] = "QEMU image not configured; set AGENTA_QEMU_IMAGE."
    else:
      status["notes"] = "QEMU ready."
    return status

  def _status_libvirt(self) -> Dict[str, object]:
    virsh = shutil.which("virsh")
    domain = self.environment.get("AGENTA_LIBVIRT_DOMAIN")
    execute_flag = self.environment.get("AGENTA_LIBVIRT_EXECUTE", "0")
    status: Dict[str, object] = {
      "provider": "libvirt",
      "configured": bool(virsh),
      "metadata": {
        "binary": virsh,
        "domain": domain,
        "execute_flag": execute_flag,
      },
      "notes": "",
      "domains": [],
    }
    if not virsh:
      status["notes"] = "virsh command not available."
      return status
    command = [virsh, "list", "--all"]
    try:
      completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT,
        check=False,
      )
      if completed.returncode != 0:
        status["notes"] = completed.stderr.strip() or "Failed to query libvirt domains."
        return status
      domains = self._parse_libvirt_list(completed.stdout)
      status["domains"] = domains
      if not domains:
        status["notes"] = "No libvirt domains found."
      else:
        status["notes"] = "Libvirt domains enumerated."
    except subprocess.TimeoutExpired:
      status["notes"] = f"virsh list timed out after {DEFAULT_TIMEOUT}s."
    except Exception as exc:
      status["notes"] = f"virsh inspection failed: {exc}"
    return status

  def _detect_processes(self, patterns: Sequence[str]) -> List[Dict[str, object]]:
    try:
      completed = subprocess.run(
        ["ps", "-Ao", "pid,command"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
      )
    except Exception:
      return []
    results: List[Dict[str, object]] = []
    for line in completed.stdout.strip().splitlines():
      line = line.strip()
      if not line or line.startswith("PID"):
        continue
      for pattern in patterns:
        if pattern in line:
          parts = line.split(None, 1)
          if not parts:
            continue
          pid = parts[0]
          command = parts[1] if len(parts) > 1 else ""
          results.append({"pid": pid, "command": command, "pattern": pattern})
          break
    return results

  @staticmethod
  def _parse_libvirt_list(output: str) -> List[Dict[str, object]]:
    domains: List[Dict[str, object]] = []
    for line in output.strip().splitlines():
      line = line.strip()
      if not line or line.lower().startswith("id"):
        continue
      segments = line.split()
      if len(segments) < 3:
        continue
      domain_id = segments[0]
      name = segments[1]
      state = " ".join(segments[2:])
      domains.append({"id": domain_id, "name": name, "state": state})
    return domains
