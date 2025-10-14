"""
Resource provider abstractions for AgentaOS scalability workflows.

These helpers allow the scalability meta-agent to interrogate different
execution substrates—local Docker daemons, Multipass virtual machines, or
placeholder cloud APIs—without hard-coding CLI logic in the agent itself.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_TIMEOUT = 10
MAX_OUTPUT = 2048


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request."""


def _run(
    command: List[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ProviderError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ProviderError(
            f"Command '{' '.join(command)}' failed with code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"Command '{' '.join(command)}' timed out after {timeout}s") from exc
    except PermissionError as exc:
        raise ProviderError(f"Permission denied executing '{' '.join(command)}': {exc}") from exc


def _truncate(value: str, limit: int = MAX_OUTPUT) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


@dataclass
class ProviderReport:
    name: str
    healthy: bool
    message: str
    details: Dict[str, object]


class ResourceProvider:
    """Abstract base class for resource providers."""

    name: str = "unknown"

    def available(self) -> bool:  # pragma: no cover - interface definition
        raise NotImplementedError

    def inventory(self) -> ProviderReport:  # pragma: no cover - interface
        raise NotImplementedError

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        raise NotImplementedError

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        raise NotImplementedError


class DockerProvider(ResourceProvider):
    name = "docker"

    def available(self) -> bool:
        return shutil.which("docker") is not None

    def inventory(self) -> ProviderReport:
        try:
            info = _run(["docker", "info"])
            containers = _run(["docker", "ps", "--format", "{{json .}}"])
            items = []
            for line in containers.stdout.strip().splitlines():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            stats_entries: List[Dict[str, object]] = []
            try:
                stats = _run(["docker", "stats", "--no-stream", "--format", "{{json .}}"])
                for line in stats.stdout.strip().splitlines():
                    try:
                        stats_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except ProviderError:
                stats_entries = []
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="Docker daemon responsive.",
                details={
                    "info": _truncate(info.stdout),
                    "containers": items,
                    "container_count": len(items),
                    "stats": stats_entries[:10],
                },
            )
        except ProviderError as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=str(exc),
                details={},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        # Scaling is highly workload-specific; we surface intent for external orchestrators.
        details = {"intent": intent or {}, "action": "scale_up_request"}
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Scale-up request logged for Docker orchestrator.",
            details=details,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        details = {"intent": intent or {}, "action": "scale_down_policy"}
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Scale-down policy evaluated for Docker workloads.",
            details=details,
        )


class MultipassProvider(ResourceProvider):
    name = "multipass"

    def available(self) -> bool:
        return shutil.which("multipass") is not None

    def inventory(self) -> ProviderReport:
        try:
            listing = _run(["multipass", "list", "--format", "json"])
            payload = json.loads(listing.stdout)
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="Multipass inventory collected.",
                details=payload,
            )
        except (ProviderError, json.JSONDecodeError) as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=f"Multipass inspection failed: {exc}",
                details={},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Multipass scale-up request recorded.",
            details={"intent": intent or {}},
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Multipass scale-down request recorded.",
            details={"intent": intent or {}},
        )


class CloudStubProvider(ResourceProvider):
    """
    Placeholder provider representing an external cloud/VPS API.

    The provider reads expected API endpoints or auth tokens from the supplied
    intent dictionary.  Since outbound networking may be restricted, we limit
    the implementation to assembling request metadata for operators to review.
    """

    name = "cloud_stub"

    def __init__(self, defaults: Optional[Dict[str, object]] = None) -> None:
        self.defaults = defaults or {}

    def available(self) -> bool:
        # The stub is always available; the runtime will supply the metadata.
        return True

    def inventory(self) -> ProviderReport:
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Cloud inventory requires external API call.",
            details={
                "instructions": "Integrate with your IaaS provider SDK or REST API here.",
                "defaults": self.defaults,
            },
        )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {**self.defaults, **(intent or {})}
        payload["timestamp"] = time.time()
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Cloud scale-up request constructed; send to provider API.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {**self.defaults, **(intent or {})}
        payload["timestamp"] = time.time()
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Cloud scale-down request constructed; send to provider API.",
            details=payload,
        )

class AwsCliProvider(ResourceProvider):
    """
    AWS provider backed by the awscli binary.

    The CLI must be configured with credentials/profile.  The provider queries
    EC2 instances and emits a summary suitable for the scalability agent.
    """

    name = "aws"

    def __init__(self, environment: Dict[str, str]):
        self.environment = environment
        self.aws_cli = shutil.which("aws")
        self.region = environment.get("AGENTA_AWS_REGION") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.profile = environment.get("AGENTA_AWS_PROFILE")

    def available(self) -> bool:
        return self.aws_cli is not None

    def _command(self, *args: str) -> List[str]:
        command = [self.aws_cli] + list(args)
        if self.profile:
            command.extend(["--profile", self.profile])
        if self.region:
            command.extend(["--region", self.region])
        command.extend(["--output", "json"])
        return command

    def inventory(self) -> ProviderReport:
        try:
            instances = _run(self._command("ec2", "describe-instances"))
            payload = json.loads(instances.stdout) if instances.stdout else {}
            summary = {
                "reservations": len(payload.get("Reservations", [])),
                "instances": sum(len(reservation.get("Instances", [])) for reservation in payload.get("Reservations", [])),
                "region": self.region,
                "profile": self.profile,
            }
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="AWS EC2 inventory retrieved.",
                details=summary,
            )
        except (ProviderError, json.JSONDecodeError) as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=f"AWS inventory unavailable: {exc}",
                details={"region": self.region, "profile": self.profile},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "region": self.region,
            "profile": self.profile,
            "next_steps": "Invoke infrastructure orchestration (CloudFormation/Terraform) outside of runtime.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="AWS scale-up request prepared; manual orchestration required.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "region": self.region,
            "profile": self.profile,
            "next_steps": "Decommission target instances with IaC/CLI.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="AWS scale-down request prepared; manual orchestration required.",
            details=payload,
        )


class AzureCliProvider(ResourceProvider):
    """Azure provider backed by the az CLI."""

    name = "azure"

    def __init__(self, environment: Dict[str, str]):
        self.az_cli = shutil.which("az")
        self.subscription = environment.get("AGENTA_AZURE_SUBSCRIPTION")

    def available(self) -> bool:
        return self.az_cli is not None

    def _command(self, *args: str) -> List[str]:
        command = [self.az_cli] + list(args)
        if self.subscription:
            command.extend(["--subscription", self.subscription])
        command.extend(["--output", "json"])
        return command

    def inventory(self) -> ProviderReport:
        try:
            vms = _run(self._command("vm", "list"))
            payload = json.loads(vms.stdout) if vms.stdout else []
            summary = {
                "vm_count": len(payload),
                "subscription": self.subscription,
            }
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="Azure VM inventory retrieved.",
                details=summary,
            )
        except (ProviderError, json.JSONDecodeError) as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=f"Azure inventory unavailable: {exc}",
                details={"subscription": self.subscription},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "subscription": self.subscription,
            "next_steps": "Trigger deployment pipeline (ARM/Bicep/Terraform) externally.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Azure scale-up request prepared; manual orchestration required.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "subscription": self.subscription,
            "next_steps": "Deallocate or delete VMs via deployment tooling.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="Azure scale-down request prepared; manual orchestration required.",
            details=payload,
        )


class GcloudCliProvider(ResourceProvider):
    """Google Cloud provider backed by the gcloud CLI."""

    name = "gcloud"

    def __init__(self, environment: Dict[str, str]):
        self.gcloud_cli = shutil.which("gcloud")
        self.project = environment.get("AGENTA_GCP_PROJECT")
        self.zone = environment.get("AGENTA_GCP_ZONE")

    def available(self) -> bool:
        return self.gcloud_cli is not None

    def _command(self, *args: str) -> List[str]:
        command = [self.gcloud_cli] + list(args)
        if self.project:
            command.extend(["--project", self.project])
        if self.zone:
            command.extend(["--zone", self.zone])
        command.extend(["--format", "json"])
        return command

    def inventory(self) -> ProviderReport:
        try:
            instances = _run(self._command("compute", "instances", "list"))
            payload = json.loads(instances.stdout) if instances.stdout else []
            summary = {
                "instance_count": len(payload),
                "project": self.project,
                "zone": self.zone,
            }
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="GCP compute inventory retrieved.",
                details=summary,
            )
        except (ProviderError, json.JSONDecodeError) as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=f"GCP inventory unavailable: {exc}",
                details={"project": self.project, "zone": self.zone},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "project": self.project,
            "zone": self.zone,
            "next_steps": "Leverage deployment manager or Terraform to create instances.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="GCP scale-up request prepared; manual orchestration required.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "project": self.project,
            "zone": self.zone,
            "next_steps": "Decommission compute resources via automation tooling.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="GCP scale-down request prepared; manual orchestration required.",
            details=payload,
        )


class QemuProvider(ResourceProvider):
    """Local virtualization provider backed by qemu-system binaries."""

    name = "qemu"

    def __init__(self, environment: Dict[str, str]):
        self.environment = environment
        self.qemu_path = shutil.which(environment.get("AGENTA_QEMU_BINARY", "qemu-system-x86_64"))
        self.image_path = environment.get("AGENTA_QEMU_IMAGE")
        self.image_type = environment.get("AGENTA_QEMU_IMAGE_TYPE", "disk")
        self.drive_interface = environment.get("AGENTA_QEMU_DRIVE_INTERFACE", "virtio")
        self.boot_target = environment.get("AGENTA_QEMU_BOOT")
        self.snapshot = environment.get("AGENTA_QEMU_SNAPSHOT", "1")
        self.memory = environment.get("AGENTA_QEMU_MEMORY", "8192")
        self.cpus = environment.get("AGENTA_QEMU_CPUS", "4")
        self.headless = environment.get("AGENTA_QEMU_HEADLESS", "1")
        self.display = environment.get("AGENTA_QEMU_DISPLAY")
        self.daemonize = environment.get("AGENTA_QEMU_DAEMONIZE", "1")
        self.extra_args = environment.get("AGENTA_QEMU_EXTRA_ARGS", "")

    def available(self) -> bool:
        return self.qemu_path is not None

    def inventory(self) -> ProviderReport:
        details = {
            "qemu_binary": self.qemu_path,
            "image": self.image_path,
            "image_type": self.image_type,
            "drive_interface": self.drive_interface,
            "boot": self.boot_target,
            "snapshot": self.snapshot,
            "memory": self.memory,
            "cpus": self.cpus,
            "headless": self.headless,
            "display": self.display,
            "daemonize": self.daemonize,
            "extra_args": self.extra_args,
            "instructions": "Use qemu-system-* with provided image and resources.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True if self.qemu_path else False,
            message="QEMU binary detected." if self.qemu_path else "QEMU binary not found.",
            details=details,
        )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "command_example": self._example_command(intent),
            "notes": "Execute outside forensic mode to launch virtual machine.",
        }
        return ProviderReport(
            name=self.name,
            healthy=bool(self.qemu_path),
            message="QEMU scale-up instructions prepared.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "notes": "Shut down QEMU guests via monitor or process manager.",
        }
        return ProviderReport(
            name=self.name,
            healthy=True,
            message="QEMU scale-down instructions prepared.",
            details=payload,
        )

    def _example_command(self, intent: Optional[Dict[str, object]]) -> Optional[str]:
        if not self.qemu_path or not self.image_path:
            return None
        memory = intent.get("memory_mb") if intent else None
        cpus = intent.get("cpus") if intent else None
        memory_value = str(memory or self.memory)
        cpu_value = str(cpus or self.cpus)
        command = [
            self.qemu_path,
            "-m",
            memory_value,
            "-smp",
            cpu_value,
        ]
        if (self.image_type or "").lower() == "cdrom":
            command.extend(["-cdrom", self.image_path])
        else:
            command.extend(["-drive", f"file={self.image_path},if={self.drive_interface or 'virtio'}"])
        if self.boot_target:
            command.extend(["-boot", self.boot_target])
        if str(self.snapshot).lower() not in {"0", "false", "no"}:
            command.append("-snapshot")
        if str(self.headless).lower() not in {"0", "false", "no"}:
            command.append("-nographic")
        elif self.display:
            command.extend(["-display", self.display])
        if str(self.daemonize).lower() not in {"0", "false", "no"}:
            command.append("-daemonize")
        if self.extra_args:
            command.extend(shlex.split(self.extra_args))
        return " ".join(command)


class LibvirtProvider(ResourceProvider):
    """Local virtualization provider backed by virsh/libvirt."""

    name = "libvirt"

    def __init__(self, environment: Dict[str, str]):
        self.environment = environment
        self.virsh = shutil.which("virsh")
        self.domain = environment.get("AGENTA_LIBVIRT_DOMAIN")

    def available(self) -> bool:
        return self.virsh is not None

    def inventory(self) -> ProviderReport:
        if not self.virsh:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message="virsh command not found.",
                details={},
            )
        try:
            domains = _run([self.virsh, "list", "--all"])
            details = {
                "domains": domains.stdout.strip().splitlines(),
                "active_domain": self.domain,
            }
            return ProviderReport(
                name=self.name,
                healthy=True,
                message="Libvirt domain inventory retrieved.",
                details=details,
            )
        except ProviderError as exc:
            return ProviderReport(
                name=self.name,
                healthy=False,
                message=f"Libvirt inspection failed: {exc}",
                details={},
            )

    def scale_up(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "domain": self.domain,
            "command_example": f"{self.virsh} start {self.domain}" if self.virsh and self.domain else None,
        }
        return ProviderReport(
            name=self.name,
            healthy=bool(self.virsh and self.domain),
            message="Libvirt scale-up instructions prepared.",
            details=payload,
        )

    def scale_down(self, intent: Optional[Dict[str, object]] = None) -> ProviderReport:
        payload = {
            "intent": intent or {},
            "domain": self.domain,
            "command_example": f"{self.virsh} shutdown {self.domain}" if self.virsh and self.domain else None,
        }
        return ProviderReport(
            name=self.name,
            healthy=bool(self.virsh and self.domain),
            message="Libvirt scale-down instructions prepared.",
            details=payload,
        )


def discover_providers(environment: Dict[str, str]) -> List[ResourceProvider]:
    """
    Determine which providers should be wired into the scalability agent.

    Environment variables can hint at preferred providers.  Supported keys:
      - `AGENTA_PROVIDER` (comma-separated names: docker,multipass,aws,azure,gcloud,qemu,libvirt,cloud)
      - `AGENTA_CLOUD_REGION`, `AGENTA_CLOUD_PROJECT` (used by cloud stub)
      - `AGENTA_AWS_REGION`, `AGENTA_AWS_PROFILE`
      - `AGENTA_AZURE_SUBSCRIPTION`
      - `AGENTA_GCP_PROJECT`, `AGENTA_GCP_ZONE`
    """

    preference = [p.strip() for p in environment.get("AGENTA_PROVIDER", "").split(",") if p.strip()]
    providers: List[ResourceProvider] = []

    registry: Dict[str, ResourceProvider] = {
        "docker": DockerProvider(),
        "multipass": MultipassProvider(),
        "aws": AwsCliProvider(environment),
        "azure": AzureCliProvider(environment),
        "gcloud": GcloudCliProvider(environment),
        "qemu": QemuProvider(environment),
        "libvirt": LibvirtProvider(environment),
        "cloud": CloudStubProvider(
            defaults={
                "region": environment.get("AGENTA_CLOUD_REGION", environment.get("AGENTA_AWS_REGION", "us-east-1")),
                "project": environment.get("AGENTA_CLOUD_PROJECT"),
            }
        ),
    }

    def include(name: str) -> None:
        provider = registry.get(name)
        if provider and provider.available():
            providers.append(provider)

    if preference:
        for name in preference:
            include(name)
    else:
        # Auto-detect available local providers.
        for name in ("docker", "multipass", "aws", "azure", "gcloud", "qemu", "libvirt"):
            include(name)

    # Always append cloud stub last if explicitly requested.
    if "cloud" in preference and registry["cloud"] not in providers:
        providers.append(registry["cloud"])

    # Fallback to cloud stub when nothing else is available.
    if not providers:
        providers.append(registry["cloud"])

    return providers
