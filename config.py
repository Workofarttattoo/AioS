"""
Configuration and manifest definitions for AgentaOS.

A manifest describes which meta-agents participate in the runtime and the
ordered sequences of actions required for boot and shutdown.  The default
manifest defined here mirrors the earlier prototype but enriches the metadata
with human descriptions and criticality flags so orchestration logic can make
decisions about failure handling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ActionConfig:
    """Metadata describing a single sub-agent action."""

    key: str
    description: str
    critical: bool = True


@dataclass(frozen=True)
class MetaAgentConfig:
    """Metadata describing a meta-agent and its available actions."""

    name: str
    actions: List[ActionConfig]
    description: str = ""

    def action_keys(self) -> Iterable[str]:
        return (action.key for action in self.actions)


@dataclass(frozen=True)
class Manifest:
    """
    Represents the complete runtime manifest.

    Attributes:
        meta_agents: Mapping of meta-agent name to configuration.
        boot_sequence: Ordered list of meta-agent.action keys executed at boot.
        shutdown_sequence: Ordered list of meta-agent.action keys executed at shutdown.
    """

    meta_agents: Dict[str, MetaAgentConfig]
    boot_sequence: List[str]
    shutdown_sequence: List[str]

    def action_config(self, action_path: str) -> ActionConfig:
        meta, action = action_path.split(".", maxsplit=1)
        return _lookup_action(self.meta_agents, meta, action)


def load_manifest(path: Optional[str] = None) -> Manifest:
    """
    Load a manifest from disk or fall back to the default bundle.

    When a path is provided we expect a JSON file with schema:
        {
          "meta_agents": {
            "kernel": {
              "description": "...",
              "actions": [
                {"key": "process_management", "description": "...", "critical": true}
              ]
            },
            ...
          },
          "boot_sequence": ["kernel.process_management", ...],
          "shutdown_sequence": [...]
        }

    For production usage the JSON could be generated from a richer authoring
    format (TOML/YAML) or managed by a policy engine.
    """

    if path:
        with Path(path).expanduser().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        meta_agents = {
            name: MetaAgentConfig(
                name=name,
                description=entry.get("description", ""),
                actions=[
                    ActionConfig(
                        key=action["key"],
                        description=action.get("description", ""),
                        critical=action.get("critical", True),
                    )
                    for action in entry.get("actions", [])
                ],
            )
            for name, entry in data["meta_agents"].items()
        }
        return Manifest(
            meta_agents=meta_agents,
            boot_sequence=data["boot_sequence"],
            shutdown_sequence=data["shutdown_sequence"],
        )

    return DEFAULT_MANIFEST


def _lookup_action(meta_agents: Dict[str, MetaAgentConfig], meta: str, action: str) -> ActionConfig:
    if meta not in meta_agents:
        raise KeyError(f"Meta-agent '{meta}' not defined.")
    for config in meta_agents[meta].actions:
        if config.key == action:
            return config
    raise KeyError(f"Action '{action}' not defined for meta-agent '{meta}'.")


def _actions(*configs: ActionConfig) -> List[ActionConfig]:
    return list(configs)


DEFAULT_MANIFEST = Manifest(
    meta_agents={
        "kernel": MetaAgentConfig(
            name="kernel",
            description="Supervises process scheduling, memory, and system calls.",
            actions=_actions(
                ActionConfig("process_management", "Bring up the scheduler."),
                ActionConfig("memory_management", "Initialize memory subsystem."),
                ActionConfig("device_drivers", "Load hardware drivers."),
                ActionConfig("system_calls", "Expose syscall interface."),
                ActionConfig("audit", "Start kernel audit logging.", critical=False),
            ),
        ),
        "security": MetaAgentConfig(
            name="security",
            description="Coordinates all security hardening services.",
            actions=_actions(
                ActionConfig("access_control", "Activate RBAC policies."),
                ActionConfig("encryption", "Start cryptographic services."),
                ActionConfig("firewall", "Enable network firewall."),
                ActionConfig("threat_detection", "Launch anomaly detection.", critical=False),
                ActionConfig("audit_review", "Stream security audit logs.", critical=False),
                ActionConfig("integrity_survey", "Capture forensic integrity snapshot.", critical=False),
                ActionConfig("sovereign_suite", "Assess Sovereign toolkit readiness.", critical=False),
            ),
        ),
        "networking": MetaAgentConfig(
            name="networking",
            description="Configures and monitors networking capabilities.",
            actions=_actions(
                ActionConfig("network_configuration", "Configure interfaces."),
                ActionConfig("protocol_handling", "Enable protocol stack."),
                ActionConfig("data_transmission", "Start data plane."),
                ActionConfig("dns_resolver", "Prime DNS cache."),
            ),
        ),
        "storage": MetaAgentConfig(
            name="storage",
            description="Manages persistent storage and recovery pipelines.",
            actions=_actions(
                ActionConfig("file_system", "Mount filesystems."),
                ActionConfig("disk_management", "Validate disk health."),
                ActionConfig("recovery", "Verify recovery checkpoints."),
                ActionConfig("backup", "Schedule backup jobs.", critical=False),
                ActionConfig("volume_inventory", "Enumerate mounted volumes and digests.", critical=False),
            ),
        ),
        "application": MetaAgentConfig(
            name="application",
            description="Hosts user-space services and application lifecycle.",
            actions=_actions(
                ActionConfig("package_manager", "Sync package registry."),
                ActionConfig("dependency_resolver", "Resolve dependencies."),
                ActionConfig("service_manager", "Start background services."),
                ActionConfig("application_launcher", "Expose launcher gateway."),
                ActionConfig("supervisor", "Run application supervisor.", critical=False),
            ),
        ),
        "user": MetaAgentConfig(
            name="user",
            description="Owns user identity, sessions, and preferences.",
            actions=_actions(
                ActionConfig("authentication", "Start authentication service."),
                ActionConfig("profile_manager", "Load user profiles."),
                ActionConfig("preferences", "Sync preference store."),
                ActionConfig("session_manager", "Enable session management."),
            ),
        ),
        "gui": MetaAgentConfig(
            name="gui",
            description="Provides dynamic graphical environment.",
            actions=_actions(
                ActionConfig("window_management", "Initialize compositor."),
                ActionConfig("event_handling", "Route input events."),
                ActionConfig("gui_design", "Render adaptive UI theme."),
                ActionConfig("theme_management", "Load theme assets."),
            ),
        ),
        "scalability": MetaAgentConfig(
            name="scalability",
            description="Scales workloads horizontally and vertically.",
            actions=_actions(
                ActionConfig("monitor_load", "Observe resource utilization.", critical=False),
                ActionConfig("scale_up", "Provision additional capacity.", critical=False),
                ActionConfig("load_balancing", "Distribute workloads."),
                ActionConfig("scale_down", "Retire unused capacity.", critical=False),
                ActionConfig("virtualization_inspect", "Assess virtualization readiness.", critical=False),
                ActionConfig("virtualization_domains", "List available virtualization domains.", critical=False),
            ),
        ),
        "oracle": MetaAgentConfig(
            name="oracle",
            description="Probabilistic forecasting and quantum-inspired projections.",
            actions=_actions(
                ActionConfig("probabilistic_forecast", "Synthesize operational forecast.", critical=False),
                ActionConfig("risk_heatmap", "Assess residual security risk.", critical=False),
                ActionConfig("quantum_projection", "Simulate small-state quantum projection.", critical=False),
                ActionConfig("adaptive_guidance", "Generate cross-domain guidance.", critical=False),
            ),
        ),
        "ai_os": MetaAgentConfig(
            name="ai_os",
            description="AI Operating System core - ultra-fast inference, quantum computing, autonomous updates.",
            actions=_actions(
                ActionConfig("initialize", "Initialize AI OS components.", critical=False),
                ActionConfig("ultrafast_reasoning", "Use speculative decoding for 2-4x speedup.", critical=False),
                ActionConfig("quantum_forecast", "Quantum-enhanced issue forecasting.", critical=False),
                ActionConfig("autonomous_research", "Self-directed research on topics.", critical=False),
                ActionConfig("self_update_status", "Check self-update system status.", critical=False),
                ActionConfig("quantum_backends", "List available quantum backends.", critical=False),
            ),
        ),
        "orchestration": MetaAgentConfig(
            name="orchestration",
            description="Provides policy enforcement and telemetry fan-out.",
            actions=_actions(
                ActionConfig("policy_engine", "Apply runtime policies."),
                ActionConfig("telemetry", "Emit telemetry to observers."),
                ActionConfig("health_monitor", "Consolidate health signals."),
                ActionConfig("supervisor_report", "Synthesize init-style supervisor report.", critical=False),
            ),
        ),
    },
    boot_sequence=[
        "ai_os.initialize",  # Initialize AI OS first
        "kernel.process_management",
        "kernel.memory_management",
        "kernel.device_drivers",
        "kernel.system_calls",
        "kernel.audit",
        "security.access_control",
        "security.encryption",
        "security.firewall",
        "security.threat_detection",
        "security.integrity_survey",
        "security.sovereign_suite",
        "networking.network_configuration",
        "networking.protocol_handling",
        "networking.data_transmission",
        "networking.dns_resolver",
        "storage.file_system",
        "storage.disk_management",
        "storage.recovery",
        "storage.backup",
        "storage.volume_inventory",
        "application.package_manager",
        "application.dependency_resolver",
        "application.service_manager",
        "application.application_launcher",
        "user.authentication",
        "user.profile_manager",
        "user.preferences",
        "user.session_manager",
        "gui.window_management",
        "gui.event_handling",
        "gui.gui_design",
        "gui.theme_management",
        "scalability.monitor_load",
        "scalability.scale_up",
        "scalability.load_balancing",
        "scalability.scale_down",
        "oracle.probabilistic_forecast",
        "oracle.risk_heatmap",
        "oracle.quantum_projection",
        "oracle.adaptive_guidance",
        "ai_os.quantum_backends",  # Check quantum hardware availability
        "ai_os.quantum_forecast",  # AI-powered issue prediction
        "ai_os.self_update_status",  # Verify auto-update system
        "orchestration.supervisor_report",
        "orchestration.policy_engine",
        "orchestration.telemetry",
        "orchestration.health_monitor",
    ],
    shutdown_sequence=[
        "orchestration.supervisor_report",
        "orchestration.telemetry",
        "orchestration.health_monitor",
        "scalability.load_balancing",
        "scalability.scale_down",
        "scalability.monitor_load",
        "oracle.adaptive_guidance",
        "oracle.quantum_projection",
        "oracle.risk_heatmap",
        "oracle.probabilistic_forecast",
        "gui.theme_management",
        "gui.event_handling",
        "gui.window_management",
        "user.session_manager",
        "user.preferences",
        "application.service_manager",
        "application.application_launcher",
        "storage.backup",
        "storage.volume_inventory",
        "storage.file_system",
        "networking.data_transmission",
        "networking.network_configuration",
        "security.firewall",
        "security.integrity_survey",
        "security.sovereign_suite",
        "security.access_control",
        "kernel.audit",
        "kernel.process_management",
    ],
)
