"""
Init-style supervisor abstraction for AgentaOS.

The supervisor consumes telemetry stored on the execution context and synthesizes
an orchestration report without mutating host state.  This helps model how a
future kernel/service manager could reason about process health, resource
pressure, and guidance from the oracle.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class InitSupervisor:
    """Read-only supervisor that summarises runtime metadata."""

    def __init__(self, metadata: Dict[str, dict], forensic_mode: bool) -> None:
        self.metadata = metadata
        self.forensic_mode = forensic_mode

    def build_report(self) -> Dict[str, object]:
        process = self.metadata.get("kernel_process_snapshot", {})
        memory = self.metadata.get("kernel_memory_snapshot", {})
        providers = self.metadata.get("scalability.monitor_load_summary", {})
        integrity = self.metadata.get("security_integrity", {})
        volume = self.metadata.get("storage_volume_inventory", {})
        forecast = self.metadata.get("oracle_forecast", {})
        risk = self.metadata.get("oracle_risk", {})
        guidance = self.metadata.get("oracle_guidance", {}).get("guidance", [])
        app_sup = self.metadata.get("application_supervisor", {}).get("summary", {})
        virtualization = self.metadata.get("virtualization.status", {})

        process_summary = process.get("summary", {})
        memory_pressure = memory.get("pressure")
        provider_alerts = [
            provider for provider in providers.get("providers", [])
            if not provider.get("healthy", True)
        ]
        virtualization_status = virtualization.get("status", [])
        virtualization_configured = sum(1 for entry in virtualization_status if entry.get("configured"))

        return {
            "forensic_mode": self.forensic_mode,
            "process_count": process_summary.get("count"),
            "process_anomalies": process_summary.get("anomalies", []),
            "memory_pressure": memory_pressure,
            "provider_alerts": provider_alerts[:5],
            "integrity_artifacts": integrity.get("artifacts", []),
            "volume_alerts": volume.get("alerts", []),
            "forecast_probability": forecast.get("probability"),
            "residual_risk": risk.get("probability"),
            "oracle_guidance": guidance,
            "application_summary": app_sup,
            "virtualization_status": virtualization_status,
            "virtualization_ready": virtualization_configured,
            "virtualization_plans": virtualization.get("plans", {}),
        }
