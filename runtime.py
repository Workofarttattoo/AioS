"""
Runtime orchestration layer for AgentaOS.

This module translates the declarative manifest into executable sub-agents,
coordinates lifecycle events, and exposes a high-level runtime facade used by
the CLI or other supervisors.  It intentionally separates configuration from
execution logic so real subsystems can be integrated later without modifying
the sequencing rules.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .agents import (
    ApplicationAgent,
    GuiAgent,
    KernelAgent,
    NetworkingAgent,
    OrchestrationAgent,
    ScalabilityAgent,
    SecurityAgent,
    StorageAgent,
    UserAgent,
)
from .config import ActionConfig, Manifest, MetaAgentConfig
from .wizard import build_post_boot_guidance
from .model import ActionResult, AgentActionError
from .oracle import ProbabilisticOracle
from .supervisor import InitSupervisor


LOG = logging.getLogger("AgentaOS")


ActionCallable = Callable[["ExecutionContext"], ActionResult]


@dataclass
class SubAgent:
    """Concrete executable unit referenced by a meta-agent action."""

    key: str
    description: str
    handler: ActionCallable
    critical: bool = True

    def execute(self, ctx: "ExecutionContext") -> ActionResult:
        ctx.push_scope(self.key)
        start = time.perf_counter()
        try:
            LOG.debug("[debug] executing sub-agent %s", ctx.action_path)
            result = self.handler(ctx)
        except Exception as exc:
            LOG.exception("Sub-agent '%s' failed", ctx.action_path)
            result = ActionResult(
                success=False,
                message=f"[error] {ctx.action_path}: {exc}",
                payload={"exception": repr(exc)},
            )
        finally:
            ctx.pop_scope()

        result.elapsed = time.perf_counter() - start
        return result


@dataclass
class MetaAgent:
    """Collection of sub-agents contributing to a core subsystem."""

    name: str
    actions: Dict[str, SubAgent] = field(default_factory=dict)
    description: str = ""

    def execute(self, action: str, ctx: "ExecutionContext") -> ActionResult:
        if action not in self.actions:
            raise AgentActionError(f"Action '{action}' not supported by {self.name} meta-agent.")
        return self.actions[action].execute(ctx)

    def available_actions(self) -> Iterable[str]:
        return self.actions.keys()


@dataclass
class ExecutionContext:
    """
    Shared execution context for sub-agents.

    Holds manifest information, environment metadata, and scoped action path
    to allow sub-agents to publish telemetry or inspect dependencies.  In a
    real deployment the context would manage IPC channels, secrets, and shared
    resources.
    """

    manifest: Manifest
    environment: Dict[str, str] = field(default_factory=dict)
    action_stack: list[str] = field(default_factory=list)
    metadata: Dict[str, dict] = field(default_factory=dict)

    @property
    def action_path(self) -> str:
        return ".".join(self.action_stack)

    def push_scope(self, component: str) -> None:
        self.action_stack.append(component)

    def pop_scope(self) -> None:
        if self.action_stack:
            self.action_stack.pop()

    def publish_metadata(self, key: str, value: dict) -> None:
        self.metadata[key] = value


class SuperUserAgent:
    """
    Supervisory agent responsible for coordinating all meta-agents.

    In a production system this agent would enforce security policies,
    approve privileged operations, and provide an external control plane.
    """

    def __init__(self, manifest: Manifest, meta_agents: Dict[str, MetaAgent]):
        self.manifest = manifest
        self.meta_agents = meta_agents

    def boot_sequence(self) -> Iterable[str]:
        return self.manifest.boot_sequence

    def shutdown_sequence(self) -> Iterable[str]:
        return self.manifest.shutdown_sequence

    def execute_action_path(self, action_path: str, ctx: ExecutionContext) -> ActionResult:
        meta_name, action = resolve_action_path(action_path)
        if meta_name not in self.meta_agents:
            raise AgentActionError(f"Meta-agent '{meta_name}' not registered.")
        ctx.push_scope(meta_name)
        try:
            return self.meta_agents[meta_name].execute(action, ctx)
        finally:
            ctx.pop_scope()

    def list_capabilities(self) -> Dict[str, Iterable[str]]:
        return {name: agent.available_actions() for name, agent in self.meta_agents.items()}


class AgentaRuntime:
    """High-level runtime facade used by the CLI to manage lifecycle."""

    SECURITY_SUITE_ACTION = "security.sovereign_suite"

    def __init__(
        self,
        manifest: Manifest,
        action_library: Optional[Dict[str, ActionCallable]] = None,
    ):
        self.manifest = manifest
        self.meta_agents = build_meta_agents(manifest, action_library or build_default_action_library(manifest))
        self.superuser = SuperUserAgent(manifest, self.meta_agents)
        self.context = ExecutionContext(manifest=manifest)
        self.booted = False

    def boot(self) -> None:
        LOG.info("[info] Initiating AgentaOS boot sequence")
        executed: List[str] = []
        for action_path in self.superuser.boot_sequence():
            self._execute_path(action_path, stage="boot")
            executed.append(action_path)
        self.booted = True
        LOG.info("[info] AgentaOS boot complete")
        self._auto_run_security_suite(stage="boot", executed=executed)
        self._maybe_display_post_boot_guidance()

    def shutdown(self) -> None:
        if not self.booted:
            LOG.warning("[warn] Shutdown requested before boot; ignoring")
            return
        LOG.info("[info] Initiating AgentaOS shutdown sequence")
        for action_path in self.superuser.shutdown_sequence():
            self._execute_path(action_path, stage="shutdown", tolerate_failure=True)
        self.booted = False
        LOG.info("[info] AgentaOS shutdown complete")

    def status(self) -> str:
        return "[info] AgentaOS status: booted" if self.booted else "[info] AgentaOS status: offline"

    def execute(self, action_path: str) -> ActionResult:
        if not self.booted:
            raise AgentActionError("System must be booted before executing actions.")
        result = self._execute_path(action_path, stage="exec")
        self._auto_run_security_suite(stage="exec", executed=[action_path])
        return result

    def run_sequence(
        self,
        action_paths: Iterable[str],
        *,
        tolerate_failures: bool = False,
        auto_boot: bool = False,
        stage: str = "sequence",
        environment_overrides: Optional[Dict[str, str]] = None,
    ) -> List[ActionResult]:
        """
        Execute a collection of action paths while preserving runtime metadata.

        Args:
            action_paths: Iterable of manifest action paths to execute in order.
            tolerate_failures: When True, continue after non-critical failures.
            auto_boot: Automatically boot the system if it is not already booted.
            stage: Stage label recorded alongside telemetry metadata.
            environment_overrides: Optional environment values merged into the execution context.
        """
        if environment_overrides:
            self.context.environment.update(environment_overrides)
        if auto_boot and not self.booted:
            self.boot()
        if not self.booted:
            raise AgentActionError("System must be booted before executing a sequence.")

        results: List[ActionResult] = []
        executed: List[str] = []
        for action_path in action_paths:
            result = self._execute_path(action_path, stage=stage, tolerate_failure=tolerate_failures)
            results.append(result)
            executed.append(action_path)
        self._auto_run_security_suite(stage=stage, executed=executed)
        return results

    def _auto_run_security_suite(self, *, stage: str, executed: Iterable[str]) -> Optional[ActionResult]:
        """
        Automatically invoke the Sovereign Security Toolkit health check.

        When the environment includes ``AGENTA_SECURITY_TOOLS`` (typically set by
        the setup wizard) we ensure the corresponding runtime action runs once so
        metadata and diagnostics are captured for dashboards and follow-up flows.
        """

        if not self._should_run_security_suite(executed):
            return None

        try:
            LOG.info("[info] %s: verifying Sovereign Security Toolkit health.", stage)
            return self._execute_path(self.SECURITY_SUITE_ACTION, stage=stage, tolerate_failure=True)
        except AgentActionError as exc:
            LOG.warning("[warn] %s: Sovereign toolkit verification skipped (%s).", stage, exc)
            return None

    def _should_run_security_suite(self, executed: Iterable[str]) -> bool:
        raw_tools = self.context.environment.get("AGENTA_SECURITY_TOOLS", "")
        if not raw_tools or not raw_tools.strip():
            return False
        if self.SECURITY_SUITE_ACTION in executed:
            return False
        if self.SECURITY_SUITE_ACTION in self.context.metadata:
            return False
        security_config = self.manifest.meta_agents.get("security")
        if not security_config:
            return False
        action_name = self.SECURITY_SUITE_ACTION.split(".", maxsplit=1)[1]
        if action_name not in {action.key for action in security_config.actions}:
            return False
        return True

    def _maybe_display_post_boot_guidance(self) -> None:
        flag = str(self.context.environment.get("AGENTA_DISABLE_POST_BOOT_GUIDANCE", "")).lower()
        if flag in {"1", "true", "yes", "off"}:
            return
        checklist = build_post_boot_guidance()
        if not checklist:
            return
        LOG.info("[info] Host preparation checklist:")
        for entry in checklist:
            LOG.info("  - %s :: %s", entry.get("title", "item"), entry.get("summary", ""))

    def metadata_snapshot(self, keys: Optional[Iterable[str]] = None) -> Dict[str, object]:
        """
        Return a deep-copied snapshot of metadata published by executed actions.

        Args:
            keys: Optional iterable that limits the snapshot to specific metadata keys.
        """
        selected_keys = set(keys) if keys else None
        snapshot: Dict[str, object] = {}
        for meta_key, meta_value in self.context.metadata.items():
            if selected_keys and meta_key not in selected_keys:
                continue
            snapshot[meta_key] = copy.deepcopy(meta_value)
        return snapshot

    def metadata_summary(self, forensic_mode: Optional[bool] = None) -> Dict[str, object]:
        """
        Synthesize a supervisor-style summary of captured telemetry.

        Args:
            forensic_mode: Optional override for forensic posture detection. Defaults to the current environment flag.
        """
        if forensic_mode is None:
            value = self.context.environment.get("AGENTA_FORENSIC_MODE", "")
            forensic_mode = value.lower() in {"1", "true", "yes", "on"}
        supervisor = InitSupervisor(self.context.metadata, forensic_mode)
        return supervisor.build_report()

    def _execute_path(self, action_path: str, stage: str, tolerate_failure: bool = False) -> ActionResult:
        LOG.info("[info] %s: executing %s", stage, action_path)
        result = self.superuser.execute_action_path(action_path, self.context)
        LOG.info("%s (%.3fs)", result.message, result.elapsed)
        metadata_payload: Dict[str, object] = {}
        if isinstance(result.payload, dict):
            metadata_payload = dict(result.payload)
        elif result.payload is not None:
            metadata_payload = {"value": result.payload}
        metadata_payload.setdefault("_message", result.message)
        metadata_payload.setdefault("_stage", stage)
        self.context.publish_metadata(action_path, metadata_payload)
        if not result and not tolerate_failure and self.manifest.action_config(action_path).critical:
            raise AgentActionError(f"{stage} sequence halted: {result.message}")
        return result



def resolve_action_path(action_path: str) -> tuple[str, str]:
    parts = action_path.split(".", maxsplit=1)
    if len(parts) != 2:
        raise AgentActionError(f"Invalid action path '{action_path}'. Use meta.action syntax.")
    return parts[0], parts[1]


def build_meta_agents(manifest: Manifest, library: Dict[str, ActionCallable]) -> Dict[str, MetaAgent]:
    meta_agents: Dict[str, MetaAgent] = {}
    for name, config in manifest.meta_agents.items():
        actions: Dict[str, SubAgent] = {}
        for action in config.actions:
            path = f"{name}.{action.key}"
            handler = library.get(path, default_stub(path, action))
            actions[action.key] = SubAgent(
                key=action.key,
                description=action.description,
                handler=handler,
                critical=action.critical,
            )
        meta_agents[name] = MetaAgent(name=name, description=config.description, actions=actions)
    return meta_agents


def default_stub(action_path: str, action: ActionConfig) -> ActionCallable:
    def _handler(ctx: ExecutionContext) -> ActionResult:
        return ActionResult(
            success=True,
            message=f"[info] {action_path}: {action.description or 'completed'}",
            payload={"auto_generated": True},
        )

    return _handler


# --- Default action implementations ----------------------------------------------------

def build_default_action_library(manifest: Manifest) -> Dict[str, ActionCallable]:
    """
    Provide the default mapping of action paths to executable handlers.

    Each handler receives an ExecutionContext and may publish metadata or
    inspect previously recorded state.  The current implementations are still
    stubs but model what a production integration could look like.
    """

    library: Dict[str, ActionCallable] = {}
    agents = {
        "kernel": KernelAgent(),
        "security": SecurityAgent(),
        "networking": NetworkingAgent(),
        "storage": StorageAgent(),
        "application": ApplicationAgent(),
        "user": UserAgent(),
        "gui": GuiAgent(),
        "scalability": ScalabilityAgent(),
        "orchestration": OrchestrationAgent(),
    }

    for meta_name, config in manifest.meta_agents.items():
        agent = agents.get(meta_name)
        if not agent:
            continue
        for action in config.actions:
            handler = getattr(agent, action.key, None)
            if handler:
                library[f"{meta_name}.{action.key}"] = handler

    def forensic_mode(ctx: ExecutionContext) -> bool:
        value = ctx.environment.get("AGENTA_FORENSIC_MODE", "")
        return value.lower() in {"1", "true", "yes", "on"}

    # Oracle meta-agent
    def make_oracle_action(method: str) -> ActionCallable:
        def _handler(ctx: ExecutionContext) -> ActionResult:
            oracle = ProbabilisticOracle(forensic_mode=forensic_mode(ctx))
            telemetry = dict(ctx.metadata)
            if method == "probabilistic_forecast":
                forecast = oracle.forecast(telemetry)
                ctx.publish_metadata("oracle_forecast", {"probability": forecast.probability, "signals": forecast.signals})
                message = f"[info] oracle: Forecast probability {forecast.probability:.2%}"
                return ActionResult(success=True, message=message, payload={
                    "summary": forecast.summary,
                    "signals": forecast.signals,
                    "guidance": forecast.guidance,
                })
            if method == "risk_heatmap":
                risk = oracle.risk_assessment(telemetry)
                ctx.publish_metadata("oracle_risk", {"residual": risk.probability, "signals": risk.signals})
                message = f"[info] oracle: Residual risk {risk.probability:.2%}"
                return ActionResult(success=True, message=message, payload={
                    "summary": risk.summary,
                    "signals": risk.signals,
                    "guidance": risk.guidance,
                })
            if method == "quantum_projection":
                # Default to 10 qubits for quick simulation.
                quantum = oracle.quantum_projection(qubits=10, telemetry=telemetry)
                ctx.publish_metadata("oracle_quantum", {
                    "qubits": quantum.qubits,
                    "entropy": quantum.entropy,
                    "measurements": quantum.measurements,
                })
                return ActionResult(success=True, message=f"[info] oracle: Quantum entropy {quantum.entropy:.2f}", payload={
                    "qubits": quantum.qubits,
                    "measurements": quantum.measurements,
                    "narrative": quantum.narrative,
                })
            if method == "adaptive_guidance":
                forecast = oracle.forecast(telemetry)
                risk = oracle.risk_assessment(telemetry)
                quantum = oracle.quantum_projection(qubits=8, telemetry=telemetry)
                guidance = oracle.adaptive_guidance(forecast, risk, quantum)
                ctx.publish_metadata("oracle_guidance", {"guidance": guidance})
                return ActionResult(success=True, message="[info] oracle: Adaptive guidance generated.", payload={
                    "guidance": guidance,
                })
            raise AgentActionError(f"Oracle method '{method}' not implemented.")

        return _handler

    library["oracle.probabilistic_forecast"] = make_oracle_action("probabilistic_forecast")
    library["oracle.risk_heatmap"] = make_oracle_action("risk_heatmap")
    library["oracle.quantum_projection"] = make_oracle_action("quantum_projection")
    library["oracle.adaptive_guidance"] = make_oracle_action("adaptive_guidance")

    def make_supervisor_action() -> ActionCallable:
        def _handler(ctx: ExecutionContext) -> ActionResult:
            supervisor = InitSupervisor(ctx.metadata, forensic_mode(ctx))
            report = supervisor.build_report()
            ctx.publish_metadata("init_supervisor", report)
            return ActionResult(
                success=True,
                message="[info] orchestration: Supervisor report generated.",
                payload=report,
            )

        return _handler

    library["orchestration.supervisor_report"] = make_supervisor_action()

    return library
