"""
OpenAGI Forensic Mode - Dry-Run Execution System

Implements read-only execution paths for safe rehearsal and planning without
system mutations. Enables simulation of workflows, validation of approaches,
and comprehensive planning before actual execution.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4

try:
    from aios.runtime import ExecutionContext, ActionResult
except Exception:
    ExecutionContext = Any
    ActionResult = Any

LOG = logging.getLogger(__name__)


class SimulationOutcome(Enum):
    """Outcome of a simulated action."""
    SUCCESS = "success"  # Would succeed
    WOULD_FAIL = "would_fail"  # Would fail due to error
    BLOCKED = "blocked"  # Would be blocked (approval, permissions)
    UNSAFE = "unsafe"  # Unsafe operation detected
    UNKNOWN = "unknown"  # Outcome unknown


@dataclass
class SimulationStep:
    """Single step in a simulated workflow."""
    step_number: int
    action_path: str
    action_name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0
    estimated_tokens: int = 0
    outcome: SimulationOutcome = SimulationOutcome.UNKNOWN
    reason: str = ""
    side_effects: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    blocked_by: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['outcome'] = self.outcome.value
        return data


@dataclass
class ForensicSimulation:
    """Complete forensic simulation/dry-run of a workflow."""
    simulation_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_name: str = ""
    description: str = ""
    steps: List[SimulationStep] = field(default_factory=list)
    overall_outcome: SimulationOutcome = SimulationOutcome.UNKNOWN
    total_estimated_duration: float = 0.0
    total_estimated_tokens: int = 0
    safety_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    simulation_duration: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['steps'] = [s.to_dict() for s in self.steps]
        data['overall_outcome'] = self.overall_outcome.value
        return data

    def add_step(self, step: SimulationStep) -> None:
        """Add a simulation step."""
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.total_estimated_duration += step.estimated_duration
        self.total_estimated_tokens += step.estimated_tokens

    def has_safety_issues(self) -> bool:
        """Check if simulation detected safety issues."""
        return len(self.safety_issues) > 0

    def has_blocks(self) -> bool:
        """Check if workflow has blocking steps."""
        return any(step.blocked_by is not None for step in self.steps)


class ForensicSimulationStore:
    """Persistent storage for forensic simulations."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize forensic simulation storage."""
        self.storage_path = storage_path or Path.home() / ".aios" / "forensics"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.simulations_file = self.storage_path / "simulations.jsonl"
        self.recommendations_file = self.storage_path / "recommendations.jsonl"

    def save_simulation(self, simulation: ForensicSimulation) -> None:
        """Save forensic simulation to storage."""
        try:
            with open(self.simulations_file, 'a') as f:
                f.write(json.dumps(simulation.to_dict()) + '\n')
        except Exception as e:
            LOG.error(f"Error saving forensic simulation: {e}")

    def save_recommendation(
        self,
        simulation_id: str,
        workflow_name: str,
        recommendation: str,
        severity: str = "info"
    ) -> None:
        """Save recommendation from simulation."""
        try:
            with open(self.recommendations_file, 'a') as f:
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'simulation_id': simulation_id,
                    'workflow_name': workflow_name,
                    'recommendation': recommendation,
                    'severity': severity
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            LOG.error(f"Error saving recommendation: {e}")

    def get_simulations(self, workflow_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve saved simulations."""
        simulations = []
        try:
            if not self.simulations_file.exists():
                return simulations

            with open(self.simulations_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if workflow_name is None or data.get('workflow_name') == workflow_name:
                        simulations.append(data)
        except Exception as e:
            LOG.error(f"Error retrieving simulations: {e}")
        return simulations

    def get_recommendations(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve saved recommendations."""
        recommendations = []
        try:
            if not self.recommendations_file.exists():
                return recommendations

            with open(self.recommendations_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if severity is None or data.get('severity') == severity:
                        recommendations.append(data)
        except Exception as e:
            LOG.error(f"Error retrieving recommendations: {e}")
        return recommendations

    def get_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific simulation."""
        try:
            if not self.simulations_file.exists():
                return None

            with open(self.simulations_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get('simulation_id') == simulation_id:
                        return data
        except Exception as e:
            LOG.error(f"Error retrieving simulation: {e}")
        return None


class ForensicModeExecutor:
    """
    Executes workflows in forensic mode (read-only, safe simulation).

    Key responsibilities:
    1. Simulate action execution without mutations
    2. Predict outcomes and side effects
    3. Detect safety issues and blockers
    4. Generate comprehensive reports
    5. Store simulations for comparison
    6. Provide decision support for actual execution
    """

    def __init__(self, storage_path: Optional[Path] = None, approval_manager=None):
        """
        Initialize forensic mode executor.

        Args:
            storage_path: Path for storing simulation results
            approval_manager: Optional ApprovalWorkflowManager for checking approvals
        """
        self.storage = ForensicSimulationStore(storage_path)
        self.approval_manager = approval_manager
        self.action_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default simulation handlers for common actions."""
        # Example handlers for different action types
        self.register_handler("kernel.*", self._simulate_kernel_action)
        self.register_handler("security.*", self._simulate_security_action)
        self.register_handler("storage.*", self._simulate_storage_action)
        self.register_handler("user.*", self._simulate_user_action)

    def register_handler(self, action_pattern: str, handler: Callable) -> None:
        """
        Register a simulation handler for action pattern.

        Args:
            action_pattern: Pattern like "security.*" or "kernel.process_management"
            handler: Callable that simulates the action
        """
        self.action_handlers[action_pattern] = handler

    def simulate_workflow(
        self,
        workflow_name: str,
        description: str,
        actions: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ForensicSimulation:
        """
        Simulate a complete workflow execution.

        Args:
            workflow_name: Name of the workflow
            description: Description of what the workflow does
            actions: List of actions with their details
            parameters: Additional parameters for simulation

        Returns:
            ForensicSimulation with complete analysis
        """
        start_time = time.time()
        simulation = ForensicSimulation(
            workflow_name=workflow_name,
            description=description,
            parameters=parameters or {}
        )

        # Simulate each action
        for i, action_spec in enumerate(actions):
            step = self._simulate_action(
                action_spec.get('action_path', ''),
                action_spec.get('action_name', ''),
                action_spec.get('description', ''),
                action_spec.get('parameters', {})
            )
            simulation.add_step(step)

            # Check for blocking issues
            if step.blocked_by:
                LOG.warning(f"Workflow would be blocked at step {i+1}: {step.blocked_by}")
                simulation.safety_issues.append(f"Step {i+1} blocked: {step.blocked_by}")

        # Analyze overall outcome
        self._analyze_simulation(simulation)

        # Save simulation
        simulation.simulation_duration = time.time() - start_time
        self.storage.save_simulation(simulation)

        # Save recommendations
        for rec in simulation.recommendations:
            severity = "error" if simulation.has_safety_issues() else "info"
            self.storage.save_recommendation(
                simulation.simulation_id,
                workflow_name,
                rec,
                severity
            )

        return simulation

    def _simulate_action(
        self,
        action_path: str,
        action_name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> SimulationStep:
        """Simulate a single action."""
        step = SimulationStep(
            step_number=0,  # Will be set by add_step
            action_path=action_path,
            action_name=action_name,
            description=description,
            parameters=parameters,
            estimated_duration=0.5,  # Default estimate
            estimated_tokens=100  # Default estimate
        )

        # Find matching handler
        handler = self._find_handler(action_path)
        if handler:
            try:
                handler(step)
            except Exception as e:
                step.outcome = SimulationOutcome.WOULD_FAIL
                step.reason = f"Simulation error: {str(e)}"
                LOG.error(f"Error simulating action {action_path}: {e}")
        else:
            # Default simulation
            step.outcome = SimulationOutcome.SUCCESS
            step.reason = "Action would execute successfully"

        return step

    def _find_handler(self, action_path: str) -> Optional[Callable]:
        """Find simulation handler for action path."""
        import fnmatch
        for pattern, handler in self.action_handlers.items():
            if fnmatch.fnmatch(action_path, pattern):
                return handler
        return None

    def _simulate_kernel_action(self, step: SimulationStep) -> None:
        """Simulate kernel action."""
        step.estimated_duration = 1.0
        step.estimated_tokens = 150
        step.outcome = SimulationOutcome.SUCCESS
        step.reason = "Kernel action would execute"
        step.side_effects = ["System state change", "Process management"]

        if "reboot" in step.action_path.lower():
            step.side_effects.append("System will reboot")

    def _simulate_security_action(self, step: SimulationStep) -> None:
        """Simulate security action."""
        step.estimated_duration = 0.5
        step.estimated_tokens = 120
        step.outcome = SimulationOutcome.SUCCESS
        step.reason = "Security action would execute"
        step.side_effects = ["Security policy updated"]

        # Check if approval is needed
        if self.approval_manager:
            req = self.approval_manager.get_sensitivity_requirement(step.action_path)
            if req.sensitivity_level.value != "none":
                step.blocked_by = f"Requires {req.sensitivity_level.value} approval"
                step.outcome = SimulationOutcome.BLOCKED

    def _simulate_storage_action(self, step: SimulationStep) -> None:
        """Simulate storage action."""
        step.estimated_duration = 2.0
        step.estimated_tokens = 200

        if "delete" in step.action_path.lower() or "remove" in step.action_path.lower():
            step.side_effects = ["Data deletion", "Irreversible operation"]
            if self.approval_manager:
                req = self.approval_manager.get_sensitivity_requirement(step.action_path)
                if req.sensitivity_level.value in ["high", "critical"]:
                    step.blocked_by = f"Requires {req.sensitivity_level.value} approval"
                    step.outcome = SimulationOutcome.BLOCKED
                else:
                    step.outcome = SimulationOutcome.SUCCESS
            else:
                step.outcome = SimulationOutcome.SUCCESS
        else:
            step.outcome = SimulationOutcome.SUCCESS
            step.side_effects = ["Storage operation"]

    def _simulate_user_action(self, step: SimulationStep) -> None:
        """Simulate user management action."""
        step.estimated_duration = 0.3
        step.estimated_tokens = 100
        step.outcome = SimulationOutcome.SUCCESS
        step.reason = "User management action would execute"
        step.side_effects = ["User state change", "Access control update"]

    def _analyze_simulation(self, simulation: ForensicSimulation) -> None:
        """Analyze simulation results and generate recommendations."""
        # Check for safety issues
        if simulation.has_blocks():
            blocked_steps = [s for s in simulation.steps if s.blocked_by]
            simulation.overall_outcome = SimulationOutcome.BLOCKED
            simulation.recommendations.append(
                f"Workflow has {len(blocked_steps)} blocking step(s). "
                "Approval required before execution."
            )
        elif any(s.outcome == SimulationOutcome.UNSAFE for s in simulation.steps):
            simulation.overall_outcome = SimulationOutcome.UNSAFE
            unsafe_steps = [s for s in simulation.steps if s.outcome == SimulationOutcome.UNSAFE]
            simulation.recommendations.append(
                f"Detected {len(unsafe_steps)} unsafe operation(s). "
                "Review before execution."
            )
        elif any(s.outcome == SimulationOutcome.WOULD_FAIL for s in simulation.steps):
            simulation.overall_outcome = SimulationOutcome.WOULD_FAIL
            failed_steps = [s for s in simulation.steps if s.outcome == SimulationOutcome.WOULD_FAIL]
            simulation.recommendations.append(
                f"Simulation predicts {len(failed_steps)} failure(s). "
                "Verify conditions before execution."
            )
        else:
            simulation.overall_outcome = SimulationOutcome.SUCCESS
            simulation.recommendations.append(
                f"Workflow simulation successful. "
                f"Estimated duration: {simulation.total_estimated_duration:.1f}s, "
                f"Estimated tokens: {simulation.total_estimated_tokens}"
            )

        # Check for dangerous operations
        dangerous_ops = [
            step for step in simulation.steps
            if any(danger in step.action_path.lower()
                   for danger in ["delete", "remove", "destroy", "drop", "truncate"])
        ]
        if dangerous_ops:
            simulation.safety_issues.append(
                f"Detected {len(dangerous_ops)} potentially destructive operation(s)"
            )
            simulation.recommendations.append(
                "Destructive operations detected. Ensure backups exist before execution."
            )

    def compare_simulations(
        self,
        simulation_id_1: str,
        simulation_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two simulations to understand differences.

        Args:
            simulation_id_1: First simulation ID
            simulation_id_2: Second simulation ID

        Returns:
            Comparison report
        """
        sim1 = self.storage.get_simulation(simulation_id_1)
        sim2 = self.storage.get_simulation(simulation_id_2)

        if not sim1 or not sim2:
            return {"error": "Simulation not found"}

        comparison = {
            "simulation_1": simulation_id_1,
            "simulation_2": simulation_id_2,
            "workflow_name": sim1.get('workflow_name'),
            "differences": []
        }

        # Compare steps
        steps1 = sim1.get('steps', [])
        steps2 = sim2.get('steps', [])

        if len(steps1) != len(steps2):
            comparison["differences"].append(
                f"Step count differs: {len(steps1)} vs {len(steps2)}"
            )

        # Compare outcomes
        if sim1.get('overall_outcome') != sim2.get('overall_outcome'):
            comparison["differences"].append(
                f"Overall outcome differs: {sim1.get('overall_outcome')} vs {sim2.get('overall_outcome')}"
            )

        # Compare duration
        dur1 = sim1.get('total_estimated_duration', 0)
        dur2 = sim2.get('total_estimated_duration', 0)
        if abs(dur1 - dur2) > 0.1:
            comparison["differences"].append(
                f"Duration differs: {dur1:.1f}s vs {dur2:.1f}s"
            )

        # Compare tokens
        tok1 = sim1.get('total_estimated_tokens', 0)
        tok2 = sim2.get('total_estimated_tokens', 0)
        if tok1 != tok2:
            comparison["differences"].append(
                f"Token estimate differs: {tok1} vs {tok2}"
            )

        return comparison

    def get_simulation_report(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get a formatted report for a simulation."""
        simulation_data = self.storage.get_simulation(simulation_id)
        if not simulation_data:
            return None

        steps = simulation_data.get('steps', [])
        return {
            "simulation_id": simulation_id,
            "workflow_name": simulation_data.get('workflow_name'),
            "description": simulation_data.get('description'),
            "outcome": simulation_data.get('overall_outcome'),
            "step_count": len(steps),
            "success_steps": sum(1 for s in steps if s.get('outcome') == 'success'),
            "blocked_steps": sum(1 for s in steps if s.get('outcome') == 'blocked'),
            "failed_steps": sum(1 for s in steps if s.get('outcome') == 'would_fail'),
            "total_duration": simulation_data.get('total_estimated_duration'),
            "total_tokens": simulation_data.get('total_estimated_tokens'),
            "safety_issues": simulation_data.get('safety_issues', []),
            "recommendations": simulation_data.get('recommendations', []),
            "created_at": simulation_data.get('created_at'),
            "steps": steps[:5]  # Include first 5 steps in report
        }

    def get_forensic_statistics(self) -> Dict[str, Any]:
        """Get statistics about forensic simulations."""
        simulations = self.storage.get_simulations()
        recommendations = self.storage.get_recommendations()

        success_count = sum(1 for s in simulations if s.get('overall_outcome') == 'success')
        blocked_count = sum(1 for s in simulations if s.get('overall_outcome') == 'blocked')
        unsafe_count = sum(1 for s in simulations if s.get('overall_outcome') == 'unsafe')
        failed_count = sum(1 for s in simulations if s.get('overall_outcome') == 'would_fail')

        return {
            "total_simulations": len(simulations),
            "successful_outcomes": success_count,
            "blocked_outcomes": blocked_count,
            "unsafe_outcomes": unsafe_count,
            "failed_outcomes": failed_count,
            "success_rate": success_count / len(simulations) if simulations else 0.0,
            "total_recommendations": len(recommendations),
            "warning_count": sum(1 for r in recommendations if r.get('severity') == 'warn'),
            "error_count": sum(1 for r in recommendations if r.get('severity') == 'error')
        }
