"""
OrchestrationAgent - Policy Engine & Agent Coordination

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

LOG = logging.getLogger(__name__)


@dataclass
class PolicyRule:
    """Policy rule structure."""
    rule_id: str
    name: str
    condition: str
    action: str
    priority: int
    enabled: bool


class OrchestrationAgent:
    """
    Meta-agent for orchestration, policy coordination, and agent management.

    Responsibilities:
    - Policy engine and evaluation
    - Telemetry aggregation from sub-agents
    - Health monitoring and reporting
    - Agent coordination and sequencing
    - Incident response orchestration
    """

    def __init__(self):
        self.name = "orchestration"
        self.policies = {}
        self.policy_history = []
        self.agent_health = {}
        self.telemetry_buffer = defaultdict(list)
        self.orchestration_log = []
        LOG.info("OrchestrationAgent initialized")

    def register_policy(
        self,
        rule_id: str,
        name: str,
        condition: str,
        action: str,
        priority: int = 50,
    ) -> bool:
        """Register a policy rule."""
        try:
            policy = PolicyRule(
                rule_id=rule_id,
                name=name,
                condition=condition,
                action=action,
                priority=priority,
                enabled=True,
            )
            self.policies[rule_id] = policy
            LOG.info(f"Registered policy: {name} (priority={priority})")
            return True
        except Exception as e:
            LOG.error(f"Failed to register policy {rule_id}: {e}")
            return False

    def evaluate_policies(self, context: Dict) -> List[str]:
        """Evaluate all policies against current context, return triggered actions."""
        triggered_actions = []

        try:
            # Sort policies by priority (higher first)
            sorted_policies = sorted(
                self.policies.values(),
                key=lambda p: p.priority,
                reverse=True,
            )

            for policy in sorted_policies:
                if not policy.enabled:
                    continue

                # Simple condition evaluation
                try:
                    # Check if condition is satisfied
                    if self._evaluate_condition(policy.condition, context):
                        triggered_actions.append(policy.action)
                        LOG.info(f"Policy '{policy.name}' triggered action: {policy.action}")

                        # Record policy evaluation
                        self.policy_history.append({
                            "timestamp": time.time(),
                            "policy_id": policy.rule_id,
                            "condition": policy.condition,
                            "action": policy.action,
                            "triggered": True,
                        })
                except Exception as e:
                    LOG.warning(f"Error evaluating policy {policy.rule_id}: {e}")

        except Exception as e:
            LOG.error(f"Policy evaluation failed: {e}")

        return triggered_actions

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Simple condition evaluator (can be extended with DSL)."""
        try:
            # Support basic conditions like: "cpu > 80", "memory < 50", "uptime > 3600"
            if ">" in condition:
                key, threshold = condition.split(">")
                key = key.strip()
                threshold = float(threshold.strip())
                value = context.get(key, 0)
                return float(value) > threshold

            elif "<" in condition:
                key, threshold = condition.split("<")
                key = key.strip()
                threshold = float(threshold.strip())
                value = context.get(key, 0)
                return float(value) < threshold

            elif "==" in condition:
                key, expected = condition.split("==")
                key = key.strip()
                expected = expected.strip()
                value = str(context.get(key, "")).strip()
                return value == expected

            else:
                # Simple presence check
                return bool(context.get(condition.strip()))

        except Exception as e:
            LOG.warning(f"Condition evaluation error: {e}")
            return False

    def enable_policy(self, rule_id: str) -> bool:
        """Enable a policy rule."""
        if rule_id in self.policies:
            self.policies[rule_id].enabled = True
            LOG.info(f"Enabled policy: {rule_id}")
            return True
        return False

    def disable_policy(self, rule_id: str) -> bool:
        """Disable a policy rule."""
        if rule_id in self.policies:
            self.policies[rule_id].enabled = False
            LOG.info(f"Disabled policy: {rule_id}")
            return True
        return False

    def aggregate_telemetry(self, source: str, data: Dict) -> None:
        """Aggregate telemetry from sub-agents."""
        try:
            self.telemetry_buffer[source].append({
                "timestamp": time.time(),
                "data": data,
            })
            LOG.debug(f"Aggregated telemetry from {source}")
        except Exception as e:
            LOG.error(f"Telemetry aggregation failed: {e}")

    def get_aggregated_telemetry(self, source: Optional[str] = None, limit: int = 100) -> Dict:
        """Get aggregated telemetry."""
        if source:
            return {
                "source": source,
                "entries": self.telemetry_buffer[source][-limit:],
            }
        else:
            return {
                "total_sources": len(self.telemetry_buffer),
                "sources": list(self.telemetry_buffer.keys()),
                "entry_count": sum(len(v) for v in self.telemetry_buffer.values()),
            }

    def update_agent_health(self, agent_name: str, status: str, details: Dict) -> None:
        """Update health status of a sub-agent."""
        try:
            self.agent_health[agent_name] = {
                "status": status,  # "healthy", "degraded", "unhealthy"
                "details": details,
                "last_update": time.time(),
            }
            LOG.info(f"Agent health update: {agent_name} -> {status}")
        except Exception as e:
            LOG.error(f"Failed to update agent health: {e}")

    def get_agent_health(self, agent_name: Optional[str] = None) -> Dict:
        """Get agent health status."""
        if agent_name:
            return self.agent_health.get(
                agent_name,
                {"status": "unknown", "details": {}}
            )
        else:
            # Summary
            total_agents = len(self.agent_health)
            healthy = sum(
                1 for v in self.agent_health.values()
                if v["status"] == "healthy"
            )
            degraded = sum(
                1 for v in self.agent_health.values()
                if v["status"] == "degraded"
            )
            unhealthy = sum(
                1 for v in self.agent_health.values()
                if v["status"] == "unhealthy"
            )

            return {
                "total_agents": total_agents,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "agents": self.agent_health,
            }

    def coordinate_agents(self, agents: List[str], sequence: List[str]) -> Dict:
        """Coordinate execution sequence across multiple agents."""
        results = {
            "agents": agents,
            "sequence": sequence,
            "executed": [],
            "failed": [],
            "timestamp": time.time(),
        }

        try:
            for action in sequence:
                # Parse action format: "agent.action"
                if "." in action:
                    agent_name, action_name = action.split(".", 1)
                    if agent_name in agents:
                        try:
                            results["executed"].append(action)
                            LOG.info(f"Coordinated action: {action}")
                        except Exception as e:
                            results["failed"].append({"action": action, "error": str(e)})
                            LOG.error(f"Failed to execute coordinated action {action}: {e}")

            self.orchestration_log.append(results)

        except Exception as e:
            LOG.error(f"Agent coordination failed: {e}")
            results["error"] = str(e)

        return results

    def handle_incident(self, incident_name: str, severity: str, context: Dict) -> Dict:
        """Orchestrate incident response."""
        incident = {
            "name": incident_name,
            "severity": severity,  # "critical", "high", "medium", "low"
            "context": context,
            "timestamp": time.time(),
            "response_actions": [],
            "status": "initiated",
        }

        try:
            # Determine response based on severity
            if severity == "critical":
                incident["response_actions"] = [
                    "security.integrity_check",
                    "kernel.system_status",
                    "application.list_applications",
                ]
            elif severity == "high":
                incident["response_actions"] = [
                    "security.firewall_status",
                    "networking.check_connectivity",
                ]
            elif severity == "medium":
                incident["response_actions"] = [
                    "kernel.list_processes",
                ]
            else:
                incident["response_actions"] = [
                    "kernel.get_system_status",
                ]

            incident["status"] = "actions_prepared"
            LOG.info(f"Incident response prepared: {incident_name} ({severity})")

            self.orchestration_log.append(incident)

        except Exception as e:
            LOG.error(f"Incident handling failed: {e}")
            incident["status"] = "failed"
            incident["error"] = str(e)

        return incident

    def get_orchestration_summary(self) -> Dict:
        """Get summary of orchestration state."""
        return {
            "policies": {
                "total": len(self.policies),
                "enabled": sum(1 for p in self.policies.values() if p.enabled),
                "disabled": sum(1 for p in self.policies.values() if not p.enabled),
            },
            "agent_health": self.get_agent_health(),
            "telemetry_sources": len(self.telemetry_buffer),
            "policy_history_count": len(self.policy_history),
            "incident_count": sum(
                1 for log in self.orchestration_log
                if "severity" in log
            ),
            "last_update": time.time(),
        }

    def get_policy_history(self, limit: int = 50) -> List[Dict]:
        """Get recent policy evaluations."""
        return self.policy_history[-limit:]

    def get_orchestration_log(self, limit: int = 50) -> List[Dict]:
        """Get orchestration log."""
        return self.orchestration_log[-limit:]

    def export_policies(self) -> str:
        """Export policies as JSON."""
        try:
            policies_dict = {
                policy_id: asdict(policy)
                for policy_id, policy in self.policies.items()
            }
            return json.dumps(policies_dict, indent=2)
        except Exception as e:
            LOG.error(f"Failed to export policies: {e}")
            return "{}"

    def import_policies(self, policies_json: str) -> bool:
        """Import policies from JSON."""
        try:
            policies_dict = json.loads(policies_json)

            for policy_id, policy_data in policies_dict.items():
                self.register_policy(
                    rule_id=policy_data["rule_id"],
                    name=policy_data["name"],
                    condition=policy_data["condition"],
                    action=policy_data["action"],
                    priority=policy_data.get("priority", 50),
                )

            LOG.info(f"Imported {len(policies_dict)} policies")
            return True
        except Exception as e:
            LOG.error(f"Failed to import policies: {e}")
            return False

    def create_coordination_plan(
        self,
        goal: str,
        available_agents: List[str],
    ) -> Dict:
        """Create coordination plan to achieve a goal."""
        plan = {
            "goal": goal,
            "available_agents": available_agents,
            "planned_sequence": [],
            "estimated_duration": 0,
            "created_at": time.time(),
        }

        try:
            # Example: create simple coordination plans based on goal
            if "security" in goal.lower():
                plan["planned_sequence"] = [
                    "security.get_firewall_status",
                    "security.check_encryption_status",
                    "security.verify_system_integrity",
                ]
            elif "performance" in goal.lower():
                plan["planned_sequence"] = [
                    "kernel.get_system_status",
                    "scalability.get_current_load",
                    "application.list_applications",
                ]
            elif "health" in goal.lower():
                plan["planned_sequence"] = [
                    "kernel.get_system_status",
                    "networking.get_network_statistics",
                    "security.get_firewall_status",
                ]
            else:
                plan["planned_sequence"] = [
                    "kernel.get_system_status",
                ]

            # Estimate duration (rough: 1 second per action)
            plan["estimated_duration"] = len(plan["planned_sequence"])

            LOG.info(f"Created coordination plan for goal: {goal}")

        except Exception as e:
            LOG.error(f"Failed to create coordination plan: {e}")
            plan["error"] = str(e)

        return plan
