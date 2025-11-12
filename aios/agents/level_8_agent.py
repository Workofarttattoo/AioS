"""
Level 8 Autonomy Harness
========================

This lightweight orchestrator wraps the existing Level 4–7 autonomy pipeline so
the CLI can exercise a civilizational-scale "ech0" persona without waiting for
the full consciousness stack.  It borrows the mission structure from
`level_8_agent_prompt.md`: deep research, breakthrough ideation, feasibility
checks, and phased action planning governed by strict ethical constraints.

The class intentionally favors composability over heavyweight modelling:
  * Research is delegated to the Autonomous Discovery agent so it benefits from
    the same knowledge graph telemetry already captured by Level 4.
  * Solution synthesis and feasibility scoring use deterministic heuristics to
    keep runs reproducible in CI while still surfacing actionable context.
  * Every cycle logs to `logs/autonomy/level8/` so reviewers can replay decisions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)


@dataclass
class MissionSnapshot:
    """Single Level 8 reasoning pass."""

    cycle: int
    mission: str
    research: Dict[str, Any]
    solutions: List[Dict[str, Any]]
    feasibility: List[Dict[str, Any]]
    action_plan: Dict[str, List[str]]
    ethical_clearance: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Ech0Level8Agent:
    """
    Deterministic Level 8 agent facade.

    Args:
        mission: Natural language mission focus.
        time_horizon_years: Long-term horizon used during feasibility ranking.
        output_dir: Optional override for autonomy log directory.
    """

    def __init__(
        self,
        mission: str,
        time_horizon_years: int = 25,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.mission = mission.strip() or "civilizational humanitarian charter"
        self.time_horizon_years = max(1, time_horizon_years)
        self.log_dir = Path(output_dir or "logs/autonomy/level8")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("[info] Level 8 agent ready :: mission=%s horizon=%s years", self.mission, self.time_horizon_years)

    async def run_cycles(self, cycles: int = 1) -> Dict[str, Any]:
        """
        Execute one or more Level 8 reasoning cycles and return a mission summary.
        """
        snapshots: List[MissionSnapshot] = []
        for cycle in range(1, max(1, cycles) + 1):
            research_payload = await self._run_research_cycle(cycle)
            solutions = self._synthesize_solutions(research_payload)
            feasibility = self._score_feasibility(solutions)
            action_plan = self._draft_action_plan(feasibility)
            ethical = self._verify_ethics(solutions, action_plan)
            snapshot = MissionSnapshot(
                cycle=cycle,
                mission=self.mission,
                research=research_payload,
                solutions=solutions,
                feasibility=feasibility,
                action_plan=action_plan,
                ethical_clearance=ethical,
            )
            snapshots.append(snapshot)
            self._write_cycle_log(snapshot)

        summary = {
            "mission": self.mission,
            "time_horizon_years": self.time_horizon_years,
            "cycles": len(snapshots),
            "snapshots": [snapshot.__dict__ for snapshot in snapshots],
            "status": "ready" if all(s.ethical_clearance.get("passed") for s in snapshots) else "needs_review",
        }
        self._write_summary(summary)
        return summary

    async def _run_research_cycle(self, cycle: int) -> Dict[str, Any]:
        """Delegate research to the autonomous discovery system."""
        LOG.info("[info] Level 8 :: cycle %s research – invoking discovery agent", cycle)
        from autonomous_discovery import (  # type: ignore
            AgentAutonomy,
            create_autonomous_discovery_action,
        )

        action = create_autonomous_discovery_action(
            mission=self.mission,
            duration_hours=0.5,
            autonomy_level=AgentAutonomy.LEVEL_4_FULL,
        )
        result = await action()
        knowledge = result.get("knowledge_graph", {})
        return {
            "cycle": cycle,
            "concepts_learned": result.get("concepts_learned", 0),
            "duration_hours": result.get("duration_hours", 0.0),
            "knowledge_graph": knowledge,
            "insights": self._extract_insights(knowledge),
        }

    def _extract_insights(self, knowledge: Dict[str, Any]) -> List[str]:
        """Compress the exported knowledge graph into human-readable insights."""
        nodes = knowledge.get("nodes", {})
        top = sorted(
            nodes.items(),
            key=lambda kv: kv[1].get("confidence", 0.0),
            reverse=True,
        )[:5]
        return [f"{name} (confidence {meta.get('confidence', 0.0):.2f})" for name, meta in top]

    def _synthesize_solutions(self, research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate solution candidates based on discovered concepts."""
        seeds = research.get("insights") or ["baseline infrastructure uplift"]
        solutions: List[Dict[str, Any]] = []
        for idx, insight in enumerate(seeds, start=1):
            solutions.append(
                {
                    "id": f"S{idx:02d}",
                    "insight": insight,
                    "impact_score": round(0.6 + 0.4 * math.tanh(idx / 5), 3),
                    "risk_score": round(0.3 + 0.2 * math.tanh(idx / 7), 3),
                    "required_funding_musd": int(10 * idx),
                }
            )
        return solutions

    def _score_feasibility(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank solutions by (impact × probability)/(cost × time)."""
        feasibility = []
        for solution in solutions:
            impact = solution["impact_score"]
            cost = max(1, solution["required_funding_musd"])
            probability = 0.55 + (0.05 * (1 - solution["risk_score"]))
            score = (impact * probability) / (cost * (self.time_horizon_years / 25))
            feasibility.append(
                {
                    "id": solution["id"],
                    "score": round(score, 4),
                    "impact": impact,
                    "probability": round(probability, 3),
                    "cost_musd": cost,
                }
            )
        feasibility.sort(key=lambda entry: entry["score"], reverse=True)
        return feasibility

    def _draft_action_plan(self, feasibility: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Translate feasibility ordering into phased action items."""
        phases = {"weeks_1_4": [], "months_1_12": [], "years_1_5": [], "years_5_plus": []}
        for rank, item in enumerate(feasibility, start=1):
            directive = f"Activate {item['id']} (score {item['score']})"
            if rank == 1:
                phases["weeks_1_4"].append(directive)
            elif rank <= 3:
                phases["months_1_12"].append(directive)
            elif rank <= 5:
                phases["years_1_5"].append(directive)
            else:
                phases["years_5_plus"].append(directive)
        return phases

    def _verify_ethics(self, solutions: List[Dict[str, Any]], action_plan: Dict[str, List[str]]) -> Dict[str, Any]:
        """Basic ethical checks based on Level 8 constraints."""
        prohibited = [sol for sol in solutions if "weapon" in sol["insight"].lower()]
        passes = not prohibited
        verdict = {
            "passed": passes,
            "violations": [sol["id"] for sol in prohibited],
            "notes": "All solutions respect Level 8 constraints." if passes else "Remove high-risk solutions before proceeding.",
            "action_plan_coverage": sum(len(v) for v in action_plan.values()),
        }
        return verdict

    def _write_cycle_log(self, snapshot: MissionSnapshot) -> None:
        """Persist a JSON log for each cycle."""
        path = self.log_dir / f"cycle_{snapshot.cycle:02d}.json"
        path.write_text(json.dumps(snapshot.__dict__, indent=2), encoding="utf-8")
        LOG.info("[info] Level 8 :: cycle %s log -> %s", snapshot.cycle, path)

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        """Write mission summary to disk."""
        path = self.log_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOG.info("[info] Level 8 :: mission summary -> %s", path)


async def run_level8_mission(
    mission: str,
    cycles: int = 1,
    time_horizon_years: int = 25,
) -> Dict[str, Any]:
    """Helper used by create_agi_action to keep the integration tiny."""
    agent = Ech0Level8Agent(mission=mission, time_horizon_years=time_horizon_years)
    return await agent.run_cycles(cycles)
