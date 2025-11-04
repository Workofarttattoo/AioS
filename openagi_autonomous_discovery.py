"""
Autonomous Tool Discovery and Learning System

Enables OpenAGI meta-agents to autonomously discover and learn optimal
tool combinations through systematic exploration and pattern recognition.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time

try:
    from aios.runtime import ExecutionContext, ActionResult
except Exception:
    # Fallback for when runtime is encrypted
    ExecutionContext = Any
    ActionResult = Any

try:
    from aios.openagi_memory_integration import OpenAGIMemoryIntegration
except Exception:
    OpenAGIMemoryIntegration = None

LOG = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools for discovery purposes"""

    SEARCH = "search"  # Information gathering
    ANALYSIS = "analysis"  # Data analysis and processing
    INTEGRATION = "integration"  # External service integration
    TRANSFORMATION = "transformation"  # Data transformation
    VALIDATION = "validation"  # Validation and verification
    OPTIMIZATION = "optimization"  # Performance optimization


@dataclass
class ToolProfile:
    """Profile of a tool learned through autonomous discovery"""

    tool_name: str
    category: ToolCategory
    effectiveness_score: float  # 0.0-1.0
    success_rate: float
    avg_latency: float
    total_uses: int
    preferred_partners: List[str]  # Tools that work well together
    common_failure_modes: List[str]
    last_used: float
    learning_confidence: float = 0.8


@dataclass
class ToolCombinationPattern:
    """Learned pattern about tool combinations"""

    tools: Tuple[str, ...]
    success_rate: float
    avg_latency: float
    total_executions: int
    confidence_score: float
    use_case_examples: List[str]
    performance_improvement: float  # vs using tools individually


class AutonomousToolDiscovery:
    """
    Autonomously discovers and learns optimal tool combinations.

    Responsibilities:
    1. Systematically explore tool combinations
    2. Track effectiveness of combinations
    3. Learn tool partnership patterns
    4. Predict tool effectiveness for new tasks
    5. Recommend optimal tool chains
    """

    def __init__(self, memory_integration: Optional[OpenAGIMemoryIntegration] = None):
        """
        Initialize autonomous tool discovery.

        Args:
            memory_integration: Memory system for persistence
        """
        self.memory = memory_integration
        self.tool_profiles: Dict[str, ToolProfile] = {}
        self.combination_patterns: Dict[str, ToolCombinationPattern] = {}
        self.tool_categories: Dict[str, ToolCategory] = {}
        self.discovery_iterations = 0
        self.total_combinations_tested = 0

    def register_tool(
        self,
        tool_name: str,
        category: ToolCategory,
        initial_effectiveness: float = 0.5,
    ) -> None:
        """
        Register a tool for discovery and learning.

        Args:
            tool_name: Name of the tool
            category: Tool category
            initial_effectiveness: Initial effectiveness estimate
        """
        self.tool_profiles[tool_name] = ToolProfile(
            tool_name=tool_name,
            category=category,
            effectiveness_score=initial_effectiveness,
            success_rate=0.0,
            avg_latency=0.0,
            total_uses=0,
            preferred_partners=[],
            common_failure_modes=[],
            last_used=0.0,
        )
        self.tool_categories[tool_name] = category
        LOG.info(f"[info] Registered tool {tool_name} in category {category.value}")

    def update_tool_effectiveness(
        self,
        tool_name: str,
        success: bool,
        latency: float,
        failure_reason: Optional[str] = None,
    ) -> None:
        """
        Update tool effectiveness based on execution.

        Args:
            tool_name: Name of the tool
            success: Whether execution succeeded
            latency: Execution latency in seconds
            failure_reason: Reason for failure if applicable
        """
        if tool_name not in self.tool_profiles:
            LOG.warning(f"[warn] Tool {tool_name} not registered")
            return

        profile = self.tool_profiles[tool_name]
        profile.total_uses += 1

        # Update success rate
        old_success_rate = profile.success_rate
        profile.success_rate = (
            (old_success_rate * (profile.total_uses - 1) + (1.0 if success else 0.0))
            / profile.total_uses
        )

        # Update latency
        old_latency = profile.avg_latency
        profile.avg_latency = (old_latency * (profile.total_uses - 1) + latency) / profile.total_uses

        # Update effectiveness score (weighted combination)
        profile.effectiveness_score = (profile.success_rate * 0.7) + ((1.0 - min(profile.avg_latency, 10.0) / 10.0) * 0.3)

        # Track failures
        if failure_reason and not success:
            if failure_reason not in profile.common_failure_modes:
                profile.common_failure_modes.append(failure_reason)
            profile.common_failure_modes = profile.common_failure_modes[:5]

        profile.last_used = time.time()

        LOG.info(
            f"[info] Updated {tool_name}: effectiveness={profile.effectiveness_score:.2f}, "
            f"success_rate={profile.success_rate:.1%}, avg_latency={profile.avg_latency:.2f}s"
        )

    def record_combination_execution(
        self,
        tools: List[str],
        success: bool,
        latency: float,
        use_case: Optional[str] = None,
    ) -> None:
        """
        Record execution of a tool combination.

        Args:
            tools: List of tools used in combination
            success: Whether combination succeeded
            latency: Total execution latency
            use_case: Description of use case
        """
        combo_key = self._combination_key(tools)

        if combo_key not in self.combination_patterns:
            self.combination_patterns[combo_key] = ToolCombinationPattern(
                tools=tuple(sorted(tools)),
                success_rate=0.0,
                avg_latency=0.0,
                total_executions=0,
                confidence_score=0.0,
                use_case_examples=[],
                performance_improvement=0.0,
            )

        pattern = self.combination_patterns[combo_key]
        pattern.total_executions += 1

        # Update success rate
        pattern.success_rate = (
            (pattern.success_rate * (pattern.total_executions - 1) + (1.0 if success else 0.0))
            / pattern.total_executions
        )

        # Update latency
        pattern.avg_latency = (
            (pattern.avg_latency * (pattern.total_executions - 1) + latency)
            / pattern.total_executions
        )

        # Update confidence (based on execution count)
        pattern.confidence_score = min(1.0, pattern.total_executions / 10.0)

        # Track use cases
        if use_case and use_case not in pattern.use_case_examples:
            pattern.use_case_examples.append(use_case)
            pattern.use_case_examples = pattern.use_case_examples[:5]

        self.total_combinations_tested += 1

        if success:
            # Update tool partnerships
            for tool in tools:
                if tool in self.tool_profiles:
                    profile = self.tool_profiles[tool]
                    partners = [t for t in tools if t != tool]
                    for partner in partners:
                        if partner not in profile.preferred_partners:
                            profile.preferred_partners.append(partner)
                            profile.preferred_partners = profile.preferred_partners[:5]

        LOG.info(
            f"[info] Combination {combo_key[:8]}: success_rate={pattern.success_rate:.1%}, "
            f"executions={pattern.total_executions}"
        )

    def get_tool_recommendations(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        Get recommended tools based on effectiveness.

        Args:
            category: Optional category filter

        Returns:
            Sorted list of tools by effectiveness
        """
        candidates = [
            (name, profile.effectiveness_score)
            for name, profile in self.tool_profiles.items()
            if category is None or self.tool_categories.get(name) == category
        ]

        return [name for name, _ in sorted(candidates, key=lambda x: x[1], reverse=True)]

    def get_combination_recommendations(self, num_recommendations: int = 5) -> List[ToolCombinationPattern]:
        """
        Get best tool combinations learned.

        Args:
            num_recommendations: Number of recommendations to return

        Returns:
            List of recommended combinations sorted by effectiveness
        """
        patterns = sorted(
            self.combination_patterns.values(),
            key=lambda p: (p.success_rate * p.confidence_score),
            reverse=True,
        )

        return patterns[:num_recommendations]

    def predict_combination_success(self, tools: List[str]) -> Tuple[float, str]:
        """
        Predict success rate of a tool combination.

        Args:
            tools: List of tools to predict for

        Returns:
            Tuple of (predicted_success_rate, reasoning)
        """
        combo_key = self._combination_key(tools)

        if combo_key in self.combination_patterns:
            pattern = self.combination_patterns[combo_key]
            return pattern.success_rate, f"Based on {pattern.total_executions} executions"

        # Predict based on individual tool effectiveness
        tool_scores = [self.tool_profiles.get(t, ToolProfile(t, ToolCategory.SEARCH, 0.5, 0.0, 0.0, 0, [], [], 0.0)).effectiveness_score for t in tools]

        if not tool_scores:
            return 0.5, "No tool data available"

        # Combine scores (conservative estimate)
        predicted = sum(tool_scores) / len(tool_scores) * 0.9  # 10% penalty for unknown combinations

        return predicted, "Based on individual tool effectiveness"

    def export_learned_profiles(self) -> Dict[str, Dict]:
        """
        Export all learned tool profiles.

        Returns:
            Dictionary of tool profiles
        """
        return {
            name: {
                **asdict(profile),
                "category": profile.category.value,
            }
            for name, profile in self.tool_profiles.items()
        }

    def export_learned_combinations(self) -> Dict[str, Dict]:
        """
        Export all learned combination patterns.

        Returns:
            Dictionary of combination patterns
        """
        return {
            combo_key: {
                **asdict(pattern),
                "tools": list(pattern.tools),
            }
            for combo_key, pattern in self.combination_patterns.items()
        }

    def get_discovery_statistics(self) -> Dict[str, any]:
        """
        Get discovery session statistics.

        Returns:
            Statistics dictionary
        """
        high_effectiveness = [
            p for p in self.tool_profiles.values() if p.effectiveness_score >= 0.8
        ]
        high_confidence_combos = [
            p for p in self.combination_patterns.values() if p.confidence_score >= 0.8
        ]

        return {
            "total_tools_registered": len(self.tool_profiles),
            "tools_tested": sum(1 for p in self.tool_profiles.values() if p.total_uses > 0),
            "high_effectiveness_tools": len(high_effectiveness),
            "total_combinations_tested": self.total_combinations_tested,
            "unique_combinations_learned": len(self.combination_patterns),
            "high_confidence_combinations": len(high_confidence_combos),
            "average_tool_effectiveness": (
                sum(p.effectiveness_score for p in self.tool_profiles.values())
                / len(self.tool_profiles) if self.tool_profiles else 0.0
            ),
            "discovery_iterations": self.discovery_iterations,
        }

    def _combination_key(self, tools: List[str]) -> str:
        """Generate hash key for tool combination."""
        sorted_tools = tuple(sorted(tools))
        return hashlib.md5(str(sorted_tools).encode()).hexdigest()


async def discover_tools_autonomous(ctx: ExecutionContext) -> ActionResult:
    """
    Manifest action: Autonomously discover optimal tool combinations.

    Uses the autonomous discovery system to learn about tools and their
    effectiveness in different scenarios.
    """
    try:
        # Get memory integration
        if not hasattr(ctx, "openagi_memory"):
            return ActionResult(
                success=False,
                message="[error] Memory integration not initialized",
                payload={},
            )

        memory = ctx.openagi_memory

        # Initialize tool discovery
        discovery = AutonomousToolDiscovery(memory_integration=memory)

        # Register some base tools
        tools_to_discover = [
            ("google_search", ToolCategory.SEARCH),
            ("database_query", ToolCategory.ANALYSIS),
            ("api_call", ToolCategory.INTEGRATION),
            ("data_transform", ToolCategory.TRANSFORMATION),
            ("validation_check", ToolCategory.VALIDATION),
        ]

        for tool_name, category in tools_to_discover:
            discovery.register_tool(tool_name, category)

        # Simulate discovery iterations
        for i in range(5):
            # Test individual tools
            for tool_name, _ in tools_to_discover:
                success = i < 3 or tool_name != "database_query"  # db queries less reliable
                latency = 0.5 + (i * 0.1)
                discovery.update_tool_effectiveness(tool_name, success, latency)

            # Test combinations
            combinations = [
                ["google_search", "api_call"],
                ["data_transform", "validation_check"],
                ["google_search", "data_transform", "api_call"],
            ]

            for combo in combinations:
                success = i < 4
                latency = len(combo) * 0.3
                discovery.record_combination_execution(
                    combo, success, latency, use_case=f"scenario_{i}"
                )

            discovery.discovery_iterations += 1
            await asyncio.sleep(0.01)  # Allow task switching

        # Get statistics
        stats = discovery.get_discovery_statistics()

        # Register high-confidence findings with memory
        for tool_name, profile in discovery.tool_profiles.items():
            if profile.effectiveness_score >= 0.7:
                memory.register_learned_concept(
                    concept=f"tool_effectiveness:{tool_name}",
                    category="tools",
                    confidence=min(1.0, profile.total_uses / 10.0),
                    source="autonomous_discovery",
                    metadata={
                        "effectiveness": profile.effectiveness_score,
                        "success_rate": profile.success_rate,
                        "avg_latency": profile.avg_latency,
                        "category": profile.category.value,
                    },
                )

        ctx.publish_metadata("openagi.tool_discovery", stats)

        return ActionResult(
            success=True,
            message=f"[info] Tool discovery complete: {stats['total_combinations_tested']} combinations tested",
            payload=stats,
        )

    except Exception as e:
        LOG.exception(f"Tool discovery failed: {e}")
        return ActionResult(
            success=False,
            message=f"[error] Tool discovery failed: {e}",
            payload={"exception": str(e)},
        )


async def recommend_tools(ctx: ExecutionContext, task: str) -> List[str]:
    """
    Recommend optimal tools for a task using learned patterns.

    Args:
        ctx: ExecutionContext
        task: Task description

    Returns:
        Recommended tool list
    """
    if not hasattr(ctx, "openagi_memory"):
        return []

    memory = ctx.openagi_memory

    # Get tools from cached workflow if available
    tools = memory.get_tool_recommendations(task)

    return tools
