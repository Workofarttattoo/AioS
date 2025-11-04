"""
Workflow-Aware Memory Manager for AIOS

Extends AIOS memory system to learn and optimize ReAct workflow patterns.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib


@dataclass
class WorkflowPattern:
    """Learned workflow pattern"""
    task_hash: str
    workflow: List[Dict]
    success_rate: float
    avg_latency: float
    total_executions: int
    first_seen: float
    last_seen: float
    preferred: bool = False


@dataclass
class ToolCombination:
    """Tool combination statistics"""
    tools: Tuple[str]
    total_uses: int = 0
    successful_uses: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    success_rate: float = 0.0


class WorkflowMemoryManager:
    """
    Manages workflow patterns and tool combination statistics for learning.

    Responsibilities:
    1. Store successful workflows indexed by task similarity
    2. Track tool combination effectiveness
    3. Recommend workflows for new tasks
    4. Learn tool chain patterns
    5. Provide analytics on workflow performance
    """

    def __init__(self, base_memory_manager=None):
        """
        Initialize workflow memory manager

        Args:
            base_memory_manager: Optional AIOS memory manager to wrap
        """
        self.base_memory_manager = base_memory_manager

        # Workflow storage
        self.workflow_library = defaultdict(list)  # task_hash → [WorkflowPattern]
        self.successful_workflows = {}  # workflow_id → WorkflowPattern
        self.failed_workflows = defaultdict(list)  # task_hash → [workflow]

        # Tool combination tracking
        self.tool_combinations = {}  # combo_hash → ToolCombination
        self.tool_pair_stats = defaultdict(lambda: {"success": 0, "total": 0})

        # Performance metrics
        self.execution_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tokens": 0,
            "total_latency": 0.0
        }

        # Learning threshold
        self.min_success_rate = 0.7  # 70% success rate threshold

    def hash_task(self, task: str) -> str:
        """Generate hash for task"""
        return hashlib.md5(task.encode()).hexdigest()

    def hash_workflow(self, workflow: List[Dict]) -> str:
        """Generate hash for workflow"""
        workflow_json = json.dumps(workflow, sort_keys=True)
        return hashlib.md5(workflow_json.encode()).hexdigest()

    def hash_tool_combination(self, tools: List[str]) -> str:
        """Generate hash for tool combination"""
        tools_sorted = tuple(sorted(tools))
        return hashlib.md5(str(tools_sorted).encode()).hexdigest()

    def add_workflow_execution(
        self,
        task_hash: str,
        workflow: List[Dict],
        success: bool,
        latency: float,
        tokens_used: int = 0
    ):
        """
        Record workflow execution for learning

        Args:
            task_hash: Hash of task
            workflow: Workflow steps
            success: Whether execution succeeded
            latency: Total execution latency
            tokens_used: Tokens consumed
        """
        workflow_id = self.hash_workflow(workflow)

        # Create workflow pattern
        pattern = WorkflowPattern(
            task_hash=task_hash,
            workflow=workflow,
            success_rate=1.0 if success else 0.0,
            avg_latency=latency,
            total_executions=1,
            first_seen=time.time(),
            last_seen=time.time()
        )

        # Update existing pattern or create new one
        if workflow_id in self.successful_workflows:
            existing = self.successful_workflows[workflow_id]
            # Update success rate
            new_total = existing.total_executions + 1
            new_successes = (existing.success_rate * existing.total_executions) + (1 if success else 0)
            existing.success_rate = new_successes / new_total
            existing.total_executions = new_total
            existing.last_seen = time.time()
            existing.avg_latency = (existing.avg_latency * (new_total - 1) + latency) / new_total
        else:
            self.successful_workflows[workflow_id] = pattern
            self.workflow_library[task_hash].append(pattern)

        # Track tool combinations
        tools = self._extract_tools(workflow)
        self._update_tool_stats(tools, success, latency)

        # Update metrics
        self.execution_metrics["total_workflows"] += 1
        self.execution_metrics["total_tokens"] += tokens_used
        self.execution_metrics["total_latency"] += latency

        if success:
            self.execution_metrics["successful_workflows"] += 1
        else:
            self.execution_metrics["failed_workflows"] += 1
            self.failed_workflows[task_hash].append(workflow)

    def _extract_tools(self, workflow: List[Dict]) -> List[str]:
        """Extract all unique tools from workflow"""
        tools = set()
        for step in workflow:
            tools.update(step.get("tool_use", []))
        return sorted(list(tools))

    def _update_tool_stats(self, tools: List[str], success: bool, latency: float):
        """Update tool combination statistics"""
        combo_hash = self.hash_tool_combination(tools)
        tools_tuple = tuple(sorted(tools))

        if combo_hash not in self.tool_combinations:
            self.tool_combinations[combo_hash] = ToolCombination(tools=tools_tuple)

        combo = self.tool_combinations[combo_hash]
        combo.total_uses += 1
        combo.total_latency += latency
        combo.avg_latency = combo.total_latency / combo.total_uses

        if success:
            combo.successful_uses += 1

        combo.success_rate = combo.successful_uses / combo.total_uses

    def recommend_workflow(
        self,
        task_hash: str,
        similarity_threshold: float = 0.85
    ) -> Optional[List[Dict]]:
        """
        Recommend best workflow for task

        Args:
            task_hash: Task hash
            similarity_threshold: Minimum similarity for recommendation

        Returns:
            Best workflow for task, or None
        """
        if task_hash not in self.workflow_library:
            return None

        workflows = self.workflow_library[task_hash]

        # Filter by success rate
        candidates = [
            w for w in workflows
            if w.success_rate >= self.min_success_rate
        ]

        if not candidates:
            return None

        # Sort by success rate, then by latency
        best = sorted(
            candidates,
            key=lambda x: (-x.success_rate, x.avg_latency)
        )[0]

        return best.workflow

    def get_preferred_tool_combinations(
        self,
        min_success_rate: float = 0.8,
        min_uses: int = 5
    ) -> List[ToolCombination]:
        """
        Get tool combinations with high success rates

        Args:
            min_success_rate: Minimum success rate (0-1)
            min_uses: Minimum number of uses

        Returns:
            List of high-performing tool combinations
        """
        candidates = [
            combo for combo in self.tool_combinations.values()
            if combo.success_rate >= min_success_rate and combo.total_uses >= min_uses
        ]

        return sorted(
            candidates,
            key=lambda x: -x.success_rate
        )

    def get_tool_pair_recommendations(self) -> Dict[str, float]:
        """
        Get recommended tool pairs based on success

        Returns:
            Dict mapping "tool1+tool2" → success_rate
        """
        recommendations = {}

        for combo, stats in self.tool_pair_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                if success_rate >= 0.8:
                    recommendations[combo] = success_rate

        return sorted(
            recommendations.items(),
            key=lambda x: -x[1]
        )

    def predict_best_tools_for_task(
        self,
        task_description: str
    ) -> Optional[List[str]]:
        """
        Predict best tool combination for task type

        Uses heuristic matching on task keywords to suggest tools.

        Args:
            task_description: Description of task

        Returns:
            List of recommended tools, or None
        """
        # Keyword → tools mapping
        keyword_map = {
            "search": ["google_search", "bing_search", "wikipedia"],
            "weather": ["weather_tool", "meteosource"],
            "math": ["wolfram_alpha"],
            "translate": ["translation_tool"],
            "image": ["image_generator", "image_analyzer"],
            "code": ["code_executor"],
            "data": ["data_processor", "csv_analyzer"]
        }

        # Find matching tools
        task_lower = task_description.lower()
        suggested_tools = set()

        for keyword, tools in keyword_map.items():
            if keyword in task_lower:
                suggested_tools.update(tools)

        if suggested_tools:
            # Return as sorted list
            return sorted(list(suggested_tools))

        return None

    def get_workflow_success_rate(self, task_hash: str) -> float:
        """Get success rate for task"""
        if task_hash not in self.workflow_library:
            return 0.0

        workflows = self.workflow_library[task_hash]
        if not workflows:
            return 0.0

        avg_success = sum(w.success_rate for w in workflows) / len(workflows)
        return avg_success

    def get_performance_report(self) -> Dict:
        """
        Get comprehensive performance report

        Returns:
            Dict with execution stats, tool recommendations, etc.
        """
        metrics = self.execution_metrics.copy()

        # Success rate
        if metrics["total_workflows"] > 0:
            metrics["success_rate"] = metrics["successful_workflows"] / metrics["total_workflows"]
            metrics["avg_latency"] = metrics["total_latency"] / metrics["total_workflows"]
        else:
            metrics["success_rate"] = 0.0
            metrics["avg_latency"] = 0.0

        # Token efficiency
        metrics["avg_tokens_per_workflow"] = (
            metrics["total_tokens"] / metrics["total_workflows"]
            if metrics["total_workflows"] > 0
            else 0
        )

        # Top tool combinations
        top_combos = self.get_preferred_tool_combinations()
        metrics["top_tool_combinations"] = [
            {
                "tools": list(combo.tools),
                "success_rate": combo.success_rate,
                "avg_latency": combo.avg_latency,
                "uses": combo.total_uses
            }
            for combo in top_combos[:5]
        ]

        # Workflow library size
        metrics["cached_workflows"] = len(self.successful_workflows)
        metrics["cached_tasks"] = len(self.workflow_library)

        return metrics

    def export_knowledge(self) -> Dict:
        """
        Export learned knowledge as JSON

        Can be persisted and loaded later for knowledge transfer.

        Returns:
            Dict containing all learned patterns
        """
        return {
            "workflows": {
                task_hash: [asdict(w) for w in workflows]
                for task_hash, workflows in self.workflow_library.items()
            },
            "tool_combinations": {
                combo_hash: asdict(combo)
                for combo_hash, combo in self.tool_combinations.items()
            },
            "metrics": self.execution_metrics.copy(),
            "timestamp": time.time()
        }

    def import_knowledge(self, knowledge_dict: Dict):
        """
        Import learned knowledge from export

        Args:
            knowledge_dict: Dict from export_knowledge()
        """
        # Import workflows
        for task_hash, workflow_list in knowledge_dict.get("workflows", {}).items():
            for workflow_data in workflow_list:
                pattern = WorkflowPattern(
                    task_hash=workflow_data["task_hash"],
                    workflow=workflow_data["workflow"],
                    success_rate=workflow_data["success_rate"],
                    avg_latency=workflow_data["avg_latency"],
                    total_executions=workflow_data["total_executions"],
                    first_seen=workflow_data["first_seen"],
                    last_seen=workflow_data["last_seen"],
                    preferred=workflow_data.get("preferred", False)
                )
                self.workflow_library[task_hash].append(pattern)

        # Import tool combinations
        for combo_hash, combo_data in knowledge_dict.get("tool_combinations", {}).items():
            combo = ToolCombination(
                tools=tuple(combo_data["tools"]),
                total_uses=combo_data["total_uses"],
                successful_uses=combo_data["successful_uses"],
                total_latency=combo_data["total_latency"],
                avg_latency=combo_data["avg_latency"],
                success_rate=combo_data["success_rate"]
            )
            self.tool_combinations[combo_hash] = combo

        # Import metrics
        for key, value in knowledge_dict.get("metrics", {}).items():
            if key in self.execution_metrics:
                self.execution_metrics[key] = value

    def clear_old_patterns(self, max_age_days: int = 30):
        """
        Remove workflow patterns older than max_age_days

        Args:
            max_age_days: Maximum age in days
        """
        max_age_seconds = max_age_days * 86400
        current_time = time.time()

        # Remove old workflows
        for task_hash in list(self.workflow_library.keys()):
            workflows = self.workflow_library[task_hash]
            # Keep only recent workflows
            self.workflow_library[task_hash] = [
                w for w in workflows
                if current_time - w.last_seen < max_age_seconds
            ]

    def get_workflow_diagnostics(self, task_hash: str) -> Dict:
        """
        Get diagnostic info for a specific task

        Args:
            task_hash: Task hash

        Returns:
            Diagnostics including success rate, common patterns, errors
        """
        workflows = self.workflow_library.get(task_hash, [])
        failed = self.failed_workflows.get(task_hash, [])

        return {
            "task_hash": task_hash,
            "total_attempts": len(workflows) + len(failed),
            "successful_workflows": len(workflows),
            "failed_workflows": len(failed),
            "success_rate": self.get_workflow_success_rate(task_hash),
            "best_workflow": workflows[0].workflow if workflows else None,
            "common_tool_patterns": self._find_common_patterns(workflows)
        }

    def _find_common_patterns(self, workflows: List[WorkflowPattern]) -> List[Dict]:
        """Find common patterns in successful workflows"""
        if not workflows:
            return []

        # Extract tool sequences
        sequences = [self._extract_tool_sequence(w.workflow) for w in workflows]

        # Count occurrences
        from collections import Counter
        seq_counts = Counter(sequences)

        return [
            {"sequence": seq, "count": count}
            for seq, count in seq_counts.most_common(5)
        ]

    def _extract_tool_sequence(self, workflow: List[Dict]) -> str:
        """Extract tool sequence from workflow"""
        sequence = []
        for step in workflow:
            tools = step.get("tool_use", [])
            if tools:
                sequence.append("+".join(sorted(tools)))
        return " → ".join(sequence) if sequence else "reasoning_only"
