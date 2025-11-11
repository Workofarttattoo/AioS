"""
OpenAGI Memory Integration with AIOS Kernel

Integrates WorkflowMemoryManager with AIOS kernel's memory system for persistent
workflow pattern storage, tool recommendation, and autonomous discovery learning.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import time

from aios.workflow_memory_manager import WorkflowMemoryManager

try:
    from aios.runtime import ExecutionContext, ActionResult
except Exception:
    # Fallback if runtime is encrypted
    ExecutionContext = Any
    ActionResult = Any

LOG = logging.getLogger(__name__)


class OpenAGIMemoryIntegration:
    """
    Bridges WorkflowMemoryManager with AIOS kernel memory subsystem.

    Responsibilities:
    1. Initialize memory manager with kernel context
    2. Persist workflow knowledge across boots
    3. Integrate with ExecutionContext for metadata publishing
    4. Support autonomous discovery tool learning
    5. Provide workflow recommendations to agents
    """

    def __init__(self, kernel=None, storage_path: Optional[Path] = None):
        """
        Initialize memory integration layer.

        Args:
            kernel: AIOS kernel reference (provides base memory manager)
            storage_path: Path for persistent workflow storage (default: ~/.aios/workflows)
        """
        self.kernel = kernel
        self.storage_path = storage_path or Path.home() / ".aios" / "workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize workflow memory manager
        self.memory = WorkflowMemoryManager(base_memory_manager=kernel.memory_manager if kernel else None)

        # Load persisted knowledge on startup
        self._load_persistent_knowledge()

        # Track learned concepts for autonomous discovery
        self.learned_concepts = {}
        self.concept_confidence_scores = {}

    def _load_persistent_knowledge(self) -> None:
        """Load previously learned workflows from disk."""
        knowledge_file = self.storage_path / "learned_knowledge.json"

        if knowledge_file.exists():
            try:
                with open(knowledge_file, "r") as f:
                    knowledge = json.load(f)
                    self.memory.import_knowledge(knowledge)
                    LOG.info(f"[info] Loaded {len(knowledge.get('workflows', {}))} learned workflows from disk")
            except Exception as e:
                LOG.warning(f"[warn] Failed to load persistent knowledge: {e}")

    def _save_persistent_knowledge(self) -> None:
        """Persist learned workflows to disk."""
        knowledge_file = self.storage_path / "learned_knowledge.json"

        try:
            knowledge = self.memory.export_knowledge()
            with open(knowledge_file, "w") as f:
                json.dump(knowledge, f, indent=2)
            LOG.info(f"[info] Persisted {len(knowledge.get('workflows', {}))} workflows to disk")
        except Exception as e:
            LOG.warning(f"[warn] Failed to persist knowledge: {e}")

    def record_workflow_execution(
        self,
        task: str,
        workflow: List[Dict],
        success: bool,
        latency: float,
        tokens_used: int = 0,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record workflow execution for learning.

        Args:
            task: Task description
            workflow: Generated workflow steps
            success: Whether workflow succeeded
            latency: Execution latency in seconds
            tokens_used: LLM tokens consumed
            metadata: Additional metadata for analytics
        """
        task_hash = self.memory.hash_task(task)

        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=success,
            latency=latency,
            tokens_used=tokens_used
        )

        # Track in kernel memory if available
        if self.kernel:
            self.kernel.memory_manager.publish_metadata(
                "openagi.workflow.execution",
                {
                    "task_hash": task_hash,
                    "success": success,
                    "latency": latency,
                    "tokens": tokens_used,
                    "timestamp": time.time(),
                    "metadata": metadata or {}
                }
            )

        LOG.info(f"[info] Recorded workflow execution: task={task_hash[:8]}, success={success}, latency={latency:.2f}s")

    def get_recommended_workflow(self, task: str) -> Optional[List[Dict]]:
        """
        Get recommended workflow for task based on history.

        Args:
            task: Task description

        Returns:
            Recommended workflow or None if no good match found
        """
        task_hash = self.memory.hash_task(task)
        recommended = self.memory.recommend_workflow(task_hash)

        if recommended:
            LOG.info(f"[info] Found cached workflow for task {task_hash[:8]}")

        return recommended

    def get_tool_recommendations(self, task: str) -> List[str]:
        """
        Get recommended tool combination for task.

        Args:
            task: Task description

        Returns:
            List of recommended tools
        """
        task_hash = self.memory.hash_task(task)

        # Get workflow from cache
        workflow = self.memory.recommend_workflow(task_hash)

        if not workflow:
            return []

        # Extract tools from workflow
        tools = set()
        for step in workflow:
            if isinstance(step, dict):
                step_tools = step.get("tool_use", [])
                if isinstance(step_tools, list):
                    tools.update(step_tools)

        return list(tools)

    def get_performance_report(self, ctx: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Args:
            ctx: ExecutionContext for publishing telemetry

        Returns:
            Performance metrics dictionary
        """
        report = self.memory.get_performance_report()

        if ctx:
            ctx.publish_metadata("openagi.memory.performance", report)

        return report

    def register_learned_concept(
        self,
        concept: str,
        category: str,
        confidence: float,
        source: str = "autonomous_discovery",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register concept learned via autonomous discovery.

        Args:
            concept: Concept name/description
            category: Category (e.g., 'security', 'performance', 'architecture')
            confidence: Confidence score 0.0-1.0
            source: Where concept was learned
            metadata: Additional context
        """
        concept_id = f"{category}:{concept}"
        self.learned_concepts[concept_id] = {
            "concept": concept,
            "category": category,
            "source": source,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.concept_confidence_scores[concept_id] = confidence

        LOG.info(f"[info] Registered learned concept: {concept_id} (confidence={confidence:.2f})")

    def get_high_confidence_concepts(self, category: Optional[str] = None, threshold: float = 0.8) -> List[Dict]:
        """
        Get concepts learned with high confidence.

        Args:
            category: Optional category filter
            threshold: Minimum confidence threshold

        Returns:
            List of high-confidence concepts
        """
        results = []

        for concept_id, data in self.learned_concepts.items():
            confidence = self.concept_confidence_scores.get(concept_id, 0.0)

            if confidence >= threshold:
                if category is None or data["category"] == category:
                    results.append({
                        **data,
                        "confidence": confidence
                    })

        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """
        Export complete knowledge graph for persistence.

        Returns:
            Knowledge graph with workflows and learned concepts
        """
        return {
            "workflows": self.memory.export_knowledge(),
            "learned_concepts": {
                concept_id: {
                    **data,
                    "confidence": self.concept_confidence_scores.get(concept_id, 0.0)
                }
                for concept_id, data in self.learned_concepts.items()
            },
            "metrics": self.memory.get_performance_report(),
            "exported_at": time.time()
        }

    def import_knowledge_graph(self, knowledge_graph: Dict[str, Any]) -> None:
        """
        Import previously exported knowledge graph.

        Args:
            knowledge_graph: Knowledge graph dictionary
        """
        # Import workflows
        if "workflows" in knowledge_graph:
            self.memory.import_knowledge(knowledge_graph["workflows"])

        # Import learned concepts
        if "learned_concepts" in knowledge_graph:
            for concept_id, data in knowledge_graph["learned_concepts"].items():
                confidence = data.pop("confidence", 0.8)
                self.learned_concepts[concept_id] = data
                self.concept_confidence_scores[concept_id] = confidence

        LOG.info(f"[info] Imported knowledge graph with {len(self.learned_concepts)} learned concepts")

    async def autonomy_ready(self, ctx: ExecutionContext) -> bool:
        """
        Check if system is ready for autonomous discovery learning.

        Args:
            ctx: ExecutionContext for checking environment

        Returns:
            True if ready for autonomous learning
        """
        # Check if autonomous discovery is enabled
        autonomy_enabled = ctx.environment.get("AGENTA_AUTONOMOUS_DISCOVERY", "").lower() in {"1", "true", "yes"}

        # Check if we have sufficient workflow history
        report = self.get_performance_report()
        has_history = report.get("total_workflows", 0) >= 5

        return autonomy_enabled and has_history


# Manifest integration - this action handler will be called from orchestration agent
async def initialize_openagi_memory(ctx: ExecutionContext) -> ActionResult:
    """
    Manifest action: Initialize OpenAGI memory integration.

    Called during boot sequence to set up workflow memory system.
    """
    try:
        # Create memory integration instance
        memory_integration = OpenAGIMemoryIntegration(kernel=ctx.kernel if hasattr(ctx, 'kernel') else None)

        # Publish to context for agent access
        ctx.publish_metadata("openagi.memory.integration", {
            "initialized": True,
            "workflows_loaded": memory_integration.memory.execution_metrics["total_workflows"],
            "concepts_learned": len(memory_integration.learned_concepts),
            "timestamp": time.time()
        })

        # Attach to context so agents can access it
        ctx.openagi_memory = memory_integration

        return ActionResult(
            success=True,
            message="[info] OpenAGI memory integration initialized",
            payload={
                "workflows_loaded": memory_integration.memory.execution_metrics["total_workflows"],
                "concepts_learned": len(memory_integration.learned_concepts)
            }
        )

    except Exception as e:
        LOG.exception(f"Failed to initialize OpenAGI memory: {e}")
        return ActionResult(
            success=False,
            message=f"[error] Memory initialization failed: {e}",
            payload={"exception": str(e)}
        )


async def persist_openagi_memory(ctx: ExecutionContext) -> ActionResult:
    """
    Manifest action: Persist OpenAGI memory on shutdown.

    Saves all learned workflows and concepts to disk.
    """
    try:
        # Get memory integration from context
        if not hasattr(ctx, 'openagi_memory'):
            return ActionResult(
                success=True,
                message="[info] OpenAGI memory not initialized, skipping persistence",
                payload={}
            )

        memory_integration = ctx.openagi_memory
        memory_integration._save_persistent_knowledge()

        # Export complete knowledge graph
        knowledge_graph = memory_integration.export_knowledge_graph()

        return ActionResult(
            success=True,
            message="[info] OpenAGI memory persisted to disk",
            payload={
                "workflows_saved": len(knowledge_graph.get("workflows", {})),
                "concepts_saved": len(knowledge_graph.get("learned_concepts", {}))
            }
        )

    except Exception as e:
        LOG.exception(f"Failed to persist OpenAGI memory: {e}")
        return ActionResult(
            success=False,
            message=f"[error] Memory persistence failed: {e}",
            payload={"exception": str(e)}
        )


async def report_openagi_memory_analytics(ctx: ExecutionContext) -> ActionResult:
    """
    Manifest action: Report memory analytics and learned insights.

    Called periodically to summarize learning progress.
    """
    try:
        if not hasattr(ctx, 'openagi_memory'):
            return ActionResult(
                success=True,
                message="[info] OpenAGI memory not initialized",
                payload={}
            )

        memory_integration = ctx.openagi_memory

        # Get performance metrics
        performance = memory_integration.get_performance_report()

        # Get high-confidence concepts
        high_confidence_concepts = memory_integration.get_high_confidence_concepts(threshold=0.85)

        # Get tool recommendations
        tool_stats = performance.get("tool_combinations", {})

        payload = {
            "total_workflows_learned": performance.get("total_workflows", 0),
            "success_rate": f"{performance.get('success_rate', 0.0):.1%}",
            "avg_latency": f"{performance.get('avg_latency', 0.0):.2f}s",
            "high_confidence_concepts": len(high_confidence_concepts),
            "top_tools": list(tool_stats.keys())[:5] if tool_stats else [],
        }

        ctx.publish_metadata("openagi.memory.analytics", payload)

        return ActionResult(
            success=True,
            message=f"[info] Memory analytics: {payload['total_workflows_learned']} workflows, {payload['success_rate']} success rate",
            payload=payload
        )

    except Exception as e:
        LOG.exception(f"Failed to report memory analytics: {e}")
        return ActionResult(
            success=False,
            message=f"[error] Analytics reporting failed: {e}",
            payload={"exception": str(e)}
        )
