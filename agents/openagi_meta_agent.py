"""
OpenAGI Meta-Agent for AIOS Kernel

Orchestrates ReAct workflow generation and execution within the AIOS kernel.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
import logging

from ..runtime import ExecutionContext, ActionResult
from ..openagi_kernel_bridge import OpenAGIKernelBridge, ToolExecutionMode
from ..workflow_memory_manager import WorkflowMemoryManager


logger = logging.getLogger(__name__)


class OpenAGIMetaAgent:
    """
    Meta-agent for ReAct workflow orchestration in AIOS.

    Responsibilities:
    1. Generate structured workflows from task descriptions
    2. Execute workflows using AIOS tools and LLM core
    3. Learn from execution results (autonomous discovery)
    4. Recommend workflows for similar tasks
    5. Provide metrics and observability
    """

    def __init__(
        self,
        kernel: Any,
        enable_learning: bool = True,
        enable_caching: bool = True,
        enable_parallelization: bool = True
    ):
        """
        Initialize OpenAGI meta-agent

        Args:
            kernel: AIOS kernel instance
            enable_learning: Enable autonomous learning from executions
            enable_caching: Enable workflow caching
            enable_parallelization: Enable parallel tool execution
        """
        self.kernel = kernel
        self.enable_learning = enable_learning
        self.enable_caching = enable_caching
        self.enable_parallelization = enable_parallelization

        # Initialize bridge components
        self.bridge = OpenAGIKernelBridge(
            llm_core=kernel.llm_core,
            context_manager=kernel.context_manager,
            memory_manager=kernel.memory_manager,
            tool_manager=kernel.tool_manager,
            logger=logger
        )

        self.memory = WorkflowMemoryManager()

        # Set execution mode
        if enable_parallelization:
            self.bridge.execution_mode = ToolExecutionMode.HYBRID
        else:
            self.bridge.execution_mode = ToolExecutionMode.SEQUENTIAL

        logger.info(
            f"OpenAGI Meta-Agent initialized "
            f"(learning={enable_learning}, caching={enable_caching}, "
            f"parallel={enable_parallelization})"
        )

    async def execute_react_workflow(self, ctx: ExecutionContext) -> ActionResult:
        """
        Execute ReAct workflow for a task.

        Args:
            ctx: AIOS ExecutionContext with task input

        Returns:
            ActionResult with workflow results
        """
        task_input = ctx.environment.get("OPENAGI_TASK_INPUT")
        if not task_input:
            return ActionResult(
                success=False,
                message="[error] OPENAGI_TASK_INPUT not set in environment",
                payload={}
            )

        task_hash = self.memory.hash_task(task_input)
        start_time = time.time()

        try:
            # Step 1: Try to get cached workflow
            workflow = None
            if self.enable_caching:
                workflow = self.memory.recommend_workflow(task_hash)
                if workflow:
                    logger.info(f"Using cached workflow for task hash {task_hash}")
                    ctx.publish_metadata("workflow.cache_hit", {"task_hash": task_hash})

            # Step 2: Generate workflow if not cached
            if workflow is None:
                logger.info(f"Generating new workflow for task: {task_input[:100]}...")
                workflow = await self.bridge.generate_workflow(
                    task=task_input,
                    ctx=ctx,
                    temperature=0.3  # Deterministic
                )

                if workflow is None:
                    return ActionResult(
                        success=False,
                        message="[error] Failed to generate workflow",
                        payload={}
                    )

                logger.info(f"Generated workflow with {len(workflow)} steps")

            # Step 3: Get approval if required
            approval_required = ctx.environment.get("OPENAGI_APPROVAL_REQUIRED", "").lower() in {
                "1", "true", "yes", "on"
            }

            if approval_required:
                approval_callback = ctx.environment.get("OPENAGI_APPROVAL_CALLBACK")
                if approval_callback:
                    logger.info("Workflow requires approval")
                else:
                    logger.warn("Approval required but no callback provided, skipping")

            # Step 4: Execute workflow
            logger.info(f"Executing workflow with {len(workflow)} steps")
            result = await self.bridge.execute_workflow(
                task=task_input,
                ctx=ctx,
                workflow=workflow,
                execution_mode=self.bridge.execution_mode
            )

            # Step 5: Learn from execution
            if self.enable_learning and result.success:
                execution_time = time.time() - start_time
                self.memory.add_workflow_execution(
                    task_hash=task_hash,
                    workflow=[step.to_dict() for step in workflow],
                    success=result.success,
                    latency=execution_time,
                    tokens_used=result.payload.get("tokens_used", 0)
                )
                logger.info(f"Recorded workflow execution (latency: {execution_time:.2f}s)")

            # Step 6: Publish metrics
            ctx.publish_metadata("openagi.workflow_execution", {
                "task_hash": task_hash,
                "steps": len(workflow),
                "success": result.success,
                "latency": time.time() - start_time,
                "learning_enabled": self.enable_learning
            })

            return result

        except Exception as e:
            logger.exception(f"Error executing workflow: {e}")
            return ActionResult(
                success=False,
                message=f"[error] Workflow execution failed: {e}",
                payload={"error": str(e), "task_hash": task_hash}
            )

    async def recommend_workflow(self, ctx: ExecutionContext) -> ActionResult:
        """
        Recommend workflow for a task based on learned patterns.

        Args:
            ctx: AIOS ExecutionContext with task input

        Returns:
            ActionResult with recommended workflow or None
        """
        task_input = ctx.environment.get("OPENAGI_TASK_INPUT")
        if not task_input:
            return ActionResult(
                success=False,
                message="[error] OPENAGI_TASK_INPUT not set",
                payload={}
            )

        task_hash = self.memory.hash_task(task_input)
        recommended = self.memory.recommend_workflow(task_hash)

        if recommended:
            logger.info(f"Found recommended workflow for task hash {task_hash}")
            return ActionResult(
                success=True,
                message="[info] Recommended workflow found",
                payload={
                    "task_hash": task_hash,
                    "workflow": recommended,
                    "success_rate": self.memory.get_workflow_success_rate(task_hash)
                }
            )
        else:
            return ActionResult(
                success=False,
                message="[info] No recommended workflow found",
                payload={"task_hash": task_hash}
            )

    async def analyze_workflow_performance(self, ctx: ExecutionContext) -> ActionResult:
        """
        Analyze workflow performance and provide recommendations.

        Args:
            ctx: AIOS ExecutionContext

        Returns:
            ActionResult with performance analysis
        """
        report = self.memory.get_performance_report()

        ctx.publish_metadata("openagi.performance_report", report)

        return ActionResult(
            success=True,
            message="[info] Performance analysis complete",
            payload={
                "total_workflows": report.get("total_workflows", 0),
                "success_rate": report.get("success_rate", 0),
                "avg_latency": report.get("avg_latency", 0),
                "avg_tokens": report.get("avg_tokens_per_workflow", 0),
                "cached_workflows": report.get("cached_workflows", 0),
                "top_tool_combinations": report.get("top_tool_combinations", [])
            }
        )

    async def execute_parallel_workflows(
        self,
        ctx: ExecutionContext,
        tasks: List[str]
    ) -> ActionResult:
        """
        Execute multiple workflows in parallel.

        Args:
            ctx: AIOS ExecutionContext
            tasks: List of task descriptions

        Returns:
            ActionResult with aggregated results
        """
        logger.info(f"Executing {len(tasks)} workflows in parallel")

        # Create task contexts for each workflow
        task_contexts = []
        for i, task in enumerate(tasks):
            task_ctx = ExecutionContext(
                parent_context=ctx,
                environment={
                    "OPENAGI_TASK_INPUT": task,
                    "OPENAGI_TASK_INDEX": str(i)
                }
            )
            task_contexts.append(task_ctx)

        # Execute in parallel
        try:
            results = await asyncio.gather(*[
                self.execute_react_workflow(task_ctx)
                for task_ctx in task_contexts
            ])

            # Aggregate results
            successful = sum(1 for r in results if r.success)
            total_latency = sum(
                r.payload.get("total_latency", 0) for r in results if r.success
            )

            ctx.publish_metadata("openagi.parallel_execution", {
                "total_tasks": len(tasks),
                "successful": successful,
                "failed": len(tasks) - successful,
                "avg_latency": total_latency / len(tasks) if tasks else 0
            })

            return ActionResult(
                success=successful == len(tasks),
                message=f"[info] Executed {successful}/{len(tasks)} workflows successfully",
                payload={
                    "results": [r.payload for r in results],
                    "successful": successful,
                    "total_tasks": len(tasks)
                }
            )

        except Exception as e:
            logger.exception(f"Error in parallel execution: {e}")
            return ActionResult(
                success=False,
                message=f"[error] Parallel execution failed: {e}",
                payload={"error": str(e)}
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get meta-agent statistics.

        Returns:
            Dict with performance and usage statistics
        """
        bridge_stats = self.bridge.get_execution_stats()
        memory_stats = self.memory.get_performance_report()

        return {
            "bridge": bridge_stats,
            "memory": memory_stats,
            "learning_enabled": self.enable_learning,
            "caching_enabled": self.enable_caching,
            "parallelization_enabled": self.enable_parallelization
        }

    async def export_learned_knowledge(self, filepath: str) -> ActionResult:
        """
        Export learned workflow patterns to file.

        Args:
            filepath: Path to save knowledge export

        Returns:
            ActionResult with export status
        """
        try:
            knowledge = self.memory.export_knowledge()
            with open(filepath, "w") as f:
                json.dump(knowledge, f, indent=2, default=str)

            logger.info(f"Exported learned knowledge to {filepath}")
            return ActionResult(
                success=True,
                message=f"[info] Knowledge exported to {filepath}",
                payload={"filepath": filepath, "size": len(json.dumps(knowledge))}
            )

        except Exception as e:
            logger.exception(f"Error exporting knowledge: {e}")
            return ActionResult(
                success=False,
                message=f"[error] Failed to export knowledge: {e}",
                payload={"error": str(e)}
            )

    async def import_learned_knowledge(self, filepath: str) -> ActionResult:
        """
        Import learned workflow patterns from file.

        Args:
            filepath: Path to load knowledge from

        Returns:
            ActionResult with import status
        """
        try:
            with open(filepath, "r") as f:
                knowledge = json.load(f)

            self.memory.import_knowledge(knowledge)

            logger.info(f"Imported learned knowledge from {filepath}")
            return ActionResult(
                success=True,
                message=f"[info] Knowledge imported from {filepath}",
                payload={"filepath": filepath}
            )

        except Exception as e:
            logger.exception(f"Error importing knowledge: {e}")
            return ActionResult(
                success=False,
                message=f"[error] Failed to import knowledge: {e}",
                payload={"error": str(e)}
            )


# Module-level initialization helper
def create_openagi_meta_agent(kernel, **kwargs) -> OpenAGIMetaAgent:
    """
    Factory function to create and initialize OpenAGI meta-agent.

    Args:
        kernel: AIOS kernel instance
        **kwargs: Additional arguments passed to OpenAGIMetaAgent

    Returns:
        Initialized OpenAGIMetaAgent instance
    """
    return OpenAGIMetaAgent(kernel, **kwargs)
