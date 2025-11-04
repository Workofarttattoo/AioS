"""
OpenAGI Kernel Bridge - Integration layer between OpenAGI and AIOS Kernel

This module provides the bridge between OpenAGI's ReAct agent pattern and AIOS's
kernel-based architecture, enabling structured workflow generation and execution
within the AIOS ecosystem.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import time
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .runtime import ExecutionContext, ActionResult


class WorkflowStep(object):
    """Represents a single step in a ReAct workflow"""

    def __init__(self, message: str, tool_use: List[str]):
        self.message = message
        self.tool_use = tool_use

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "tool_use": self.tool_use
        }

    @staticmethod
    def from_dict(data: dict):
        return WorkflowStep(
            message=data.get("message", ""),
            tool_use=data.get("tool_use", [])
        )


@dataclass
class WorkflowExecution:
    """Tracks workflow execution details"""
    workflow_id: str
    task_input: str
    steps: List[WorkflowStep]
    start_time: float
    end_time: Optional[float] = None
    step_results: List[Dict] = None
    success: bool = False
    error: Optional[str] = None
    total_tokens: int = 0
    total_latency: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ToolExecutionMode(Enum):
    """Tool execution mode enumeration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"  # Parallel within step, sequential across steps


class OpenAGIKernelBridge:
    """
    Bridges OpenAGI's ReAct workflow generation with AIOS Kernel.

    This bridge:
    1. Generates structured JSON workflows from task descriptions
    2. Executes workflows using AIOS tools and LLM core
    3. Tracks execution metrics for learning and optimization
    4. Caches successful workflows for similar tasks
    5. Integrates with AIOS memory system for pattern learning
    """

    def __init__(
        self,
        llm_core: Any,
        context_manager: Any,
        memory_manager: Any,
        tool_manager: Any,
        logger: Optional[Any] = None
    ):
        """
        Initialize the OpenAGI Kernel Bridge

        Args:
            llm_core: AIOS LLM Core instance
            context_manager: AIOS Context Manager
            memory_manager: AIOS Memory Manager
            tool_manager: AIOS Tool Manager
            logger: Optional logger instance
        """
        self.llm_core = llm_core
        self.context_manager = context_manager
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager
        self.logger = logger

        self.workflow_cache = {}  # task_hash → workflow
        self.execution_history = []
        self.tool_combo_stats = {}  # tool_combo → stats
        self.execution_mode = ToolExecutionMode.HYBRID

    def hash_task(self, task: str) -> str:
        """Generate hash for task caching"""
        import hashlib
        return hashlib.md5(task.encode()).hexdigest()

    def _log(self, message: str, level: str = "info"):
        """Log message"""
        if self.logger:
            self.logger.log(f"[OpenAGI Bridge] {message}", level=level)
        else:
            print(f"[{level.upper()}] {message}")

    async def generate_workflow(
        self,
        task: str,
        ctx: ExecutionContext,
        available_tools: Optional[List[str]] = None,
        max_retries: int = 3,
        temperature: float = 0.3
    ) -> Optional[List[WorkflowStep]]:
        """
        Generate a structured ReAct workflow for the given task.

        Args:
            task: Task description
            ctx: AIOS ExecutionContext
            available_tools: List of available tool names
            max_retries: Max retries for invalid JSON
            temperature: LLM temperature (lower = more deterministic)

        Returns:
            List of WorkflowStep objects, or None if generation fails
        """
        # Check cache
        task_hash = self.hash_task(task)
        if task_hash in self.workflow_cache:
            self._log(f"Using cached workflow for task hash {task_hash}")
            return self.workflow_cache[task_hash]

        # Get available tools from tool_manager if not provided
        if available_tools is None:
            available_tools = self.tool_manager.list_available_tools()

        # Build system instruction
        system_instruction = self._build_workflow_generation_prompt(available_tools)

        # Retry logic for invalid JSON
        for attempt in range(max_retries):
            try:
                self._log(f"Generating workflow (attempt {attempt + 1}/{max_retries})")

                # Call LLM to generate workflow
                response = await self.llm_core.call_async(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": task}
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    timeout=30
                )

                # Parse JSON response
                workflow_json = json.loads(response.text)

                # Validate workflow structure
                if not isinstance(workflow_json, list):
                    self._log(f"Invalid workflow format (attempt {attempt + 1})", level="warn")
                    continue

                # Convert to WorkflowStep objects
                steps = []
                for step_data in workflow_json:
                    if "message" not in step_data or "tool_use" not in step_data:
                        self._log(f"Invalid step format (attempt {attempt + 1})", level="warn")
                        continue
                    steps.append(WorkflowStep.from_dict(step_data))

                if steps:
                    # Cache the workflow
                    self.workflow_cache[task_hash] = steps
                    self._log(f"Generated workflow with {len(steps)} steps")

                    # Track tokens used
                    ctx.publish_metadata("workflow.generation", {
                        "task_hash": task_hash,
                        "steps": len(steps),
                        "tokens_used": getattr(response, 'usage', {}).get('total_tokens', 0)
                    })

                    return steps

            except json.JSONDecodeError:
                self._log(f"JSON decode error (attempt {attempt + 1})", level="warn")
                continue
            except Exception as e:
                self._log(f"Error generating workflow: {e}", level="error")
                if attempt == max_retries - 1:
                    return None

        self._log("Failed to generate valid workflow", level="error")
        return None

    def _build_workflow_generation_prompt(self, available_tools: List[str]) -> str:
        """Build system prompt for workflow generation"""
        tools_desc = "\n".join([f"- {tool}" for tool in available_tools])

        return f"""You are a workflow generation system for AI agents. Your task is to generate a
step-by-step execution plan for complex tasks.

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
1. Analyze the user's task and break it into logical steps
2. For each step, decide which tools to use (empty list for reasoning-only steps)
3. Order steps to respect dependencies
4. Pre-select only the most relevant tools for each step

OUTPUT FORMAT:
Return a JSON array of steps with this exact structure:
[
  {{"message": "Step description", "tool_use": ["tool_name1", "tool_name2"]}},
  ...
]

GUIDELINES:
- Each step should be atomic and independently executable
- Minimize the number of tools per step (prefer 1-2)
- Use empty tool_use array for reasoning, analysis, or synthesis steps
- Maximize tool parallelization where possible
- Prefer familiar tools over rare ones"""

    async def execute_workflow(
        self,
        task: str,
        ctx: ExecutionContext,
        workflow: Optional[List[WorkflowStep]] = None,
        approval_callback: Optional[Callable] = None,
        max_retries: int = 3,
        execution_mode: Optional[ToolExecutionMode] = None
    ) -> ActionResult:
        """
        Execute a workflow, either provided or generated.

        Args:
            task: Task description
            ctx: AIOS ExecutionContext
            workflow: Pre-generated workflow steps (generated if None)
            approval_callback: Optional callback for workflow approval
            max_retries: Max retries for failed tool calls
            execution_mode: Tool execution mode (uses default if None)

        Returns:
            ActionResult with workflow results
        """
        start_time = time.time()

        try:
            # Generate workflow if not provided
            if workflow is None:
                workflow = await self.generate_workflow(task, ctx)
                if workflow is None:
                    return ActionResult(
                        success=False,
                        message="[error] Failed to generate workflow",
                        payload={}
                    )

            # Request approval if callback provided
            if approval_callback:
                workflow_dict = [step.to_dict() for step in workflow]
                approved = await approval_callback(workflow_dict)
                if not approved:
                    return ActionResult(
                        success=False,
                        message="[info] Workflow approval denied",
                        payload={"workflow": workflow_dict}
                    )

            # Execute workflow
            execution = WorkflowExecution(
                workflow_id=self.hash_task(task),
                task_input=task,
                steps=workflow,
                start_time=start_time,
                step_results=[]
            )

            # Use provided execution mode or default
            mode = execution_mode or self.execution_mode

            # Execute each step
            for step_idx, step in enumerate(workflow):
                step_result = await self._execute_step(
                    step=step,
                    step_idx=step_idx,
                    ctx=ctx,
                    execution_mode=mode,
                    max_retries=max_retries
                )

                execution.step_results.append(step_result)

                if not step_result.get("success", False) and step.tool_use:
                    self._log(f"Step {step_idx} failed, continuing", level="warn")

            # Mark execution as complete
            execution.end_time = time.time()
            execution.total_latency = execution.end_time - execution.start_time
            execution.success = all(
                step.get("success", True) for step in execution.step_results
            )

            # Record execution for learning
            self.execution_history.append(execution)
            await self._record_workflow_execution(execution, ctx)

            # Publish execution metrics
            ctx.publish_metadata("workflow.execution", {
                "task_hash": execution.workflow_id,
                "steps": len(workflow),
                "success": execution.success,
                "total_latency": execution.total_latency,
                "tools_used": self._extract_tools_from_workflow(workflow)
            })

            return ActionResult(
                success=execution.success,
                message=f"[info] Workflow completed in {len(workflow)} steps",
                payload={
                    "workflow_id": execution.workflow_id,
                    "steps_completed": len(execution.step_results),
                    "total_steps": len(workflow),
                    "total_latency": execution.total_latency,
                    "step_results": execution.step_results
                }
            )

        except Exception as e:
            self._log(f"Error executing workflow: {e}", level="error")
            return ActionResult(
                success=False,
                message=f"[error] Workflow execution failed: {e}",
                payload={"error": str(e)}
            )

    async def _execute_step(
        self,
        step: WorkflowStep,
        step_idx: int,
        ctx: ExecutionContext,
        execution_mode: ToolExecutionMode,
        max_retries: int
    ) -> Dict:
        """Execute a single workflow step"""
        step_start = time.time()

        try:
            # If no tools, just call LLM for reasoning
            if not step.tool_use:
                response = await self.llm_core.call_async(
                    messages=ctx.messages + [
                        {"role": "user", "content": f"Step {step_idx}: {step.message}"}
                    ],
                    temperature=0.0
                )

                return {
                    "step": step_idx,
                    "message": step.message,
                    "success": True,
                    "tools_used": [],
                    "response": response.text,
                    "latency": time.time() - step_start
                }

            # Execute tools
            tool_results = {}

            if execution_mode == ToolExecutionMode.PARALLEL:
                # Execute all tools in parallel
                tasks = [
                    self._execute_tool(tool_name, ctx)
                    for tool_name in step.tool_use
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_name, result in zip(step.tool_use, results):
                    tool_results[tool_name] = result

            else:  # SEQUENTIAL or HYBRID
                # Execute tools sequentially
                for tool_name in step.tool_use:
                    result = await self._execute_tool(tool_name, ctx)
                    tool_results[tool_name] = result

            # Call LLM with tool results
            tool_messages = []
            for tool_name, result in tool_results.items():
                tool_messages.append({
                    "role": "user",
                    "content": f"{tool_name} output: {result}"
                })

            response = await self.llm_core.call_async(
                messages=ctx.messages + tool_messages + [
                    {"role": "user", "content": f"Step {step_idx}: {step.message}"}
                ],
                temperature=0.0
            )

            return {
                "step": step_idx,
                "message": step.message,
                "success": True,
                "tools_used": step.tool_use,
                "tool_results": tool_results,
                "response": response.text,
                "latency": time.time() - step_start
            }

        except Exception as e:
            return {
                "step": step_idx,
                "message": step.message,
                "success": False,
                "error": str(e),
                "latency": time.time() - step_start
            }

    async def _execute_tool(self, tool_name: str, ctx: ExecutionContext) -> Any:
        """Execute a single tool"""
        tool = self.tool_manager.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Get tool parameters from context
        params = ctx.environment.get(f"tool.{tool_name}.params", {})

        # Execute tool
        result = await tool.run_async(params) if hasattr(tool, 'run_async') else tool.run(params)

        # Record tool usage
        if tool_name not in self.tool_combo_stats:
            self.tool_combo_stats[tool_name] = {
                "executions": 0,
                "successes": 0,
                "total_latency": 0.0
            }

        self.tool_combo_stats[tool_name]["executions"] += 1
        self.tool_combo_stats[tool_name]["successes"] += 1

        return result

    async def _record_workflow_execution(
        self,
        execution: WorkflowExecution,
        ctx: ExecutionContext
    ):
        """Record workflow execution in memory for learning"""
        # Store in memory manager
        self.memory_manager.add_workflow_execution(
            task_hash=execution.workflow_id,
            workflow=execution.steps,
            success=execution.success,
            latency=execution.total_latency
        )

    def _extract_tools_from_workflow(self, workflow: List[WorkflowStep]) -> List[str]:
        """Extract all unique tools from workflow"""
        tools = set()
        for step in workflow:
            tools.update(step.tool_use)
        return sorted(list(tools))

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {}

        total_executions = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e.success)
        total_latency = sum(e.total_latency for e in self.execution_history)

        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "avg_latency": total_latency / total_executions if total_executions > 0 else 0,
            "total_latency": total_latency,
            "tool_stats": self.tool_combo_stats,
            "cache_size": len(self.workflow_cache)
        }
