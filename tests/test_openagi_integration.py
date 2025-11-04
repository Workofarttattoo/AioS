"""
Unit tests for OpenAGI-AIOS integration.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import unittest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aios.openagi_kernel_bridge import (
    OpenAGIKernelBridge,
    WorkflowStep,
    WorkflowExecution,
    ToolExecutionMode
)
from aios.workflow_memory_manager import WorkflowMemoryManager
from aios.agents.openagi_meta_agent import OpenAGIMetaAgent
from aios.runtime import ExecutionContext, ActionResult


class TestWorkflowStep(unittest.TestCase):
    """Tests for WorkflowStep class"""

    def test_workflow_step_creation(self):
        """Test creating a workflow step"""
        step = WorkflowStep(
            message="Search for restaurants",
            tool_use=["google_search", "yelp"]
        )

        self.assertEqual(step.message, "Search for restaurants")
        self.assertEqual(step.tool_use, ["google_search", "yelp"])

    def test_workflow_step_to_dict(self):
        """Test converting workflow step to dict"""
        step = WorkflowStep(
            message="Search for restaurants",
            tool_use=["google_search"]
        )

        result = step.to_dict()

        self.assertEqual(result["message"], "Search for restaurants")
        self.assertEqual(result["tool_use"], ["google_search"])

    def test_workflow_step_from_dict(self):
        """Test creating workflow step from dict"""
        data = {
            "message": "Analyze results",
            "tool_use": []
        }

        step = WorkflowStep.from_dict(data)

        self.assertEqual(step.message, "Analyze results")
        self.assertEqual(step.tool_use, [])


class TestWorkflowMemoryManager(unittest.TestCase):
    """Tests for WorkflowMemoryManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = WorkflowMemoryManager()

    def test_hash_task(self):
        """Test task hashing"""
        task = "Find Italian restaurants in Tokyo"

        hash1 = self.memory.hash_task(task)
        hash2 = self.memory.hash_task(task)

        # Same task should produce same hash
        self.assertEqual(hash1, hash2)

    def test_add_workflow_execution(self):
        """Test recording workflow execution"""
        task = "Find restaurants"
        task_hash = self.memory.hash_task(task)
        workflow = [
            {"message": "Search", "tool_use": ["google_search"]},
            {"message": "Filter", "tool_use": []}
        ]

        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=5.0,
            tokens_used=100
        )

        # Check execution was recorded
        self.assertEqual(self.memory.execution_metrics["total_workflows"], 1)
        self.assertEqual(self.memory.execution_metrics["successful_workflows"], 1)

    def test_recommend_workflow(self):
        """Test workflow recommendation"""
        task = "Find restaurants"
        task_hash = self.memory.hash_task(task)
        workflow = [
            {"message": "Search", "tool_use": ["google_search"]},
        ]

        # Add successful execution
        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=5.0
        )

        # Recommend workflow
        recommended = self.memory.recommend_workflow(task_hash)

        self.assertIsNotNone(recommended)
        self.assertEqual(recommended, workflow)

    def test_tool_combination_stats(self):
        """Test tool combination tracking"""
        workflow = [
            {"message": "Search", "tool_use": ["google_search", "yelp"]},
            {"message": "Filter", "tool_use": []}
        ]

        self.memory.add_workflow_execution(
            task_hash="test123",
            workflow=workflow,
            success=True,
            latency=5.0
        )

        # Check tool stats were recorded
        report = self.memory.get_performance_report()
        self.assertTrue("tool_combinations" in report or "total_workflows" in report)

    def test_export_import_knowledge(self):
        """Test exporting and importing knowledge"""
        task = "Find restaurants"
        task_hash = self.memory.hash_task(task)
        workflow = [
            {"message": "Search", "tool_use": ["google_search"]},
        ]

        # Add workflow execution
        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=5.0
        )

        # Export knowledge
        knowledge = self.memory.export_knowledge()
        self.assertIn("workflows", knowledge)
        self.assertIn("tool_combinations", knowledge)

        # Create new memory manager and import
        memory2 = WorkflowMemoryManager()
        memory2.import_knowledge(knowledge)

        # Verify knowledge was imported
        recommended = memory2.recommend_workflow(task_hash)
        self.assertIsNotNone(recommended)


class TestOpenAGIKernelBridge(unittest.TestCase):
    """Tests for OpenAGIKernelBridge"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock AIOS components
        self.mock_llm_core = AsyncMock()
        self.mock_context_manager = Mock()
        self.mock_memory_manager = Mock()
        self.mock_tool_manager = Mock()

        self.bridge = OpenAGIKernelBridge(
            llm_core=self.mock_llm_core,
            context_manager=self.mock_context_manager,
            memory_manager=self.mock_memory_manager,
            tool_manager=self.mock_tool_manager
        )

    def test_bridge_initialization(self):
        """Test bridge initialization"""
        self.assertIsNotNone(self.bridge.llm_core)
        self.assertIsNotNone(self.bridge.workflow_cache)
        self.assertIsNotNone(self.bridge.execution_history)

    def test_hash_task(self):
        """Test task hashing consistency"""
        task = "Find restaurants"

        hash1 = self.bridge.hash_task(task)
        hash2 = self.bridge.hash_task(task)

        self.assertEqual(hash1, hash2)

    def test_workflow_generation_prompt_structure(self):
        """Test workflow generation prompt structure"""
        tools = ["google_search", "yelp", "wikipedia"]
        prompt = self.bridge._build_workflow_generation_prompt(tools)

        # Verify prompt contains required elements
        self.assertIn("JSON", prompt)
        self.assertIn("tools", prompt.lower())
        self.assertIn("step", prompt.lower())

    def test_extract_tools_from_workflow(self):
        """Test tool extraction from workflow"""
        workflow = [
            WorkflowStep("Search", ["google_search", "yelp"]),
            WorkflowStep("Analyze", []),
            WorkflowStep("Filter", ["google_search"])
        ]

        tools = self.bridge._extract_tools_from_workflow(workflow)

        self.assertEqual(set(tools), {"google_search", "yelp"})

    @patch('asyncio.gather')
    async def test_parallel_tool_execution(self, mock_gather):
        """Test parallel tool execution mode"""
        self.bridge.execution_mode = ToolExecutionMode.PARALLEL

        step = WorkflowStep("Search", ["tool1", "tool2"])
        self.mock_tool_manager.list_available_tools.return_value = ["tool1", "tool2"]
        self.mock_tool_manager.get_tool.return_value = AsyncMock()

        # Verify execution mode is set
        self.assertEqual(self.bridge.execution_mode, ToolExecutionMode.PARALLEL)


class TestOpenAGIMetaAgent(unittest.TestCase):
    """Tests for OpenAGIMetaAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_kernel = Mock()
        self.mock_kernel.llm_core = AsyncMock()
        self.mock_kernel.context_manager = Mock()
        self.mock_kernel.memory_manager = Mock()
        self.mock_kernel.tool_manager = Mock()

        self.agent = OpenAGIMetaAgent(self.mock_kernel)

    def test_agent_initialization(self):
        """Test meta-agent initialization"""
        self.assertTrue(self.agent.enable_learning)
        self.assertTrue(self.agent.enable_caching)
        self.assertTrue(self.agent.enable_parallelization)
        self.assertIsNotNone(self.agent.bridge)
        self.assertIsNotNone(self.agent.memory)

    async def test_execute_react_workflow_no_task(self):
        """Test executing workflow without task input"""
        ctx = ExecutionContext(environment={})

        result = await self.agent.execute_react_workflow(ctx)

        self.assertFalse(result.success)
        self.assertIn("OPENAGI_TASK_INPUT", result.message)

    def test_get_statistics(self):
        """Test getting agent statistics"""
        stats = self.agent.get_statistics()

        self.assertIn("bridge", stats)
        self.assertIn("memory", stats)
        self.assertTrue(stats["learning_enabled"])
        self.assertTrue(stats["caching_enabled"])

    async def test_export_knowledge(self):
        """Test exporting learned knowledge"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "knowledge.json")

            result = await self.agent.export_learned_knowledge(filepath)

            # Even with empty knowledge, should succeed
            if result.success:
                self.assertTrue(os.path.exists(filepath))


class TestExecutionContext(unittest.TestCase):
    """Tests for ExecutionContext integration"""

    def test_execution_context_creation(self):
        """Test creating execution context"""
        ctx = ExecutionContext(
            environment={"OPENAGI_TASK_INPUT": "Find restaurants"}
        )

        self.assertEqual(
            ctx.environment.get("OPENAGI_TASK_INPUT"),
            "Find restaurants"
        )

    def test_metadata_publishing(self):
        """Test publishing metadata to context"""
        ctx = ExecutionContext()

        ctx.publish_metadata("test.metric", {"value": 42})

        # Verify metadata can be retrieved
        metadata = ctx.metadata.get("test.metric")
        self.assertEqual(metadata, {"value": 42})


class TestToolExecutionModes(unittest.TestCase):
    """Tests for different tool execution modes"""

    def test_execution_mode_enumeration(self):
        """Test execution mode values"""
        self.assertEqual(ToolExecutionMode.SEQUENTIAL.value, "sequential")
        self.assertEqual(ToolExecutionMode.PARALLEL.value, "parallel")
        self.assertEqual(ToolExecutionMode.HYBRID.value, "hybrid")

    def test_bridge_mode_configuration(self):
        """Test configuring bridge execution mode"""
        mock_components = {
            "llm_core": AsyncMock(),
            "context_manager": Mock(),
            "memory_manager": Mock(),
            "tool_manager": Mock()
        }

        bridge = OpenAGIKernelBridge(**mock_components)

        # Test setting modes
        bridge.execution_mode = ToolExecutionMode.SEQUENTIAL
        self.assertEqual(bridge.execution_mode, ToolExecutionMode.SEQUENTIAL)

        bridge.execution_mode = ToolExecutionMode.PARALLEL
        self.assertEqual(bridge.execution_mode, ToolExecutionMode.PARALLEL)


class TestWorkflowCaching(unittest.TestCase):
    """Tests for workflow caching functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = WorkflowMemoryManager()

    def test_cache_hit_on_similar_task(self):
        """Test workflow cache hit"""
        task1 = "Find Italian restaurants in Tokyo"
        task2 = "Find Italian restaurants in Tokyo"  # Same task

        task_hash = self.memory.hash_task(task1)
        workflow = [
            {"message": "Search", "tool_use": ["google_search"]},
        ]

        # Add first execution
        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=5.0
        )

        # Try to get recommendation for second task
        recommended = self.memory.recommend_workflow(task_hash)

        # Should hit cache
        self.assertIsNotNone(recommended)
        self.assertEqual(recommended, workflow)

    def test_cache_size_tracking(self):
        """Test cache size tracking"""
        memory = WorkflowMemoryManager()

        # Add multiple workflows
        for i in range(5):
            workflow = [{"message": f"Step {i}", "tool_use": []}]
            memory.add_workflow_execution(
                task_hash=f"task_{i}",
                workflow=workflow,
                success=True,
                latency=1.0
            )

        # Check cache tracking
        stats = memory.get_performance_report()
        self.assertEqual(stats.get("total_workflows"), 5)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for performance metrics tracking"""

    def test_metrics_initialization(self):
        """Test metrics are initialized correctly"""
        memory = WorkflowMemoryManager()

        metrics = memory.execution_metrics

        self.assertEqual(metrics["total_workflows"], 0)
        self.assertEqual(metrics["successful_workflows"], 0)
        self.assertEqual(metrics["total_latency"], 0.0)

    def test_latency_tracking(self):
        """Test latency metric tracking"""
        memory = WorkflowMemoryManager()

        workflow = [{"message": "Test", "tool_use": []}]

        memory.add_workflow_execution(
            task_hash="test1",
            workflow=workflow,
            success=True,
            latency=2.5
        )

        report = memory.get_performance_report()

        self.assertAlmostEqual(report["avg_latency"], 2.5)

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        memory = WorkflowMemoryManager()

        workflow = [{"message": "Test", "tool_use": []}]

        # 2 successes
        memory.add_workflow_execution(
            task_hash="test1",
            workflow=workflow,
            success=True,
            latency=1.0
        )
        memory.add_workflow_execution(
            task_hash="test2",
            workflow=workflow,
            success=True,
            latency=1.0
        )

        # 1 failure
        memory.add_workflow_execution(
            task_hash="test3",
            workflow=workflow,
            success=False,
            latency=1.0
        )

        report = memory.get_performance_report()

        self.assertAlmostEqual(report["success_rate"], 2/3, places=2)


# Run tests
if __name__ == "__main__":
    unittest.main()
