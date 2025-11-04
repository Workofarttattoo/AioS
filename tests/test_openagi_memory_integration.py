"""
Integration tests for OpenAGI memory system with AIOS kernel.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from aios.openagi_memory_integration import (
    OpenAGIMemoryIntegration,
    initialize_openagi_memory,
    persist_openagi_memory,
    report_openagi_memory_analytics,
)
from aios.workflow_memory_manager import WorkflowMemoryManager
from aios.runtime import ExecutionContext, ActionResult


class TestOpenAGIMemoryIntegration(unittest.TestCase):
    """Tests for OpenAGI memory integration layer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        self.integration = OpenAGIMemoryIntegration(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()

    def test_memory_integration_initialization(self):
        """Test memory integration initializes correctly"""
        self.assertIsNotNone(self.integration.memory)
        self.assertIsNotNone(self.integration.storage_path)
        self.assertEqual(len(self.integration.learned_concepts), 0)

    def test_record_workflow_execution(self):
        """Test recording workflow execution"""
        task = "Find restaurants in Tokyo"
        workflow = [
            {"message": "Search", "tool_use": ["google_search"]},
            {"message": "Filter", "tool_use": []},
        ]

        self.integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=5.0,
            tokens_used=150,
        )

        # Verify execution was recorded
        report = self.integration.get_performance_report()
        self.assertEqual(report["total_workflows"], 1)
        self.assertEqual(report["successful_workflows"], 1)

    def test_get_recommended_workflow(self):
        """Test getting recommended workflow"""
        task = "Find restaurants"
        workflow = [{"message": "Search", "tool_use": ["google_search"]}]

        # Record execution
        self.integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=3.0,
        )

        # Get recommendation
        recommended = self.integration.get_recommended_workflow(task)

        self.assertIsNotNone(recommended)
        self.assertEqual(recommended, workflow)

    def test_get_tool_recommendations(self):
        """Test getting tool recommendations"""
        task = "Find restaurants"
        workflow = [
            {"message": "Search", "tool_use": ["google_search", "yelp"]},
            {"message": "Filter", "tool_use": ["google_maps"]},
        ]

        self.integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=3.0,
        )

        tools = self.integration.get_tool_recommendations(task)

        self.assertIn("google_search", tools)
        self.assertIn("yelp", tools)
        self.assertIn("google_maps", tools)

    def test_register_learned_concept(self):
        """Test registering learned concepts"""
        self.integration.register_learned_concept(
            concept="microservices architecture",
            category="architecture",
            confidence=0.92,
            source="autonomous_discovery",
            metadata={"domain": "distributed_systems"},
        )

        self.assertIn("architecture:microservices architecture", self.integration.learned_concepts)
        self.assertEqual(
            self.integration.concept_confidence_scores["architecture:microservices architecture"],
            0.92,
        )

    def test_get_high_confidence_concepts(self):
        """Test filtering high-confidence concepts"""
        # Register multiple concepts with different confidence levels
        self.integration.register_learned_concept(
            "kubernetes scaling",
            "performance",
            confidence=0.95,
        )
        self.integration.register_learned_concept(
            "thread pooling",
            "performance",
            confidence=0.88,
        )
        self.integration.register_learned_concept(
            "eventual consistency",
            "architecture",
            confidence=0.72,  # Below default threshold of 0.8
        )

        # Get high-confidence concepts
        high_confidence = self.integration.get_high_confidence_concepts(threshold=0.85)

        self.assertEqual(len(high_confidence), 2)
        self.assertTrue(all(c["confidence"] >= 0.85 for c in high_confidence))

    def test_get_high_confidence_concepts_by_category(self):
        """Test filtering concepts by category"""
        self.integration.register_learned_concept(
            "kubernetes scaling",
            "performance",
            confidence=0.95,
        )
        self.integration.register_learned_concept(
            "circuit breaker pattern",
            "resilience",
            confidence=0.90,
        )

        # Get concepts only in performance category
        performance_concepts = self.integration.get_high_confidence_concepts(
            category="performance",
            threshold=0.9,
        )

        self.assertEqual(len(performance_concepts), 1)
        self.assertEqual(performance_concepts[0]["category"], "performance")

    def test_export_knowledge_graph(self):
        """Test exporting knowledge graph"""
        # Add workflows
        self.integration.record_workflow_execution(
            task="task1",
            workflow=[{"message": "Step 1", "tool_use": []}],
            success=True,
            latency=1.0,
        )

        # Add concepts
        self.integration.register_learned_concept(
            "pattern1",
            "category1",
            confidence=0.9,
        )

        # Export
        graph = self.integration.export_knowledge_graph()

        self.assertIn("workflows", graph)
        self.assertIn("learned_concepts", graph)
        self.assertIn("metrics", graph)
        self.assertIn("exported_at", graph)
        self.assertEqual(len(graph["learned_concepts"]), 1)

    def test_import_knowledge_graph(self):
        """Test importing knowledge graph"""
        # Create knowledge graph
        source_integration = OpenAGIMemoryIntegration()
        source_integration.record_workflow_execution(
            task="task1",
            workflow=[{"message": "Step 1", "tool_use": ["tool1"]}],
            success=True,
            latency=1.0,
        )
        source_integration.register_learned_concept(
            "pattern1",
            "category1",
            confidence=0.9,
        )

        graph = source_integration.export_knowledge_graph()

        # Import into new integration
        target_integration = OpenAGIMemoryIntegration()
        target_integration.import_knowledge_graph(graph)

        # Verify import
        self.assertEqual(len(target_integration.learned_concepts), 1)
        self.assertEqual(
            target_integration.concept_confidence_scores["category1:pattern1"],
            0.9,
        )

    def test_persist_and_load_knowledge(self):
        """Test persisting and loading knowledge from disk"""
        # Record execution and concept
        self.integration.record_workflow_execution(
            task="restaurant search",
            workflow=[{"message": "Search", "tool_use": ["google_search"]}],
            success=True,
            latency=2.5,
        )

        self.integration.register_learned_concept(
            "search optimization",
            "performance",
            confidence=0.88,
        )

        # Save
        self.integration._save_persistent_knowledge()

        # Load in new integration
        new_integration = OpenAGIMemoryIntegration(storage_path=self.storage_path)

        # Verify loaded
        report = new_integration.get_performance_report()
        self.assertEqual(report["total_workflows"], 1)

    def test_performance_report(self):
        """Test performance reporting"""
        # Record multiple executions
        for i in range(3):
            self.integration.record_workflow_execution(
                task=f"task_{i}",
                workflow=[{"message": f"Step {i}", "tool_use": []}],
                success=i < 2,  # 2 successes, 1 failure
                latency=1.0 + i,
            )

        report = self.integration.get_performance_report()

        self.assertEqual(report["total_workflows"], 3)
        self.assertEqual(report["successful_workflows"], 2)
        self.assertGreater(report["success_rate"], 0.5)

    @patch("aios.openagi_memory_integration.ExecutionContext")
    async def test_autonomy_ready(self, mock_ctx):
        """Test autonomy readiness check"""
        # Mock ExecutionContext
        mock_ctx.environment.get.return_value = "1"

        # Record enough workflows
        for i in range(5):
            self.integration.record_workflow_execution(
                task=f"task_{i}",
                workflow=[{"message": f"Step {i}", "tool_use": []}],
                success=True,
                latency=1.0,
            )

        # Check readiness
        is_ready = await self.integration.autonomy_ready(mock_ctx)

        # Should be ready with enough history
        self.assertTrue(is_ready)


class TestOpenAGIMemoryManifestActions(unittest.TestCase):
    """Tests for manifest action handlers"""

    def setUp(self):
        """Set up test fixtures"""
        self.ctx = Mock(spec=ExecutionContext)
        self.ctx.environment = {}
        self.ctx.publish_metadata = Mock()

    @patch("aios.openagi_memory_integration.OpenAGIMemoryIntegration")
    async def test_initialize_memory_action(self, mock_memory_class):
        """Test initialize_memory manifest action"""
        mock_memory = Mock()
        mock_memory.memory = Mock()
        mock_memory.memory.execution_metrics = {"total_workflows": 5}
        mock_memory.learned_concepts = {"concept1": "data1"}
        mock_memory_class.return_value = mock_memory

        result = await initialize_openagi_memory(self.ctx)

        self.assertTrue(result.success)
        self.assertIn("initialized", result.message.lower())
        self.ctx.publish_metadata.assert_called()

    async def test_persist_memory_action_no_integration(self):
        """Test persist_memory when no integration exists"""
        # No openagi_memory attribute
        self.ctx.openagi_memory = None

        result = await persist_openagi_memory(self.ctx)

        # Should succeed but indicate nothing to persist
        self.assertTrue(result.success)

    async def test_report_analytics_action(self):
        """Test memory analytics reporting action"""
        # Mock memory integration
        mock_memory = Mock(spec=OpenAGIMemoryIntegration)
        mock_memory.get_performance_report.return_value = {
            "total_workflows": 10,
            "successful_workflows": 9,
            "success_rate": 0.9,
            "avg_latency": 2.5,
            "tool_combinations": {"tool1": 5, "tool2": 3},
        }
        mock_memory.get_high_confidence_concepts.return_value = [
            {"concept": "pattern1", "confidence": 0.95},
            {"concept": "pattern2", "confidence": 0.88},
        ]

        self.ctx.openagi_memory = mock_memory

        result = await report_openagi_memory_analytics(self.ctx)

        self.assertTrue(result.success)
        self.assertIn("analytics", result.message.lower())
        self.assertEqual(result.payload["total_workflows_learned"], 10)


class TestWorkflowMemoryManagerIntegration(unittest.TestCase):
    """Tests for WorkflowMemoryManager integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = WorkflowMemoryManager()

    def test_memory_manager_hashing_consistency(self):
        """Test hash consistency across calls"""
        task = "Find Italian restaurants"

        hash1 = self.memory.hash_task(task)
        hash2 = self.memory.hash_task(task)

        self.assertEqual(hash1, hash2)

    def test_memory_manager_workflow_caching(self):
        """Test workflow caching mechanism"""
        task = "restaurant search"
        task_hash = self.memory.hash_task(task)
        workflow = [{"message": "Search", "tool_use": ["google_search"]}]

        # Add workflow
        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=2.0,
        )

        # Retrieve
        recommended = self.memory.recommend_workflow(task_hash)

        self.assertEqual(recommended, workflow)

    def test_memory_manager_tool_stats(self):
        """Test tool combination statistics"""
        task_hash = "task123"
        workflow1 = [
            {"message": "Step 1", "tool_use": ["google_search", "yelp"]},
            {"message": "Step 2", "tool_use": []},
        ]

        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow1,
            success=True,
            latency=2.0,
        )

        report = self.memory.get_performance_report()
        self.assertGreater(len(report.get("tool_combinations", {})), 0)

    def test_memory_manager_export_import(self):
        """Test knowledge export and import"""
        # Add data to source
        task_hash = "task1"
        workflow = [{"message": "Search", "tool_use": ["tool1"]}]

        self.memory.add_workflow_execution(
            task_hash=task_hash,
            workflow=workflow,
            success=True,
            latency=1.0,
        )

        # Export
        knowledge = self.memory.export_knowledge()

        # Import to new manager
        new_memory = WorkflowMemoryManager()
        new_memory.import_knowledge(knowledge)

        # Verify
        recommended = new_memory.recommend_workflow(task_hash)
        self.assertEqual(recommended, workflow)


# Run tests
if __name__ == "__main__":
    unittest.main()
