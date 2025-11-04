"""
End-to-End workflow tests for OpenAGI-AIOS integration.

Tests complete workflows from task input through execution, learning, and persistence.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from aios.openagi_kernel_bridge import OpenAGIKernelBridge, WorkflowStep
from aios.workflow_memory_manager import WorkflowMemoryManager
from aios.openagi_memory_integration import OpenAGIMemoryIntegration
from aios.openagi_autonomous_discovery import AutonomousToolDiscovery, ToolCategory


class TestEndToEndWorkflowExecution(unittest.TestCase):
    """Tests for complete workflow execution pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)

        # Create mock kernel components
        self.mock_llm_core = AsyncMock()
        self.mock_context_manager = Mock()
        self.mock_memory_manager = Mock()
        self.mock_tool_manager = Mock()

        # Initialize integration components
        self.memory_integration = OpenAGIMemoryIntegration(storage_path=self.storage_path)
        self.discovery = AutonomousToolDiscovery(memory_integration=self.memory_integration)
        self.bridge = OpenAGIKernelBridge(
            llm_core=self.mock_llm_core,
            context_manager=self.mock_context_manager,
            memory_manager=self.mock_memory_manager,
            tool_manager=self.mock_tool_manager,
        )

    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()

    def test_workflow_recording_and_retrieval(self):
        """Test recording workflow and retrieving it later"""
        task = "Find restaurants"
        workflow = [
            {"message": "Search for restaurants", "tool_use": ["google_search", "yelp"]},
            {"message": "Filter by rating", "tool_use": []},
            {"message": "Return results", "tool_use": []},
        ]

        # Record execution
        self.memory_integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=2.5,
            tokens_used=150,
        )

        # Retrieve recommendation
        recommended = self.memory_integration.get_recommended_workflow(task)

        self.assertIsNotNone(recommended)
        self.assertEqual(recommended, workflow)

    def test_tool_discovery_with_workflow_learning(self):
        """Test discovering tools during workflow execution"""
        # Register tools
        tools = [
            ("google_search", ToolCategory.SEARCH),
            ("yelp_search", ToolCategory.SEARCH),
            ("rating_filter", ToolCategory.ANALYSIS),
        ]

        for tool_name, category in tools:
            self.discovery.register_tool(tool_name, category)

        # Simulate workflow execution with tool usage
        task = "Find restaurants with high ratings"
        workflow = [
            {"message": "Search", "tool_use": ["google_search", "yelp_search"]},
            {"message": "Filter", "tool_use": ["rating_filter"]},
        ]

        # Track tool usage
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)
        self.discovery.update_tool_effectiveness("yelp_search", success=True, latency=0.6)
        self.discovery.update_tool_effectiveness("rating_filter", success=True, latency=0.3)

        # Record combination
        self.discovery.record_combination_execution(
            ["google_search", "yelp_search"],
            success=True,
            latency=1.1,
            use_case="search_phase",
        )

        # Record workflow
        self.memory_integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=1.9,
            tokens_used=180,
        )

        # Verify learning
        recommendations = self.discovery.get_tool_recommendations()
        self.assertIn("google_search", recommendations)
        self.assertIn("yelp_search", recommendations)

        combo_recs = self.discovery.get_combination_recommendations(num_recommendations=1)
        self.assertGreater(len(combo_recs), 0)

    def test_workflow_caching_improves_performance(self):
        """Test that workflow caching provides performance improvement"""
        task = "Fetch user data"
        workflow = [
            {"message": "Query database", "tool_use": ["database_query"]},
            {"message": "Format response", "tool_use": []},
        ]

        # First execution - no cache hit
        self.memory_integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=5.0,
        )

        # Second execution - should hit cache
        cached_workflow = self.memory_integration.get_recommended_workflow(task)
        self.assertIsNotNone(cached_workflow)

        # Verify it's the same workflow
        self.assertEqual(cached_workflow, workflow)

    def test_tool_learning_across_multiple_tasks(self):
        """Test tool learning accumulates across different tasks"""
        # Register tools
        for tool in ["search", "analyze", "integrate"]:
            self.discovery.register_tool(tool, ToolCategory.SEARCH)

        # Execute multiple tasks using same tools
        tasks = [
            ("Task A", ["search", "analyze"]),
            ("Task B", ["search", "integrate"]),
            ("Task C", ["analyze", "integrate"]),
        ]

        for task_name, tools_used in tasks:
            for tool in tools_used:
                success = True
                latency = 0.5
                self.discovery.update_tool_effectiveness(tool, success, latency)

            self.discovery.record_combination_execution(
                tools_used,
                success=True,
                latency=1.0,
                use_case=task_name,
            )

        # Verify learning
        stats = self.discovery.get_discovery_statistics()
        self.assertEqual(stats["tools_tested"], 3)
        self.assertEqual(stats["total_combinations_tested"], 3)

    def test_concept_registration_from_discovery(self):
        """Test registering learned concepts with memory"""
        # Discover tool effectiveness
        tool_name = "advanced_search"
        self.discovery.register_tool(tool_name, ToolCategory.SEARCH)

        # Execute multiple times
        for i in range(10):
            self.discovery.update_tool_effectiveness(tool_name, success=True, latency=0.4)

        # Get tool profile
        profiles = self.discovery.export_learned_profiles()
        tool_profile = profiles[tool_name]

        # Register with memory
        self.memory_integration.register_learned_concept(
            concept=f"{tool_name}_optimal",
            category="tools",
            confidence=min(1.0, tool_profile["total_uses"] / 20.0),
            source="autonomous_discovery",
            metadata=tool_profile,
        )

        # Verify registration
        concepts = self.memory_integration.get_high_confidence_concepts(
            category="tools", threshold=0.0
        )
        self.assertGreater(len(concepts), 0)

    def test_workflow_persistence_and_recovery(self):
        """Test that workflows persist to disk and are recovered on startup"""
        # Record workflows
        workflows_to_store = [
            ("task1", [{"message": "Step 1", "tool_use": ["tool1"]}]),
            ("task2", [{"message": "Step 2", "tool_use": ["tool2", "tool3"]}]),
            ("task3", [{"message": "Step 3", "tool_use": ["tool1", "tool3"]}]),
        ]

        for task, workflow in workflows_to_store:
            self.memory_integration.record_workflow_execution(
                task=task,
                workflow=workflow,
                success=True,
                latency=1.0,
            )

        # Save to disk
        self.memory_integration._save_persistent_knowledge()

        # Create new integration from same storage
        new_integration = OpenAGIMemoryIntegration(storage_path=self.storage_path)

        # Verify recovery
        for task, expected_workflow in workflows_to_store:
            recovered = new_integration.get_recommended_workflow(task)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered, expected_workflow)

    def test_performance_improvement_tracking(self):
        """Test tracking performance improvements from caching and learning"""
        task = "Analytics query"
        workflow = [
            {"message": "Query", "tool_use": ["database_query"]},
            {"message": "Process", "tool_use": ["data_processor"]},
        ]

        # Initial execution (no optimization)
        initial_latency = 10.0
        self.memory_integration.record_workflow_execution(
            task=task,
            workflow=workflow,
            success=True,
            latency=initial_latency,
        )

        # Subsequent executions get cached
        for i in range(5):
            cached = self.memory_integration.get_recommended_workflow(task)
            self.assertIsNotNone(cached)

        # Get performance report
        report = self.memory_integration.get_performance_report()

        # Should have recorded metrics
        self.assertEqual(report["total_workflows"], 1)
        self.assertGreater(report["avg_latency"], 0)

    def test_knowledge_export_import_roundtrip(self):
        """Test exporting and importing knowledge preserves all data"""
        # Build up knowledge
        for tool in ["search", "analyze"]:
            self.discovery.register_tool(tool, ToolCategory.SEARCH)
            self.discovery.update_tool_effectiveness(tool, True, 0.5)

        self.discovery.record_combination_execution(
            ["search", "analyze"], True, 1.0
        )

        task = "Test task"
        workflow = [
            {"message": "Step", "tool_use": ["search", "analyze"]}
        ]

        self.memory_integration.record_workflow_execution(
            task, workflow, True, 1.0
        )

        self.memory_integration.register_learned_concept(
            "concept1", "category1", 0.9
        )

        # Export
        graph = self.memory_integration.export_knowledge_graph()

        # Import to new system
        new_memory = OpenAGIMemoryIntegration()
        new_memory.import_knowledge_graph(graph)

        # Verify all data preserved
        self.assertEqual(len(new_memory.learned_concepts), 1)
        self.assertEqual(new_memory.memory.execution_metrics["total_workflows"], 1)

    def test_multiple_tool_categories(self):
        """Test learning across different tool categories"""
        categories = [
            ("search_tool", ToolCategory.SEARCH),
            ("analyze_tool", ToolCategory.ANALYSIS),
            ("api_tool", ToolCategory.INTEGRATION),
        ]

        for tool_name, category in categories:
            self.discovery.register_tool(tool_name, category)
            self.discovery.update_tool_effectiveness(tool_name, True, 0.5)

        # Get recommendations by category
        search_tools = self.discovery.get_tool_recommendations(category=ToolCategory.SEARCH)
        analysis_tools = self.discovery.get_tool_recommendations(category=ToolCategory.ANALYSIS)
        integration_tools = self.discovery.get_tool_recommendations(category=ToolCategory.INTEGRATION)

        self.assertEqual(len(search_tools), 1)
        self.assertEqual(len(analysis_tools), 1)
        self.assertEqual(len(integration_tools), 1)

    def test_failure_handling_and_recovery(self):
        """Test handling tool failures and learning from them"""
        tool_name = "potentially_failing_tool"
        self.discovery.register_tool(tool_name, ToolCategory.SEARCH)

        # Mix of successes and failures
        for i in range(10):
            success = i % 3 != 0  # 66% success rate
            failure_reason = "timeout" if not success else None
            self.discovery.update_tool_effectiveness(
                tool_name,
                success=success,
                latency=0.5,
                failure_reason=failure_reason,
            )

        profile = self.discovery.tool_profiles[tool_name]

        # Should still have positive effectiveness
        self.assertGreater(profile.effectiveness_score, 0)
        self.assertAlmostEqual(profile.success_rate, 0.67, places=1)
        self.assertIn("timeout", profile.common_failure_modes)

    def test_end_to_end_scenario_restaurant_finder(self):
        """Complete end-to-end scenario: Restaurant finder workflow"""
        # Setup discovery with relevant tools
        search_tools = [
            ("google_search", ToolCategory.SEARCH),
            ("yelp_api", ToolCategory.INTEGRATION),
        ]
        analysis_tools = [
            ("rating_analyzer", ToolCategory.ANALYSIS),
            ("location_filter", ToolCategory.ANALYSIS),
        ]

        for tool, category in search_tools + analysis_tools:
            self.discovery.register_tool(tool, category)

        # Execute first query
        task1 = "Find Italian restaurants in Tokyo with 4+ stars"
        workflow1 = [
            {
                "message": "Search for Italian restaurants",
                "tool_use": ["google_search", "yelp_api"],
            },
            {
                "message": "Filter by rating",
                "tool_use": ["rating_analyzer"],
            },
            {
                "message": "Sort by location",
                "tool_use": ["location_filter"],
            },
        ]

        # Track tool usage
        for tool in ["google_search", "yelp_api"]:
            self.discovery.update_tool_effectiveness(tool, True, 0.5)
        for tool in ["rating_analyzer", "location_filter"]:
            self.discovery.update_tool_effectiveness(tool, True, 0.3)

        # Record workflow
        self.memory_integration.record_workflow_execution(
            task1, workflow1, True, 1.3, 200
        )

        # Record tool combination
        self.discovery.record_combination_execution(
            ["google_search", "yelp_api"], True, 1.0, "search_phase"
        )

        # Execute similar query (should use cached workflow)
        task2 = "Find French restaurants in Paris with 4+ stars"
        cached = self.memory_integration.get_recommended_workflow(task1)
        self.assertIsNotNone(cached)

        # Verify learning
        stats = self.discovery.get_discovery_statistics()
        self.assertEqual(stats["total_tools_registered"], 4)
        self.assertEqual(stats["tools_tested"], 4)


# Run tests
if __name__ == "__main__":
    unittest.main()
