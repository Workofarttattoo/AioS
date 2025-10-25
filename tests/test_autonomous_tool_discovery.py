"""
Tests for autonomous tool discovery and learning system.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from aios.openagi_autonomous_discovery import (
    AutonomousToolDiscovery,
    ToolCategory,
    ToolProfile,
    ToolCombinationPattern,
    discover_tools_autonomous,
)
from aios.runtime import ExecutionContext


class TestToolProfile(unittest.TestCase):
    """Tests for ToolProfile dataclass"""

    def test_tool_profile_creation(self):
        """Test creating tool profile"""
        profile = ToolProfile(
            tool_name="google_search",
            category=ToolCategory.SEARCH,
            effectiveness_score=0.8,
            success_rate=0.9,
            avg_latency=0.5,
            total_uses=10,
            preferred_partners=["api_call"],
            common_failure_modes=["timeout"],
            last_used=0.0,
        )

        self.assertEqual(profile.tool_name, "google_search")
        self.assertEqual(profile.category, ToolCategory.SEARCH)
        self.assertEqual(profile.effectiveness_score, 0.8)


class TestToolCombinationPattern(unittest.TestCase):
    """Tests for ToolCombinationPattern dataclass"""

    def test_combination_pattern_creation(self):
        """Test creating combination pattern"""
        pattern = ToolCombinationPattern(
            tools=("google_search", "api_call"),
            success_rate=0.85,
            avg_latency=1.0,
            total_executions=20,
            confidence_score=0.95,
            use_case_examples=["search_and_fetch"],
            performance_improvement=0.2,
        )

        self.assertEqual(len(pattern.tools), 2)
        self.assertEqual(pattern.success_rate, 0.85)
        self.assertGreater(pattern.confidence_score, 0.9)


class TestAutonomousToolDiscovery(unittest.TestCase):
    """Tests for AutonomousToolDiscovery"""

    def setUp(self):
        """Set up test fixtures"""
        self.discovery = AutonomousToolDiscovery()

    def test_discovery_initialization(self):
        """Test discovery system initializes correctly"""
        self.assertEqual(len(self.discovery.tool_profiles), 0)
        self.assertEqual(len(self.discovery.combination_patterns), 0)
        self.assertEqual(self.discovery.discovery_iterations, 0)

    def test_register_tool(self):
        """Test registering a tool"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)

        self.assertIn("google_search", self.discovery.tool_profiles)
        self.assertEqual(self.discovery.tool_profiles["google_search"].tool_name, "google_search")
        self.assertEqual(self.discovery.tool_profiles["google_search"].category, ToolCategory.SEARCH)

    def test_register_multiple_tools(self):
        """Test registering multiple tools"""
        tools = [
            ("google_search", ToolCategory.SEARCH),
            ("database_query", ToolCategory.ANALYSIS),
            ("api_call", ToolCategory.INTEGRATION),
        ]

        for tool_name, category in tools:
            self.discovery.register_tool(tool_name, category)

        self.assertEqual(len(self.discovery.tool_profiles), 3)

    def test_update_tool_effectiveness(self):
        """Test updating tool effectiveness"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)

        # First execution - success
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)

        profile = self.discovery.tool_profiles["google_search"]
        self.assertEqual(profile.total_uses, 1)
        self.assertEqual(profile.success_rate, 1.0)
        self.assertEqual(profile.avg_latency, 0.5)

    def test_update_tool_effectiveness_multiple(self):
        """Test updating tool effectiveness over time"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)

        # Multiple executions
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)
        self.discovery.update_tool_effectiveness("google_search", success=False, latency=0.3)
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.6)

        profile = self.discovery.tool_profiles["google_search"]
        self.assertEqual(profile.total_uses, 3)
        self.assertAlmostEqual(profile.success_rate, 2.0 / 3.0, places=2)
        self.assertGreater(profile.effectiveness_score, 0.0)

    def test_update_tool_failure_reason(self):
        """Test tracking failure reasons"""
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        self.discovery.update_tool_effectiveness(
            "api_call",
            success=False,
            latency=5.0,
            failure_reason="timeout",
        )

        profile = self.discovery.tool_profiles["api_call"]
        self.assertIn("timeout", profile.common_failure_modes)

    def test_record_combination_execution(self):
        """Test recording tool combination"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )

        self.assertEqual(len(self.discovery.combination_patterns), 1)
        self.assertEqual(self.discovery.total_combinations_tested, 1)

    def test_record_combination_updates_partnerships(self):
        """Test that successful combinations record partnerships"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )

        search_profile = self.discovery.tool_profiles["google_search"]
        self.assertIn("api_call", search_profile.preferred_partners)

    def test_get_tool_recommendations(self):
        """Test getting tool recommendations"""
        # Register and track tools
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("database_query", ToolCategory.ANALYSIS)

        # Update effectiveness
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.6)
        self.discovery.update_tool_effectiveness("database_query", success=True, latency=2.0)

        recommendations = self.discovery.get_tool_recommendations()

        # google_search should be first (better effectiveness)
        self.assertEqual(recommendations[0], "google_search")

    def test_get_tool_recommendations_by_category(self):
        """Test filtering recommendations by category"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("database_query", ToolCategory.ANALYSIS)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        # Update all
        for tool in ["google_search", "database_query", "api_call"]:
            self.discovery.update_tool_effectiveness(tool, success=True, latency=1.0)

        # Get only search tools
        search_tools = self.discovery.get_tool_recommendations(category=ToolCategory.SEARCH)

        self.assertEqual(len(search_tools), 1)
        self.assertEqual(search_tools[0], "google_search")

    def test_get_combination_recommendations(self):
        """Test getting recommended combinations"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)
        self.discovery.register_tool("database_query", ToolCategory.ANALYSIS)

        # Record combinations
        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )
        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.1,
        )
        self.discovery.record_combination_execution(
            ["database_query"],
            success=False,
            latency=5.0,
        )

        recommendations = self.discovery.get_combination_recommendations(num_recommendations=2)

        # First recommendation should be the successful one
        self.assertEqual(recommendations[0].success_rate, 1.0)

    def test_predict_combination_success_known(self):
        """Test predicting success for known combination"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        # Record successful combination
        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )

        success_rate, reason = self.discovery.predict_combination_success(
            ["google_search", "api_call"]
        )

        self.assertEqual(success_rate, 1.0)
        self.assertIn("execution", reason.lower())

    def test_predict_combination_success_unknown(self):
        """Test predicting success for unknown combination"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH, initial_effectiveness=0.8)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION, initial_effectiveness=0.8)

        # Update to set effectiveness
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)
        self.discovery.update_tool_effectiveness("api_call", success=True, latency=0.5)

        success_rate, reason = self.discovery.predict_combination_success(
            ["google_search", "api_call"]
        )

        # Should be based on individual tool effectiveness
        self.assertGreater(success_rate, 0.5)

    def test_export_learned_profiles(self):
        """Test exporting tool profiles"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.update_tool_effectiveness("google_search", success=True, latency=0.5)

        profiles = self.discovery.export_learned_profiles()

        self.assertIn("google_search", profiles)
        self.assertEqual(profiles["google_search"]["tool_name"], "google_search")
        self.assertEqual(profiles["google_search"]["category"], "search")

    def test_export_learned_combinations(self):
        """Test exporting combination patterns"""
        self.discovery.register_tool("google_search", ToolCategory.SEARCH)
        self.discovery.register_tool("api_call", ToolCategory.INTEGRATION)

        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )

        combinations = self.discovery.export_learned_combinations()

        self.assertEqual(len(combinations), 1)
        pattern = list(combinations.values())[0]
        self.assertEqual(len(pattern["tools"]), 2)

    def test_get_discovery_statistics(self):
        """Test getting discovery statistics"""
        # Register and test tools
        for tool_name in ["google_search", "api_call", "database_query"]:
            self.discovery.register_tool(tool_name, ToolCategory.SEARCH)
            self.discovery.update_tool_effectiveness(tool_name, success=True, latency=0.5)

        # Record combinations
        self.discovery.record_combination_execution(
            ["google_search", "api_call"],
            success=True,
            latency=1.0,
        )

        stats = self.discovery.get_discovery_statistics()

        self.assertEqual(stats["total_tools_registered"], 3)
        self.assertEqual(stats["tools_tested"], 3)
        self.assertEqual(stats["total_combinations_tested"], 1)
        self.assertGreater(stats["average_tool_effectiveness"], 0.0)

    async def test_discover_tools_autonomous_action(self):
        """Test autonomous discovery manifest action"""
        mock_memory = Mock()
        mock_memory.register_learned_concept = Mock()

        mock_ctx = Mock(spec=ExecutionContext)
        mock_ctx.openagi_memory = mock_memory
        mock_ctx.publish_metadata = Mock()

        result = await discover_tools_autonomous(mock_ctx)

        self.assertTrue(result.success)
        self.assertIn("discovery", result.message.lower())
        self.assertIn("combinations_tested", result.payload)

    async def test_discover_tools_no_memory(self):
        """Test discovery fails without memory integration"""
        mock_ctx = Mock(spec=ExecutionContext)
        # No openagi_memory attribute

        result = await discover_tools_autonomous(mock_ctx)

        self.assertFalse(result.success)
        self.assertIn("error", result.message.lower())


# Run tests
if __name__ == "__main__":
    unittest.main()
