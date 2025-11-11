"""
Comprehensive Test Suite for Production Ai:oS Agents.

Tests following 2025 production best practices:
- End-to-end action testing
- Error handling and retry logic
- Forensic mode compliance
- Performance benchmarks (>99.9% reliability target)
- Cost tracking validation

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

import unittest
import sys
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_manifest, Manifest
from src.aios.runtime import ProductionRuntime, ActionResult, ExecutionContext
from agents.security_agent import SecurityAgent
from observability import ObservabilitySystem
from evaluation import EvaluationHarness, CodeBasedEvaluator
from model_router import ModelRouter, ResponseCache


class TestProductionRuntime(unittest.TestCase):
    """Test suite for production runtime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runtime = ProductionRuntime(
            enable_observability=True,
            enable_model_routing=True,
            enable_caching=True
        )
    
    def test_runtime_initialization(self):
        """Test runtime initializes correctly."""
        self.assertIsNotNone(self.runtime.manifest)
        self.assertIsNotNone(self.runtime.observability)
        self.assertIsNotNone(self.runtime.model_router)
        self.assertIsNotNone(self.runtime.response_cache)
    
    def test_execute_action_success(self):
        """Test successful action execution."""
        # Execute a simple action
        result = self.runtime.execute_action("security.access_control")
        
        self.assertTrue(result.success)
        self.assertIsInstance(result, ActionResult)
        self.assertGreater(result.latency_ms, 0)
        self.assertIsNotNone(result.message)
    
    def test_execute_action_with_retry(self):
        """Test action execution with retry logic."""
        # Force an error on first attempt, success on retry
        with patch.object(SecurityAgent, 'access_control') as mock_action:
            # First call fails, second succeeds
            mock_action.side_effect = [
                ActionResult(success=False, message="Transient error", error="Timeout"),
                ActionResult(success=True, message="Success after retry", payload={})
            ]
            
            result = self.runtime.execute_action("security.access_control", retry=True)
            
            # Should have retried and succeeded
            self.assertTrue(result.success)
            self.assertEqual(mock_action.call_count, 2)
    
    def test_execute_action_non_retryable_error(self):
        """Test non-retryable errors fail immediately."""
        with patch.object(SecurityAgent, 'access_control') as mock_action:
            mock_action.return_value = ActionResult(
                success=False,
                message="Not found",
                error="FileNotFoundError",
                error_type="FileNotFoundError"
            )
            
            result = self.runtime.execute_action("security.access_control", retry=True)
            
            # Should not retry permanent errors
            self.assertFalse(result.success)
            self.assertEqual(mock_action.call_count, 1)
    
    def test_forensic_mode_respected(self):
        """Test forensic mode prevents mutations."""
        forensic_runtime = ProductionRuntime(environment={"AGENTA_FORENSIC_MODE": "1"})
        
        self.assertTrue(forensic_runtime.ctx.forensic_mode)
        
        # Execute firewall action in forensic mode
        result = forensic_runtime.execute_action("security.firewall")
        
        # Should report but not mutate
        self.assertTrue(result.success)
        self.assertTrue(result.payload.get("forensic", False))
    
    def test_sequence_execution(self):
        """Test executing a sequence of actions."""
        sequence = [
            "security.access_control",
            "security.encryption",
            "security.firewall"
        ]
        
        summary = self.runtime.execute_sequence(sequence)
        
        self.assertGreater(summary["total_actions"], 0)
        self.assertGreaterEqual(summary["success_rate"], 0.0)
        self.assertLessEqual(summary["success_rate"], 1.0)
    
    def test_metrics_collection(self):
        """Test metrics are collected correctly."""
        # Execute a few actions
        self.runtime.execute_action("security.access_control")
        self.runtime.execute_action("security.encryption")
        
        metrics = self.runtime.get_metrics()
        
        self.assertIn("runtime", metrics)
        self.assertIn("observability", metrics)
        self.assertEqual(metrics["runtime"]["agents_loaded"], 1)  # SecurityAgent loaded


class TestSecurityAgent(unittest.TestCase):
    """Test suite for SecurityAgent with production patterns."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = SecurityAgent()
        self.ctx = ExecutionContext(
            manifest=load_manifest(),
            environment={}
        )
    
    def test_access_control_action(self):
        """Test access control action returns ActionResult."""
        result = self.agent.access_control(self.ctx, {})
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertIn("RBAC", result.message)
    
    def test_encryption_action(self):
        """Test encryption verification."""
        result = self.agent.encryption(self.ctx, {})
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertIsInstance(result.payload, dict)
    
    def test_firewall_action(self):
        """Test firewall management."""
        result = self.agent.firewall(self.ctx, {})
        
        self.assertIsInstance(result, ActionResult)
        # May succeed or fail depending on system, but should return ActionResult
        self.assertIsNotNone(result.message)
        self.assertIsInstance(result.payload, dict)
    
    def test_forensic_mode_prevents_mutations(self):
        """Test forensic mode is respected."""
        forensic_ctx = ExecutionContext(
            manifest=load_manifest(),
            environment={"AGENTA_FORENSIC_MODE": "1"}
        )
        
        result = self.agent.firewall(forensic_ctx, {})
        
        # Should succeed without mutation
        self.assertTrue(result.success)
        self.assertTrue(result.payload.get("forensic", False))
    
    def test_error_handling(self):
        """Test error handling returns proper ActionResult."""
        # Force an error by passing invalid context
        with patch.object(self.agent, 'get_firewall_status', side_effect=Exception("Test error")):
            result = self.agent.firewall(self.ctx, {})
            
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertIsNotNone(result.error_type)
    
    def test_confidence_scores(self):
        """Test actions include confidence scores in metadata."""
        result = self.agent.firewall(self.ctx, {})
        
        if result.success and result.metadata:
            # Successful actions should have confidence scores
            self.assertIn("confidence", result.metadata)
            confidence = result.metadata["confidence"]
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestObservability(unittest.TestCase):
    """Test suite for observability system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.obs = ObservabilitySystem()
    
    def test_trace_creation(self):
        """Test trace creation and management."""
        trace_id = self.obs.start_trace("test_trace")
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.obs.traces)
    
    def test_trace_span_context_manager(self):
        """Test trace span as context manager."""
        self.obs.start_trace("test")
        
        with self.obs.trace_span("test_action", agent_name="test_agent") as span:
            time.sleep(0.01)  # Simulate work
            span.set_tokens(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
        
        # Verify span was recorded
        self.assertEqual(len(self.obs.traces[self.obs.current_trace_id]), 1)
        recorded_span = self.obs.traces[self.obs.current_trace_id][0]
        
        self.assertEqual(recorded_span.name, "test_action")
        self.assertEqual(recorded_span.total_tokens, 150)
        self.assertGreater(recorded_span.duration_ms, 0)
        self.assertGreater(recorded_span.cost_usd, 0)
    
    def test_metrics_aggregation(self):
        """Test metrics are aggregated correctly."""
        self.obs.start_trace("test")
        
        # Simulate multiple actions
        for i in range(5):
            with self.obs.trace_span(f"action_{i}", agent_name="test") as span:
                span.set_tokens(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
        
        metrics = self.obs.get_metrics_summary()
        
        self.assertEqual(metrics["summary"]["total_calls"], 5)
        self.assertEqual(metrics["summary"]["total_tokens"], 750)  # 150 * 5
        self.assertGreater(metrics["summary"]["total_cost_usd"], 0)
    
    def test_error_tracking(self):
        """Test error tracking in observability."""
        self.obs.start_trace("test")
        
        try:
            with self.obs.trace_span("failing_action") as span:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        metrics = self.obs.get_metrics_summary()
        
        self.assertEqual(metrics["summary"]["total_errors"], 1)
        self.assertGreater(metrics["summary"]["error_rate"], 0)


class TestModelRouter(unittest.TestCase):
    """Test suite for model routing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = ModelRouter()
    
    def test_simple_task_routing(self):
        """Test simple tasks route to cheap models."""
        decision = self.router.route(
            prompt="List all files in the directory",
            estimated_tokens=100
        )
        
        # Simple tasks should route to flash/turbo models
        self.assertIn("flash", decision.model.name.lower() or "turbo" in decision.model.name.lower())
        self.assertLess(decision.estimated_cost_usd, 0.001)  # Very cheap
    
    def test_complex_task_routing(self):
        """Test complex tasks route to capable models."""
        decision = self.router.route(
            prompt="Design a distributed system architecture for handling 1M requests/sec with multi-region failover and explain the tradeoffs",
            estimated_tokens=2000
        )
        
        # Complex tasks should route to sonnet/gpt-4
        self.assertTrue(
            "sonnet" in decision.model.name.lower() or 
            "gpt-4" in decision.model.name.lower() or
            "o3" in decision.model.name.lower()
        )
    
    def test_routing_statistics(self):
        """Test routing statistics are tracked."""
        # Route several tasks
        self.router.route("Simple task 1", estimated_tokens=50)
        self.router.route("Complex reasoning task about quantum mechanics", estimated_tokens=500)
        self.router.route("Another simple task", estimated_tokens=50)
        
        stats = self.router.get_routing_stats()
        
        self.assertEqual(stats["total_decisions"], 3)
        self.assertGreater(stats["total_estimated_cost_usd"], 0)
        self.assertIn("by_complexity", stats)
        self.assertIn("by_model", stats)


class TestResponseCache(unittest.TestCase):
    """Test suite for response caching system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ResponseCache(max_size=100, ttl_seconds=10)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get("nonexistent_key")
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)
    
    def test_cache_hit(self):
        """Test cache hit returns stored value."""
        self.cache.put("test_key", "test_response", cost_usd=0.001)
        
        result = self.cache.get("test_key")
        
        self.assertEqual(result, "test_response")
        self.assertEqual(self.cache.hits, 1)
    
    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        cache = ResponseCache(max_size=100, ttl_seconds=1)  # 1 second TTL
        
        cache.put("test_key", "test_response", cost_usd=0.001)
        
        # Should hit immediately
        self.assertIsNotNone(cache.get("test_key"))
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should miss after expiration
        self.assertIsNone(cache.get("test_key"))
    
    def test_cache_statistics(self):
        """Test cache statistics calculation."""
        self.cache.put("key1", "response1", cost_usd=0.001)
        self.cache.put("key2", "response2", cost_usd=0.002)
        
        # Hit cache multiple times
        self.cache.get("key1")
        self.cache.get("key1")
        self.cache.get("key2")
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 1)
        self.assertGreater(stats["hit_rate"], 0.5)  # 3/4 = 0.75


class TestEvaluationHarness(unittest.TestCase):
    """Test suite for evaluation harness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.harness = EvaluationHarness()
    
    def test_add_test_case(self):
        """Test adding test cases."""
        test_case = {
            "id": "test_1",
            "agent_name": "security",
            "action_path": "security.firewall",
            "inputs": {},
            "expected_output": "firewall enabled"
        }
        
        self.harness.add_test_case(test_case)
        
        self.assertEqual(len(self.harness.test_cases), 1)
    
    def test_code_based_evaluator(self):
        """Test code-based evaluation."""
        def validate(expected, actual):
            return str(expected) in str(actual), 1.0 if str(expected) in str(actual) else 0.0
        
        evaluator = CodeBasedEvaluator(validate)
        
        test_case = {
            "id": "test_1",
            "agent_name": "security",
            "action_path": "security.firewall",
            "inputs": {},
            "expected_output": "enabled"
        }
        
        result = evaluator.evaluate(test_case, "firewall enabled successfully")
        
        self.assertTrue(result.success)
        self.assertEqual(result.accuracy_score, 1.0)
    
    def test_evaluation_summary(self):
        """Test evaluation summary generation."""
        # Add test cases
        for i in range(5):
            self.harness.add_test_case({
                "id": f"test_{i}",
                "agent_name": "test",
                "action_path": "test.action",
                "inputs": {},
                "expected_output": "success"
            })
        
        # Mock agent function
        def mock_agent(inputs):
            return "success"
        
        # Evaluate
        def validate(expected, actual):
            return expected == actual, 1.0 if expected == actual else 0.0
        
        evaluator = CodeBasedEvaluator(validate)
        summary = self.harness.evaluate(mock_agent, evaluator)
        
        self.assertEqual(summary["total_tests"], 5)
        self.assertEqual(summary["success_rate"], 1.0)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_boot_sequence(self):
        """Test complete boot sequence execution."""
        runtime = ProductionRuntime(
            enable_observability=True,
            enable_model_routing=True,
            enable_caching=True
        )
        
        # Execute a subset of boot sequence
        sequence = [
            "security.access_control",
            "security.encryption",
            "security.firewall"
        ]
        
        summary = runtime.execute_sequence(sequence)
        
        # Should complete with high success rate
        self.assertGreater(summary["success_rate"], 0.5)
        
        # Check metrics were collected
        metrics = runtime.get_metrics()
        self.assertGreater(metrics["runtime"]["agents_loaded"], 0)
    
    def test_reliability_target(self):
        """Test system achieves >99% success rate on critical actions."""
        runtime = ProductionRuntime()
        
        successes = 0
        total = 100
        
        for _ in range(total):
            result = runtime.execute_action("security.access_control")
            if result.success:
                successes += 1
        
        success_rate = successes / total
        
        # Target: >99% success rate for critical actions
        self.assertGreater(success_rate, 0.99, 
                          f"Success rate {success_rate:.1%} below 99% target")
    
    def test_cost_optimization(self):
        """Test model routing provides cost optimization."""
        runtime_with_routing = ProductionRuntime(
            enable_model_routing=True,
            enable_caching=True
        )
        
        runtime_without_routing = ProductionRuntime(
            enable_model_routing=False,
            enable_caching=False
        )
        
        # Execute same sequence with both
        # (In real test, this would execute LLM calls and measure actual costs)
        
        # For now, verify routing infrastructure is in place
        self.assertIsNotNone(runtime_with_routing.model_router)
        self.assertIsNotNone(runtime_with_routing.response_cache)
        self.assertIsNone(runtime_without_routing.model_router)


def run_test_suite():
    """Run the complete test suite with detailed reporting."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProductionRuntime))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestObservability))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseCache))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationHarness))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun:.1%}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)

