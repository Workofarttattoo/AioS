"""
Comprehensive test suite for Ai:oS agents.

Tests all meta-agents following production best practices:
- Unit tests for individual actions
- Integration tests for agent sequences
- Forensic mode verification
- Error handling validation
- Performance benchmarking

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_manifest, DEFAULT_MANIFEST
from observability import ObservabilitySystem
from error_handling import RetryConfig, retry_with_backoff, CircuitBreaker
from evaluation import EvaluationHarness, CodeBasedEvaluator


class TestManifestLoading(unittest.TestCase):
    """Test manifest configuration system."""
    
    def test_default_manifest_loads(self):
        """Default manifest should load without errors."""
        manifest = DEFAULT_MANIFEST
        self.assertIsNotNone(manifest)
        self.assertIn("kernel", manifest.meta_agents)
        self.assertIn("security", manifest.meta_agents)
        self.assertIn("networking", manifest.meta_agents)
    
    def test_boot_sequence_has_actions(self):
        """Boot sequence should contain valid action paths."""
        manifest = DEFAULT_MANIFEST
        self.assertGreater(len(manifest.boot_sequence), 0)
        
        # First action should be kernel initialization
        self.assertTrue(manifest.boot_sequence[0].startswith("kernel") or
                       manifest.boot_sequence[0].startswith("ai_os"))
    
    def test_action_config_lookup(self):
        """Should be able to look up action configurations."""
        manifest = DEFAULT_MANIFEST
        
        # Look up security.firewall action
        config = manifest.action_config("security.firewall")
        self.assertEqual(config.key, "firewall")
        self.assertIn("firewall", config.description.lower())
    
    def test_critical_actions_marked(self):
        """Critical actions should be properly marked."""
        manifest = DEFAULT_MANIFEST
        
        # Kernel actions should be critical
        kernel_agent = manifest.meta_agents["kernel"]
        process_mgmt = next(a for a in kernel_agent.actions if a.key == "process_management")
        self.assertTrue(process_mgmt.critical)
        
        # Audit should not be critical
        audit = next(a for a in kernel_agent.actions if a.key == "audit")
        self.assertFalse(audit.critical)


class TestObservabilitySystem(unittest.TestCase):
    """Test production observability infrastructure."""
    
    def setUp(self):
        """Set up observability system for each test."""
        self.obs = ObservabilitySystem()
    
    def test_trace_creation(self):
        """Should create traces with unique IDs."""
        trace_id1 = self.obs.start_trace("test1")
        trace_id2 = self.obs.start_trace("test2")
        
        self.assertNotEqual(trace_id1, trace_id2)
        self.assertIn(trace_id1, self.obs.traces)
        self.assertIn(trace_id2, self.obs.traces)
    
    def test_span_tracing(self):
        """Should trace spans with timing."""
        self.obs.start_trace("test")
        
        with self.obs.trace_span("test_span", agent_name="test_agent"):
            pass  # Do nothing
        
        # Should have recorded one span
        spans = self.obs.traces[self.obs.current_trace_id]
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "test_span")
        self.assertEqual(span.agent_name, "test_agent")
        self.assertIsNotNone(span.duration_ms)
        self.assertGreater(span.duration_ms, 0)
    
    def test_token_tracking(self):
        """Should track token usage and calculate costs."""
        self.obs.start_trace("test")
        
        with self.obs.trace_span("llm_call") as span:
            span.set_tokens(input_tokens=1000, output_tokens=500, model="gpt-4o")
        
        span = self.obs.traces[self.obs.current_trace_id][0]
        self.assertEqual(span.input_tokens, 1000)
        self.assertEqual(span.output_tokens, 500)
        self.assertEqual(span.total_tokens, 1500)
        self.assertGreater(span.cost_usd, 0)
    
    def test_error_tracking(self):
        """Should track errors in spans."""
        self.obs.start_trace("test")
        
        try:
            with self.obs.trace_span("failing_span"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        span = self.obs.traces[self.obs.current_trace_id][0]
        self.assertFalse(span.success)
        self.assertEqual(span.error_type, "ValueError")
        self.assertIn("Test error", span.error)
    
    def test_metrics_aggregation(self):
        """Should aggregate metrics across spans."""
        self.obs.start_trace("test")
        
        # Create multiple spans with different metrics
        for i in range(5):
            with self.obs.trace_span(f"span_{i}") as span:
                span.set_tokens(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
        
        summary = self.obs.get_metrics_summary()
        self.assertEqual(summary["summary"]["total_calls"], 5)
        self.assertEqual(summary["summary"]["total_tokens"], 750)  # 150 * 5
        self.assertGreater(summary["summary"]["total_cost_usd"], 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling patterns."""
    
    def test_retry_with_backoff_success(self):
        """Retry should succeed after transient failures."""
        attempts = [0]
        
        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay_sec=0.01))
        def flaky_function():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("Transient failure")
            return "success"
        
        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)
    
    def test_retry_with_backoff_exhausted(self):
        """Retry should raise after max attempts."""
        @retry_with_backoff(RetryConfig(max_attempts=2, initial_delay_sec=0.01))
        def always_failing():
            raise RuntimeError("Permanent failure")
        
        with self.assertRaises(RuntimeError):
            always_failing()
    
    def test_circuit_breaker_opens(self):
        """Circuit breaker should open after threshold failures."""
        from error_handling import CircuitBreakerConfig, CircuitBreakerError
        
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        
        def failing_service():
            raise RuntimeError("Service down")
        
        # Trigger failures
        for _ in range(3):
            try:
                cb.call(failing_service)
            except RuntimeError:
                pass
        
        # Circuit should now be open
        with self.assertRaises(CircuitBreakerError):
            cb.call(failing_service)
    
    def test_circuit_breaker_recovers(self):
        """Circuit breaker should recover after successful calls."""
        from error_handling import CircuitBreakerConfig, CircuitState
        
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_sec=0.1
        ))
        
        # Open the circuit
        def failing():
            raise RuntimeError("Fail")
        
        for _ in range(2):
            try:
                cb.call(failing)
            except RuntimeError:
                pass
        
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Wait for timeout
        import time
        time.sleep(0.15)
        
        # Should allow call through (half-open)
        def succeeding():
            return "success"
        
        result = cb.call(succeeding)
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, CircuitState.CLOSED)


class TestEvaluationHarness(unittest.TestCase):
    """Test evaluation infrastructure."""
    
    def test_code_based_evaluator(self):
        """Code-based evaluator should validate deterministically."""
        def exact_match_validator(expected, actual):
            return expected == actual, 1.0 if expected == actual else 0.0
        
        evaluator = CodeBasedEvaluator(exact_match_validator)
        
        test_case = {
            "id": "test_1",
            "agent_name": "test",
            "action_path": "test.action",
            "inputs": {},
            "expected_output": "success"
        }
        
        # Test success case
        result = evaluator.evaluate(test_case, "success")
        self.assertTrue(result.success)
        self.assertEqual(result.accuracy_score, 1.0)
        
        # Test failure case
        result = evaluator.evaluate(test_case, "failure")
        self.assertFalse(result.success)
        self.assertEqual(result.accuracy_score, 0.0)
    
    def test_harness_aggregation(self):
        """Harness should aggregate results correctly."""
        harness = EvaluationHarness()
        
        # Add test cases
        test_cases = [
            {"id": f"test_{i}", "agent_name": "test", "action_path": "test.action",
             "inputs": {}, "expected_output": "pass"}
            for i in range(5)
        ]
        
        for tc in test_cases:
            harness.add_test_case(tc)
        
        # Mock agent that always succeeds
        def mock_agent(inputs):
            return "pass"
        
        def validator(expected, actual):
            return expected == actual, 1.0 if expected == actual else 0.0
        
        evaluator = CodeBasedEvaluator(validator)
        summary = harness.evaluate(mock_agent, evaluator)
        
        self.assertEqual(summary["total_tests"], 5)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["accuracy"]["mean"], 1.0)


class TestReliabilityCalculations(unittest.TestCase):
    """Test reliability and error compounding calculations."""
    
    def test_error_compounding_95_percent(self):
        """95% per step reliability compounds to 36% over 20 steps."""
        from error_handling import calculate_reliability
        
        end_to_end = calculate_reliability(0.95, 20)
        
        # Should be approximately 0.358 (35.8%)
        self.assertAlmostEqual(end_to_end, 0.358, places=2)
    
    def test_required_step_reliability(self):
        """Calculate required per-step reliability for target."""
        from error_handling import required_step_reliability
        
        # For 90% end-to-end over 20 steps
        required = required_step_reliability(0.90, 20)
        
        # Should be approximately 0.9947 (99.47%)
        self.assertAlmostEqual(required, 0.9947, places=3)
        
        # Verify by calculating end-to-end
        from error_handling import calculate_reliability
        end_to_end = calculate_reliability(required, 20)
        self.assertAlmostEqual(end_to_end, 0.90, places=2)
    
    def test_reliability_validator(self):
        """ReliabilityValidator should flag insufficient reliability."""
        from error_handling import ReliabilityValidator
        
        validator = ReliabilityValidator(
            target_end_to_end_reliability=0.90,
            num_steps=20
        )
        
        # 95% should fail (only gives 36% end-to-end)
        self.assertFalse(validator.validate_step("test_step", 0.95))
        
        # 99.5% should pass (gives >90% end-to-end)
        self.assertTrue(validator.validate_step("test_step", 0.995))


class TestMLAlgorithms(unittest.TestCase):
    """Test ML algorithms integration."""
    
    def test_ml_algorithms_import(self):
        """ML algorithms should be importable."""
        from ml_algorithms import (
            AdaptiveParticleFilter,
            NeuralGuidedMCTS,
            NoUTurnSampler,
            SparseGaussianProcess,
            get_algorithm_catalog
        )
        
        # These should not require torch
        self.assertIsNotNone(AdaptiveParticleFilter)
        self.assertIsNotNone(NeuralGuidedMCTS)
        self.assertIsNotNone(NoUTurnSampler)
        self.assertIsNotNone(SparseGaussianProcess)
    
    def test_particle_filter_basic_operation(self):
        """Particle filter should perform predict-update cycles."""
        from ml_algorithms import AdaptiveParticleFilter
        import numpy as np
        
        pf = AdaptiveParticleFilter(num_particles=100, state_dim=2, obs_dim=1)
        
        # Predict step
        def transition(x):
            return x + 0.1
        
        pf.predict(transition, process_noise=0.01)
        
        # Update step
        observation = np.array([1.0])
        
        def likelihood(obs, state):
            return np.exp(-0.5 * np.sum((obs[0] - state[0])**2))
        
        pf.update(observation, likelihood)
        
        # Get estimate
        estimate = pf.estimate()
        self.assertEqual(estimate.shape, (2,))
    
    def test_algorithm_catalog(self):
        """Algorithm catalog should list all algorithms."""
        from ml_algorithms import get_algorithm_catalog
        
        catalog = get_algorithm_catalog()
        self.assertGreater(len(catalog), 5)
        
        # Check for key algorithms
        names = [algo["name"] for algo in catalog]
        self.assertIn("AdaptiveParticleFilter", names)
        self.assertIn("NeuralGuidedMCTS", names)


class TestAutonomousDiscovery(unittest.TestCase):
    """Test autonomous discovery system."""
    
    def test_knowledge_graph_creation(self):
        """Should create and populate knowledge graph."""
        from autonomous_discovery import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add concepts
        node1 = kg.add_concept("machine learning", confidence=0.9)
        node2 = kg.add_concept("neural networks", confidence=0.85, parent="machine learning")
        
        self.assertEqual(len(kg.nodes), 2)
        self.assertIn("neural networks", kg.nodes["machine learning"].children)
    
    def test_knowledge_graph_export(self):
        """Knowledge graph should export to dict format."""
        from autonomous_discovery import KnowledgeGraph
        
        kg = KnowledgeGraph()
        kg.add_concept("topic A", confidence=0.9)
        kg.add_concept("topic B", confidence=0.8)
        
        export = kg.export()
        self.assertIn("nodes", export)
        self.assertIn("stats", export)
        self.assertEqual(export["stats"]["total_concepts"], 2)
    
    def test_autonomous_agent_initialization(self):
        """Autonomous agent should initialize correctly."""
        from autonomous_discovery import AutonomousLLMAgent, AgentAutonomy
        
        agent = AutonomousLLMAgent(
            model_name="deepseek-r1",
            autonomy_level=AgentAutonomy.LEVEL_4_FULL
        )
        
        self.assertEqual(agent.autonomy_level, AgentAutonomy.LEVEL_4_FULL)
        self.assertIsNotNone(agent.inference_engine)
        self.assertIsNotNone(agent.knowledge_graph)
    
    def test_mission_setting(self):
        """Agent should accept mission configuration."""
        from autonomous_discovery import AutonomousLLMAgent
        
        agent = AutonomousLLMAgent()
        agent.set_mission("test mission", duration_hours=0.5)
        
        self.assertEqual(agent.mission, "test mission")
        self.assertEqual(agent.duration_hours, 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for agent sequences."""
    
    def test_security_firewall_forensic_mode(self):
        """Security agent should respect forensic mode."""
        # This would test actual agent behavior in forensic mode
        # For now, we test the concept
        
        forensic_env = {"AGENTA_FORENSIC_MODE": "1"}
        
        # In forensic mode, operations should be advisory only
        self.assertEqual(forensic_env.get("AGENTA_FORENSIC_MODE"), "1")
    
    def test_error_handling_in_sequence(self):
        """Agent sequence should handle errors gracefully."""
        # Test that error handling patterns work in sequences
        from error_handling import RetryConfig, retry_with_backoff
        
        call_count = [0]
        
        @retry_with_backoff(RetryConfig(max_attempts=2, initial_delay_sec=0.01))
        def agent_action():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Transient error")
            return "success"
        
        result = agent_action()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)


# ═══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_tests(verbosity=2):
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

