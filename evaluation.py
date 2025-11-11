"""
Production Evaluation Harness for Ai:oS Agents.

Implements comprehensive evaluation following 2025 production best practices:
- End-to-end workflow evaluation
- Component-level testing
- LLM-as-judge scoring
- Code-based deterministic checks
- Human annotation queues
- A/B testing framework
- Continuous evaluation on production data

Two-level evaluation strategy:
1. End-to-end: Treat system as black box, measure task completion
2. Component-level: Test individual agent components for precise debugging

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import statistics

LOG = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════

class MetricType(Enum):
    """Types of evaluation metrics."""
    TASK_COMPLETION = "task_completion"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    ERROR_RATE = "error_rate"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    test_id: str
    agent_name: str
    action_path: str
    
    # Inputs and outputs
    inputs: Dict[str, Any]
    expected_output: Any
    actual_output: Any
    
    # Metrics
    success: bool
    accuracy_score: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    
    # Detailed scoring
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "agent_name": self.agent_name,
            "action_path": self.action_path,
            "success": self.success,
            "accuracy_score": self.accuracy_score,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "scores": self.scores,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


# ═══════════════════════════════════════════════════════════════════════
# EVALUATORS
# ═══════════════════════════════════════════════════════════════════════

class BaseEvaluator:
    """Base class for evaluators."""
    
    def evaluate(
        self,
        test_case: Dict[str, Any],
        actual_output: Any
    ) -> EvaluationResult:
        """Evaluate a test case."""
        raise NotImplementedError


class CodeBasedEvaluator(BaseEvaluator):
    """
    Deterministic code-based evaluator.
    
    Most reliable for objective criteria:
    - JSON parsing
    - Format validation
    - Exact matches
    - Regex patterns
    """
    
    def __init__(self, validation_fn: Callable[[Any, Any], Tuple[bool, float]]):
        """
        Initialize with validation function.
        
        Args:
            validation_fn: Function (expected, actual) -> (success, score)
        """
        self.validation_fn = validation_fn
    
    def evaluate(
        self,
        test_case: Dict[str, Any],
        actual_output: Any
    ) -> EvaluationResult:
        """Evaluate using code-based validation."""
        start_time = time.time()
        
        try:
            expected = test_case.get("expected_output")
            success, score = self.validation_fn(expected, actual_output)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                test_id=test_case.get("id", "unknown"),
                agent_name=test_case.get("agent_name", "unknown"),
                action_path=test_case.get("action_path", "unknown"),
                inputs=test_case.get("inputs", {}),
                expected_output=expected,
                actual_output=actual_output,
                success=success,
                accuracy_score=score,
                latency_ms=latency_ms,
                scores={"code_based": score}
            )
        except Exception as exc:
            return EvaluationResult(
                test_id=test_case.get("id", "unknown"),
                agent_name=test_case.get("agent_name", "unknown"),
                action_path=test_case.get("action_path", "unknown"),
                inputs=test_case.get("inputs", {}),
                expected_output=test_case.get("expected_output"),
                actual_output=actual_output,
                success=False,
                accuracy_score=0.0,
                error=str(exc),
                error_type=type(exc).__name__
            )


class LLMAsJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-judge evaluator for subjective criteria.
    
    Cost-effective at scale but requires validation against human judgment.
    Good for:
    - Semantic similarity
    - Quality assessment
    - Coherence
    - Helpfulness
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        rubric: Optional[str] = None
    ):
        """
        Initialize LLM judge.
        
        Args:
            model: LLM model to use for judging
            rubric: Scoring rubric (criteria and scale)
        """
        self.model = model
        self.rubric = rubric or self._default_rubric()
    
    def _default_rubric(self) -> str:
        """Default scoring rubric."""
        return """
        Score the agent output on a scale of 0-10:
        10: Perfect - meets all requirements
        7-9: Good - meets most requirements with minor issues
        4-6: Fair - partially meets requirements
        1-3: Poor - fails to meet requirements
        0: Failure - completely incorrect or error
        """
    
    def evaluate(
        self,
        test_case: Dict[str, Any],
        actual_output: Any
    ) -> EvaluationResult:
        """Evaluate using LLM judge (simulated for MVP)."""
        start_time = time.time()
        
        try:
            # In production, this would call actual LLM API
            # For MVP, we simulate with heuristic scoring
            expected = test_case.get("expected_output")
            score = self._simulate_llm_judge(expected, actual_output)
            
            latency_ms = (time.time() - start_time) * 1000
            success = score >= 7.0  # Threshold for success
            
            return EvaluationResult(
                test_id=test_case.get("id", "unknown"),
                agent_name=test_case.get("agent_name", "unknown"),
                action_path=test_case.get("action_path", "unknown"),
                inputs=test_case.get("inputs", {}),
                expected_output=expected,
                actual_output=actual_output,
                success=success,
                accuracy_score=score / 10.0,  # Normalize to 0-1
                latency_ms=latency_ms,
                scores={"llm_judge": score},
                metadata={"model": self.model}
            )
        except Exception as exc:
            return EvaluationResult(
                test_id=test_case.get("id", "unknown"),
                agent_name=test_case.get("agent_name", "unknown"),
                action_path=test_case.get("action_path", "unknown"),
                inputs=test_case.get("inputs", {}),
                expected_output=test_case.get("expected_output"),
                actual_output=actual_output,
                success=False,
                accuracy_score=0.0,
                error=str(exc),
                error_type=type(exc).__name__
            )
    
    def _simulate_llm_judge(self, expected: Any, actual: Any) -> float:
        """
        Simulate LLM judge scoring.
        In production, replace with actual API call to GPT-4o-mini.
        """
        # Simple heuristic: string similarity
        if expected is None or actual is None:
            return 0.0
        
        expected_str = str(expected).lower()
        actual_str = str(actual).lower()
        
        # Check for key terms
        if expected_str in actual_str or actual_str in expected_str:
            return 9.0  # High score for containing expected content
        
        # Check for partial match
        expected_words = set(expected_str.split())
        actual_words = set(actual_str.split())
        overlap = len(expected_words & actual_words)
        total = len(expected_words | actual_words)
        
        if total > 0:
            similarity = overlap / total
            return similarity * 10.0
        
        return 3.0  # Default fair score


class HumanEvaluator(BaseEvaluator):
    """
    Human evaluator for annotation queues.
    
    Most expensive but highest quality.
    Used for:
    - Validating LLM judge accuracy
    - Edge cases
    - Safety-critical decisions
    """
    
    def evaluate(
        self,
        test_case: Dict[str, Any],
        actual_output: Any
    ) -> EvaluationResult:
        """Queue for human evaluation."""
        # In production, this would add to annotation queue
        # For MVP, we log and return pending status
        LOG.info(
            f"Test case {test_case.get('id')} queued for human evaluation"
        )
        
        return EvaluationResult(
            test_id=test_case.get("id", "unknown"),
            agent_name=test_case.get("agent_name", "unknown"),
            action_path=test_case.get("action_path", "unknown"),
            inputs=test_case.get("inputs", {}),
            expected_output=test_case.get("expected_output"),
            actual_output=actual_output,
            success=False,  # Unknown until human review
            accuracy_score=0.0,
            metadata={"status": "pending_human_review"}
        )


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """
    Comprehensive evaluation harness for agent testing.
    
    Supports:
    - Multiple evaluator types
    - Test case management
    - Result aggregation
    - Continuous evaluation
    """
    
    def __init__(
        self,
        test_cases_path: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        self.test_cases_path = test_cases_path
        self.results_dir = results_dir or Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[EvaluationResult] = []
        
        if test_cases_path and test_cases_path.exists():
            self.load_test_cases(test_cases_path)
        
        LOG.info(f"EvaluationHarness initialized with {len(self.test_cases)} test cases")
    
    def load_test_cases(self, path: Path) -> None:
        """Load test cases from JSON file."""
        with path.open() as f:
            self.test_cases = json.load(f)
        LOG.info(f"Loaded {len(self.test_cases)} test cases from {path}")
    
    def add_test_case(self, test_case: Dict[str, Any]) -> None:
        """Add a test case."""
        self.test_cases.append(test_case)
    
    def evaluate(
        self,
        agent_fn: Callable,
        evaluator: BaseEvaluator,
        test_filter: Optional[Callable[[Dict], bool]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on test cases.
        
        Args:
            agent_fn: Function to execute agent action
            evaluator: Evaluator to score results
            test_filter: Optional filter for test cases
        
        Returns:
            Evaluation summary with metrics
        """
        test_cases = self.test_cases
        if test_filter:
            test_cases = [tc for tc in test_cases if test_filter(tc)]
        
        LOG.info(f"Running evaluation on {len(test_cases)} test cases")
        results = []
        
        for test_case in test_cases:
            try:
                # Execute agent
                actual_output = agent_fn(test_case["inputs"])
                
                # Evaluate
                result = evaluator.evaluate(test_case, actual_output)
                results.append(result)
                
                status = "✓" if result.success else "✗"
                LOG.info(
                    f"  {status} {test_case['id']}: "
                    f"accuracy={result.accuracy_score:.2%}, "
                    f"latency={result.latency_ms:.1f}ms"
                )
            except Exception as exc:
                LOG.error(f"  ✗ {test_case['id']}: {exc}")
                results.append(EvaluationResult(
                    test_id=test_case["id"],
                    agent_name=test_case.get("agent_name", "unknown"),
                    action_path=test_case.get("action_path", "unknown"),
                    inputs=test_case["inputs"],
                    expected_output=test_case.get("expected_output"),
                    actual_output=None,
                    success=False,
                    error=str(exc),
                    error_type=type(exc).__name__
                ))
        
        self.results.extend(results)
        return self.get_summary(results)
    
    def get_summary(self, results: Optional[List[EvaluationResult]] = None) -> Dict[str, Any]:
        """Get evaluation summary with aggregate metrics."""
        results = results or self.results
        
        if not results:
            return {"total": 0, "message": "No results to summarize"}
        
        total = len(results)
        successes = sum(1 for r in results if r.success)
        
        accuracies = [r.accuracy_score for r in results]
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        costs = [r.cost_usd for r in results if r.cost_usd > 0]
        
        return {
            "total_tests": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "accuracy": {
                "mean": statistics.mean(accuracies) if accuracies else 0.0,
                "median": statistics.median(accuracies) if accuracies else 0.0,
                "min": min(accuracies) if accuracies else 0.0,
                "max": max(accuracies) if accuracies else 0.0,
            },
            "latency_ms": {
                "mean": statistics.mean(latencies) if latencies else 0.0,
                "median": statistics.median(latencies) if latencies else 0.0,
                "p95": self._percentile(latencies, 95) if latencies else 0.0,
            },
            "cost_usd": {
                "total": sum(costs),
                "mean": statistics.mean(costs) if costs else 0.0,
            }
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def export_results(self, filename: Optional[str] = None) -> Path:
        """Export results to JSON file."""
        filename = filename or f"evaluation_{int(time.time())}.json"
        output_path = self.results_dir / filename
        
        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results]
        }
        
        output_path.write_text(json.dumps(data, indent=2))
        LOG.info(f"Exported {len(self.results)} results to {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# A/B TESTING FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ABTestVariant:
    """A/B test variant configuration."""
    name: str
    agent_fn: Callable
    traffic_split: float = 0.5  # Fraction of traffic to route here


class ABTest:
    """
    A/B testing framework for prompt optimization.
    
    Enables:
    - Testing different prompts
    - Comparing model versions
    - Optimizing agent parameters
    """
    
    def __init__(self, name: str, variants: List[ABTestVariant]):
        if not variants or len(variants) < 2:
            raise ValueError("At least 2 variants required for A/B test")
        
        self.name = name
        self.variants = variants
        self.results_by_variant: Dict[str, List[EvaluationResult]] = {
            v.name: [] for v in variants
        }
    
    def run(
        self,
        test_cases: List[Dict[str, Any]],
        evaluator: BaseEvaluator
    ) -> Dict[str, Any]:
        """
        Run A/B test on test cases.
        
        Returns:
            Comparison summary between variants
        """
        LOG.info(f"Running A/B test '{self.name}' with {len(self.variants)} variants")
        
        for test_case in test_cases:
            # Route to variant based on traffic split
            variant = self._route_variant()
            
            try:
                actual_output = variant.agent_fn(test_case["inputs"])
                result = evaluator.evaluate(test_case, actual_output)
                self.results_by_variant[variant.name].append(result)
            except Exception as exc:
                LOG.error(f"Variant {variant.name} failed: {exc}")
        
        return self.get_comparison()
    
    def _route_variant(self) -> ABTestVariant:
        """Route to variant based on traffic split."""
        import random
        r = random.random()
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.traffic_split
            if r <= cumulative:
                return variant
        return self.variants[-1]
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison between variants."""
        comparison = {}
        
        for variant in self.variants:
            results = self.results_by_variant[variant.name]
            if not results:
                continue
            
            accuracies = [r.accuracy_score for r in results]
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            successes = sum(1 for r in results if r.success)
            
            comparison[variant.name] = {
                "total_tests": len(results),
                "success_rate": successes / len(results) if results else 0.0,
                "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
                "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            }
        
        return comparison


# ═══════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """Demonstrate evaluation harness."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PRODUCTION EVALUATION HARNESS - DEMONSTRATION                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create sample test cases
    test_cases = [
        {
            "id": "test_1",
            "agent_name": "security",
            "action_path": "security.firewall",
            "inputs": {"command": "check_status"},
            "expected_output": "firewall enabled"
        },
        {
            "id": "test_2",
            "agent_name": "security",
            "action_path": "security.firewall",
            "inputs": {"command": "enable"},
            "expected_output": "firewall enabled successfully"
        },
        {
            "id": "test_3",
            "agent_name": "security",
            "action_path": "security.encryption",
            "inputs": {"check": "status"},
            "expected_output": "encryption enabled"
        }
    ]
    
    # Create harness
    harness = EvaluationHarness()
    for tc in test_cases:
        harness.add_test_case(tc)
    
    # Mock agent function
    def mock_agent(inputs):
        if "firewall" in str(inputs):
            return "firewall enabled"
        return "encryption enabled"
    
    # Evaluate with code-based evaluator
    def validate(expected, actual):
        match = str(expected).lower() in str(actual).lower()
        return match, 1.0 if match else 0.0
    
    evaluator = CodeBasedEvaluator(validate)
    summary = harness.evaluate(mock_agent, evaluator)
    
    print("Evaluation Summary:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Mean accuracy: {summary['accuracy']['mean']:.1%}")
    print(f"  Mean latency: {summary['latency_ms']['mean']:.1f}ms")
    print()
    
    # Export results
    output_path = harness.export_results()
    print(f"Results exported to: {output_path}")


if __name__ == "__main__":
    _demo()

