"""
Production Observability System for Ai:oS.

Provides comprehensive observability following production best practices:
- Token usage and cost tracking
- Latency monitoring
- Error rate tracking
- Structured logging with traces
- Performance metrics
- Agent-specific telemetry

Based on 2025 production AI agent patterns emphasizing:
- LangSmith-style tracing
- Token-level cost analysis
- Sub-millisecond latency tracking
- Error compounding prevention (95%^20 = 36% problem)

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

LOG = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# COST MODELS (Per Million Tokens)
# ═══════════════════════════════════════════════════════════════════════

class ModelPricing:
    """
    Token pricing for major LLM providers (as of 2025).
    Prices in USD per million tokens.
    """
    PRICING = {
        # OpenAI
        "gpt-4o": {"input": 30.0, "output": 60.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "o3-mini": {"input": 1.0, "output": 4.0},
        
        # Anthropic
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        
        # Google
        "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        
        # Open Source (self-hosted, electricity only)
        "llama-3.x": {"input": 0.01, "output": 0.01},
        "mistral": {"input": 0.01, "output": 0.01},
        "deepseek-r1": {"input": 0.01, "output": 0.01},
    }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost in USD for token usage."""
        pricing = cls.PRICING.get(model, {"input": 10.0, "output": 30.0})  # Default fallback
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════
# TRACE SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class TraceLevel(Enum):
    """Trace event levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TraceEvent:
    """
    Individual trace event in agent execution.
    
    Captures everything needed for debugging and analysis:
    - Timing (start, end, duration)
    - Inputs and outputs
    - Token usage and costs
    - Errors and stack traces
    - Agent context
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    level: TraceLevel
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Agent context
    agent_name: Optional[str] = None
    action_path: Optional[str] = None
    
    # Inputs/Outputs
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None
    cost_usd: float = 0.0
    
    # Performance
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None  # Time to first token
    
    # Error tracking
    success: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def end(self) -> None:
        """Mark trace event as ended."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def set_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> None:
        """Set token usage and calculate cost."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.model = model
        self.cost_usd = ModelPricing.calculate_cost(model, input_tokens, output_tokens)
    
    def set_error(self, error: Exception) -> None:
        """Record error information."""
        self.success = False
        self.error = str(error)
        self.error_type = type(error).__name__
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "level": self.level.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "agent_name": self.agent_name,
            "action_path": self.action_path,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "tags": self.tags
        }


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABILITY SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class ObservabilitySystem:
    """
    Production observability system for Ai:oS agents.
    
    Features:
    - Full execution tracing (LangSmith-style)
    - Token usage and cost tracking
    - Latency monitoring (p50, p95, p99)
    - Error rate tracking
    - Agent performance metrics
    - Structured logging
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.current_trace_id: Optional[str] = None
        self.traces: Dict[str, List[TraceEvent]] = {}
        self.metrics: Dict[str, Any] = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_errors": 0,
            "latencies_ms": [],
            "agent_metrics": {}
        }
        
        LOG.info(f"ObservabilitySystem initialized. Logs: {self.log_dir}")
    
    def start_trace(self, name: str) -> str:
        """Start a new trace session."""
        trace_id = str(uuid.uuid4())
        self.current_trace_id = trace_id
        self.traces[trace_id] = []
        LOG.info(f"Started trace: {trace_id} ({name})")
        return trace_id
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        agent_name: Optional[str] = None,
        action_path: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **kwargs
    ):
        """
        Context manager for tracing a span of execution.
        
        Usage:
            with observability.trace_span("agent.action", agent_name="security"):
                # Do work
                pass
        """
        if not self.current_trace_id:
            self.current_trace_id = self.start_trace(name)
        
        span_id = str(uuid.uuid4())
        event = TraceEvent(
            trace_id=self.current_trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            level=TraceLevel.INFO,
            start_time=time.time(),
            agent_name=agent_name,
            action_path=action_path,
            inputs=kwargs
        )
        
        try:
            yield event
            event.end()
        except Exception as exc:
            event.set_error(exc)
            event.end()
            self.metrics["total_errors"] += 1
            raise
        finally:
            # Record metrics
            if event.duration_ms:
                self.metrics["latencies_ms"].append(event.duration_ms)
            self.metrics["total_calls"] += 1
            self.metrics["total_tokens"] += event.total_tokens
            self.metrics["total_cost_usd"] += event.cost_usd
            
            # Agent-specific metrics
            if agent_name:
                if agent_name not in self.metrics["agent_metrics"]:
                    self.metrics["agent_metrics"][agent_name] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "errors": 0,
                        "latencies_ms": []
                    }
                agent_metrics = self.metrics["agent_metrics"][agent_name]
                agent_metrics["calls"] += 1
                agent_metrics["tokens"] += event.total_tokens
                agent_metrics["cost_usd"] += event.cost_usd
                if not event.success:
                    agent_metrics["errors"] += 1
                if event.duration_ms:
                    agent_metrics["latencies_ms"].append(event.duration_ms)
            
            # Store trace event
            self.traces[self.current_trace_id].append(event)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        latencies = self.metrics["latencies_ms"]
        
        # Calculate percentiles
        p50 = self._percentile(latencies, 50) if latencies else 0
        p95 = self._percentile(latencies, 95) if latencies else 0
        p99 = self._percentile(latencies, 99) if latencies else 0
        
        # Calculate error rate
        error_rate = (
            self.metrics["total_errors"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0.0
        )
        
        # Per-agent summaries
        agent_summaries = {}
        for agent_name, agent_data in self.metrics["agent_metrics"].items():
            agent_latencies = agent_data["latencies_ms"]
            agent_summaries[agent_name] = {
                "calls": agent_data["calls"],
                "tokens": agent_data["tokens"],
                "cost_usd": round(agent_data["cost_usd"], 4),
                "errors": agent_data["errors"],
                "error_rate": agent_data["errors"] / agent_data["calls"] if agent_data["calls"] > 0 else 0.0,
                "p50_latency_ms": self._percentile(agent_latencies, 50) if agent_latencies else 0,
                "p95_latency_ms": self._percentile(agent_latencies, 95) if agent_latencies else 0,
            }
        
        return {
            "summary": {
                "total_calls": self.metrics["total_calls"],
                "total_tokens": self.metrics["total_tokens"],
                "total_cost_usd": round(self.metrics["total_cost_usd"], 4),
                "total_errors": self.metrics["total_errors"],
                "error_rate": round(error_rate, 4),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
            },
            "agents": agent_summaries
        }
    
    def export_trace(self, trace_id: str, output_path: Optional[Path] = None) -> Path:
        """Export trace to JSON file."""
        if trace_id not in self.traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        output_path = output_path or (self.log_dir / f"trace_{trace_id}.json")
        
        trace_data = {
            "trace_id": trace_id,
            "spans": [event.to_dict() for event in self.traces[trace_id]]
        }
        
        output_path.write_text(json.dumps(trace_data, indent=2))
        LOG.info(f"Exported trace to {output_path}")
        return output_path
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_errors": 0,
            "latencies_ms": [],
            "agent_metrics": {}
        }
        LOG.info("Metrics reset")


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════

# Global observability instance for easy access
_global_observability: Optional[ObservabilitySystem] = None


def get_observability() -> ObservabilitySystem:
    """Get or create global observability instance."""
    global _global_observability
    if _global_observability is None:
        _global_observability = ObservabilitySystem()
    return _global_observability


# ═══════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """Demonstrate observability system."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PRODUCTION OBSERVABILITY SYSTEM - DEMONSTRATION                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    obs = ObservabilitySystem()
    
    # Simulate agent execution
    trace_id = obs.start_trace("security_scan")
    
    # Span 1: Firewall check
    with obs.trace_span("security.firewall", agent_name="security", action_path="security.firewall"):
        time.sleep(0.01)  # Simulate work
    
    # Span 2: Encryption check (with token usage)
    with obs.trace_span("security.encryption", agent_name="security", action_path="security.encryption") as span:
        time.sleep(0.02)  # Simulate work
        span.set_tokens(input_tokens=1500, output_tokens=500, model="gpt-4o")
    
    # Span 3: Integrity check (with error)
    try:
        with obs.trace_span("security.integrity", agent_name="security") as span:
            time.sleep(0.01)
            raise RuntimeError("Simulated integrity failure")
    except RuntimeError:
        pass  # Expected
    
    # Get metrics
    print("Metrics Summary:")
    metrics = obs.get_metrics_summary()
    
    summary = metrics["summary"]
    print(f"  Total calls: {summary['total_calls']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Total cost: ${summary['total_cost_usd']:.4f}")
    print(f"  Error rate: {summary['error_rate']:.1%}")
    print(f"  P50 latency: {summary['p50_latency_ms']:.2f}ms")
    print(f"  P95 latency: {summary['p95_latency_ms']:.2f}ms")
    print()
    
    # Agent-specific metrics
    print("Agent Metrics:")
    for agent_name, agent_metrics in metrics["agents"].items():
        print(f"  {agent_name}:")
        print(f"    Calls: {agent_metrics['calls']}")
        print(f"    Tokens: {agent_metrics['tokens']}")
        print(f"    Cost: ${agent_metrics['cost_usd']:.4f}")
        print(f"    Errors: {agent_metrics['errors']} ({agent_metrics['error_rate']:.1%})")
    print()
    
    # Export trace
    trace_path = obs.export_trace(trace_id)
    print(f"Trace exported to: {trace_path}")


if __name__ == "__main__":
    _demo()

