"""
Production Error Handling Patterns for Ai:oS Agents.

Implements production-grade error handling patterns from 2025 best practices:
- Retry with exponential backoff
- Circuit breakers
- Fallback chains
- Error compounding mitigation

Key insight from production deployments:
95% reliability per step → only 36% success over 20 steps!
Production agents require 99.9%+ reliability per component.

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union
import random

LOG = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════
# RETRY WITH EXPONENTIAL BACKOFF
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retry with exponential backoff.
    
    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=5))
        def unstable_operation():
            # May fail transiently
            pass
    
    For transient failures like:
    - Network timeouts
    - Rate limiting
    - Temporary service unavailability
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    
                    if attempt == config.max_attempts - 1:
                        # Final attempt failed
                        LOG.error(
                            f"[{func.__name__}] All {config.max_attempts} attempts failed. "
                            f"Last error: {exc}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay_sec * (config.exponential_base ** attempt),
                        config.max_delay_sec
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    LOG.warning(
                        f"[{func.__name__}] Attempt {attempt + 1}/{config.max_attempts} failed: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(exc, attempt + 1)
                    
                    time.sleep(delay)
            
            # Should never reach here, but satisfy type checker
            raise last_exception
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """Async version of retry_with_backoff decorator."""
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    
                    if attempt == config.max_attempts - 1:
                        LOG.error(
                            f"[{func.__name__}] All {config.max_attempts} attempts failed. "
                            f"Last error: {exc}"
                        )
                        raise
                    
                    delay = min(
                        config.initial_delay_sec * (config.exponential_base ** attempt),
                        config.max_delay_sec
                    )
                    
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    LOG.warning(
                        f"[{func.__name__}] Attempt {attempt + 1}/{config.max_attempts} failed: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(exc, attempt + 1)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes in half-open before closing
    timeout_sec: float = 60.0   # Time in open state before trying half-open
    window_sec: float = 60.0    # Rolling window for failure counting


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, reject calls immediately
    - HALF_OPEN: Testing recovery, allow limited calls
    
    Prevents:
    - Cascading failures
    - Wasting resources on failing services
    - Overloading already-struggling systems
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.opened_at: Optional[float] = None
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if (
                self.opened_at
                and time.time() - self.opened_at >= self.config.timeout_sec
            ):
                LOG.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker OPEN. Service unavailable. "
                    f"Retry after {self.config.timeout_sec}s."
                )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Async version of call."""
        if self.state == CircuitState.OPEN:
            if (
                self.opened_at
                and time.time() - self.opened_at >= self.config.timeout_sec
            ):
                LOG.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker OPEN. Service unavailable."
                )
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure()
            raise
    
    def _record_success(self) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                LOG.info("Circuit breaker transitioning to CLOSED (recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _record_failure(self) -> None:
        """Record failed call."""
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, go back to OPEN
            LOG.warning("Circuit breaker transitioning to OPEN (recovery failed)")
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                LOG.error(
                    f"Circuit breaker transitioning to OPEN "
                    f"({self.failure_count} failures)"
                )
                self.state = CircuitState.OPEN
                self.opened_at = time.time()


# ═══════════════════════════════════════════════════════════════════════
# FALLBACK CHAIN
# ═══════════════════════════════════════════════════════════════════════

class FallbackChain:
    """
    Fallback chain pattern for degraded operation.
    
    Tries strategies in order until one succeeds:
    1. Optimal strategy (e.g., GPT-4o, full features)
    2. Good strategy (e.g., GPT-3.5, reduced features)
    3. Minimal strategy (e.g., rule-based, basic functionality)
    
    Ensures system continues operating even when optimal path fails.
    """
    
    def __init__(self, strategies: List[Callable]):
        """
        Initialize fallback chain.
        
        Args:
            strategies: List of callables to try in order, from best to worst
        """
        if not strategies:
            raise ValueError("At least one strategy required")
        self.strategies = strategies
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute strategies in order until one succeeds.
        
        Returns:
            Result from first successful strategy
        
        Raises:
            Exception from last strategy if all fail
        """
        last_exception = None
        
        for i, strategy in enumerate(self.strategies):
            try:
                LOG.info(
                    f"Trying fallback strategy {i + 1}/{len(self.strategies)}: "
                    f"{strategy.__name__}"
                )
                result = strategy(*args, **kwargs)
                
                if i > 0:
                    LOG.warning(
                        f"Using degraded strategy {i + 1}/{len(self.strategies)}"
                    )
                
                return result
            except Exception as exc:
                last_exception = exc
                LOG.warning(
                    f"Strategy {i + 1}/{len(self.strategies)} failed: {exc}"
                )
                
                if i == len(self.strategies) - 1:
                    # Last strategy failed
                    LOG.error("All fallback strategies failed")
                    raise
        
        # Should never reach here
        raise last_exception
    
    async def execute_async(self, *args, **kwargs) -> Any:
        """Async version of execute."""
        last_exception = None
        
        for i, strategy in enumerate(self.strategies):
            try:
                LOG.info(
                    f"Trying fallback strategy {i + 1}/{len(self.strategies)}"
                )
                result = await strategy(*args, **kwargs)
                
                if i > 0:
                    LOG.warning(f"Using degraded strategy {i + 1}")
                
                return result
            except Exception as exc:
                last_exception = exc
                LOG.warning(f"Strategy {i + 1} failed: {exc}")
                
                if i == len(self.strategies) - 1:
                    LOG.error("All fallback strategies failed")
                    raise
        
        raise last_exception


# ═══════════════════════════════════════════════════════════════════════
# ERROR COMPOUNDING MITIGATION
# ═══════════════════════════════════════════════════════════════════════

def calculate_reliability(step_reliability: float, num_steps: int) -> float:
    """
    Calculate end-to-end reliability for multi-step workflow.
    
    Example: 95% per step reliability over 20 steps = 0.95^20 = 35.8%
    
    This is the fundamental challenge for production autonomous agents!
    """
    return step_reliability ** num_steps


def required_step_reliability(target_reliability: float, num_steps: int) -> float:
    """
    Calculate required per-step reliability to achieve target.
    
    Example: For 90% end-to-end reliability over 20 steps:
    required = 0.90^(1/20) = 99.47% per step!
    """
    return target_reliability ** (1 / num_steps)


class ReliabilityValidator:
    """
    Validates agent workflows meet reliability targets.
    
    Helps prevent error compounding by ensuring each step meets
    the required reliability threshold.
    """
    
    def __init__(
        self,
        target_end_to_end_reliability: float = 0.90,
        num_steps: int = 20
    ):
        self.target_reliability = target_end_to_end_reliability
        self.num_steps = num_steps
        self.required_step_reliability = required_step_reliability(
            target_end_to_end_reliability,
            num_steps
        )
        
        LOG.info(
            f"ReliabilityValidator: Target {target_end_to_end_reliability:.1%} "
            f"over {num_steps} steps requires {self.required_step_reliability:.2%} per step"
        )
    
    def validate_step(self, step_name: str, actual_reliability: float) -> bool:
        """
        Validate that a step meets required reliability.
        
        Returns:
            True if step meets threshold, False otherwise
        """
        meets_threshold = actual_reliability >= self.required_step_reliability
        
        if not meets_threshold:
            LOG.error(
                f"Step '{step_name}' reliability {actual_reliability:.2%} "
                f"below required {self.required_step_reliability:.2%}. "
                f"This will cause error compounding!"
            )
        
        return meets_threshold
    
    def calculate_end_to_end_reliability(
        self,
        step_reliabilities: List[float]
    ) -> float:
        """Calculate end-to-end reliability from step reliabilities."""
        end_to_end = 1.0
        for reliability in step_reliabilities:
            end_to_end *= reliability
        return end_to_end


# ═══════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """Demonstrate error handling patterns."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PRODUCTION ERROR HANDLING PATTERNS - DEMONSTRATION              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # 1. Retry with backoff
    print("1. Retry with Exponential Backoff:")
    print("   " + "─" * 60)
    
    attempt_count = [0]
    
    @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay_sec=0.1))
    def flaky_operation():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError(f"Transient failure #{attempt_count[0]}")
        return "Success!"
    
    result = flaky_operation()
    print(f"   Result: {result} (took {attempt_count[0]} attempts)")
    print()
    
    # 2. Circuit breaker
    print("2. Circuit Breaker:")
    print("   " + "─" * 60)
    
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, timeout_sec=1.0))
    
    def failing_service():
        raise RuntimeError("Service unavailable")
    
    # Trigger circuit breaker
    for i in range(5):
        try:
            cb.call(failing_service)
        except (RuntimeError, CircuitBreakerError) as exc:
            status = "OPEN" if isinstance(exc, CircuitBreakerError) else "CLOSED"
            print(f"   Call {i + 1}: {exc.__class__.__name__} (circuit: {status})")
    
    print()
    
    # 3. Fallback chain
    print("3. Fallback Chain:")
    print("   " + "─" * 60)
    
    def optimal_strategy():
        raise RuntimeError("Optimal unavailable")
    
    def good_strategy():
        raise RuntimeError("Good unavailable")
    
    def minimal_strategy():
        return "Minimal strategy result"
    
    chain = FallbackChain([optimal_strategy, good_strategy, minimal_strategy])
    result = chain.execute()
    print(f"   Result: {result}")
    print()
    
    # 4. Reliability calculation
    print("4. Error Compounding Analysis:")
    print("   " + "─" * 60)
    
    for step_reliability in [0.95, 0.99, 0.999]:
        end_to_end = calculate_reliability(step_reliability, 20)
        print(
            f"   {step_reliability:.1%} per step over 20 steps = "
            f"{end_to_end:.1%} end-to-end"
        )
    
    print()
    print("   For 90% end-to-end over 20 steps:")
    required = required_step_reliability(0.90, 20)
    print(f"   Required per-step reliability: {required:.2%}")
    print()


if __name__ == "__main__":
    _demo()

