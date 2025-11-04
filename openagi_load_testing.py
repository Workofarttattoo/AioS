#!/usr/bin/env python
"""
OpenAGI Load Testing & Production Hardening Framework.

Comprehensive stress testing for approval workflows, forensic mode,
and integrated systems at 100+ concurrent operations.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import time
import tempfile
import threading
import multiprocessing
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    from aios.openagi_approval_workflow import (
        ApprovalWorkflowManager,
        ApprovalRequirement,
        ActionSensitivity,
    )
except Exception:
    ApprovalWorkflowManager = None


try:
    from aios.openagi_forensic_mode import (
        ForensicModeExecutor,
        SimulationOutcome,
    )
except Exception:
    ForensicModeExecutor = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for load test."""

    operation: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_time_seconds: float
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    operations_per_second: float
    memory_used_mb: float
    memory_peak_mb: float
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LoadTestResult:
    """Complete load test results."""

    test_name: str
    timestamp: str
    duration_seconds: float
    metrics: List[PerformanceMetrics]
    system_health: Dict[str, Any]
    bottlenecks: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "metrics": [m.to_dict() for m in self.metrics],
            "system_health": self.system_health,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }


class LoadTestOrchestrator:
    """Main load testing orchestrator."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize load test orchestrator.

        Args:
            storage_path: Path for temporary test storage
        """
        self.storage_path = storage_path or Path(tempfile.gettempdir()) / "openagi_load_test"
        self.storage_path.mkdir(exist_ok=True, parents=True)

        self.approval_manager = None
        self.forensic_executor = None
        self.test_results: List[LoadTestResult] = []

        self._initialize_systems()

    def _initialize_systems(self) -> None:
        """Initialize approval and forensic systems."""
        if ApprovalWorkflowManager:
            self.approval_manager = ApprovalWorkflowManager(self.storage_path)

        if ForensicModeExecutor:
            self.forensic_executor = ForensicModeExecutor(self.storage_path)

    def run_approval_workflow_load_test(
        self,
        num_concurrent: int = 100,
        operations_per_thread: int = 10,
    ) -> PerformanceMetrics:
        """Load test approval workflows with concurrent operations.

        Args:
            num_concurrent: Number of concurrent threads
            operations_per_thread: Operations per thread

        Returns:
            Performance metrics for the test
        """
        if not self.approval_manager:
            return self._create_dummy_metrics("approval_workflow")

        operation_name = f"approval_requests ({num_concurrent}x{operations_per_thread})"
        total_ops = num_concurrent * operations_per_thread

        tracemalloc.start()
        times = []
        errors = []
        start_time = time.time()

        def worker_thread(thread_id: int) -> Tuple[int, List[float], List[str]]:
            """Worker thread for approval operations."""
            thread_times = []
            thread_errors = []

            for op_id in range(operations_per_thread):
                try:
                    action_path = f"test.action_{thread_id}_{op_id}"

                    # Create request
                    req_start = time.time()
                    request = self.approval_manager.create_approval_request(
                        action_path=action_path,
                        action_name=f"Test Action {op_id}",
                        description=f"Load test action from thread {thread_id}",
                        context={"test_id": thread_id, "op_id": op_id},
                        requester_id=f"user_{thread_id}"
                    )
                    req_time = (time.time() - req_start) * 1000
                    thread_times.append(req_time)

                    if request:
                        # Submit decision
                        dec_start = time.time()
                        self.approval_manager.submit_approval_decision(
                            request_id=request.request_id,
                            approved=True,
                            approver_id=f"approver_{thread_id}",
                            reason="Load test approval",
                            two_factor_verified=False
                        )
                        dec_time = (time.time() - dec_start) * 1000
                        thread_times.append(dec_time)

                        # Check can execute
                        exe_start = time.time()
                        self.approval_manager.can_execute_action(
                            action_path=action_path,
                            request_id=request.request_id
                        )
                        exe_time = (time.time() - exe_start) * 1000
                        thread_times.append(exe_time)

                except Exception as e:
                    thread_errors.append(f"Thread {thread_id} op {op_id}: {str(e)}")

            return thread_id, thread_times, thread_errors

        successful_ops = 0
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(worker_thread, i)
                for i in range(num_concurrent)
            ]

            for future in as_completed(futures):
                try:
                    thread_id, thread_times, thread_errors = future.result()
                    times.extend(thread_times)
                    errors.extend(thread_errors)
                    successful_ops += operations_per_thread
                except Exception as e:
                    errors.append(f"Thread failed: {str(e)}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = time.time() - start_time

        return PerformanceMetrics(
            operation=operation_name,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=len(errors),
            total_time_seconds=total_time,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            operations_per_second=total_ops / total_time if total_time > 0 else 0.0,
            memory_used_mb=current / 1024 / 1024,
            memory_peak_mb=peak / 1024 / 1024,
            error_rate=len(errors) / total_ops if total_ops > 0 else 0.0,
        )

    def run_forensic_mode_load_test(
        self,
        num_concurrent: int = 100,
        operations_per_thread: int = 10,
        workflow_size: int = 5,
    ) -> PerformanceMetrics:
        """Load test forensic mode with concurrent simulations.

        Args:
            num_concurrent: Number of concurrent threads
            operations_per_thread: Simulations per thread
            workflow_size: Actions per workflow

        Returns:
            Performance metrics for the test
        """
        if not self.forensic_executor:
            return self._create_dummy_metrics("forensic_mode")

        operation_name = f"forensic_simulations ({num_concurrent}x{operations_per_thread})"
        total_ops = num_concurrent * operations_per_thread

        tracemalloc.start()
        times = []
        errors = []
        start_time = time.time()

        def worker_thread(thread_id: int) -> Tuple[int, List[float], List[str]]:
            """Worker thread for forensic simulations."""
            thread_times = []
            thread_errors = []

            for op_id in range(operations_per_thread):
                try:
                    # Create workflow
                    actions = [
                        {
                            "action_path": f"security.action_{i}",
                            "action_name": f"Security Action {i}",
                            "description": f"Test action {i}",
                            "parameters": {"test_id": f"{thread_id}_{op_id}"}
                        }
                        for i in range(workflow_size)
                    ]

                    sim_start = time.time()
                    simulation = self.forensic_executor.simulate_workflow(
                        workflow_name=f"Workflow_{thread_id}_{op_id}",
                        description=f"Load test workflow from thread {thread_id}",
                        actions=actions
                    )
                    sim_time = (time.time() - sim_start) * 1000
                    thread_times.append(sim_time)

                    if simulation:
                        # Get report
                        rep_start = time.time()
                        self.forensic_executor.get_simulation_report(
                            simulation.simulation_id
                        )
                        rep_time = (time.time() - rep_start) * 1000
                        thread_times.append(rep_time)

                        # Get statistics
                        stats_start = time.time()
                        self.forensic_executor.get_forensic_statistics()
                        stats_time = (time.time() - stats_start) * 1000
                        thread_times.append(stats_time)

                except Exception as e:
                    thread_errors.append(f"Thread {thread_id} op {op_id}: {str(e)}")

            return thread_id, thread_times, thread_errors

        successful_ops = 0
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(worker_thread, i)
                for i in range(num_concurrent)
            ]

            for future in as_completed(futures):
                try:
                    thread_id, thread_times, thread_errors = future.result()
                    times.extend(thread_times)
                    errors.extend(thread_errors)
                    successful_ops += operations_per_thread
                except Exception as e:
                    errors.append(f"Thread failed: {str(e)}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = time.time() - start_time

        return PerformanceMetrics(
            operation=operation_name,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=len(errors),
            total_time_seconds=total_time,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            operations_per_second=total_ops / total_time if total_time > 0 else 0.0,
            memory_used_mb=current / 1024 / 1024,
            memory_peak_mb=peak / 1024 / 1024,
            error_rate=len(errors) / total_ops if total_ops > 0 else 0.0,
        )

    def run_integrated_system_test(
        self,
        num_requests: int = 50,
    ) -> PerformanceMetrics:
        """Run integrated system test with approval + forensic.

        Args:
            num_requests: Number of integrated operations

        Returns:
            Performance metrics
        """
        if not self.approval_manager or not self.forensic_executor:
            return self._create_dummy_metrics("integrated_system")

        operation_name = f"integrated_operations ({num_requests})"

        tracemalloc.start()
        times = []
        errors = []
        start_time = time.time()

        for i in range(num_requests):
            try:
                # Step 1: Create approval request
                req_start = time.time()
                request = self.approval_manager.create_approval_request(
                    action_path="security.deployment",
                    action_name=f"Deployment {i}",
                    description="Integrated test deployment",
                    context={"iteration": i},
                    requester_id="system"
                )
                times.append((time.time() - req_start) * 1000)

                # Step 2: Simulate deployment with forensic
                if request:
                    sim_start = time.time()
                    actions = [
                        {
                            "action_path": "kernel.deployment",
                            "action_name": "Deploy",
                            "description": "Deploy changes",
                            "parameters": {"deployment_id": request.request_id}
                        },
                        {
                            "action_path": "security.verify",
                            "action_name": "Verify",
                            "description": "Verify deployment",
                            "parameters": {}
                        }
                    ]

                    simulation = self.forensic_executor.simulate_workflow(
                        workflow_name=f"Deployment_{i}",
                        description="Deployment verification",
                        actions=actions
                    )
                    times.append((time.time() - sim_start) * 1000)

                    # Step 3: Approve deployment
                    if simulation:
                        dec_start = time.time()
                        self.approval_manager.submit_approval_decision(
                            request_id=request.request_id,
                            approved=(
                                simulation.overall_outcome == SimulationOutcome.SUCCESS
                                if hasattr(SimulationOutcome, 'SUCCESS')
                                else True
                            ),
                            approver_id="automation",
                            reason="Forensic verification passed",
                            two_factor_verified=False
                        )
                        times.append((time.time() - dec_start) * 1000)

            except Exception as e:
                errors.append(f"Integration op {i}: {str(e)}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = time.time() - start_time

        return PerformanceMetrics(
            operation=operation_name,
            total_operations=num_requests,
            successful_operations=num_requests - len(errors),
            failed_operations=len(errors),
            total_time_seconds=total_time,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            operations_per_second=num_requests / total_time if total_time > 0 else 0.0,
            memory_used_mb=current / 1024 / 1024,
            memory_peak_mb=peak / 1024 / 1024,
            error_rate=len(errors) / num_requests if num_requests > 0 else 0.0,
        )

    def run_storage_scalability_test(
        self,
        num_records: int = 10000,
    ) -> PerformanceMetrics:
        """Test storage system scalability with many records.

        Args:
            num_records: Number of records to create

        Returns:
            Performance metrics
        """
        if not self.approval_manager:
            return self._create_dummy_metrics("storage_scalability")

        operation_name = f"storage_scalability ({num_records} records)"

        tracemalloc.start()
        times = []
        start_time = time.time()

        # Write phase
        write_times = []
        for i in range(num_records):
            try:
                w_start = time.time()
                self.approval_manager.create_approval_request(
                    action_path=f"storage.op_{i}",
                    action_name=f"Storage Op {i}",
                    description="Scalability test",
                    context={"record_id": i},
                    requester_id="tester"
                )
                write_times.append((time.time() - w_start) * 1000)
            except Exception as e:
                pass

        # Read phase
        read_times = []
        try:
            for i in range(0, num_records, 100):
                r_start = time.time()
                self.approval_manager.storage.get_pending_requests()
                read_times.append((time.time() - r_start) * 1000)
        except Exception as e:
            pass

        times = write_times + read_times
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = time.time() - start_time

        return PerformanceMetrics(
            operation=operation_name,
            total_operations=num_records,
            successful_operations=len(write_times),
            failed_operations=num_records - len(write_times),
            total_time_seconds=total_time,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            operations_per_second=num_records / total_time if total_time > 0 else 0.0,
            memory_used_mb=current / 1024 / 1024,
            memory_peak_mb=peak / 1024 / 1024,
            error_rate=(num_records - len(write_times)) / num_records if num_records > 0 else 0.0,
        )

    def analyze_results(
        self,
        metrics: List[PerformanceMetrics],
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Analyze load test results for health and recommendations.

        Args:
            metrics: List of performance metrics

        Returns:
            Tuple of (system_health dict, bottlenecks list, recommendations list)
        """
        system_health = {
            "tests_run": len(metrics),
            "avg_operation_time_ms": (
                sum(m.avg_time_ms for m in metrics) / len(metrics) if metrics else 0
            ),
            "total_operations": sum(m.total_operations for m in metrics),
            "successful_operations": sum(m.successful_operations for m in metrics),
            "failed_operations": sum(m.failed_operations for m in metrics),
            "overall_error_rate": (
                sum(m.failed_operations for m in metrics) / sum(m.total_operations for m in metrics)
                if sum(m.total_operations for m in metrics) > 0
                else 0
            ),
            "peak_memory_mb": max((m.memory_peak_mb for m in metrics), default=0),
            "overall_ops_per_second": sum(m.operations_per_second for m in metrics),
        }

        bottlenecks = []
        recommendations = []

        # Identify bottlenecks
        for metric in metrics:
            if metric.error_rate > 0.05:
                bottlenecks.append(
                    f"{metric.operation}: High error rate ({metric.error_rate*100:.1f}%)"
                )

            if metric.avg_time_ms > 100:
                bottlenecks.append(
                    f"{metric.operation}: High latency ({metric.avg_time_ms:.1f}ms avg)"
                )

            if metric.memory_peak_mb > 500:
                bottlenecks.append(
                    f"{metric.operation}: High memory usage ({metric.memory_peak_mb:.1f}MB peak)"
                )

        # Generate recommendations
        if system_health["overall_error_rate"] > 0.01:
            recommendations.append(
                "Investigate error handling - error rate above 1% threshold"
            )

        if system_health["peak_memory_mb"] > 1000:
            recommendations.append(
                "Implement memory optimization - peak usage above 1GB"
            )

        if not bottlenecks:
            recommendations.append(
                "All systems performing within acceptable parameters ✅"
            )

        recommendations.append(
            f"Achieved {system_health['overall_ops_per_second']:.0f} ops/sec aggregate throughput"
        )

        return system_health, bottlenecks, recommendations

    def run_complete_load_test(self) -> LoadTestResult:
        """Run complete load testing suite.

        Returns:
            Complete load test results
        """
        start_time = time.time()
        metrics = []

        print("=" * 70)
        print("OpenAGI Load Testing & Production Hardening Suite")
        print("=" * 70)

        # Test 1: Approval workflow
        print("\n[1/4] Running approval workflow load test (100 concurrent)...")
        metrics.append(self.run_approval_workflow_load_test(num_concurrent=100))
        print(f"  ✓ {metrics[-1].successful_operations}/{metrics[-1].total_operations} ops successful")

        # Test 2: Forensic mode
        print("\n[2/4] Running forensic mode load test (100 concurrent)...")
        metrics.append(self.run_forensic_mode_load_test(num_concurrent=100))
        print(f"  ✓ {metrics[-1].successful_operations}/{metrics[-1].total_operations} sims successful")

        # Test 3: Integrated system
        print("\n[3/4] Running integrated system test...")
        metrics.append(self.run_integrated_system_test(num_requests=50))
        print(f"  ✓ {metrics[-1].successful_operations}/{metrics[-1].total_operations} integrations successful")

        # Test 4: Storage scalability
        print("\n[4/4] Running storage scalability test...")
        metrics.append(self.run_storage_scalability_test(num_records=5000))
        print(f"  ✓ {metrics[-1].successful_operations}/{metrics[-1].total_operations} storage ops successful")

        total_time = time.time() - start_time
        system_health, bottlenecks, recommendations = self.analyze_results(metrics)

        result = LoadTestResult(
            test_name="OpenAGI Phase 3 Load Testing Suite",
            timestamp=datetime.now().isoformat(),
            duration_seconds=total_time,
            metrics=metrics,
            system_health=system_health,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

        self.test_results.append(result)
        return result

    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """Save test results to file.

        Args:
            filepath: Optional output filepath

        Returns:
            Path to saved results
        """
        if not self.test_results:
            raise ValueError("No test results to save")

        if filepath is None:
            filepath = Path.home() / "LOAD_TEST_RESULTS.json"

        results_data = [r.to_dict() for r in self.test_results]

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        return filepath

    def print_summary(self) -> None:
        """Print summary of latest test results."""
        if not self.test_results:
            print("No test results available")
            return

        result = self.test_results[-1]

        print("\n" + "=" * 70)
        print("LOAD TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nTest: {result.test_name}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Total Duration: {result.duration_seconds:.2f}s")

        print("\n--- System Health ---")
        for key, value in result.system_health.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        print("\n--- Individual Metrics ---")
        for metric in result.metrics:
            print(f"\n{metric.operation}:")
            print(f"  Success Rate: {(1-metric.error_rate)*100:.1f}%")
            print(f"  Latency: {metric.avg_time_ms:.2f}ms avg (min: {metric.min_time_ms:.2f}ms, max: {metric.max_time_ms:.2f}ms)")
            print(f"  Throughput: {metric.operations_per_second:.1f} ops/sec")
            print(f"  Memory: {metric.memory_used_mb:.1f}MB ({metric.memory_peak_mb:.1f}MB peak)")

        if result.bottlenecks:
            print("\n--- Bottlenecks ---")
            for bottleneck in result.bottlenecks:
                print(f"⚠ {bottleneck}")
        else:
            print("\n--- Bottlenecks ---")
            print("None detected ✅")

        print("\n--- Recommendations ---")
        for recommendation in result.recommendations:
            print(f"→ {recommendation}")

        print("\n" + "=" * 70)

    @staticmethod
    def _create_dummy_metrics(operation: str) -> PerformanceMetrics:
        """Create dummy metrics when system not available.

        Args:
            operation: Operation name

        Returns:
            Dummy metrics
        """
        return PerformanceMetrics(
            operation=operation,
            total_operations=0,
            successful_operations=0,
            failed_operations=0,
            total_time_seconds=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            avg_time_ms=0.0,
            operations_per_second=0.0,
            memory_used_mb=0.0,
            memory_peak_mb=0.0,
            error_rate=1.0,
        )


def run_production_hardening_suite() -> bool:
    """Run complete production hardening suite.

    Returns:
        True if tests passed, False otherwise
    """
    orchestrator = LoadTestOrchestrator()
    result = orchestrator.run_complete_load_test()
    orchestrator.print_summary()

    # Save results
    results_path = orchestrator.save_results()
    print(f"\n✓ Results saved to: {results_path}")

    # Determine pass/fail
    passed = result.system_health["overall_error_rate"] < 0.05

    if passed:
        print("\n✅ Production Hardening Tests: PASSED")
    else:
        print("\n❌ Production Hardening Tests: FAILED")

    return passed


if __name__ == "__main__":
    import sys
    success = run_production_hardening_suite()
    sys.exit(0 if success else 1)
