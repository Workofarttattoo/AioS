"""
Probabilistic forecasting and quantum simulation utilities for AgentaOS.

The Oracle module keeps all computation in-memory so the operating system can
operate in forensic/immutable environments without mutating host state.  The
implementations favour transparent, explainable scoring over heavyweight
dependencies so the runtime remains portable.

Enhanced with 23 quantum algorithms including HHL and Schrödinger dynamics
for exponential speedup in forecasting and linear system solving.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# Import quantum suite if available
try:
    from .quantum_hhl_algorithm import hhl_linear_system_solver
    from .quantum_schrodinger_dynamics import quantum_dynamics_forecasting
    from .quantum_advanced_synthesis import (
        quantum_temporal_forecasting,
        quantum_kalman_filtering,
    )
    QUANTUM_ENHANCED = True
except ImportError:
    QUANTUM_ENHANCED = False


MAX_QUBITS = 15


@dataclass
class ForecastResult:
    probability: float
    summary: str
    signals: Dict[str, float]
    guidance: List[str]


@dataclass
class QuantumResult:
    qubits: int
    measurements: Dict[str, float]
    entropy: float
    narrative: str


class ProbabilisticOracle:
    """
    Lightweight probabilistic reasoning engine.

    The oracle ingests telemetry stored on the ExecutionContext and emits
    forecast probabilities, risk heatmaps, and quantum-inspired projections.
    """

    def __init__(self, forensic_mode: bool = False):
        self.forensic_mode = forensic_mode

    def forecast(self, telemetry: Dict[str, dict]) -> ForecastResult:
        features = self._collect_features(telemetry)
        load_signal = features["load"]
        memory_signal = features["memory"]
        provider_health = features["provider"]
        container_pressure = features["container"]
        network_signal = features["network"]
        dns_signal = features["dns"]
        apps_signal = features["apps"]

        alpha = 1.0
        beta = 1.0
        weighted_signals = [
            (load_signal, 6.0),
            (memory_signal, 4.0),
            (provider_health, 3.0),
            (max(0.0, 1 - container_pressure), 3.0),
            (network_signal, 2.5),
            (dns_signal, 2.0),
            (apps_signal, 2.0),
        ]
        for value, weight in weighted_signals:
            alpha += value * weight
            beta += (1 - value) * weight
        probability = alpha / (alpha + beta)

        guidance = []
        if probability > 0.8:
            guidance.append("High probability of resource pressure; prepare scale-out playbooks.")
        elif probability > 0.6:
            guidance.append("Moderate probability of contention; validate standby capacity.")
        else:
            guidance.append("Low probability of pressure; maintain forensic observation.")
        if apps_signal < 0.4:
            guidance.append("Application supervisor reports low success rate; review managed tools.")

        if self.forensic_mode:
            guidance.append("Forensic mode active: defer any automated remediation.")

        summary = (
            "Forecast probability integrates system load, memory telemetry, provider health, container pressure, and application signals "
            f"to estimate short-term resource contention at {probability:.2%}."
        )

        signals = {
            "load_signal": load_signal,
            "memory_signal": memory_signal,
            "provider_health": provider_health,
            "container_pressure": container_pressure,
            "network_signal": network_signal,
            "dns_signal": dns_signal,
            "apps_signal": apps_signal,
        }
        return ForecastResult(probability=probability, summary=summary, signals=signals, guidance=guidance)

    def risk_assessment(self, telemetry: Dict[str, dict]) -> ForecastResult:
        features = self._collect_features(telemetry)
        firewall_status = features["firewall"]
        process_anomalies = features["process"]
        backup_health = features["backup"]
        audit_signal = features["audit"]
        apps_success = features["apps"]

        alpha = 1.0
        beta = 1.0
        weighted_risks = [
            (1 - firewall_status, 5.0),
            (process_anomalies, 4.0),
            (1 - backup_health, 3.0),
            (1 - audit_signal, 3.0),
        ]
        for value, weight in weighted_risks:
            alpha += value * weight
            beta += (1 - value) * weight
        residual_risk = alpha / (alpha + beta)

        guidance = []
        if residual_risk > 0.7:
            guidance.append("Elevated security residual risk; recommend manual log review.")
        elif residual_risk > 0.4:
            guidance.append("Moderate residual risk; monitor anomalous processes.")
        else:
            guidance.append("Low residual risk based on current telemetry.")
        if apps_success < 0.5:
            guidance.append("Investigate failed supervised applications for security impact.")

        if self.forensic_mode:
            guidance.append("Ensure evidence retention remains immutable; avoid altering host artefacts.")

        summary = (
            "Security residual risk calculated from firewall posture, anomalous process ratios, backup health, audit events, and application success "
            f"evaluates to {residual_risk:.2%}."
        )

        signals = {
            "firewall_health": firewall_status,
            "process_anomaly_ratio": process_anomalies,
            "backup_health": backup_health,
            "audit_signal": audit_signal,
            "apps_success": apps_success,
        }
        return ForecastResult(probability=residual_risk, summary=summary, signals=signals, guidance=guidance)

    def quantum_projection(
        self,
        qubits: int = 10,
        telemetry: Optional[Dict[str, dict]] = None,
        seed: Optional[int] = None,
    ) -> QuantumResult:
        qubits = max(1, min(MAX_QUBITS, qubits))
        features = self._collect_features(telemetry or {})
        forecast_hint = (
            features["load"] * 0.4
            + features["memory"] * 0.2
            + features["provider"] * 0.2
            + (1 - features["container"]) * 0.2
        )
        residual_hint = (
            (1 - features["firewall"]) * 0.3
            + features["process"] * 0.25
            + (1 - features["backup"]) * 0.25
            + (1 - features["audit"]) * 0.2
        )
        simulator = QuantumCircuitSimulator(qubits, seed=seed)
        simulator.apply_all_hadamard()
        simulator.apply_entanglement_chain()
        simulator.apply_feature_phases(features, forecast_hint, residual_hint)
        simulator.renormalize()
        measurements = simulator.measure_distribution(max_states=32)
        entropy = simulator.entropy()

        narrative = (
            f"Simulated {qubits}-qubit circuit with forecast bias {forecast_hint:.2f} and residual risk {residual_hint:.2f}. "
            f"Entropy of resulting state: {entropy:.2f} bits."
        )
        if self.forensic_mode:
            narrative += " Forensic mode ensures amplitudes are retained in-memory only."

        return QuantumResult(qubits=qubits, measurements=measurements, entropy=entropy, narrative=narrative)

    def adaptive_guidance(self, forecast: ForecastResult, risk: ForecastResult, quantum: QuantumResult) -> List[str]:
        guidance = []
        if forecast.probability > 0.7 and risk.probability > 0.5:
            guidance.append("Synthesize cross-discipline response team; both operations and security show elevated signals.")
        if quantum.entropy < quantum.qubits / 2:
            guidance.append("Quantum projection indicates coherent state; predictions more deterministic.")
        else:
            guidance.append("Quantum projection entropy high; maintain probabilistic posture.")
        guidance.extend(forecast.guidance[:1])
        guidance.extend(risk.guidance[:1])
        return guidance

    # --- Quantum-Enhanced Forecasting (23 Algorithms) -------------------------------------------------

    def quantum_hhl_forecast(self, telemetry: Dict[str, dict]) -> Dict:
        """
        Use HHL algorithm for exponential speedup in solving linear forecasting systems.
        Complexity: O(log(N)κ²) vs O(N³) classical.
        """
        if not QUANTUM_ENHANCED:
            return {"available": False, "message": "Quantum suite not available"}

        import numpy as np

        # Extract features and build linear system A·x = b for forecasting
        features = self._collect_features(telemetry)

        # Simple 2x2 system: predict load and memory pressure
        # A represents transition dynamics, b is current state
        A = np.array([
            [2.0, -0.5],  # Load dynamics with memory coupling
            [-0.3, 2.0]   # Memory dynamics with load coupling
        ])
        b = np.array([features["load"], features["memory"]])

        try:
            result = hhl_linear_system_solver(A, b)
            return {
                "available": True,
                "forecast": result["expectation_z"],
                "success_probability": result["success_probability"],
                "quantum_advantage": result["quantum_advantage"],
                "complexity": f"O(log({A.shape[0]})κ²) vs O({A.shape[0]}³)",
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

    def quantum_schrodinger_forecast(self, telemetry: Dict[str, dict], time_horizon: float = 1.0) -> Dict:
        """
        Use Schrödinger dynamics for time evolution forecasting.
        Predicts system state evolution over time.
        """
        if not QUANTUM_ENHANCED:
            return {"available": False, "message": "Quantum suite not available"}

        import numpy as np

        features = self._collect_features(telemetry)

        # Build 2-state Hamiltonian (bull/bear market analogy)
        load_bias = features["load"] - 0.5
        memory_bias = features["memory"] - 0.5

        H = np.array([
            [load_bias, 0.3],      # State coupling
            [0.3, -memory_bias]
        ])

        # Initial state (current system state)
        psi0 = np.array([
            np.sqrt(features["load"]),
            np.sqrt(1 - features["load"])
        ])
        psi0 = psi0 / np.linalg.norm(psi0)

        try:
            result = quantum_dynamics_forecasting(H, psi0, time_horizon)
            return {
                "available": True,
                "final_state": result["final_state"].tolist(),
                "probabilities": result["probabilities"],
                "energy": result["energy"],
                "time_horizon": time_horizon,
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

    def quantum_kalman_filter(self, telemetry: Dict[str, dict], measurements: List[float]) -> Dict:
        """
        Quantum-enhanced Kalman filtering for state estimation.
        O(log(N)κ²) per time step vs O(N³) classical.
        """
        if not QUANTUM_ENHANCED:
            return {"available": False, "message": "Quantum suite not available"}

        import numpy as np

        features = self._collect_features(telemetry)

        # State: [load, memory]
        # Use quantum Kalman filter for exponential speedup
        try:
            initial_state = np.array([features["load"], features["memory"]])
            result = quantum_kalman_filtering(
                initial_state=initial_state,
                measurements=measurements,
                num_steps=len(measurements)
            )
            return {
                "available": True,
                "final_estimate": result["final_state"].tolist(),
                "estimation_error": result["error"],
                "quantum_speedup": result["speedup"],
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

    # --- Internal signal extraction helpers -------------------------------------------------

    def _collect_features(self, telemetry: Dict[str, dict]) -> Dict[str, float]:
        return {
            "load": self._extract_load_signal(telemetry),
            "memory": self._extract_memory_signal(telemetry),
            "provider": self._extract_provider_health(telemetry),
            "container": self._extract_container_pressure(telemetry),
            "firewall": self._extract_firewall_health(telemetry),
            "process": self._extract_process_anomalies(telemetry),
            "backup": self._extract_backup_health(telemetry),
            "audit": self._extract_security_events(telemetry),
            "network": self._extract_network_reachability(telemetry),
            "dns": self._extract_dns_health(telemetry),
            "apps": self._extract_app_success(telemetry),
        }

    def _extract_load_signal(self, telemetry: Dict[str, dict]) -> float:
        load = telemetry.get("scalability.monitor_load", {})
        values = [
            load.get("load_1m", 0.0),
            load.get("load_5m", 0.0),
            load.get("load_15m", 0.0),
        ]
        cleaned = [v for v in values if isinstance(v, (int, float))]
        if not cleaned:
            return 0.3
        normalized = [min(1.5, max(0.0, v)) / 1.5 for v in cleaned]
        return sum(normalized) / len(normalized)

    def _extract_memory_signal(self, telemetry: Dict[str, dict]) -> float:
        kernel_mem = telemetry.get("kernel.memory_management", {})
        free_mb = kernel_mem.get("free_mb")
        total_mb = kernel_mem.get("total_mb") or kernel_mem.get("active_mb")
        if isinstance(free_mb, (int, float)) and isinstance(total_mb, (int, float)) and total_mb > 0:
            used_ratio = 1 - (free_mb / total_mb)
            return max(0.0, min(1.0, used_ratio))
        return 0.4

    def _extract_provider_health(self, telemetry: Dict[str, dict]) -> float:
        providers = telemetry.get("scalability.monitor_load", {}).get("providers", [])
        if not providers:
            return 0.5
        scores = []
        for provider in providers:
            healthy = provider.get("healthy")
            if healthy is True:
                scores.append(1.0)
            elif healthy is False:
                scores.append(0.0)
        if not scores:
            return 0.5
        return statistics.mean(scores)

    def _extract_container_pressure(self, telemetry: Dict[str, dict]) -> float:
        providers = telemetry.get("scalability.monitor_load", {}).get("providers", [])
        docker_stats = []
        for provider in providers:
            if provider.get("provider") == "docker":
                stats = provider.get("details", {}).get("stats") or []
                docker_stats.extend(stats)
        if not docker_stats:
            return 0.4
        pressures = []
        for entry in docker_stats:
            cpu = entry.get("CPUPerc") or entry.get("CPUPerc".lower())
            mem = entry.get("MemPerc") or entry.get("MemPerc".lower())
            cpu_value = self._parse_percentage(cpu)
            mem_value = self._parse_percentage(mem)
            if cpu_value is not None or mem_value is not None:
                combined = max(cpu_value or 0.0, mem_value or 0.0) / 100.0
                pressures.append(max(0.0, min(1.0, combined)))
        if not pressures:
            return 0.4
        return statistics.mean(pressures)

    def _extract_firewall_health(self, telemetry: Dict[str, dict]) -> float:
        firewall = telemetry.get("security.firewall", {})
        profiles = firewall.get("profiles")
        if isinstance(profiles, list) and profiles:
            enabled = [profile.get("Enabled") for profile in profiles if isinstance(profile, dict)]
            if enabled:
                ratio = sum(1 for flag in enabled if str(flag).lower() in {"true", "1"}) / len(enabled)
                return max(0.0, min(1.0, ratio))
        raw = firewall.get("raw")
        if isinstance(raw, str) and "State = 1" in raw:
            return 1.0
        return 0.6

    def _extract_process_anomalies(self, telemetry: Dict[str, dict]) -> float:
        scan = telemetry.get("security.threat_detection", {})
        anomalies = scan.get("high_cpu_candidates") or []
        sample = scan.get("sample") or []
        if sample:
            ratio = len(anomalies) / max(1, len(sample))
            return max(0.0, min(1.0, ratio))
        return 0.2

    def _extract_backup_health(self, telemetry: Dict[str, dict]) -> float:
        backup = telemetry.get("storage.backup", {})
        status = backup.get("status") or backup.get("status_code")
        if status and isinstance(status, str):
            if "error" in status.lower():
                return 0.2
            if "success" in status.lower():
                return 0.9
        return 0.5

    def _extract_security_events(self, telemetry: Dict[str, dict]) -> float:
        audit = telemetry.get("security.audit_review", {})
        events = audit.get("events") or audit.get("log_tail")
        if not events:
            return 0.6
        text = " ".join(str(event) for event in events)
        lowered = text.lower()
        if any(token in lowered for token in ["error", "fail", "denied", "critical"]):
            return 0.3
        return 0.8

    def _extract_network_reachability(self, telemetry: Dict[str, dict]) -> float:
        net = telemetry.get("networking.data_transmission", {})
        summary = net.get("ping_summary") or net.get("_message")
        if isinstance(summary, str):
            lowered = summary.lower()
            if any(term in lowered for term in ["ttl", "time=", "bytes"]):
                return 0.9
            if any(term in lowered for term in ["timeout", "unreachable", "host down"]):
                return 0.2
        return 0.5

    def _extract_dns_health(self, telemetry: Dict[str, dict]) -> float:
        resolver = telemetry.get("networking.dns_resolver", {})
        address = resolver.get("address")
        message = resolver.get("_message")
        if address:
            return 0.9
        if isinstance(message, str) and "error" in message.lower():
            return 0.3
        return 0.5

    def _extract_app_success(self, telemetry: Dict[str, dict]) -> float:
        supervisor = telemetry.get("application_supervisor", {})
        summary = supervisor.get("summary", {})
        total_specs = summary.get("total_specs") or 0
        if total_specs:
            completed = summary.get("completed") or 0
            failed = summary.get("failed") or 0
            errors = summary.get("errors") or 0
            success_ratio = max(0.0, min(1.0, completed / total_specs))
            penalty = max(0.0, min(1.0, (failed + errors) / total_specs))
            return max(0.0, min(1.0, success_ratio - penalty * 0.5))
        return 0.5

    def _parse_percentage(self, value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip().strip("%")
            try:
                return float(stripped)
            except ValueError:
                return None
        return None


class QuantumCircuitSimulator:
    def __init__(self, qubits: int, seed: Optional[int] = None) -> None:
        self.qubits = qubits
        self.size = 1 << qubits
        self.state: List[complex] = [0j] * self.size
        self.state[0] = 1 + 0j
        self.rng = random.Random(seed)

    def apply_hadamard(self, qubit: int) -> None:
        step = 1 << qubit
        for block in range(0, self.size, step * 2):
            for offset in range(step):
                idx0 = block + offset
                idx1 = idx0 + step
                a = self.state[idx0]
                b = self.state[idx1]
                factor = 1 / math.sqrt(2)
                self.state[idx0] = (a + b) * factor
                self.state[idx1] = (a - b) * factor

    def apply_all_hadamard(self) -> None:
        for qubit in range(self.qubits):
            self.apply_hadamard(qubit)

    def apply_cnot(self, control: int, target: int) -> None:
        control_mask = 1 << control
        target_mask = 1 << target
        for idx in range(self.size):
            if idx & control_mask:
                partner = idx ^ target_mask
                if idx < partner:
                    self.state[idx], self.state[partner] = self.state[partner], self.state[idx]

    def apply_entanglement_chain(self) -> None:
        for ctrl in range(self.qubits - 1):
            self.apply_cnot(ctrl, ctrl + 1)

    def apply_phase(self, qubit: int, angle: float) -> None:
        phase = complex(math.cos(angle), math.sin(angle))
        mask = 1 << qubit
        for idx in range(self.size):
            if idx & mask:
                self.state[idx] *= phase

    def apply_feature_phases(self, features: Dict[str, float], forecast_hint: float, residual_hint: float) -> None:
        network_bias = features["network"]
        dns_bias = features["dns"]
        for qubit in range(self.qubits):
            angle = (
                (features["load"] * 0.25 + features["memory"] * 0.15) * (qubit + 1) / self.qubits
                + (network_bias + dns_bias) * 0.1 * (qubit + 1)
            ) * math.pi / 4
            self.apply_phase(qubit, angle)

        load_bias = forecast_hint - 0.5
        risk_bias = residual_hint - 0.5
        container_bias = features["container"] - 0.5
        for idx in range(self.size):
            pop = bin(idx).count("1")
            load_component = load_bias * (pop / max(1, self.qubits))
            risk_component = risk_bias * ((self.qubits - pop) / max(1, self.qubits))
            container_component = -container_bias * (pop / max(1, self.qubits)) * 0.5
            adjustment = 1 + load_component + risk_component + container_component
            if adjustment < 0:
                adjustment = 0
            self.state[idx] *= adjustment

        # Introduce small stochastic variation for diversity
        noise_scale = 0.02
        for idx in range(self.size):
            noise = (self.rng.random() - 0.5) * noise_scale
            self.state[idx] *= (1 + noise)

    def renormalize(self) -> None:
        norm = math.sqrt(sum((abs(amplitude) ** 2 for amplitude in self.state))) or 1.0
        self.state = [amplitude / norm for amplitude in self.state]

    def measure_distribution(self, max_states: int = 32) -> Dict[str, float]:
        probabilities = []
        for idx, amplitude in enumerate(self.state):
            probability = abs(amplitude) ** 2
            if probability > 0:
                probabilities.append((probability, idx))
        probabilities.sort(reverse=True)
        top = probabilities[: max_states]
        return {format(idx, f"0{self.qubits}b"): round(prob, 6) for prob, idx in top}

    def entropy(self) -> float:
        entropy = 0.0
        for amplitude in self.state:
            probability = abs(amplitude) ** 2
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def _parse_percentage(self, value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip().strip("%")
            try:
                return float(stripped)
            except ValueError:
                return None
        return None
