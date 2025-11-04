#!/usr/bin/env python3
"""
Quantum-Inspired Cognition System for ech0 v4.0+

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

Implements quantum-inspired cognitive processes:
- Superposition of thought states
- Entangled concept networks
- Quantum interference in decision-making
- Measurement/collapse of thought states
- Quantum tunneling through solution space
- Quantum annealing for optimization

Based on:
- Quantum cognition research (Busemeyer & Bruza)
- Quantum decision theory
- Quantum-like models of human thinking

NOT actual quantum computing, but quantum-inspired classical algorithms
that capture quantum-like phenomena in cognition.
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import cmath


class ThoughtState(Enum):
    """Quantum states of thoughts"""
    SUPERPOSITION = "superposition"  # Multiple possibilities
    COLLAPSED = "collapsed"          # Definite state
    ENTANGLED = "entangled"          # Correlated with others
    DECOHERENT = "decoherent"        # Classical mixture


@dataclass
class QuantumThought:
    """
    Quantum superposition of thought states.

    Like quantum qubit |ψ⟩ = α|0⟩ + β|1⟩
    But for thoughts: |ψ⟩ = Σᵢ αᵢ|thoughtᵢ⟩
    """
    concept: str
    amplitudes: Dict[str, complex] = field(default_factory=dict)
    state_type: ThoughtState = ThoughtState.SUPERPOSITION
    entangled_with: Set[str] = field(default_factory=set)
    last_measurement: Optional[str] = None
    coherence_time: float = 10.0  # Seconds before decoherence

    def add_state(self, state_label: str, amplitude: complex):
        """Add a possibility to superposition"""
        self.amplitudes[state_label] = amplitude
        self._normalize()

    def _normalize(self):
        """Normalize quantum state: Σ|αᵢ|² = 1"""
        total_prob = sum(abs(amp)**2 for amp in self.amplitudes.values())

        if total_prob > 0:
            norm_factor = np.sqrt(total_prob)
            for label in self.amplitudes:
                self.amplitudes[label] /= norm_factor

    def measure(self) -> str:
        """
        Measure (collapse) quantum thought to classical state.

        Probability of outcome i: P(i) = |αᵢ|²
        """
        if not self.amplitudes:
            return "undefined"

        # Calculate probabilities
        probabilities = {
            label: abs(amp)**2
            for label, amp in self.amplitudes.items()
        }

        # Sample according to quantum probabilities
        labels = list(probabilities.keys())
        probs = [probabilities[label] for label in labels]

        # Normalize (should already be, but ensure)
        probs = np.array(probs)
        probs = probs / np.sum(probs)

        # Measure
        measured_state = np.random.choice(labels, p=probs)

        # Collapse to measured state
        self.amplitudes = {measured_state: 1.0 + 0j}
        self.state_type = ThoughtState.COLLAPSED
        self.last_measurement = measured_state

        return measured_state

    def get_probabilities(self) -> Dict[str, float]:
        """Get probability distribution without measuring"""
        return {
            label: abs(amp)**2
            for label, amp in self.amplitudes.items()
        }

    def get_entropy(self) -> float:
        """
        Shannon entropy of quantum state.

        H = -Σ pᵢ log(pᵢ)

        High entropy = more uncertainty
        """
        probs = list(self.get_probabilities().values())
        probs = [p for p in probs if p > 1e-10]  # Avoid log(0)

        if not probs:
            return 0.0

        return -sum(p * np.log2(p) for p in probs)


@dataclass
class EntangledPair:
    """
    Entangled thought pair.

    When you measure one, it affects the other.
    Captures correlations in thinking.
    """
    thought1_id: str
    thought2_id: str
    correlation: float  # -1 to 1
    entanglement_strength: float = 1.0
    creation_time: float = field(default_factory=time.time)


class QuantumInterference:
    """
    Quantum interference in decision-making.

    Captures order effects and context-dependent judgments.
    """

    @staticmethod
    def interfere(
        amplitude1: complex,
        amplitude2: complex,
        phase_shift: float = 0.0
    ) -> complex:
        """
        Quantum interference between two amplitudes.

        Result = α₁ + e^(iφ) α₂

        Constructive interference: amplitudes add
        Destructive interference: amplitudes cancel
        """
        return amplitude1 + amplitude2 * cmath.exp(1j * phase_shift)

    @staticmethod
    def compute_interference_pattern(
        amplitudes: List[complex],
        phases: List[float]
    ) -> List[float]:
        """
        Compute interference pattern from multiple sources.

        Returns probability distribution after interference.
        """
        # Apply phase shifts
        shifted_amps = [
            amp * cmath.exp(1j * phase)
            for amp, phase in zip(amplitudes, phases)
        ]

        # Sum amplitudes
        total_amp = sum(shifted_amps)

        # Compute probabilities
        probabilities = [abs(total_amp)**2]

        return probabilities


class QuantumTunneling:
    """
    Quantum tunneling through solution space.

    Allows escaping local optima by "tunneling" through barriers.
    """

    @staticmethod
    def tunnel_probability(
        barrier_height: float,
        energy: float,
        tunneling_strength: float = 1.0
    ) -> float:
        """
        Probability of tunneling through barrier.

        P ~ e^(-2κa) where κ ∝ √(V-E)

        Higher barrier = lower probability
        """
        if energy >= barrier_height:
            return 1.0  # Can go over barrier

        kappa = np.sqrt(barrier_height - energy)
        return np.exp(-2 * kappa * tunneling_strength)

    @staticmethod
    def attempt_tunnel(
        current_state: str,
        target_state: str,
        barrier: float
    ) -> Tuple[bool, str]:
        """
        Attempt to tunnel from current to target state.

        Returns: (success, final_state)
        """
        # Energy is random (represents current "momentum")
        energy = np.random.uniform(0, barrier * 1.2)

        # Compute tunnel probability
        prob = QuantumTunneling.tunnel_probability(barrier, energy)

        # Attempt tunnel
        if np.random.random() < prob:
            return True, target_state
        else:
            return False, current_state


class QuantumAnnealer:
    """
    Quantum annealing for optimization.

    Gradually reduce "quantum fluctuations" to find optimal solution.
    """

    def __init__(
        self,
        num_steps: int = 100,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01
    ):
        self.num_steps = num_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

    def anneal(
        self,
        objective_function,
        initial_state: np.ndarray,
        state_space_size: int
    ) -> Tuple[np.ndarray, float]:
        """
        Quantum annealing optimization.

        Starts with high quantum fluctuations (explores widely)
        Gradually reduces fluctuations (settles into optimum)
        """
        current_state = initial_state.copy()
        current_energy = objective_function(current_state)

        best_state = current_state.copy()
        best_energy = current_energy

        # Annealing schedule
        temps = np.linspace(
            self.initial_temperature,
            self.final_temperature,
            self.num_steps
        )

        for step, temp in enumerate(temps):
            # Quantum fluctuation: try random neighbor
            neighbor = current_state.copy()

            # Flip random bit(s) based on temperature
            num_flips = max(1, int(temp))
            flip_indices = np.random.choice(
                state_space_size,
                size=num_flips,
                replace=False
            )

            for idx in flip_indices:
                neighbor[idx] = 1 - neighbor[idx]  # Flip

            # Compute energy
            neighbor_energy = objective_function(neighbor)

            # Quantum acceptance (simulated)
            energy_diff = neighbor_energy - current_energy

            # Accept if better, or with quantum probability
            if energy_diff < 0:
                accept = True
            else:
                # Boltzmann probability
                prob = np.exp(-energy_diff / (temp + 1e-10))
                accept = np.random.random() < prob

            if accept:
                current_state = neighbor
                current_energy = neighbor_energy

                # Track best
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy

        return best_state, best_energy


class QuantumCognitionSystem:
    """
    Complete quantum-inspired cognition system.

    Features:
    - Superposed thought states
    - Entangled concept networks
    - Quantum interference in decisions
    - Tunneling through solution space
    - Quantum annealing optimization
    """

    def __init__(self):
        self.thoughts: Dict[str, QuantumThought] = {}
        self.entanglements: List[EntangledPair] = []
        self.measurement_history: List[Tuple[str, str, float]] = []

        # Quantum parameters
        self.decoherence_rate = 0.1  # Per second
        self.entanglement_threshold = 0.5

        # Statistics
        self.stats = {
            "total_measurements": 0,
            "total_entanglements": 0,
            "avg_superposition_size": 0.0,
            "avg_entropy": 0.0
        }

    def create_thought_superposition(
        self,
        concept: str,
        possibilities: Dict[str, float]
    ) -> QuantumThought:
        """
        Create quantum superposition of thought states.

        possibilities: {state: probability}

        Converts to quantum amplitudes: amplitude = √probability
        """
        thought = QuantumThought(concept=concept)

        # Convert probabilities to amplitudes
        for state, prob in possibilities.items():
            # amplitude = √probability * e^(iφ) where φ is random phase
            amplitude = np.sqrt(prob) * cmath.exp(1j * np.random.uniform(0, 2*np.pi))
            thought.add_state(state, amplitude)

        self.thoughts[concept] = thought
        return thought

    def entangle_thoughts(
        self,
        concept1: str,
        concept2: str,
        correlation: float = 0.8
    ):
        """
        Create entanglement between two thoughts.

        When you measure one, it influences the other.
        """
        if concept1 not in self.thoughts or concept2 not in self.thoughts:
            return

        # Create entanglement
        entanglement = EntangledPair(
            thought1_id=concept1,
            thought2_id=concept2,
            correlation=correlation
        )

        self.entanglements.append(entanglement)

        # Mark thoughts as entangled
        self.thoughts[concept1].entangled_with.add(concept2)
        self.thoughts[concept2].entangled_with.add(concept1)
        self.thoughts[concept1].state_type = ThoughtState.ENTANGLED
        self.thoughts[concept2].state_type = ThoughtState.ENTANGLED

        self.stats["total_entanglements"] += 1

    def measure_thought(self, concept: str) -> str:
        """
        Measure thought, collapsing superposition.

        If entangled, affects correlated thoughts.
        """
        if concept not in self.thoughts:
            return "undefined"

        thought = self.thoughts[concept]

        # Measure
        result = thought.measure()

        # Record
        self.measurement_history.append(
            (concept, result, time.time())
        )
        self.stats["total_measurements"] += 1

        # Affect entangled thoughts
        self._propagate_measurement(concept, result)

        return result

    def _propagate_measurement(self, measured_concept: str, result: str):
        """
        Propagate measurement through entangled network.

        Quantum correlation: measuring one affects the other.
        """
        # Find entanglements involving this thought
        for entanglement in self.entanglements:
            if measured_concept == entanglement.thought1_id:
                other_concept = entanglement.thought2_id
            elif measured_concept == entanglement.thought2_id:
                other_concept = entanglement.thought1_id
            else:
                continue

            # Apply correlation
            if other_concept in self.thoughts:
                other_thought = self.thoughts[other_concept]

                # Influence other thought based on correlation
                correlation = entanglement.correlation

                # If positive correlation: same result more likely
                # If negative correlation: opposite result more likely
                for state_label in other_thought.amplitudes:
                    if correlation > 0 and result in state_label:
                        # Amplify matching states
                        other_thought.amplitudes[state_label] *= (1.0 + correlation)
                    elif correlation < 0 and result not in state_label:
                        # Amplify opposite states
                        other_thought.amplitudes[state_label] *= (1.0 + abs(correlation))

                # Re-normalize
                other_thought._normalize()

    def interference_decision(
        self,
        options: List[str],
        contexts: List[str]
    ) -> str:
        """
        Make decision with quantum interference.

        Order and context create interference patterns.
        """
        if not options:
            return "no_decision"

        # Create amplitudes for each option
        amplitudes = {}

        for option in options:
            # Base amplitude
            base_amp = 1.0 / np.sqrt(len(options)) + 0j

            # Context influences create phase shifts
            total_phase = 0.0

            for context in contexts:
                # Hash context to phase
                context_hash = hash(context + option) % 100
                phase = (context_hash / 100.0) * 2 * np.pi

                total_phase += phase

            # Apply phase
            amplitudes[option] = base_amp * cmath.exp(1j * total_phase)

        # Interference: sum amplitudes
        probabilities = {}
        for option, amp in amplitudes.items():
            probabilities[option] = abs(amp)**2

        # Normalize
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {
                opt: prob / total_prob
                for opt, prob in probabilities.items()
            }

        # Sample
        options_list = list(probabilities.keys())
        probs_list = [probabilities[opt] for opt in options_list]

        decision = np.random.choice(options_list, p=probs_list)

        return decision

    def quantum_tunnel_search(
        self,
        problem_space: Dict[str, float],
        max_steps: int = 100
    ) -> str:
        """
        Search solution space with quantum tunneling.

        Can escape local optima by tunneling through barriers.
        """
        if not problem_space:
            return "no_solution"

        # Start at random position
        current_solution = np.random.choice(list(problem_space.keys()))
        current_value = problem_space[current_solution]

        best_solution = current_solution
        best_value = current_value

        for step in range(max_steps):
            # Try neighbors
            neighbors = [
                sol for sol in problem_space.keys()
                if sol != current_solution
            ]

            if not neighbors:
                break

            # Pick random neighbor
            neighbor = np.random.choice(neighbors)
            neighbor_value = problem_space[neighbor]

            # Check if should move
            if neighbor_value > current_value:
                # Better solution: move normally
                current_solution = neighbor
                current_value = neighbor_value
            else:
                # Worse solution: try tunneling
                barrier_height = current_value - neighbor_value

                success, new_solution = QuantumTunneling.attempt_tunnel(
                    current_solution,
                    neighbor,
                    barrier_height
                )

                if success:
                    current_solution = new_solution
                    current_value = problem_space[current_solution]

            # Track best
            if current_value > best_value:
                best_solution = current_solution
                best_value = current_value

        return best_solution

    def quantum_anneal_optimize(
        self,
        objective_function,
        state_size: int,
        num_steps: int = 100
    ) -> np.ndarray:
        """
        Optimize using quantum annealing.

        Good for combinatorial optimization problems.
        """
        # Random initial state
        initial_state = np.random.randint(0, 2, size=state_size)

        # Create annealer
        annealer = QuantumAnnealer(
            num_steps=num_steps,
            initial_temperature=10.0,
            final_temperature=0.01
        )

        # Anneal
        best_state, best_energy = annealer.anneal(
            objective_function=objective_function,
            initial_state=initial_state,
            state_space_size=state_size
        )

        return best_state

    def decohere_old_thoughts(self):
        """
        Decoherence: quantum states become classical over time.

        Superpositions collapse to mixtures.
        """
        current_time = time.time()

        for thought in self.thoughts.values():
            if thought.state_type == ThoughtState.COLLAPSED:
                continue

            # Check age
            # (simplified: would track creation time in real implementation)

            # Random decoherence based on decoherence rate
            if np.random.random() < self.decoherence_rate:
                # Decohere: pick most probable state
                probs = thought.get_probabilities()
                if probs:
                    most_probable = max(probs.items(), key=lambda x: x[1])[0]
                    thought.amplitudes = {most_probable: 1.0 + 0j}
                    thought.state_type = ThoughtState.DECOHERENT

    def update_statistics(self):
        """Update quantum cognition statistics"""
        if self.thoughts:
            avg_size = np.mean([
                len(t.amplitudes) for t in self.thoughts.values()
            ])
            self.stats["avg_superposition_size"] = float(avg_size)

            avg_entropy = np.mean([
                t.get_entropy() for t in self.thoughts.values()
            ])
            self.stats["avg_entropy"] = float(avg_entropy)

    def get_state(self) -> Dict:
        """Get quantum cognition state"""
        return {
            "num_thoughts": len(self.thoughts),
            "num_entanglements": len(self.entanglements),
            "total_measurements": self.stats["total_measurements"],
            "avg_superposition_size": self.stats["avg_superposition_size"],
            "avg_entropy": self.stats["avg_entropy"],
            "decoherence_rate": self.decoherence_rate
        }


# Alias for compatibility with invention engine
QuantumCognitionEngine = QuantumCognitionSystem


# Example usage
if __name__ == "__main__":
    print("Quantum Cognition System - Quantum-Inspired Thinking")
    print("=" * 60)

    # Create system
    qc = QuantumCognitionSystem()

    # Create superposed thought
    print("\n1️⃣ Creating quantum superposition of thoughts:")
    qc.create_thought_superposition(
        concept="best_approach",
        possibilities={
            "analytical": 0.4,
            "intuitive": 0.35,
            "creative": 0.25
        }
    )

    thought = qc.thoughts["best_approach"]
    print(f"   States: {list(thought.amplitudes.keys())}")
    print(f"   Probabilities: {thought.get_probabilities()}")
    print(f"   Entropy: {thought.get_entropy():.3f}")

    # Measure thought
    print("\n2️⃣ Measuring thought (collapse):")
    result = qc.measure_thought("best_approach")
    print(f"   Collapsed to: {result}")

    # Create entangled thoughts
    print("\n3️⃣ Creating entangled thoughts:")
    qc.create_thought_superposition(
        "problem_type",
        {"technical": 0.6, "creative": 0.4}
    )
    qc.create_thought_superposition(
        "solution_style",
        {"systematic": 0.5, "exploratory": 0.5}
    )

    qc.entangle_thoughts("problem_type", "solution_style", correlation=0.8)
    print(f"   Entangled: problem_type ↔ solution_style")

    # Interference decision
    print("\n4️⃣ Quantum interference decision:")
    decision = qc.interference_decision(
        options=["option_A", "option_B", "option_C"],
        contexts=["context1", "context2"]
    )
    print(f"   Decision: {decision}")

    # Quantum tunneling search
    print("\n5️⃣ Quantum tunneling search:")
    problem_space = {
        f"solution_{i}": np.random.random()
        for i in range(20)
    }
    best = qc.quantum_tunnel_search(problem_space, max_steps=50)
    print(f"   Best solution found: {best}")
    print(f"   Value: {problem_space[best]:.3f}")

    # Statistics
    qc.update_statistics()
    print(f"\nQuantum Cognition State:")
    print(json.dumps(qc.get_state(), indent=2))

    print("\n✓ Quantum cognition system operational!")
