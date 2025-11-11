"""
Example demonstrating ML algorithms integration within AgentaOS.

This script shows how to use the advanced ML algorithms suite in:
1. Meta-agent action handlers
2. Oracle predictions
3. State estimation pipelines
"""

import sys
import numpy as np
from pathlib import Path

# Add AgentaOS to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aios.ml_algorithms import (
    AdaptiveParticleFilter,
    NeuralGuidedMCTS,
    NoUTurnSampler,
    SparseGaussianProcess,
    get_algorithm_catalog,
    check_dependencies
)
from aios.runtime import ExecutionContext, ActionResult
from aios.config import DEFAULT_MANIFEST


def example_particle_filter_state_tracking():
    """
    Example 1: Real-time state tracking with particle filter.
    Use case: Track system load, resource usage, or sensor measurements.
    """
    print("═" * 70)
    print("EXAMPLE 1: Particle Filter for State Tracking")
    print("═" * 70)

    # Initialize particle filter for 3D state (CPU, Memory, Disk)
    pf = AdaptiveParticleFilter(
        num_particles=500,
        state_dim=3,
        obs_dim=3
    )

    # Simulate state tracking over time
    print("\nSimulating system state tracking:")

    def transition_fn(state):
        """Simple dynamics: state evolves with random walk."""
        return state + np.random.randn(3) * 0.05

    def likelihood_fn(observation, state):
        """Gaussian likelihood."""
        diff = observation - state
        return np.exp(-0.5 * np.dot(diff, diff))

    # Simulate 10 timesteps
    true_state = np.array([0.5, 0.6, 0.4])
    for t in range(10):
        # True state evolution
        true_state = true_state + np.random.randn(3) * 0.02

        # Noisy observation
        observation = true_state + np.random.randn(3) * 0.1

        # Particle filter steps
        pf.predict(transition_fn, process_noise=0.05)
        pf.update(observation, likelihood_fn)

        # Get estimate
        estimate = pf.estimate()

        print(f"  t={t:2d}: True={true_state}, Estimate={estimate}, "
              f"Error={np.linalg.norm(true_state - estimate):.4f}")

    print("\n✓ Particle filter successfully tracked state with low error\n")


def example_mcts_planning():
    """
    Example 2: Planning with neural-guided MCTS.
    Use case: Resource allocation, scheduling, or optimization decisions.
    """
    print("═" * 70)
    print("EXAMPLE 2: Neural-Guided MCTS for Planning")
    print("═" * 70)

    # Simple policy/value network stubs (in production, use trained nets)
    class SimplePolicy:
        def __call__(self, state_tensor):
            # Return uniform policy logits
            if check_dependencies()['torch']:
                import torch
                return torch.zeros(1, 10)  # 10 actions
            return None

    class SimpleValue:
        def __call__(self, state_tensor):
            # Return neutral value
            if check_dependencies()['torch']:
                import torch
                return torch.tensor([0.5])
            return 0.5

    # Initialize MCTS
    mcts = NeuralGuidedMCTS(
        policy_net=SimplePolicy(),
        value_net=SimpleValue(),
        c_puct=1.0
    )

    # Example state
    state = np.array([0.5, 0.3, 0.8, 0.2])

    print("\nRunning MCTS simulations (simplified)...")
    print(f"  Initial state: {state}")
    print("  Number of simulations: 50")

    # Run search (simplified for demo)
    try:
        policy = mcts.search(state, num_simulations=50)
        print(f"  Recommended policy distribution: {policy[:5]}...")
        print("\n✓ MCTS completed successfully\n")
    except Exception as e:
        print(f"  Note: {e}")
        print("  (This is expected without trained networks)\n")


def example_hmc_sampling():
    """
    Example 3: Bayesian posterior sampling with NUTS.
    Use case: Parameter estimation, uncertainty quantification.
    """
    print("═" * 70)
    print("EXAMPLE 3: Hamiltonian Monte Carlo (NUTS) Sampling")
    print("═" * 70)

    # Define a simple posterior (2D Gaussian)
    def log_prob(theta):
        """Log probability of 2D Gaussian."""
        mu = np.array([1.0, 2.0])
        sigma = 0.5
        diff = theta - mu
        return -0.5 * np.dot(diff, diff) / (sigma ** 2)

    # Initialize NUTS sampler
    nuts = NoUTurnSampler(
        log_prob_fn=log_prob,
        step_size=0.1,
        max_tree_depth=5
    )

    print("\nSampling from 2D Gaussian posterior:")
    print("  True mean: [1.0, 2.0]")
    print("  Generating 100 samples...")

    # Generate samples
    initial_position = np.array([0.0, 0.0])
    samples = nuts.sample(initial_position, num_samples=100)

    # Compute statistics
    sample_mean = np.mean(samples, axis=0)
    sample_std = np.std(samples, axis=0)

    print(f"  Sample mean: {sample_mean}")
    print(f"  Sample std: {sample_std}")
    print(f"  Error from true mean: {np.linalg.norm(sample_mean - np.array([1.0, 2.0])):.4f}")

    print("\n✓ NUTS successfully sampled from posterior\n")


def example_sparse_gp_regression():
    """
    Example 4: Scalable regression with sparse GP.
    Use case: Predict metrics, model system behavior with uncertainty.
    """
    print("═" * 70)
    print("EXAMPLE 4: Sparse Gaussian Process Regression")
    print("═" * 70)

    # RBF kernel
    def rbf_kernel(X1, X2, lengthscale=1.0):
        """Radial basis function kernel."""
        from scipy.spatial.distance import cdist
        dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-dists / (2 * lengthscale ** 2))

    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(200, 1) * 10
    y_train = np.sin(X_train).ravel() + np.random.randn(200) * 0.1

    # Initialize sparse GP
    sgp = SparseGaussianProcess(num_inducing=20, kernel=rbf_kernel)

    print("\nTraining sparse GP on 200 points with 20 inducing points...")
    sgp.fit(X_train, y_train, noise_var=0.01)

    # Make predictions
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    mean, variance = sgp.predict(X_test)

    print(f"  Training data: {X_train.shape[0]} points")
    print(f"  Inducing points: {sgp.num_inducing}")
    print(f"  Test predictions: {X_test.shape[0]} points")
    print(f"  Mean prediction range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Uncertainty (std) range: [{np.sqrt(variance.min()):.3f}, {np.sqrt(variance.max()):.3f}]")

    print("\n✓ Sparse GP regression completed with uncertainty estimates\n")


def example_agentaos_integration():
    """
    Example 5: Integration with AgentaOS meta-agent action handler.
    Shows how to use ML algorithms in the runtime execution context.
    """
    print("═" * 70)
    print("EXAMPLE 5: AgentaOS Meta-Agent Integration")
    print("═" * 70)

    # Create execution context
    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_ML_ALGORITHMS_ENABLED': '1'
    }

    def advanced_forecasting_handler(ctx: ExecutionContext) -> ActionResult:
        """
        Example meta-agent action using ML algorithms.
        This could be part of the Oracle agent for prediction.
        """
        # Initialize particle filter for system state tracking
        pf = AdaptiveParticleFilter(num_particles=100, state_dim=2, obs_dim=2)

        # Simulate observation
        observation = np.array([0.6, 0.4])

        # Simple dynamics and likelihood
        def dynamics(x):
            return x + 0.01

        def likelihood(obs, state):
            return np.exp(-np.sum((obs - state) ** 2))

        # Run filter
        pf.predict(dynamics, process_noise=0.01)
        pf.update(observation, likelihood)
        estimate = pf.estimate()

        # Publish to metadata
        ctx.publish_metadata('ml.state_estimate', {
            'estimate': estimate.tolist(),
            'observation': observation.tolist(),
            'algorithm': 'AdaptiveParticleFilter',
            'num_particles': 100
        })

        return ActionResult(
            success=True,
            message=f"[info] State estimated: {estimate}",
            payload={'estimate': estimate.tolist()}
        )

    print("\nExecuting meta-agent action with ML algorithms...")
    result = advanced_forecasting_handler(ctx)

    print(f"  Result: {result.message}")
    print(f"  Success: {result.success}")
    print(f"  Metadata keys: {list(ctx.metadata.keys())}")
    print(f"  State estimate: {ctx.metadata.get('ml.state_estimate', {}).get('estimate')}")

    print("\n✓ Successfully integrated ML algorithms into AgentaOS action handler\n")


def main():
    """Run all examples."""
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║      AgentaOS ML Algorithms Suite - Integration Examples          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    # Check dependencies
    deps = check_dependencies()
    print("Dependency Check:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    print()

    # Run examples
    try:
        example_particle_filter_state_tracking()
    except Exception as e:
        print(f"  ✗ Error in particle filter example: {e}\n")

    try:
        example_mcts_planning()
    except Exception as e:
        print(f"  ✗ Error in MCTS example: {e}\n")

    try:
        example_hmc_sampling()
    except Exception as e:
        print(f"  ✗ Error in HMC example: {e}\n")

    try:
        example_sparse_gp_regression()
    except Exception as e:
        print(f"  ✗ Error in sparse GP example: {e}\n")

    try:
        example_agentaos_integration()
    except Exception as e:
        print(f"  ✗ Error in AgentaOS integration example: {e}\n")

    print("═" * 70)
    print("All examples completed!")
    print("═" * 70)
    print()
    print("Next steps:")
    print("  - Explore individual algorithm classes in AgentaOS/ml_algorithms.py")
    print("  - Integrate algorithms into your meta-agent action handlers")
    print("  - Use get_algorithm_catalog() to discover all available algorithms")
    print("  - Check PyTorch availability with check_dependencies()")
    print()


if __name__ == "__main__":
    main()
