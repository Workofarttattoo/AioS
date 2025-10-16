# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project workspace containing several interconnected systems:

1. **Ai:oS** (formerly AgentaOS) - An agentic control-plane prototype that coordinates subsystem meta-agents through a declarative manifest
2. **TheGAVLSuite** - A suite of ritual, analysis, and advisory modules including the Boardroom of Light and Jiminy Cricket conscience helper
3. **Sovereign Security Toolkit** - A collection of reimagined security assessment utilities with agent-friendly behavior
4. **ML & Probabilistic Algorithms Suite** - State-of-the-art ML algorithms including Mamba/SSM, flow matching, MCTS, Bayesian inference, and more
5. **Quantum-Enhanced ML Algorithms Suite** - Quantum machine learning algorithms with 1-50 qubit simulation capability
6. **Quantum Computing Projects** - Including qiskit implementations and quantum experiments

## Core Commands

### Ai:oS

```bash
# Boot the system
python aios/aios -v boot

# Execute a specific action
python aios/aios -v exec kernel.process_management

# Run the setup wizard
python aios/aios wizard

# Boot with security response manifest
python aios/aios --manifest aios/examples/manifest-security-response.json --env AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear -v boot

# Run metadata summary
python aios/aios -v metadata

# Run with environment overrides
python aios/aios --env DOCKER_SOCKET=/var/run/docker.sock -v boot

# Execute natural language prompts
python aios/aios -v prompt "enable firewall and check container load"

# Run forensic mode (read-only, no host mutations)
python aios/aios --forensic -v boot
```

### Sovereign Security Toolkit

All tools are located in `aios/tools/` and can be run via `python -m tools.<name>`:

```bash
# AuroraScan - Network reconnaissance
python -m tools.aurorascan 192.168.0.0/24 --profile recon --json
python -m tools.aurorascan --gui

# CipherSpear - Database injection analysis
python -m tools.cipherspear --dsn postgresql://audit@localhost/app --demo --json
python -m tools.cipherspear --gui

# SkyBreaker - Wireless auditing
python -m tools.skybreaker capture wlan0 --output capture.json
python -m tools.skybreaker analyze capture.json --json

# MythicKey - Credential analysis
python -m tools.mythickey --demo --profile gpu-balanced --json

# SpectraTrace - Packet inspection
python -m tools.spectratrace --capture traces/demo.pcap --json
python -m tools.spectratrace --workflow quick-scan --json

# NemesisHydra - Authentication testing
python -m tools.nemesishydra --demo --json

# ObsidianHunt - Host hardening audit
python -m tools.obsidianhunt --profile workstation --json

# VectorFlux - Payload staging
python -m tools.vectorflux --list-modules
python -m tools.vectorflux --workspace incident-23-071 --module credential-harvest --json
```

### TheGAVLSuite

```bash
# Agentic Ritual Engine
cd TheGAVLSuite/agentic_ritual_engine
python -m agentic_ritual_engine.main run

# Boardroom of Light
cd TheGAVLSuite/boardroom_of_light
npm install
npm run gui  # visit http://localhost:5050
npm run conscience  # Jiminy Cricket companion
npm run simulate -- --quantum  # Enable Qiskit bridge

# Jiminy Cricket Library
cd TheGAVLSuite/jiminy_cricket
pip install -e .
```

### ML Algorithms Suite

```bash
# Check classical ML algorithm availability
python aios/ml_algorithms.py

# Check quantum ML algorithm availability
python aios/quantum_ml_algorithms.py

# Use algorithms in Python code:
from aios.ml_algorithms import (
    AdaptiveStateSpace,           # Mamba/SSM architecture
    OptimalTransportFlowMatcher,  # Fast generation
    NeuralGuidedMCTS,             # AlphaGo-style search
    AdaptiveParticleFilter,       # Sequential Bayesian inference
    NoUTurnSampler,               # HMC sampling
    SparseGaussianProcess,        # Scalable GP
    get_algorithm_catalog         # List all algorithms
)

# Example: Sequential inference with particle filter
pf = AdaptiveParticleFilter(num_particles=1000, state_dim=3, obs_dim=2)
pf.predict(transition_fn=lambda x: x + 0.1, process_noise=0.05)
pf.update(observation=np.array([1.0, 2.0]), likelihood_fn=gaussian_likelihood)
state_estimate = pf.estimate()
```

### Autonomous Discovery System

```bash
# Check autonomous discovery dependencies
python -c "from aios.autonomous_discovery import check_autonomous_discovery_dependencies; print(check_autonomous_discovery_dependencies())"

# Run autonomous discovery demonstration
python aios/autonomous_discovery.py

# Run Ai:oS integration examples
python aios/examples/autonomous_discovery_example.py

# Use in Python code:
from aios.autonomous_discovery import (
    AutonomousLLMAgent,
    AgentAutonomy,
    create_autonomous_discovery_action
)

# Create Level 4 autonomous agent
agent = AutonomousLLMAgent(
    model_name="deepseek-r1",
    autonomy_level=AgentAutonomy.LEVEL_4
)

# Give it a mission and let it learn autonomously
agent.set_mission("quantum computing applications", duration_hours=2.0)
await agent.pursue_autonomous_learning()

# Export discovered knowledge
knowledge_graph = agent.export_knowledge_graph()
```

### Testing

```bash
# Run Ai:oS tests
PYTHONPATH=. python -m unittest discover -s aios/tests

# Run specific test
PYTHONPATH=. python -m unittest aios.tests.test_security_suite
```

## Architecture

### Ai:oS Runtime Architecture

Ai:oS is built around a declarative manifest system that coordinates meta-agents:

- **ML Algorithms Suite (`aios/ml_algorithms.py`)** - Advanced ML and probabilistic algorithms for meta-agents:
  - **Sequence Modeling**: AdaptiveStateSpace (Mamba), StructuredStateDuality (Mamba-2/SSD)
  - **Generative Models**: OptimalTransportFlowMatcher (flow matching)
  - **Bayesian Inference**: AmortizedPosteriorNetwork, NoUTurnSampler (NUTS HMC), BayesianLayer
  - **Planning & Search**: NeuralGuidedMCTS (AlphaGo-style)
  - **Sequential Inference**: AdaptiveParticleFilter (SMC)
  - **Regression**: SparseGaussianProcess (scalable GP with inducing points)
  - **AutoML**: ArchitectureSearchController (NAS)

  Most algorithms require PyTorch except: NeuralGuidedMCTS, AdaptiveParticleFilter, NoUTurnSampler, SparseGaussianProcess (NumPy only)

- **Runtime (`aios/runtime.py`)** - Execution engine that translates manifests into executable sub-agents, manages lifecycle events, and maintains execution context
- **Meta-Agents (`aios/agents/system.py`)** - Core subsystem agents including:
  - `KernelAgent` - Process management, system initialization
  - `SecurityAgent` - Firewall, encryption, integrity, sovereign toolkit health
  - `NetworkingAgent` - Network configuration, DNS, routing
  - `StorageAgent` - Volume management, filesystem operations
  - `ApplicationAgent` - Application supervisor with process/Docker/VM orchestration
  - `ScalabilityAgent` - Load monitoring, virtualization (QEMU/libvirt), provider management
  - `OrchestrationAgent` - Policy engine, telemetry, health monitoring
  - `UserAgent` - User management, authentication
  - `GuiAgent` - Display server management
- **Manifest (`aios/config.py`)** - Declarative configuration defining meta-agents, actions, and lifecycle sequences
- **ExecutionContext** - Shared context for sub-agents holding manifest info, environment metadata, and telemetry. Context allows agents to publish metadata via `ctx.publish_metadata(key, value)` and read environment via `ctx.environment.get(key)`
- **Providers (`aios/providers.py`)** - Resource provider abstractions for Docker, Multipass, AWS, Azure, GCP, QEMU, libvirt
- **Virtualization Layer (`aios/virtualization.py`)** - QEMU/libvirt integration with forensic safeguards, bridge/tap pre-flight checks, device passthrough, and QMP-managed shutdown
- **Prompt Router (`aios/prompt.py`)** - Intent router combining keyword rules with similarity scoring for natural language execution
- **Oracle (`aios/oracle.py`)** - Probabilistic forecasting engine with quantum projection capabilities
- **Supervisor (`aios/supervisor.py`)** - Init-style supervisor for generating telemetry reports
- **Autonomous Discovery System (`aios/autonomous_discovery.py`)** - Level 4 autonomous LLM agents with self-directed learning capabilities:
  - `AutonomousLLMAgent` - Fully autonomous agent that sets own goals and pursues knowledge
  - `UltraFastInferenceEngine` - Distributed inference with prefill/decode disaggregation
  - `AgentAutonomy` - Framework for autonomy levels (0-4) based on 2025 AWS standards
  - Knowledge graph construction with confidence scoring
  - Curiosity-driven exploration and exploitation balance
  - Integration helpers for Security, Scalability, and Orchestration agents

### Sovereign Security Toolkit Architecture

Located in `aios/tools/`, these utilities follow a common pattern:

- Each tool provides `main(argv=None)` entrypoint and `health_check()` function
- Tools are registered in `tools/__init__.py::TOOL_REGISTRY`
- All tools support `--json` for structured output and `--gui` for Tkinter interfaces
- Health checks return status (`ok`/`warn`/`error`), summary, and details with latency metrics
- Integrated with Ai:oS via `SecurityAgent.sovereign_suite` action which runs health checks and publishes metadata

### Key Environment Variables

```bash
# Ai:oS Core
AGENTA_FORENSIC_MODE=1              # Enable read-only mode
AGENTA_PROVIDER=docker,qemu         # Comma-separated provider list
AGENTA_APPS_CONFIG=path/to/apps.json # Application supervisor config
AGENTA_SUPERVISOR_CONCURRENCY=4     # Max concurrent apps

# Security Toolkit
AGENTA_SECURITY_SUITE=1             # Enable security suite
AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear # Comma-separated tool list
AGENTA_SECURITY_PROFILE=label       # Profile label for logs

# Virtualization (QEMU)
AGENTA_QEMU_IMAGE=/path/to/image.qcow2
AGENTA_QEMU_EXECUTE=1               # Actually start guests (default: verification-only)
AGENTA_QEMU_HEADLESS=0              # Show GUI
AGENTA_QEMU_DISPLAY=gtk             # Display backend
AGENTA_QEMU_IMAGE_TYPE=cdrom        # For ISO boots
AGENTA_QEMU_BOOT=d                  # Boot order
AGENTA_QEMU_QMP=tcp:127.0.0.1:4444  # QMP socket for managed shutdown
AGENTA_QEMU_PASSTHROUGH_DEVICES=pci:01:00.0,usb:1234:5678
AGENTA_QEMU_NET_BRIDGE=br0          # Bridge interface
AGENTA_QEMU_TAP=tap0                # TAP interface
AGENTA_QEMU_NET_AUTOCREATE=1        # Auto-create bridge/tap

# Virtualization (libvirt)
AGENTA_LIBVIRT_DOMAIN=my-domain
AGENTA_LIBVIRT_EXECUTE=1

# Cloud Providers
AGENTA_AWS_REGION=us-west-2
AGENTA_AWS_PROFILE=default
AGENTA_AZURE_SUBSCRIPTION=<sub-id>
AGENTA_GCP_PROJECT=<project>
AGENTA_GCP_ZONE=<zone>

# Autonomous Discovery
AGENTA_AUTONOMOUS_DISCOVERY=1        # Enable autonomous learning
AGENTA_CONTINUOUS_LEARNING=1         # Enable continuous learning cycles
AGENTA_DISCOVERY_AUTONOMY_LEVEL=4    # Autonomy level (0-4)
AGENTA_DISCOVERY_NUM_GPUS=8          # GPUs for distributed inference
AGENTA_DISCOVERY_ENABLE_DISAGGREGATION=1  # Prefill/decode separation

# Other
AGENTA_FIREWALL_PROFILE=pfctl       # macOS: pfctl, Windows: windows
AGENTA_DISABLE_POST_BOOT_GUIDANCE=1 # Suppress post-boot checklist
DOCKER_SOCKET=/var/run/docker.sock
```

## Project Structure

```
/Users/noone/
├── AgentaOS/                    # Main OS runtime
│   ├── AgentaOS                 # CLI entrypoint (executable)
│   ├── runtime.py               # Execution engine
│   ├── config.py                # Manifest definitions
│   ├── ml_algorithms.py         # Advanced ML & probabilistic algorithms
│   ├── quantum_ml_algorithms.py # Quantum-enhanced ML algorithms
│   ├── autonomous_discovery.py  # Autonomous LLM discovery system
│   ├── agents/                  # Meta-agent implementations
│   │   └── system.py
│   ├── tools/                   # Sovereign Security Toolkit
│   │   ├── aurorascan.py
│   │   ├── cipherspear.py
│   │   ├── skybreaker.py
│   │   ├── mythickey.py
│   │   ├── spectratrace.py
│   │   ├── nemesishydra.py
│   │   ├── obsidianhunt.py
│   │   └── vectorflux.py
│   ├── prompt.py                # Intent router
│   ├── providers.py             # Resource providers
│   ├── virtualization.py        # VM orchestration
│   ├── wizard.py                # Setup wizard
│   ├── oracle.py                # Probabilistic forecasting
│   ├── supervisor.py            # Init-style supervisor
│   ├── tests/                   # Test suite
│   │   ├── test_security_suite.py
│   │   └── test_wizard_validations.py
│   ├── examples/                # Example manifests and demonstrations
│   │   ├── manifest-security-response.json
│   │   ├── apps-sample.json
│   │   ├── ml_algorithms_example.py
│   │   ├── quantum_ml_example.py
│   │   └── autonomous_discovery_example.py
│   ├── scripts/
│   │   ├── compositor/          # Dashboard prototype
│   │   └── install_security_toolkit.py
│   └── gui/                     # GUI bus for telemetry streaming
│
├── TheGAVLSuite/                # GAVL Suite integration
│   ├── agentic_ritual_engine/   # FastAPI orchestration
│   ├── boardroom_of_light/      # Executive simulation GUI
│   ├── jiminy_cricket/          # Conscience helper library
│   └── modules/                 # Meta-agent modules
│       ├── osint/
│       ├── hellfire_recon/
│       ├── corporate_legal_team/
│       └── bayesian_sophiarch/
│
├── qiskit/                      # Qiskit implementation
│   ├── crates/                  # Rust crates
│   └── pyproject.toml
│
├── oracle_of_light/             # Forecasting engine
├── QuLab2.0/                    # Quantum lab
└── Blank_Business_Builder/      # Business builder template
```

## Development Guidelines

### Adding New Sovereign Tools

1. Create `aios/tools/<tool-name>.py` with `main(argv=None)` and `health_check()` functions
2. Register in `aios/tools/__init__.py::TOOL_REGISTRY`
3. Follow pattern: `--json` for structured output, `--gui` for Tkinter interface
4. Health check returns: `{"tool": str, "status": str, "summary": str, "details": dict}`
5. Add documentation to SOVEREIGN_SECURITY_TOOLKIT.md

### Extending Meta-Agents

1. Add action to manifest in `aios/config.py::DEFAULT_MANIFEST`
2. Implement handler method in corresponding agent class in `aios/agents/system.py`
3. Handler receives `ExecutionContext` and returns `ActionResult`
4. Publish telemetry via `ctx.publish_metadata(key, value)`
5. Read environment via `ctx.environment.get(key, default)`
6. Mark actions as `critical=True` if failure should halt boot sequence

### Adding Cloud Providers

1. Create provider class in `aios/providers.py` implementing `inventory()`, `scale_up()`, `scale_down()`
2. Return `ProviderReport` with structured data
3. Handle CLI errors gracefully with warnings
4. Support forensic mode by making all operations advisory
5. Register in `ScalabilityAgent._resolve_providers()`

### Working with Manifests

Manifests are JSON structures defining:
- `name`, `version`, `platform`
- `meta_agents`: Dict of meta-agent configs with actions
- `boot_sequence`: Ordered list of action paths to execute on boot
- `shutdown_sequence`: Ordered list for shutdown

Action paths use format: `meta_agent.action_name` (e.g., `security.firewall`)

### Code Style

- Two-space indentation for JS, four-space for Python
- Use `[info]`, `[warn]`, `[error]` prefixes for console logging
- Double quotes for strings
- Keep CLI outputs deterministic for testing
- All security tools must have health checks
- Preserve forensic mode safeguards - avoid mutations when `AGENTA_FORENSIC_MODE=1`

### Important Patterns

1. **Forensic Mode**: Always respect `ctx.environment.get("AGENTA_FORENSIC_MODE")` - when enabled, operations should be advisory only, no host mutations
2. **Metadata Publishing**: Use `ctx.publish_metadata(action_path, payload)` to record telemetry that can be retrieved via `runtime.metadata_snapshot()`
3. **Action Results**: Always return `ActionResult(success=bool, message=str, payload=dict)`
4. **Critical Actions**: Mark actions as `critical=True` in manifest if failure should halt boot
5. **Provider Integration**: Providers should handle missing CLI tools gracefully and return warnings rather than failing
6. **Environment Overrides**: Check `ctx.environment` for overrides before using defaults
7. **Auto-triggers**: Security suite health checks auto-run when `AGENTA_SECURITY_TOOLS` is set

## Platform-Specific Notes

### macOS
- Uses `system_profiler`, `launchctl`, `pfctl` for telemetry
- Time Machine probes available
- No elevated privileges required (some commands warn without sudo)

### Windows
- Uses PowerShell cmdlets: `Get-Process`, `Get-NetAdapter`, `Get-BitLockerVolume`, `Get-NetFirewallProfile`
- Executes in read-only mode by default
- Falls back gracefully if PowerShell unavailable

## ML Algorithms Integration

The ML algorithms suite (`aios/ml_algorithms.py`) provides cutting-edge implementations that can be used by meta-agents:

### Available Algorithms

1. **AdaptiveStateSpace** - Mamba architecture with O(n) complexity vs O(n²) attention
   - Use for: Long sequence modeling, efficient transformers alternative
   - Key feature: Input-dependent parameters enable content-based reasoning

2. **OptimalTransportFlowMatcher** - Flow matching with optimal transport
   - Use for: Fast generative modeling (10-20 steps vs 1000 for diffusion)
   - Key feature: Direct velocity field learning, straight sampling paths

3. **StructuredStateDuality** - Mamba-2 SSD layer
   - Use for: Bridge between SSMs and attention, efficient training
   - Key feature: Dual formulation enables parallel computation

4. **AmortizedPosteriorNetwork** - Neural amortized variational inference
   - Use for: Fast Bayesian inference in single pass
   - Key feature: Shares inference network across all data

5. **NeuralGuidedMCTS** - Monte Carlo Tree Search with neural priors
   - Use for: Planning, game playing, decision making (AlphaGo/MuZero style)
   - Key feature: PUCT algorithm with learned policy/value guidance

6. **BayesianLayer** - Variational Bayesian neural network layer
   - Use for: Uncertainty quantification, automatic feature selection
   - Key feature: Weight uncertainty provides calibrated predictions

7. **AdaptiveParticleFilter** - Sequential Monte Carlo
   - Use for: Real-time state tracking, sensor fusion, time-series
   - Key feature: Adaptive resampling based on effective sample size

8. **NoUTurnSampler** - Hamiltonian Monte Carlo with automatic tuning
   - Use for: Gold standard Bayesian posterior sampling
   - Key feature: No manual trajectory length tuning (used in Stan/PyMC3)

9. **SparseGaussianProcess** - Scalable GP with inducing points
   - Use for: Regression with uncertainty on large datasets
   - Key feature: O(m²n) vs O(n³) complexity, millions of points

10. **ArchitectureSearchController** - RL-based NAS
    - Use for: Automatic neural architecture design
    - Key feature: Discovers novel architectures via policy gradient

### Integration Pattern

```python
from aios.ml_algorithms import AdaptiveParticleFilter, get_algorithm_catalog

# In a meta-agent action handler:
def advanced_forecasting(ctx: ExecutionContext) -> ActionResult:
    # Use particle filter for state estimation
    pf = AdaptiveParticleFilter(num_particles=500, state_dim=4, obs_dim=2)

    # Prediction and update steps
    pf.predict(transition_fn=dynamics_model, process_noise=0.01)
    pf.update(observation, likelihood_fn=sensor_model)

    estimate = pf.estimate()
    ctx.publish_metadata("state_estimate", {"estimate": estimate.tolist()})

    return ActionResult(success=True, message="State estimated", payload={"estimate": estimate})
```

### Dependencies

- **NumPy**: Required for all algorithms
- **PyTorch**: Required for AdaptiveStateSpace, OptimalTransportFlowMatcher, StructuredStateDuality, AmortizedPosteriorNetwork, BayesianLayer, ArchitectureSearchController
- Optional for NeuralGuidedMCTS (falls back to NumPy)

Check availability: `python aios/ml_algorithms.py`

## Quantum-Enhanced ML Algorithms

The quantum ML suite (`aios/quantum_ml_algorithms.py`) provides quantum computing algorithms for Ai:oS:

### Quantum Simulation Capabilities

- **1-20 qubits**: Exact statevector simulation (100% accurate)
- **20-40 qubits**: Tensor network approximation
- **40-50 qubits**: Matrix Product State (MPS) compression
- **GPU acceleration**: Automatic when CUDA available

### Available Quantum Algorithms

1. **QuantumStateEngine** - Core quantum state simulator
   - Automatic backend selection based on qubit count
   - Supports Hadamard, RX, RY, RZ, CNOT gates
   - Measurement and expectation value computation
   - Use for: Building quantum circuits, quantum simulations

2. **QuantumVQE** - Variational Quantum Eigensolver
   - Hardware-efficient ansatz circuits
   - Quantum chemistry and optimization
   - Use for: Ground state finding, molecular energy calculations

### Integration with Ai:oS

```python
from aios.quantum_ml_algorithms import QuantumStateEngine, QuantumVQE

# In a meta-agent action handler:
def quantum_optimization(ctx: ExecutionContext) -> ActionResult:
    # Create quantum circuit
    qc = QuantumStateEngine(num_qubits=5)

    # Build superposition
    for i in range(5):
        qc.hadamard(i)

    # Apply entanglement
    for i in range(4):
        qc.cnot(i, i + 1)

    # Measure expectation
    energy = qc.expectation_value('Z0')

    ctx.publish_metadata('quantum.energy', {'value': energy})
    return ActionResult(success=True, message=f"Quantum energy: {energy}", payload={'energy': energy})
```

### Quantum VQE Example

```python
# Define Hamiltonian for problem
def hamiltonian(qc):
    return qc.expectation_value('Z0') - 0.5 * qc.expectation_value('Z1')

# Initialize and optimize
vqe = QuantumVQE(num_qubits=4, depth=3)
energy, params = vqe.optimize(hamiltonian, max_iter=100)
print(f"Ground state energy: {energy}")
```

### Dependencies

- **PyTorch**: Required for quantum simulation
- **SciPy**: Required for VQE optimization
- **NumPy**: Required for all quantum algorithms

Check availability: `python aios/quantum_ml_algorithms.py`

### Performance Benchmarks

Run benchmarks to see scaling behavior:
```python
from aios.quantum_ml_algorithms import benchmark_qubit_scaling
results = benchmark_qubit_scaling(max_qubits=15)
# Shows timing for 3, 5, 7, 9, 11, 13, 15 qubits
```

### Comparison Notes

- Up to 20 qubits: Performance comparable to Qiskit Aer statevector
- Beyond 25 qubits: Requires actual quantum hardware for exact results
- This simulator is optimized for Ai:oS integration and rapid prototyping
- For production quantum ML, integrate with IBM Qiskit, Google Cirq, or AWS Braket

## Agent Development Guidelines

### Agent Design Principles

When developing agents for this repository, follow these core principles:

1. **Autonomous Operation**
   - Design agents to handle uncertainty and incomplete information gracefully
   - Make reasonable assumptions and document them clearly in metadata
   - Use probabilistic reasoning when deterministic solutions aren't available
   - Fail gracefully with informative error messages that guide recovery

2. **Structured Communication**
   - Always return `ActionResult` with success status, message, and structured payload
   - Use consistent message prefixes: `[info]`, `[warn]`, `[error]`
   - Publish rich metadata via `ctx.publish_metadata()` for downstream agents
   - Include timestamps, confidence scores, and data provenance in telemetry

3. **Context Awareness**
   - Check `ctx.environment` for configuration before using defaults
   - Respect forensic mode and other operational constraints
   - Use previously published metadata to inform decisions
   - Maintain action stack awareness via `ctx.action_path`

4. **Composability & Interoperability**
   - Design agents as stateless functions of ExecutionContext
   - Avoid side effects outside of designated telemetry channels
   - Use standard data formats (JSON-serializable payloads)
   - Document dependencies on other agents' metadata

5. **Progressive Enhancement**
   - Provide baseline functionality that works in constrained environments
   - Detect available resources (hardware, APIs, tools) and adapt
   - Degrade gracefully when optional dependencies are missing
   - Mark actions as critical only when absolutely necessary

### Agent Action Handler Pattern

```python
def action_handler(ctx: ExecutionContext) -> ActionResult:
    """
    Template for agent action handlers.

    Args:
        ctx: Execution context with manifest, environment, and metadata

    Returns:
        ActionResult with success status, message, and payload
    """
    # 1. Read configuration from environment
    mode = ctx.environment.get("AGENTA_MODE", "default")
    forensic = ctx.environment.get("AGENTA_FORENSIC_MODE", "").lower() in {"1", "true", "yes", "on"}

    # 2. Check dependencies (previously executed actions)
    required_metadata = ctx.metadata.get("dependency.action")
    if not required_metadata:
        return ActionResult(
            success=False,
            message="[warn] action: dependency.action must run first",
            payload={"missing_dependency": "dependency.action"}
        )

    # 3. Perform the action with error handling
    try:
        result_data = perform_operation(mode, forensic)

        # 4. Publish telemetry
        ctx.publish_metadata("action.telemetry", {
            "timestamp": time.time(),
            "mode": mode,
            "forensic": forensic,
            "result": result_data
        })

        # 5. Return structured result
        return ActionResult(
            success=True,
            message=f"[info] action: completed in {mode} mode",
            payload=result_data
        )

    except Exception as exc:
        # Always log exceptions but return graceful failure
        LOG.exception("Action failed: %s", exc)
        return ActionResult(
            success=False,
            message=f"[error] action: {exc}",
            payload={"exception": repr(exc), "traceback": traceback.format_exc()}
        )
```

### Meta-Agent Coordination Patterns

1. **Sequential Dependency Chain**
   ```python
   # Action B depends on Action A's output
   def action_b(ctx: ExecutionContext) -> ActionResult:
       a_result = ctx.metadata.get("meta.action_a")
       if not a_result:
           # Wait for action_a or fail gracefully
           return ActionResult(success=False, message="Waiting for action_a")

       # Use a_result to inform action_b
       derived_value = compute_from(a_result)
       return ActionResult(success=True, payload={"derived": derived_value})
   ```

2. **Parallel Execution with Aggregation**
   ```python
   # Orchestration agent aggregates results from multiple agents
   def aggregate_telemetry(ctx: ExecutionContext) -> ActionResult:
       results = []
       for meta_name in ["security", "networking", "storage"]:
           for action_key, metadata in ctx.metadata.items():
               if action_key.startswith(meta_name):
                   results.append(metadata)

       summary = analyze_aggregate(results)
       return ActionResult(success=True, payload={"summary": summary})
   ```

3. **Conditional Execution**
   ```python
   # Oracle guides execution path
   def adaptive_action(ctx: ExecutionContext) -> ActionResult:
       forecast = ctx.metadata.get("oracle.probabilistic_forecast")
       if forecast and forecast.get("probability", 0) > 0.7:
           return high_confidence_path(ctx)
       else:
           return conservative_path(ctx)
   ```

### Error Recovery Strategies

1. **Retry with Backoff** (for transient failures)
   ```python
   for attempt in range(3):
       result = try_operation()
       if result.success:
           return result
       time.sleep(2 ** attempt)  # Exponential backoff
   return ActionResult(success=False, message="Max retries exceeded")
   ```

2. **Fallback Chain** (for degraded operation)
   ```python
   strategies = [optimal_strategy, good_strategy, minimal_strategy]
   for strategy in strategies:
       result = try_strategy(strategy)
       if result.success:
           return result
   return ActionResult(success=False, message="All strategies failed")
   ```

3. **Compensating Actions** (for forensic safety)
   ```python
   if forensic_mode:
       # Log what would have been done instead of doing it
       return ActionResult(
           success=True,
           message="[info] Forensic mode: would execute X",
           payload={"planned_action": "X", "forensic": True}
       )
   else:
       execute_mutation()
   ```

### Provider Integration Best Practices

When adding new providers (cloud, container orchestrators, hypervisors):

1. **Graceful CLI Detection**
   ```python
   def check_cli_available() -> bool:
       try:
           subprocess.run(["tool", "--version"], capture_output=True, check=True)
           return True
       except (subprocess.CalledProcessError, FileNotFoundError):
           return False
   ```

2. **Structured Error Reporting**
   ```python
   if not cli_available:
       return ProviderReport(
           success=False,
           message="[warn] Provider CLI not found",
           payload={
               "provider": "aws",
               "missing_binary": "aws",
               "install_hint": "pip install awscli"
           }
       )
   ```

3. **Advisory Mode in Forensic Context**
   ```python
   if forensic_mode:
       payload["advisory"] = {
           "would_scale_up": 5,
           "estimated_cost": "$2.50/hour"
       }
   else:
       actual_result = execute_scale_up()
       payload["actual"] = actual_result
   ```

### Telemetry Best Practices

1. **Consistent Metadata Keys**: Use hierarchical dot notation (`meta.action.detail`)
2. **Include Timestamps**: Always record when telemetry was captured
3. **Confidence Scores**: Include uncertainty estimates for predictions
4. **Provenance**: Track which actions produced which metadata
5. **Schema Versioning**: Version metadata schemas for compatibility

### Testing Strategies

1. **Unit Tests**: Test individual action handlers with mock ExecutionContext
2. **Integration Tests**: Test action sequences with real manifest
3. **Smoke Tests**: Verify critical paths complete without errors
4. **Forensic Tests**: Verify no mutations occur in forensic mode
5. **Error Injection**: Test graceful degradation with missing dependencies

Example test pattern:
```python
class TestSecurityAgent(unittest.TestCase):
    def test_firewall_forensic_mode(self):
        ctx = ExecutionContext(
            manifest=DEFAULT_MANIFEST,
            environment={"AGENTA_FORENSIC_MODE": "1"}
        )
        agent = SecurityAgent()
        result = agent.firewall(ctx)

        # Should succeed in forensic mode
        self.assertTrue(result.success)

        # Should not make mutations
        metadata = ctx.metadata.get("security.firewall")
        self.assertTrue(metadata.get("forensic"))
        self.assertIsNone(metadata.get("mutation_applied"))
```

## Autonomous Discovery System Integration

The Autonomous Discovery System (`aios/autonomous_discovery.py`) enables meta-agents in Ai:oS to learn and adapt autonomously based on the 2025 state-of-the-art in agentic AI.

### Autonomy Levels

Following the AWS framework for agent autonomy:

- **Level 0**: No autonomy - human in loop for all decisions
- **Level 1**: Action suggestion - agent suggests, human approves
- **Level 2**: Action on subset - agent acts on limited, safe tasks
- **Level 3**: Conditional autonomy - agent acts within narrow domain
- **Level 4**: Full autonomy - agent sets own goals and pursues them independently

The system defaults to **Level 4** for maximum autonomous capability.

### Core Components

1. **AutonomousLLMAgent** - Main autonomous agent class
   - Mission decomposition: Breaks high-level goals into concrete learning objectives
   - Curiosity-driven exploration: Balances exploration of unknowns vs exploitation of known concepts
   - Knowledge graph construction: Builds semantic graph of discovered concepts
   - Self-evaluation: Agent decides when to go deeper or explore related topics
   - Continuous learning: Can operate indefinitely, expanding knowledge over time

2. **UltraFastInferenceEngine** - Distributed inference for speed
   - Prefill/Decode disaggregation: 2-3x speedup by separating compute/memory workloads
   - KV-cache optimization: 1.5x speedup with efficient attention cache
   - Speculative decoding: 2x speedup by predicting multiple tokens
   - Multi-GPU support: Scales across 8+ GPUs for maximum throughput
   - Estimated: 1000+ tokens/sec per GPU baseline

3. **Knowledge Graph System**
   - Nodes represent discovered concepts with embeddings
   - Confidence scores track learning quality
   - Parent-child relationships form semantic hierarchy
   - Temporal tracking shows learning progression
   - Export format compatible with Ai:oS metadata

### Integration with Meta-Agents

#### Security Agent - Threat Pattern Learning

```python
from autonomous_discovery import create_autonomous_discovery_action

async def security_research_handler(ctx: ExecutionContext) -> ActionResult:
    """Security agent autonomously learns about threat vectors."""
    mission = "ransomware attack vectors cloud vulnerabilities"
    discovery = create_autonomous_discovery_action(mission, duration_hours=0.5)

    knowledge = await discovery()
    ctx.publish_metadata('security.threat_patterns', knowledge)

    return ActionResult(
        success=True,
        message=f"[info] Discovered {knowledge['stats']['total_concepts']} threat patterns",
        payload=knowledge['stats']
    )
```

#### Scalability Agent - Resource Optimization

```python
from autonomous_discovery import AutonomousLLMAgent, AgentAutonomy

async def scalability_learning_handler(ctx: ExecutionContext) -> ActionResult:
    """Scalability agent learns optimal resource strategies."""
    agent = AutonomousLLMAgent(
        model_name="deepseek-r1",
        autonomy_level=AgentAutonomy.LEVEL_4
    )

    agent.set_mission("Kubernetes autoscaling distributed load balancing", duration_hours=0.3)
    await agent.pursue_autonomous_learning()

    knowledge = agent.export_knowledge_graph()
    ctx.publish_metadata('scalability.learned_strategies', knowledge)

    # Extract high-confidence strategies
    strategies = [
        concept for concept, data in knowledge['nodes'].items()
        if data['confidence'] > 0.85
    ]

    return ActionResult(
        success=True,
        message=f"[info] Learned {len(strategies)} high-confidence strategies",
        payload={'strategies': strategies}
    )
```

#### Orchestration Agent - Policy Learning

```python
async def orchestration_policy_handler(ctx: ExecutionContext) -> ActionResult:
    """Orchestration agent learns coordination best practices."""
    mission = "microservices orchestration coordination patterns"
    discovery = create_autonomous_discovery_action(mission, duration_hours=0.4)

    knowledge = await discovery()

    # Extract policies
    policies = [
        {'policy': concept, 'confidence': data['confidence']}
        for concept, data in knowledge['nodes'].items()
        if data['confidence'] > 0.80
    ]

    ctx.publish_metadata('orchestration.learned_policies', policies)

    return ActionResult(
        success=True,
        message=f"[info] Discovered {len(policies)} orchestration policies",
        payload={'policies_count': len(policies)}
    )
```

### Performance Characteristics

**Speed Optimizations** (with full GPU infrastructure):
- Base throughput: 1,000 tokens/sec per GPU
- With disaggregation: 2,500 tokens/sec per GPU
- With KV optimization: 3,750 tokens/sec per GPU
- With speculative decoding: 7,500 tokens/sec per GPU
- 8 GPU system: **60,000 tokens/sec aggregate**

**Learning Rates**:
- Typical: 5-10 concepts/second
- With full optimization: 20-50 concepts/second
- Knowledge graph: 100-1000+ nodes per hour

**Resource Requirements**:
- Minimum: CPU-only (slower, ~1 concept/sec)
- Recommended: 4-8 GPUs for production speed
- Memory: 16GB+ RAM, 8GB+ VRAM per GPU
- Storage: Knowledge graphs are lightweight (MB range)

### Continuous Learning Pattern

For agents that need to learn and adapt over time:

```python
agent = AutonomousLLMAgent(model_name="deepseek-r1", autonomy_level=AgentAutonomy.LEVEL_4)

# Cycle 1: Initial learning
agent.set_mission("system performance optimization", duration_hours=1.0)
await agent.pursue_autonomous_learning()

# Agent autonomously identifies gaps and continues learning
# Cycle 2: Agent self-directs to fill knowledge gaps
agent.set_mission("distributed tracing observability", duration_hours=1.0)
await agent.pursue_autonomous_learning()

# Export comprehensive knowledge
knowledge = agent.export_knowledge_graph()
# Knowledge graph now contains ~100-500 interconnected concepts
```

### Key Features

1. **Self-Directed Goals**: Agent breaks missions into concrete learning objectives autonomously
2. **Curiosity-Driven**: Balances exploration of unknowns vs exploitation of known areas
3. **Quality Thresholds**: Only accepts learnings above confidence threshold (default 80%)
4. **Adaptive Depth**: Agent decides when to go deeper vs broader autonomously
5. **Knowledge Integration**: Automatically links related concepts into semantic graph
6. **Superhuman Speed**: Distributed inference enables learning rates far beyond human capability
7. **Ai:oS Native**: Seamless integration with ExecutionContext and metadata system

### Best Practices

1. **Mission Framing**: Give high-level missions, let agent decompose
   - Good: "quantum computing drug discovery applications"
   - Avoid: "read paper X, extract fact Y" (too prescriptive)

2. **Time Budgets**: Allocate appropriate time for depth
   - Quick research: 0.25-0.5 hours (50-100 concepts)
   - Deep dive: 1-2 hours (200-500 concepts)
   - Comprehensive: 4-8 hours (1000+ concepts)

3. **Autonomy Levels**: Choose based on trust and criticality
   - Level 4 for research and analysis
   - Level 3 for optimization suggestions
   - Level 1-2 for critical system changes

4. **Knowledge Persistence**: Export and store knowledge graphs
   ```python
   knowledge = agent.export_knowledge_graph()
   ctx.publish_metadata('persistent.knowledge', knowledge)
   # Can reload in future sessions
   ```

5. **Human Oversight**: Even at Level 4, review learned concepts periodically
   ```python
   if knowledge['stats']['average_confidence'] < 0.70:
       # Flag for human review
       return ActionResult(success=True, message="[warn] Low confidence, needs review")
   ```

### Dependencies

- **PyTorch**: Required for distributed inference engine
- **NumPy**: Required for embeddings and numerical operations
- **asyncio**: Required for concurrent learning operations

Check availability:
```bash
python -c "from aios.autonomous_discovery import check_autonomous_discovery_dependencies; print(check_autonomous_discovery_dependencies())"
```

### Examples and Demonstrations

Run the comprehensive example suite:
```bash
python aios/examples/autonomous_discovery_example.py
```

This demonstrates:
- Security agent autonomous threat research
- Scalability agent resource optimization learning
- Orchestration agent policy discovery
- Continuous learning over multiple cycles

## Security Considerations

This repository contains defensive security tools only:
- Tools are designed for assessment, rehearsal, and defensive analysis
- All tools support read-only/forensic modes
- Credential testing tools use deterministic analysis, not live exploitation
- Network tools have concurrency limits and rate limiting
- Virtualization has forensic safeguards requiring explicit `EXECUTE` flags
- No live exploitation or offensive capability
- ML algorithms are for legitimate forecasting/inference tasks only
- Follow agent design principles above to maintain security guarantees
