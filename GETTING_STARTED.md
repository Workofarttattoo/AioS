# Ai:oS Getting Started Guide

**Version:** 2.0
**Last Updated:** 2025-10-24
**Difficulty:** Beginner to Advanced
**Time to Complete:** 15-30 minutes (quick start) to 2+ hours (full exploration)

---

## Quick Start Decision Tree

*Choose your primary use case to find the fastest path forward:*

```
Are you...

┌─ Orchestrating cloud infrastructure?
│  └─ YES → Go to: Cloud Orchestration (p. AWS/Docker providers)
│  └─ NO ↓
│
├─ Assessing security or running penetration tests?
│  └─ YES → Go to: Security Toolkit (p. 15+ tools)
│  └─ NO ↓
│
├─ Working with quantum computing?
│  └─ YES → Go to: Quantum Computing (p. 1-50 qubits)
│  └─ NO ↓
│
├─ Building ML forecasting/inference systems?
│  └─ YES → Go to: Machine Learning Algorithms (p. 10+ algorithms)
│  └─ NO ↓
│
└─ Exploring autonomous AI agents?
   └─ YES → Go to: Autonomous Discovery (p. Level 4 autonomy)
```

---

## Entry Points Overview

### 1. **CLI (Command Line)**
Best for: Developers, scripting, automation

```bash
# Boot the system with default manifest
python aios/aios -v boot

# Run setup wizard
python aios/aios wizard

# Execute specific action
python aios/aios -v exec kernel.process_management

# Natural language commands
python aios/aios -v prompt "check system health"

# List available resources
python aios/aios --list-tools
python aios/aios --list-algorithms
python aios/aios --list-providers
```

### 2. **Manifest-Driven (Declarative)**
Best for: Complex orchestration, reproducible deployments

```bash
# Execute custom manifest
python aios/aios --manifest path/to/manifest.json boot

# With environment overrides
python aios/aios --manifest custom.json --env DOCKER_SOCKET=/var/run/docker.sock boot

# Forensic mode (read-only)
python aios/aios --forensic -v boot
```

### 3. **Direct Python API**
Best for: Integrating into Python applications

```python
from aios.runtime import Runtime
from aios.config import DEFAULT_MANIFEST

# Create runtime
rt = Runtime(DEFAULT_MANIFEST)

# Boot and execute
result = rt.boot()
action_result = rt.execute_action("security.firewall")

# Query state
metadata = rt.metadata_snapshot()
```

### 4. **Individual Tools**
Best for: Single-purpose security assessments

```bash
python -m tools.aurorascan 192.168.0.0/24 --json
python -m tools.cipherspear --demo --gui
python -m tools.skybreaker capture wlan0
```

---

## Use Case Paths

### Path 1: Cloud Infrastructure Orchestration

**Time:** 30 minutes
**Goal:** Coordinate multiple cloud providers and containers

**Steps:**

1. **Install providers**
   ```bash
   # Docker (Linux/Mac)
   brew install docker

   # Or: AWS CLI
   pip install awscli-v2

   # Or: Azure CLI
   brew install azure-cli
   ```

2. **Create orchestration manifest** (`my-infra.json`)
   ```json
   {
     "name": "Multi-Cloud Infrastructure",
     "version": "1.0",
     "meta_agents": {
       "orchestration": {
         "actions": [
           "provision_resources",
           "health_check",
           "scale_deployment"
         ]
       },
       "scalability": {
         "actions": [
           "detect_load",
           "recommend_scaling",
           "execute_scaling"
         ]
       }
     },
     "boot_sequence": [
       "orchestration.provision_resources",
       "scalability.detect_load"
     ]
   }
   ```

3. **Boot with manifest**
   ```bash
   python aios/aios --manifest my-infra.json -v boot
   ```

4. **Monitor execution**
   ```bash
   python aios/aios metadata | jq '.orchestration'
   ```

**Next Steps:**
- Explore `aios/examples/manifest-security-response.json` for patterns
- Read: `aios/providers.py` for provider-specific options
- Advanced: Multi-provider failover strategies

---

### Path 2: Security Assessment Toolkit

**Time:** 20 minutes
**Goal:** Run comprehensive security audits

**Available Tools:**

| Tool | Use Case | Speed | Complexity |
|------|----------|-------|------------|
| **AuroraScan** | Network reconnaissance | Fast | Low |
| **CipherSpear** | Database injection analysis | Medium | Medium |
| **SkyBreaker** | Wireless auditing | Medium | High |
| **MythicKey** | Credential analysis | Fast | Medium |
| **SpectraTrace** | Packet inspection | Slow | High |
| **NemesisHydra** | Authentication testing | Medium | Medium |
| **ObsidianHunt** | Host hardening audit | Fast | Low |
| **VectorFlux** | Payload staging | Medium | High |
| **DirReaper** | Directory fuzzing | Fast | Low |

**Quick Start:**

```bash
# Network scan
python -m tools.aurorascan 192.168.0.0/24 --profile recon --json

# Database analysis
python -m tools.cipherspear --dsn postgresql://user@host/db --demo

# With GUI
python -m tools.aurorascan --gui

# Full assessment
python aios/aios --manifest aios/examples/manifest-security-response.json \
  --env AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear,ObsidianHunt \
  -v boot
```

**Next Steps:**
- Read: `aios/tools/QUICK_START.md` for tool specifics
- Advanced: Chain tools for multi-stage assessments
- Customize: `manifest-security-response.json`

---

### Path 3: Machine Learning & Forecasting

**Time:** 45 minutes
**Goal:** Use advanced ML algorithms for prediction/inference

**Available Algorithms:**

**Sequence Modeling:**
- `AdaptiveStateSpace` (Mamba) - O(n) complexity, content-aware
- `StructuredStateDuality` (Mamba-2) - Parallel training, sequential inference

**Generative Models:**
- `OptimalTransportFlowMatcher` - 10-20 steps vs 1000 for diffusion

**Bayesian Inference:**
- `AmortizedPosteriorNetwork` - Fast single-pass inference
- `NoUTurnSampler` (NUTS HMC) - Gold standard posterior sampling
- `AdaptiveParticleFilter` (SMC) - Real-time state tracking

**Planning & Search:**
- `NeuralGuidedMCTS` - AlphaGo-style planning with neural priors

**Regression:**
- `SparseGaussianProcess` - Scalable GP for millions of points

**AutoML:**
- `ArchitectureSearchController` - Neural architecture search

**Quick Start:**

```python
from aios.ml_algorithms import (
    AdaptiveParticleFilter,
    NeuralGuidedMCTS,
    SparseGaussianProcess
)
import numpy as np

# Particle filter for time series
pf = AdaptiveParticleFilter(num_particles=500, state_dim=3, obs_dim=2)
pf.predict(transition_fn=lambda x: x + 0.1, process_noise=0.05)
pf.update(observation=np.array([1.0, 2.0]), likelihood_fn=gaussian_likelihood)
state = pf.estimate()

# MCTS for planning
mcts = NeuralGuidedMCTS(num_simulations=100, depth=10)
best_action = mcts.search(initial_state, policy_network, value_network)

# Gaussian Process regression
gp = SparseGaussianProcess(num_inducing=100)
gp.fit(X_train, y_train)
predictions = gp.predict(X_test)
```

**Check Availability:**
```bash
python aios/ml_algorithms.py  # Shows installed algorithms
python aios/quantum_ml_algorithms.py  # Quantum ML availability
```

**Next Steps:**
- See: `aios/examples/ml_algorithms_example.py`
- Dependencies: PyTorch (most algorithms), NumPy (all)
- GPU: Automatically detected and used when available

---

### Path 4: Quantum Computing

**Time:** 1 hour
**Goal:** Run quantum circuits and algorithms

**Capabilities:**

| Qubits | Backend | Speed | Accuracy |
|--------|---------|-------|----------|
| 1-20 | Statevector | Fast | 100% |
| 20-40 | Tensor Network | Medium | 95%+ |
| 40-50 | MPS Compression | Slow | 80%+ |
| 50+ | Real Hardware | Varies | Hardware-limited |

**Supported Hardware:**

- **Local:** Qiskit Aer (free, unlimited qubits up to memory)
- **Cloud:** IonQ Harmony (11 qubits, 99.9% fidelity)
- **Custom:** Connect your own backend

**Quick Start:**

```bash
# Quantum ML algorithms
python aios/quantum_ml_algorithms.py

# Run quantum circuit
python -c "
from aios.quantum_ml_algorithms import QuantumStateEngine

qc = QuantumStateEngine(num_qubits=5)
for i in range(5):
    qc.hadamard(i)  # Superposition
for i in range(4):
    qc.cnot(i, i+1)  # Entanglement

energy = qc.expectation_value('Z0')
print(f'Energy: {energy}')
"

# VQE optimization
python -c "
from aios.quantum_ml_algorithms import QuantumVQE

def hamiltonian(qc):
    return qc.expectation_value('Z0') - 0.5 * qc.expectation_value('Z1')

vqe = QuantumVQE(num_qubits=4, depth=3)
energy, params = vqe.optimize(hamiltonian, max_iter=100)
print(f'Ground state: {energy}')
"
```

**Next Steps:**
- See: `aios/examples/quantum_ml_example.py`
- Dependencies: PyTorch, SciPy
- Real Hardware: Set `IONQ_API_KEY` environment variable

---

### Path 5: Autonomous Discovery & Learning

**Time:** 1+ hour
**Goal:** Enable agents to learn autonomously

**Autonomy Levels:**

- **Level 0:** Human approval for every decision
- **Level 1:** Agent suggests, human approves
- **Level 2:** Agent acts on limited safe tasks
- **Level 3:** Agent acts within narrow domain
- **Level 4:** Full autonomy - agent sets own goals ⭐

**Quick Start:**

```python
from aios.autonomous_discovery import (
    AutonomousLLMAgent,
    AgentAutonomy,
    create_autonomous_discovery_action
)
import asyncio

async def main():
    # Quick research (30 minutes)
    discovery = create_autonomous_discovery_action(
        mission="quantum cryptography applications",
        duration_hours=0.5
    )
    knowledge = await discovery()
    print(f"Discovered {knowledge['stats']['total_concepts']} concepts")

    # Deep dive (2 hours)
    agent = AutonomousLLMAgent(
        model_name="deepseek-r1",
        autonomy_level=AgentAutonomy.LEVEL_4
    )
    agent.set_mission("Kubernetes autoscaling strategies", duration_hours=2.0)
    await agent.pursue_autonomous_learning()

    knowledge_graph = agent.export_knowledge_graph()
    print(f"Knowledge graph: {len(knowledge_graph['nodes'])} concepts")

asyncio.run(main())
```

**Performance:**

- **Base:** 5-10 concepts/second
- **Optimized:** 20-50 concepts/second
- **Multi-GPU:** Up to 60,000 tokens/sec (8 GPUs)

**Next Steps:**
- See: `aios/examples/autonomous_discovery_example.py`
- Integrate with meta-agents (Security, Scalability, Orchestration)
- Dependencies: PyTorch, asyncio

---

## Common Workflows

### Workflow 1: Security Assessment + Remediation

```bash
# 1. Run assessment
python aios/aios --manifest aios/examples/manifest-security-response.json \
  --env AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear -v boot

# 2. Check results
python aios/aios metadata | jq '.security'

# 3. Execute remediation
python aios/aios -v exec security.remediation
```

### Workflow 2: Multi-Provider Infrastructure Deployment

```bash
# 1. Validate manifest
python aios/aios --manifest multi-cloud.json --validate

# 2. Dry run (forensic mode)
AGENTA_FORENSIC_MODE=1 python aios/aios --manifest multi-cloud.json -v boot

# 3. Execute deployment
python aios/aios --manifest multi-cloud.json -v boot
```

### Workflow 3: Quantum ML Training

```bash
# 1. Prepare data
python -c "import numpy as np; X = np.random.randn(1000, 5); np.save('data.npy', X)"

# 2. Run quantum VQE
python -c "
from aios.quantum_ml_algorithms import QuantumVQE
import numpy as np

X = np.load('data.npy')
vqe = QuantumVQE(num_qubits=4, depth=5)
# ... VQE training ...
"

# 3. Compare with classical
from aios.ml_algorithms import SparseGaussianProcess
gp = SparseGaussianProcess()
# ... Classical training ...
```

---

## Architecture Quick Reference

```
Ai:oS Runtime Architecture
│
├─ Meta-Agents (9 types)
│  ├─ KernelAgent (process management)
│  ├─ SecurityAgent (firewall, encryption, sovereign toolkit)
│  ├─ NetworkingAgent (configuration, DNS, routing)
│  ├─ StorageAgent (volume management, filesystems)
│  ├─ ApplicationAgent (process/Docker/VM orchestration)
│  ├─ ScalabilityAgent (load monitoring, virtualization)
│  ├─ OrchestrationAgent (policy engine, telemetry)
│  ├─ UserAgent (authentication, user management)
│  └─ GuiAgent (display server management)
│
├─ ML & Algorithms
│  ├─ Classical ML (10+ algorithms)
│  ├─ Quantum ML (1-50 qubits)
│  └─ Autonomous Discovery (Level 4 autonomy)
│
├─ Resource Providers
│  ├─ Docker
│  ├─ AWS, Azure, GCP
│  ├─ QEMU, libvirt
│  └─ Custom integrations
│
└─ Tools
   └─ Sovereign Security Toolkit (15+ tools)
```

---

## Troubleshooting

### "Command not found: aios"
The CLI is located at `aios/aios` (an executable). Add to your shell:
```bash
alias aios="python /Users/noone/aios/aios"
```

### "No module named 'aios'"
Make sure you're in the QuLab2.0 directory with proper PYTHONPATH:
```bash
cd /Users/noone && PYTHONPATH=. python -m aios.aios -v boot
```

### "Provider not available"
Check which providers are installed:
```bash
python -c "from aios.providers import check_provider_availability; check_provider_availability()"
```

### "Git-encrypted file"
Some files (runtime.py, tools/__init__.py) require decryption:
```bash
git-crypt unlock aios-git-crypt.key
```

### "ImportError: No module named 'qiskit'"
Install quantum dependencies:
```bash
pip install qiskit qiskit-aer qiskit-machine-learning
```

---

## Next Steps by Interest

**Security-focused?**
→ Read `aios/tools/QUICK_START.md`
→ Try `manifest-security-response.json`
→ Explore individual tool docs: `tools/*.py`

**Quantum-curious?**
→ Read `aios/QUANTUM_VQE_USAGE.md`
→ Try `aios/examples/quantum_ml_example.py`
→ Run `python aios/quantum_ml_algorithms.py` for capabilities

**ML researcher?**
→ Read `aios/ml_algorithms.py` docstrings
→ Try `aios/examples/ml_algorithms_example.py`
→ Check dependencies: `python aios/ml_algorithms.py`

**Building production systems?**
→ Study `aios/examples/manifest-*.json` patterns
→ Read `aios/config.py` for manifest schema
→ Try forensic mode first: `AGENTA_FORENSIC_MODE=1`

**Exploring autonomy?**
→ Read `aios/autonomous_discovery.py`
→ Try `aios/examples/autonomous_discovery_example.py`
→ Understand autonomy levels in this guide (p. 5)

---

## Quick Command Reference

```bash
# System Status
python aios/aios -v boot                          # Full boot
python aios/aios metadata                         # Current metadata
python aios/aios -v exec kernel.process_management  # Single action

# Tools
python -m tools.aurorascan --help                 # Tool help
python -m tools.aurorascan 192.168.0.0/24 --json  # JSON output
python -m tools.aurorascan --gui                  # GUI interface

# Algorithms
python aios/ml_algorithms.py                      # Check available
python aios/quantum_ml_algorithms.py              # Quantum check

# Manifests
python aios/aios --manifest custom.json --validate  # Validate
AGENTA_FORENSIC_MODE=1 python aios/aios --manifest custom.json -v boot  # Dry run

# Autonomous Learning
python aios/examples/autonomous_discovery_example.py  # Examples
```

---

## Additional Resources

- **Full Documentation:** See `aios/CLAUDE.md`
- **Examples:** `aios/examples/` directory
- **Tool Specifics:** `aios/tools/` and `aios/red-team-tools/`
- **Quantum:** `aios/qulab/README.md` and QuLab2.0 project
- **Tests:** `aios/tests/` for real-world examples
- **Web Interface:** `aios/docs/` (served by `npm start`)

---

## Support & Feedback

Having trouble? Check the CLAUDE.md file for detailed architecture documentation, or examine the example manifests in `aios/examples/`.

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
