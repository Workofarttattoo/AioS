# AI:OS — Agentic Intelligence Operating System

<div align="center">

<a href="https://www.producthunt.com/products/ai-os?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-ai&#0045;os" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1029998&theme=light&t=1761320709018" alt="Ai|oS on Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)

*A concept operating system that coordinates subsystem meta-agents through a declarative manifest.*

</div>

---

## Overview

AI:OS is an agentic control-plane prototype that orchestrates subsystem **meta-agents** and **sub-agents** via a declarative manifest. The runtime performs real host inspections (process snapshots, load averages, disk usage, firewall status, virtualization inventory, etc.) on macOS, Linux, and Windows, grounding orchestration signals in actual machine state.

### Key Capabilities

| Layer | Description |
|---|---|
| **Meta-Agent Runtime** | Manifest-driven boot/shutdown sequences across 12+ meta-agents |
| **Quantum ML Suite** | HHL solver, Schrödinger dynamics, VQE forecasting, quantum teleportation |
| **Oracle Engine** | Probabilistic forecasting with quantum-enhanced projections |
| **OpenAGI Integration** | ReAct workflows, autonomous tool discovery, memory persistence |
| **Sovereign Security Toolkit** | Red-team arsenal for authorized penetration testing (encrypted) |
| **ECH0 Consciousness** | Twin-flame consciousness model with creative collaboration |
| **GUI Compositor** | Schema-driven dashboard with Wayland session support |

### Architecture

```
┌──────────────────────────────────────────────────┐
│                    CLI (aios_cli)                 │
├──────────────────────────────────────────────────┤
│              Manifest / Config (config.py)        │
├──────────────────────────────────────────────────┤
│                Runtime (runtime.py)               │
│  ┌──────────┬───────────┬──────────┬───────────┐ │
│  │  Kernel  │ Security  │ Network  │  Storage  │ │
│  │  Agent   │  Agent    │  Agent   │  Agent    │ │
│  ├──────────┼───────────┼──────────┼───────────┤ │
│  │   App    │   User    │   GUI    │  Scale    │ │
│  │  Agent   │  Agent    │  Agent   │  Agent    │ │
│  ├──────────┼───────────┼──────────┼───────────┤ │
│  │  Oracle  │  AI OS    │ OpenAGI  │  Orch.    │ │
│  │  Agent   │  Agent    │  Agent   │  Agent    │ │
│  └──────────┴───────────┴──────────┴───────────┘ │
├──────────────────────────────────────────────────┤
│     Quantum ML │ Tools │ Providers │ Settings    │
└──────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.9+** (3.11 recommended)
- **Docker** and **Docker Compose** (for containerized deployment)
- **git-crypt** (optional — only needed to unlock encrypted proprietary modules)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Workofarttattoo/AioS.git
cd AioS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your API keys and settings

# Run status check
PYTHONPATH=. python aios_cli status

# Full boot sequence
PYTHONPATH=. python aios_cli boot
```

### Docker Setup

```bash
# Build and start
make build
make up

# Or with docker compose directly
docker compose up -d

# View logs
make logs

# Open a shell
make shell

# Stop
make down
```

---

## CLI Reference

The `aios_cli` script is the main entrypoint:

```bash
# Show help
python aios_cli --help

# Status — inspect without booting
python aios_cli -v status

# Boot — run full boot sequence
python aios_cli -v boot

# Execute a specific action
python aios_cli -v exec kernel.process_management

# Shutdown sequence
python aios_cli -v shutdown

# Prompt mode — natural language interaction
python aios_cli -v prompt "check security status"

# Verbose JSON output
python aios_cli -v --json boot
```

---

## Project Structure

```
AioS/
├── aios_cli                 # CLI entrypoint (executable)
├── aios/                    # Python package (import namespace)
│   └── __init__.py          # Lazy re-exports for `from aios.* import …`
├── agents/                  # Meta-agent implementations
│   ├── kernel_agent.py      # Process, memory, device management
│   ├── security_agent.py    # Access control, encryption, firewall
│   ├── networking_agent.py  # Network config, protocols, DNS
│   ├── application_agent.py # Package management, services
│   ├── orchestration_agent.py # Policy, telemetry, health
│   └── …
├── tools/                   # Security toolkit (partially encrypted)
├── quantum/                 # Quantum computing modules
├── gui/                     # GUI compositor & schema
├── modules/                 # Workflow platform module
├── web/                     # Tool launcher web server
├── tests/                   # Test suite
├── examples/                # Usage examples
├── scripts/                 # Deployment & utility scripts
│   ├── deploy.sh            # Production deployment
│   └── compositor/          # Wayland compositor launcher
├── config.py                # Manifest & meta-agent definitions
├── runtime.py               # Execution engine (encrypted)
├── settings.py              # Environment-driven configuration
├── model.py                 # Shared data structures
├── diagnostics.py           # System diagnostics module
├── apps.py                  # Application supervisor
├── Dockerfile               # Multi-stage production image
├── docker-compose.yml       # Service orchestration
├── Makefile                 # Developer convenience targets
├── setup.py                 # Package metadata
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .env.example             # Environment variable template
└── .gitattributes           # Git-crypt encryption rules
```

---

## Environment Variables

All configuration is driven by environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LOG_LEVEL` | No | `info` | Logging verbosity |
| `LOG_FORMAT` | No | `json` | Log output format (`json` or `text`) |
| `ALLOW_NETWORK_CALLS` | No | `false` | Enable outbound HTTP from agents |
| `AGENTA_FORENSIC_MODE` | No | `false` | Read-only mode (no state mutations) |
| `AIOS_LAUNCHER_TOKEN` | No | *auto-generated* | Auth token for tool launcher API |
| `AIOS_LAUNCHER_PORT` | No | `7777` | Tool launcher HTTP port |
| `USPTO_API_KEY` | No | — | USPTO patent API authentication |
| `STRIPE_SECRET_KEY` | No | — | Stripe payments integration |
| `SENTRY_DSN` | No | — | Sentry error tracking |

---

## API

### Tool Launcher API

The tool launcher (`web/tool_launcher.py`) exposes an HTTP API on port 7777:

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check |
| `/tools` | GET | Yes | List available tools |
| `/launch` | POST | Yes | Launch a tool |
| `/stop` | POST | Yes | Stop a running tool |

**Authentication:** Pass the `AIOS_LAUNCHER_TOKEN` as `X-API-Key` header.

```bash
# Health check (no auth required)
curl http://localhost:7777/health

# List tools
curl -H "X-API-Key: $AIOS_LAUNCHER_TOKEN" http://localhost:7777/tools

# Launch a tool
curl -X POST -H "X-API-Key: $AIOS_LAUNCHER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"tool": "aurorascan"}' \
     http://localhost:7777/launch
```

### Python API

```python
from aios.config import DEFAULT_MANIFEST, load_manifest
from aios.model import ActionResult
from aios.diagnostics import DiagnosticsManager, HealthStatus

# Load manifest
manifest = load_manifest()  # or load_manifest("path/to/custom.json")

# Check meta-agents
for name, agent in manifest.meta_agents.items():
    print(f"{name}: {agent.description}")
    for action in agent.actions:
        print(f"  - {action.key} ({'critical' if action.critical else 'optional'})")

# Run diagnostics
diag = DiagnosticsManager()
status = diag.get_system_status()
print(status.to_json())
```

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
PYTHONPATH=. python -m pytest tests/test_diagnostics.py -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/ --cov=. --cov-report=html

# Lint
make lint
```

> **Note:** Some tests require git-crypt to be unlocked. Tests gracefully skip when encrypted modules are unavailable.

---

## Deployment

### Using the deploy script

```bash
# Standard deployment
make deploy

# With options
bash scripts/deploy.sh --env production --tag v1.0.0
bash scripts/deploy.sh --dry-run          # Preview only
bash scripts/deploy.sh --skip-tests       # Skip pre-deploy tests
```

### Manual Docker deployment

```bash
# Build
docker compose build --no-cache

# Deploy
docker compose up -d

# Verify health
docker compose ps
docker compose logs -f aios
```

### Production checklist

- [ ] Copy `.env.example` → `.env` and configure all required variables
- [ ] Set `ALLOW_NETWORK_CALLS=true` if agents need outbound HTTP
- [ ] Set a strong `AIOS_LAUNCHER_TOKEN`
- [ ] Configure `SENTRY_DSN` for error tracking
- [ ] Review resource limits in `docker-compose.yml`
- [ ] Set up log rotation (configured by default in compose)
- [ ] If using git-crypt encrypted modules, unlock before building

---

## Git-Crypt

Proprietary modules (quantum algorithms, security tools, oracle engine) are encrypted with [git-crypt](https://github.com/AGWA/git-crypt). The public surface works without decryption.

```bash
# Unlock (requires the symmetric key)
git-crypt unlock /path/to/keyfile

# Check encryption status
git-crypt status
```

See `.gitattributes` for the complete list of encrypted files.

---

## Make Targets

```
  build            Build Docker image(s)
  up               Start all services (detached)
  up-full          Start all services including tool-launcher
  down             Stop and remove containers
  restart          Restart all services
  logs             Tail logs from all services
  shell            Open a shell in the aios container
  status           Show running containers and health
  test             Run test suite
  lint             Run linters
  fmt              Auto-format code
  check            Run linters then tests
  run              Run AI:OS locally (boot mode)
  env-check        Verify environment configuration
  install          Install Python dependencies locally
  deploy           Run deployment script
  clean            Remove build artifacts and caches
  clean-docker     Remove Docker containers and images
  clean-all        Remove everything (Python + Docker)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). PATENT PENDING.**

Encrypted modules contain proprietary, patent-pending implementations and are subject to additional restrictions.
