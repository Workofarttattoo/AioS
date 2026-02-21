# Project Context: AgentaOS (AI:OS)

## Overview
AgentaOS is a prototype for an agentic operating system that coordinates subsystem meta-agents and sub-agents through a declarative manifest. It aims to provide real host inspections, virtualization orchestration, and a comprehensive security toolkit.

## Core Architecture
The system is built around a "Meta-Agent" architecture where distinct agents manage specific domains:
- **Kernel Agent**: Process management, system initialization.
- **Security Agent**: Firewall, encryption, integrity, sovereign toolkit.
- **Networking Agent**: Network configuration, DNS, routing.
- **Storage Agent**: Volume management, filesystem operations.
- **Application Agent**: Supervisor, Docker/VM orchestration.
- **Scalability Agent**: Load monitoring, virtualization (QEMU/Libvirt).
- **Orchestration Agent**: Policy engine, telemetry.

These agents are configured via `config.py` which defines a "Manifest" of actions and criticalities.

## File Structure & Organization
- **Root Directory**: Contains core modules (`config.py`, `apps.py`, `runtime.py`) and agent definitions (`agents/`).
- **`agents/`**: Contains Python implementations of the meta-agents (e.g., `kernel_agent.py`, `system.py`).
- **`tools/`**: Hosts the "Sovereign Security Toolkit" (e.g., `aurorascan.py`, `cipherspear.py`).
- **`tests/`**: Contains unit and integration tests, though some appear to rely on hardcoded paths or external environments.

## Current State & Challenges
- **Directory Structure**: The project root is cluttered. Core modules should ideally be in a package (e.g., `agentaos/`).
- **Runtime Integrity**: `runtime.py` appears to be a binary file, which is critical to investigate/restore.
- **Testing**: Tests like `tests/test_meta_agents_tier3.py` reference hardcoded paths (`/Users/noone/...`) and rely heavily on mocks.
- **Documentation**: While extensive READMEs exist, they reference directories (`AgentaOS/`) that do not match the current file structure.

## Development Standards
- **Language**: Python 3.10+
- **Configuration**: Declarative manifests (Python dataclasses or JSON).
- **Concurrency**: `asyncio` for the supervisor and agent operations.
- **Dependencies**: `psutil`, `docker`, `libvirt`, `qemu`.
