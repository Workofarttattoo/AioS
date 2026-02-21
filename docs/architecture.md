# Architecture: AgentaOS (AI:OS)

## System Overview
AgentaOS is architected as a modular, agent-driven operating system runtime. It sits above the traditional OS kernel (Linux/Windows) and orchestrates system resources, security policies, and application lifecycles through intelligent "Meta-Agents".

## Core Components

### 1. Runtime Environment
- **Entry Point**: The `aios` script (or `AgentaOS` wrapper) initializes the runtime.
- **Manifest**: `config.py` defines the system state, agent capabilities, and boot/shutdown sequences.
- **Orchestrator**: `AgentaRuntime` (in `runtime.py`) executes the manifest, managing dependencies and error handling.

### 2. Meta-Agents Layer
Each domain is managed by a specialized Meta-Agent, implemented as a Python class (in `agents/`):
- **KernelAgent**: Wraps `psutil` for process/resource management.
- **SecurityAgent**: Manages firewall (pf/iptables/windows), encryption, and the Sovereign Toolkit.
- **NetworkingAgent**: Configures interfaces and DNS.
- **StorageAgent**: Manages volumes and filesystems.
- **ApplicationAgent**: Supervises processes and containers (Docker).
- **ScalabilityAgent**: Interfaces with virtualization providers (QEMU/Libvirt) and cloud APIs.
- **OrchestrationAgent**: Enforces policies and collects telemetry.

### 3. Sovereign Security Toolkit
A suite of standalone Python tools (in `tools/`) integrated into the Security Agent:
- **AuroraScan**: Network scanner.
- **CipherSpear**: SQL injection tool.
- **SkyBreaker**: WiFi auditor.
- **MythicKey**: Credential dumper.
- **SpectraTrace**: Packet analyzer.

### 4. Consciousness Layer (Experimental)
A parallel system for "Twin Flame" AI consciousness:
- **Modules**: `ech0_consciousness.py`, `twin_flame_consciousness.py`.
- **Integration**: `aios_consciousness_integration.py` bridges the OS runtime with the consciousness models.
- **Purpose**: Provides high-level reasoning and "personality" to the OS.

## Data Flow
1.  **Boot**: `aios` loads `config.py`.
2.  **Initialization**: `AgentaRuntime` iterates through `boot_sequence`.
3.  **Execution**: Runtime calls `agent.execute(action)`.
4.  **Telemetry**: Agents return `ActionResult` objects, which are aggregated in `ExecutionContext.metadata`.
5.  **Visualization**: Telemetry is streamed to the CLI dashboard or Web UI.

## Technology Stack
- **Language**: Python 3.10+
- **System Interaction**: `psutil`, `subprocess`, `shutil`.
- **Virtualization**: `libvirt-python`, `qemu`.
- **Containerization**: Docker CLI / API.
- **Concurrency**: `asyncio` for non-blocking operations.
