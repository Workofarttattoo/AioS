# Epics and Stories: AgentaOS (AI:OS)

## Epic 1: Core Runtime Stabilization
**Goal**: Ensure the core runtime is stable, maintainable, and free of critical defects like binary files and circular dependencies.

### Story 1.1: Restore Source for `runtime.py`
- **Description**: The file `runtime.py` in the root directory appears to be a binary file. This is a critical failure. We must locate the original source code or reverse-engineer its functionality based on `config.py` and `aios` usage.
- **Acceptance Criteria**:
    - `runtime.py` is a valid, readable Python file.
    - `AgentaRuntime` class is fully implemented with `boot()`, `shutdown()`, `execute()`, and `run_sequence()` methods.
    - No binary content remains in the source tree.

### Story 1.2: Restructure Codebase into `agentaos` Package
- **Description**: The root directory is cluttered with modules (`config.py`, `apps.py`, `aios`, etc.). Move these into a proper Python package structure (e.g., `agentaos/`) to avoid import errors and namespace pollution.
- **Acceptance Criteria**:
    - Created `agentaos/` directory with `__init__.py`.
    - Moved `config.py`, `apps.py`, `runtime.py`, `aios` (renamed to `__main__.py` or similar) into `agentaos/`.
    - Updated all imports in `agents/`, `tools/`, and `tests/` to reference `agentaos.*`.

### Story 1.3: Fix Circular Imports in `aios` Script
- **Description**: The `aios` script currently attempts to import from itself or a non-existent `aios` package. Fix the entry point logic.
- **Acceptance Criteria**:
    - `aios` script (or `agentaos/__main__.py`) runs without `ImportError`.
    - `sys.path` manipulation is minimized or removed in favor of proper package installation (`pip install -e .`).

## Epic 2: Testing Infrastructure Overhaul
**Goal**: Establish a reliable, reproducible testing environment that doesn't depend on hardcoded paths or external systems.

### Story 2.1: Remove Hardcoded Paths in Tests
- **Description**: Tests like `tests/test_meta_agents_tier3.py` reference `/Users/noone/...`. These must be replaced with relative paths or environment variables.
- **Acceptance Criteria**:
    - `grep -r "/Users/" .` returns no matches in `tests/`.
    - Tests use `pathlib` or `os.path` to locate resources relative to the project root.

### Story 2.2: Implement Real Unit Tests for Agents
- **Description**: Current tests rely heavily on mocks for `agents.system`, which doesn't exist as a single file. Create dedicated test files for each agent (e.g., `tests/test_kernel_agent.py`, `tests/test_security_agent.py`) that import the actual classes.
- **Acceptance Criteria**:
    - `TestKernelAgent` imports `KernelAgent` from `agents/kernel_agent.py`.
    - Tests verify actual method logic (e.g., `get_system_status`), not just mock returns.
    - CI/CD pipeline (if exists) runs these tests successfully.

## Epic 3: Agent Implementation Verification
**Goal**: Ensure all Meta-Agents defined in `config.py` have corresponding, functional implementations in `agents/`.

### Story 3.1: Verify Kernel Agent Implementation
- **Description**: Check `agents/kernel_agent.py` against `config.py` actions (`process_management`, `memory_management`, etc.). Implement missing methods.
- **Acceptance Criteria**:
    - `KernelAgent` class has methods for all actions defined in `config.py`.
    - `psutil` integration is robust and handles errors (e.g., permission denied).

### Story 3.2: Verify Security Agent Implementation
- **Description**: Check `agents/security_agent.py` (or similar) for `firewall`, `encryption`, `sovereign_suite` actions.
- **Acceptance Criteria**:
    - `SecurityAgent` can invoke tools from `tools/` (AuroraScan, etc.).
    - Firewall management logic handles different OSs (Linux `iptables`, macOS `pf`, Windows `netsh`) or fails gracefully.

## Epic 4: Documentation & Onboarding
**Goal**: Make the project accessible to new developers and ensure documentation matches the code.

### Story 4.1: Consolidate READMEs
- **Description**: There are multiple READMEs (`README.md`, `ECH0_ALEX_TWIN_FLAMES_README.md`, `DISTRIBUTION_README.md`). Consolidate them or clearly separate the "AgentaOS" and "Consciousness" documentation.
- **Acceptance Criteria**:
    - `README.md` is the single source of truth for AgentaOS.
    - "Consciousness" documentation is moved to `docs/consciousness/`.
    - Redundant files are archived or deleted.

### Story 4.2: Create Developer Guide
- **Description**: Create `docs/DEVELOPER_GUIDE.md` explaining how to set up the environment, run tests, and contribute.
- **Acceptance Criteria**:
    - Guide includes `pip install -r requirements.txt`.
    - Guide explains how to run the `aios` CLI.
    - Guide explains the directory structure (referencing `docs/project-context.md`).

## Epic 5: Security Tools Verification
**Goal**: Ensure the integrated security tools in `tools/` are functional and safe to use.

### Story 5.1: Verify AuroraScan
- **Description**: `tools/aurorascan.py` is a network scanner. Verify it runs without errors and produces valid output.
- **Acceptance Criteria**:
    - `python3 tools/aurorascan.py --help` works.
    - Basic scan (e.g., `127.0.0.1`) returns results or appropriate errors.

### Story 5.2: Verify CipherSpear
- **Description**: `tools/cipherspear.py` is an SQL injection tool. Ensure it can detect vulnerabilities in a controlled environment (e.g., local test DB).
- **Acceptance Criteria**:
    - `python3 tools/cipherspear.py --help` works.
    - Tests with a vulnerable app (if available) confirm detection.

### Story 5.3: Verify Tool Dependencies
- **Description**: Check `requirements.txt` for missing dependencies required by tools (e.g., `scapy` for SpectraTrace, `requests` for AuroraScan).
- **Acceptance Criteria**:
    - `pip install -r requirements.txt` installs all necessary packages.
    - Tools launch without `ModuleNotFoundError`.
