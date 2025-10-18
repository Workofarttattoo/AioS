# AI:OS Runtime Prototype

This package hosts the agentic control-plane prototype for **AI:OS**, a concept operating system that coordinates subsystem meta-agents and sub-agents through a declarative manifest.  The runtime now performs real host inspections (process snapshots, load averages, disk free space, firewall status, Docker/Multipass inventory, BitLocker/FileVault state, etc.) on both macOS and Windows so orchestration signals are rooted in the machine state rather than placeholder strings.  The structure is ready for layering virtualization, container orchestration, or hardware drivers on top.

## Layout

- `AgentaOS/AgentaOS` — CLI entrypoint for booting, shutting down, prompt-driven execution, or capability inspection.
- `AgentaOS/__init__.py` — exposes helpers for building the runtime programmatically.
- `AgentaOS/config.py` — declarative manifest describing meta-agents and lifecycle sequences.
- `AgentaOS/runtime.py` — execution engine, context management, and manifest-aware action wiring.
- `AgentaOS/agents/system.py` — concrete meta-agent implementations that query the host OS, networking stack, Docker/virtualization providers, and user profiles while enforcing security checks.
- `AgentaOS/prompt.py` — ensemble intent router combining keyword rules with similarity scoring and pluggable classifiers.
- `AgentaOS/providers.py` — resource provider abstractions that surface Docker, Multipass, and cloud/VPS orchestration hooks.
- `AgentaOS/virtualization.py` — virtualization planning and execution layer for QEMU/libvirt integrations with forensic safeguards.

## Quick Start

```bash
# Inspect status without booting.
python AgentaOS/AgentaOS -v status

# Perform a full boot sequence using the built-in manifest.
python AgentaOS/AgentaOS -v boot

# Execute a specific action after boot (boots implicitly before running exec).
python AgentaOS/AgentaOS -v exec kernel.process_management

# Run a targeted sequence of manifest actions.
python AgentaOS/AgentaOS -v sequence "security.firewall,networking.network_configuration"

# Summarise runtime telemetry without additional commands.
python AgentaOS/AgentaOS -v metadata

# Route a natural-language prompt through the intent router and execute the matching actions.
python AgentaOS/AgentaOS -v prompt "enable firewall and check container load"

# Load a custom manifest authored in JSON.
python AgentaOS/AgentaOS --manifest path/to/manifest.json -v boot

# Focus on the Sovereign Security Toolkit using the bundled response manifest.
python AgentaOS/AgentaOS --manifest AgentaOS/examples/manifest-security-response.json --env AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear -v boot

# Provide environment overrides exposed to sub-agents.
python AgentaOS/AgentaOS --env DOCKER_SOCKET=/var/run/docker.sock -v boot

# Hint the scalability layer to prefer Docker and a cloud stub provider.
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=docker,cloud -v boot

# Query AWS via the configured awscli (credentials/profile must already be set).
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=aws --env AGENTA_AWS_REGION=us-west-2 -v exec scalability.monitor_load

# Query Azure or GCP inventories via local CLIs.
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=azure --env AGENTA_AZURE_SUBSCRIPTION=<sub-id> -v exec scalability.monitor_load
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=gcloud --env AGENTA_GCP_PROJECT=<project> --env AGENTA_GCP_ZONE=<zone> -v exec scalability.monitor_load

# Surface local virtualization readiness (QEMU/libvirt).
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=qemu --env AGENTA_QEMU_IMAGE=/path/to/image.qcow2 -v exec scalability.monitor_load
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=libvirt --env AGENTA_LIBVIRT_DOMAIN=my-domain -v exec scalability.monitor_load
# Set `AGENTA_QEMU_EXECUTE=1` or `AGENTA_LIBVIRT_EXECUTE=1` to actually start guests; defaults are verification-only.
# Virtualization outcomes are recorded under `virtualization.up` / `virtualization.down` metadata keys.
# Launch a graphical QEMU guest (example boots a qcow2 desktop image).
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=qemu --env AGENTA_QEMU_IMAGE=/path/to/desktop.qcow2 \
  --env AGENTA_QEMU_EXECUTE=1 --env AGENTA_QEMU_HEADLESS=0 --env AGENTA_QEMU_DISPLAY=gtk -v exec scalability.monitor_load
# For ISO boots set `AGENTA_QEMU_IMAGE_TYPE=cdrom` and optionally `AGENTA_QEMU_BOOT=d`.
# Toggle `AGENTA_QEMU_DAEMONIZE=0` to keep the QEMU window attached to the invoking terminal.
# Emit virtualization readiness report directly from the CLI.
python AgentaOS/AgentaOS -v virtualization

# Run the auto-detect setup wizard (profiles + Sovereign Security Toolkit prompts).
python AgentaOS/AgentaOS wizard

# Run AuroraScan directly against a subnet using the reimagined toolkit.
# (Run from the repository root so the `tools` package is on PYTHONPATH.)
python -m tools.aurorascan --list-profiles
python -m tools.aurorascan 192.168.0.0/24 --profile recon --zap-targets zap-targets.txt --json
# Import `zap-targets.txt` into OWASP ZAP (File > Import > URLs) to queue passive scans.
# Launch the graphical interface instead of CLI.
python -m tools.aurorascan --gui

# Rehearse database injection chains with CipherSpear.
python -m tools.cipherspear --dsn postgresql://audit@localhost/app --demo --json
python -m tools.cipherspear --gui

# Capture and analyse wireless telemetry with SkyBreaker.
python -m tools.skybreaker capture wlan0 --output capture.json
python -m tools.skybreaker analyze capture.json --json
python -m tools.skybreaker --gui

# Evaluate credential resilience with MythicKey.
python -m tools.mythickey --demo --profile gpu-balanced --json
python -m tools.mythickey --gui

# Inspect packet captures with SpectraTrace (PCAP or JSON inputs).
python -m tools.spectratrace --capture traces/demo.pcap --json
python -m tools.spectratrace --gui

# Drive the SpectraTrace workflow + recipe builder pipeline.
# Quick situational scan with workflow presets and JSON export.
python -m tools.spectratrace --capture traces/demo.pcap --workflow quick-scan --json --output quick-scan.json
# Stream NDJSON packets from stdin and apply the suspicious-http workflow.
cat traces/http.ndjson | python -m tools.spectratrace --stream --workflow suspicious-http --json
# Launch the dockable UI (timeline, heatmap, annotations, plugin insights, recipe composer).
python -m tools.spectratrace --gui

# Coordinate authentication rehearsals using NemesisHydra.
python -m tools.nemesishydra --demo --json
python -m tools.nemesishydra --gui

# Audit host hardening baselines with ObsidianHunt.
python -m tools.obsidianhunt --profile workstation --json
python -m tools.obsidianhunt --gui

# Stage signed payload manifests via VectorFlux.
python -m tools.vectorflux --list-modules
python -m tools.vectorflux --workspace incident-23-071 --module credential-harvest --json
python -m tools.vectorflux --gui

# Inspect virtualization readiness and planned commands.
python AgentaOS/AgentaOS --env AGENTA_PROVIDER=qemu,libvirt -v exec scalability.virtualization_inspect

# Enumerate available libvirt domains for selection.
python AgentaOS/AgentaOS -v virtualization-domains

# Manage application set via supervisor.
python AgentaOS/AgentaOS --env AGENTA_APPS_CONFIG=AgentaOS/examples/apps-sample.json --env AGENTA_SUPERVISOR_CONCURRENCY=4 -v exec application.supervisor

# Activate forensic collection mode so agents avoid stateful mutations.
python AgentaOS/AgentaOS --forensic -v boot
# or disable safeguards explicitly when safe to mutate.
python AgentaOS/AgentaOS --mutable -v boot
# Skip interactive menu or dashboard if running unattended.
python AgentaOS/AgentaOS --no-menu --no-dashboard -v boot

# Capture integrity and volume snapshots on demand.
python AgentaOS/AgentaOS --forensic -v exec security.integrity_survey
python AgentaOS/AgentaOS --forensic -v exec storage.volume_inventory

# Generate an init-style supervisor report from captured telemetry.
python AgentaOS/AgentaOS --forensic -v exec orchestration.supervisor_report

# Enforce a pf firewall profile during the security pass (view-only without sudo).
python AgentaOS/AgentaOS --env AGENTA_FIREWALL_PROFILE=pfctl -v exec security.firewall

# Windows-only: surface BitLocker + Defender firewall details.
python AgentaOS/AgentaOS --env AGENTA_FIREWALL_PROFILE=windows -v exec security.firewall
```

The CLI accepts `-v` (info) and `-vv` (debug) flags for additional logging.

## Live USB Builder

The `build/` directory includes a Debian `live-build` scaffolding so AI:OS can boot directly from a USB stick without an intermediate ISO download. Run `bash build/build-image.sh` (inside Debian/Ubuntu or the provided Docker image) to produce `dist/agentaos.img`, then flash it with `sudo dd if=dist/agentaos.img of=/dev/sdX bs=4M status=progress oflag=sync`. See `docs/live-usb.md` for detailed instructions, package overrides, and guidance on adding an encrypted persistence partition.

### Container Image

A minimal Docker image is provided so you can run AI:OS without installing Python on the host.

```bash
# Build the image
docker build -t agentaos:latest .

# Run in mutable mode with the default interactive dashboard disabled
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  agentaos:latest --no-menu --no-dashboard -v boot

# Example forensic run against the host (read-only mounts)
docker run --rm -it \
  -v /:/host:ro \
  -e AGENTA_PROVIDER=docker \
  agentaos:latest --forensic --no-menu --no-dashboard -v boot
```

The image copies only the `AgentaOS/` package, keeping size minimal.

### Dashboard Prototype

Stream worker telemetry over the in-memory GUI bus by running:

```bash
python -m scripts.compositor
```

Pass `--manifest path/to/manifest.json` to target an alternate runtime definition. The prototype prints live updates and is ready to be swapped for a full Wayland compositor once available.


### Application Supervisor

Define the tools you want the supervisor to run in a JSON manifest (see `AgentaOS/examples/apps-sample.json`) and set `AGENTA_APPS_CONFIG` to its path:

```json
[
  {
    "name": "logtail",
    "mode": "process",
    "command": ["/usr/bin/tail", "-f", "/var/log/system.log"],
    "restart": "always"
  },
  {
    "name": "reports",
    "mode": "docker",
    "image": "alpine:3.19",
    "args": ["sh", "-c", "echo report && sleep 5"],
    "restart": "on-failure",
    "max_restarts": 2
  }
]
```

Additional environment variables:

- `AGENTA_SUPERVISOR_CONCURRENCY` (default: 10) – maximum apps to run at once.
- `AGENTA_QEMU_*` / `AGENTA_LIBVIRT_*` – optional virtualization hooks for QEMU/libvirt-backed apps (e.g., `AGENTA_QEMU_IMAGE`, `AGENTA_QEMU_EXECUTE`, `AGENTA_QEMU_HEADLESS`, `AGENTA_QEMU_DISPLAY`, `AGENTA_QEMU_IMAGE_TYPE`, `AGENTA_QEMU_BOOT`, `AGENTA_QEMU_EXTRA_ARGS`, plus advanced knobs like `AGENTA_QEMU_PASSTHROUGH_DEVICES`, `AGENTA_QEMU_NET_BRIDGE`, `AGENTA_QEMU_NETDEV_ID`, `AGENTA_QEMU_NET_DEVICE`, `AGENTA_QEMU_MANAGED_SHUTDOWN`, and `AGENTA_QEMU_QMP` for device passthrough, bridge pre-flight checks, and managed shutdown signalling).
- `AGENTA_SECURITY_SUITE`, `AGENTA_SECURITY_TOOLS`, `AGENTA_SECURITY_PROFILE` – defined by the setup wizard when the Sovereign Security Toolkit is enabled.
- `AGENTA_SUPERVISOR_EXECUTE` is implied by the individual provider flags (`AGENTA_QEMU_EXECUTE`, `AGENTA_LIBVIRT_EXECUTE`).
- Virtualization telemetry appears under `virtualization.status`, `virtualization.up`, and `virtualization.down` metadata keys.

Launch the supervisor with:

```bash
python AgentaOS/AgentaOS --env AGENTA_APPS_CONFIG=apps.json -v exec application.supervisor
```

Results appear in the dashboard and metadata, including PID, exit code, stdout/stderr (truncated), and restart counts.

## Extending the Prototype

1. Author a manifest derived from `AgentaOS/config.py::DEFAULT_MANIFEST` to add or reorder meta-agents.
2. Provide concrete sub-agent handlers either by extending the classes in `AgentaOS/agents/system.py` or by supplying a custom action library to `AgentaRuntime`.
3. Use the `ExecutionContext` inside every handler to publish telemetry, cache discovery data, or read environment overrides (e.g., `DOCKER_SOCKET`, `AGENTA_PROVIDER`, `AGENTA_FIREWALL_PROFILE`, `AGENTA_CGROUP_SLICE`).
4. Hook real container or VM provisioning into `ScalabilityAgent` by implementing custom providers in `AgentaOS/providers.py` that call Kubernetes, Nomad, cloud SDKs, or hypervisor APIs.  The bundled `AwsCliProvider` demonstrates how to tie into existing CLI tooling without bundling SDK credentials.
5. Integrate resource isolation (cgroups, job objects, firewalls, sandboxing) by expanding `SecurityAgent`; privileged operations remain opt-in through environment flags so restricted environments still succeed with read-only telemetry.
6. Swap in bespoke prompt classifiers (LLM calls, intent microservices) by providing alternatives to the default ensemble when constructing `PromptRouter`.
7. Extend the probabilistic oracle in `AgentaOS/oracle.py` with domain-specific models or external inference engines if higher fidelity forecasts are required.
8. Orchestrate targeted flows via `AgentaRuntime.run_sequence(...)` to validate manifest subsets while capturing consistent telemetry.
9. Export execution context data with `AgentaRuntime.metadata_snapshot()` or synthesise dashboards using `AgentaRuntime.metadata_summary()`.
10. Extend virtualization orchestration in `AgentaOS/virtualization.py` to support additional hypervisors or custom command flows.

## Setup Wizard & Security Toolkit

- `python AgentaOS/AgentaOS wizard` auto-detects providers, discovers guest images, and writes `agenta-wizard-profile.json` with recommended overrides.
- The wizard exposes pre-build profiles (minimal telemetry, virtualization lab, security response deck) so you can bootstrap common lab layouts quickly.
- The wizard emits a validation report covering OS detection, provider fallback recommendations, image discovery counts, and display compatibility so you can spot missing binaries or misconfigured displays before boot. The structured checks live under `payload["validation"]["checks"]` for automation pipelines.
- Enabling the Sovereign Security Toolkit sets `AGENTA_SECURITY_SUITE`, `AGENTA_SECURITY_TOOLS`, and `AGENTA_SECURITY_PROFILE` while cataloguing renamed tooling for audits, then immediately runs `security.sovereign_suite` through the runtime to confirm tool health.
- Any runtime invoked with `AGENTA_SECURITY_TOOLS` now auto-triggers `security.sovereign_suite` after boot or targeted sequences, ensuring dashboards capture the Sovereign telemetry even when you skip the wizard flow.
- The ML algorithms suite (`AgentaOS/ml_algorithms.py`) provides state-of-the-art implementations including Mamba/SSM architecture, flow matching, Monte Carlo tree search with neural guidance (AlphaGo-style), Bayesian inference (NUTS HMC, particle filters, variational), sparse Gaussian processes, and neural architecture search. Run `python AgentaOS/ml_algorithms.py` to see the full catalog with dependency status.
- The quantum-enhanced ML suite (`AgentaOS/quantum_ml_algorithms.py`) provides quantum computing algorithms including quantum state simulation (1-50 qubits), Variational Quantum Eigensolver (VQE), and foundations for quantum machine learning. Run `python AgentaOS/quantum_ml_algorithms.py` or `python AgentaOS/examples/quantum_ml_example.py` to test quantum capabilities.
- A quantum patent discovery scaffold (`AgentaOS/quantum/patent_discovery.py`) bundles a Claude-aligned simulator, patent meta-agent, FastAPI app factory, and SwiftUI client source for rapid prototyping.
- A high-level overview of the four patentable innovations delivered alongside this project lives in `docs/patent_portfolio.md`.
- Agents can instantiate the lightweight quantum state engine on demand through `AgentaOS.agentaos_load("quantum.engine", num_qubits=2)` which returns the new simulator from `AgentaOS.quantum`.
- A minimal manifest optimised for the security deck lives at `AgentaOS/examples/manifest-security-response.json` and pairs with the new tests under `tests/`.
- Run `python scripts/install_security_toolkit.py` to populate `bin/` with runnable shims for the Sovereign utilities and add that directory to your `PATH` for one-line access outside the repo root.
- Set `AGENTA_QEMU_NET_AUTOCREATE=1` to have the virtualization planner emit executable `ip`/`tuntap` steps for bridge/tap wiring and cleanly tear them down on shutdown; even when disabled, the metadata now lists recommended commands for manual setup.
- At boot, the runtime prints a host-prep checklist covering antivirus exclusions and firewall adjustments; set `AGENTA_DISABLE_POST_BOOT_GUIDANCE=1` to suppress the reminders in kiosk or demo builds.
- Reimagined utilities include AuroraScan (network recon), CipherSpear (database exploitation rehearsal), SkyBreaker (wireless auditing), MythicKey (credential analysis), SpectraTrace (packet inspection), NemesisHydra (auth testing), ObsidianHunt (hardening), and VectorFlux (payload lab).
- Every Sovereign utility accepts a `--gui` switch that launches its Tk workspace when Tkinter is installed; if GUI libraries are absent the CLI now degrades gracefully with a friendly hint to stay in terminal mode.
- A quantum patent discovery scaffold (`AgentaOS/quantum/patent_discovery.py`) bundles a Claude-aligned simulator, patent meta-agent, FastAPI app factory, and SwiftUI client source for rapid prototyping.

## Deployment & Operations

- Install backend dependencies with `pip install -r requirements.txt` (use `requirements-dev.txt` for local testing).
- Containerize the API with the root `Dockerfile`; run locally via `docker compose up patent-api` or deploy with the manifests under `k8s/`.
- The FastAPI factory exposes `/healthz` for liveness/readiness probes; container health checks invoke this endpoint by default.
- Prometheus metrics are available at `/metrics` (powered by `prometheus-fastapi-instrumentator`); set `LOG_FORMAT=json` for structured logs and adjust log verbosity with `LOG_LEVEL`.
- Secrets are supplied via environment variables (`USPTO_USERNAME`, `USPTO_API_KEY`, `STRIPE_SECRET_KEY`, etc.). Populate `.env` from `.env.example` locally, and mirror the values in `k8s/secret-example.yaml` (base64-encode before applying). Outbound HTTP calls stay disabled unless `ALLOW_NETWORK_CALLS=true` is explicitly set.
- GitHub Actions CI (`.github/workflows/ci.yml`) installs dependencies and runs `pytest` to guard regressions.

### Base ISO Recommendation

- Start from the latest **Parrot Security (Home Edition)** ISO when crafting an AI:OS-branded live image. It already bundles the pen-testing toolchain we mirror, supports rolling updates, and is easy to slim down for kiosks or “lite” GUIs.
- If you prefer a Debian base, the **Kali Linux Light** ISO offers a minimal XFCE desktop and curated offensive tooling; replace Kali branding, prune heavyweight metasploit modules, and bake in the Sovereign wrappers for a turnkey experience.
- Whichever base you choose, make sure to script post-install steps (user creation, package pinning, Sovereign toolkit sync) so the environment is reproducible and can inherit the post-boot checklist described above.

### SpectraTrace Workflows & Recipe Modules

- **Workflows (`--workflow`)**: `quick-scan`, `latency-troubleshoot`, and `suspicious-http` load curated playbook rules and packet limits tailored for rapid SITREP, TCP triage, or suspicious web traffic hunts. Use `python -m tools.spectratrace --workflow quick-scan --capture traces/demo.pcap --json` to export enriched telemetry.
- **Recipe builder (GUI tab)** lets analysts compose staged pipelines:
  - **Enrich** modules (e.g., TLS JA3, HTTP metadata) extend packets with additional context.
  - **Detect** modules (behavioral scoring, custom DSL) layer heuristics and rule checks.
  - **Respond** modules (alert routing, annotation push) prime downstream automations.
- Enabling a module auto-merges its playbook rules into the live capture so alerts/annotations match the selected recipe.
- Recipe previews surface the generated JSON (module count + stage layout) for copy/paste into git or automation repos. Combine with `--stream` to pipe NDJSON data into the same analysis flow.
- Plugin architecture: drop Python scripts exposing `process_packets(packets, analysis)` into `tools/plugins/` to add bespoke insights. CLI responses now include `plugin_insights` so downstream dashboards can visualize custom findings alongside heatmaps and annotations.

### Platform Notes

- **macOS**: Existing telemetry continues to use `system_profiler`, `launchctl`, `pfctl`, and Time Machine probes when present.  No elevated privileges are required, though certain commands (e.g., `pfctl` rule dumps) may return warnings without sudo.
- **Windows**: Meta-agents rely on PowerShell cmdlets such as `Get-Process`, `Get-NetAdapter`, `Get-BitLockerVolume`, `Get-NetFirewallProfile`, `Get-Service`, and `Get-PhysicalDisk`.  They execute in read-only mode by default; administrator prompts are not triggered.  If PowerShell is missing (server core scenarios), agents fall back to telemetry warnings rather than failing the boot sequence.

Critical actions are enforced via the manifest—if a critical handler returns `success=False`, the boot sequence halts. Non-critical handlers log warnings and allow the system to continue so you can progressively harden the OS while still collecting telemetry about missing capabilities.

### Cloud Provider Integration

- **AWS CLI**: Set `AGENTA_PROVIDER=aws` and optionally `AGENTA_AWS_REGION` / `AGENTA_AWS_PROFILE` to route scalability telemetry through the locally configured AWS CLI.  The runtime will surface instance counts and mark scale-up/down intents that you can feed into IaC or deployment tooling.
- **Azure CLI**: Use `AGENTA_PROVIDER=azure` and optionally `AGENTA_AZURE_SUBSCRIPTION` to summarise VM inventory via `az vm list`.  All instructions remain advisory when forensic mode is active.
- **GCP gcloud**: Use `AGENTA_PROVIDER=gcloud` with `AGENTA_GCP_PROJECT` / `AGENTA_GCP_ZONE` to surface Compute Engine instance counts using the local `gcloud` CLI.
- **Custom Providers**: Extend `AgentaOS/providers.py` with additional classes following the bundled patterns.  Implement `inventory`, `scale_up`, and `scale_down` to return structured `ProviderReport` payloads.

### Forensic Collection Mode

Set `AGENTA_FORENSIC_MODE=1` (or pass `--forensic`) to ensure agents avoid any host mutations.  In this mode:

- Scale-up/scale-down actions log recommendations without executing orchestration.
- Oracle guidance highlights immutability expectations.
- All telemetry continues to be gathered in-memory only, leaving no persistent footprint.

You can also toggle this behaviour via CLI flags:

- `--forensic` enables read-only mode for the current command (default is mutable).
- `--mutable` explicitly disables the safeguards (equivalent to `AGENTA_FORENSIC_MODE=0`).
- `--no-menu` skips the interactive boot menu prompts.
- `--no-dashboard` skips the post-boot dashboard.
