#!/usr/bin/env bash
set -euo pipefail

# Headless smoke harness for CI/CD or cron jobs. It performs:
#   1. Non-interactive boot (forensic mode, dashboard disabled)
#   2. Level-8 autonomy mission (ech0 persona)
#   3. Metadata dump and graceful shutdown
#
# Logs land in logs/autonomy/level8 so reviewers can replay every cycle.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/autonomy/ci"
mkdir -p "${LOG_DIR}"

echo "[info] Booting AgentaOS runtime (forensic mode)..."
python "${ROOT_DIR}/aios_shell.py" boot --no-menu --no-dashboard --forensic > "${LOG_DIR}/boot.log" 2>&1

echo "[info] Running Level-8 autonomy sweep..."
python "${ROOT_DIR}/aios_shell.py" autonomy \
  --level 8 \
  --mission "stabilize climate risk via federated micro-grids" \
  --cycles 2 \
  > "${LOG_DIR}/autonomy.log" 2>&1

echo "[info] Capturing metadata snapshot..."
python "${ROOT_DIR}/aios_shell.py" metadata > "${LOG_DIR}/metadata.log" 2>&1 || true

echo "[info] Shutting down runtime..."
python "${ROOT_DIR}/aios_shell.py" shutdown --no-dashboard > "${LOG_DIR}/shutdown.log" 2>&1 || true

echo "[info] Autonomy CI sweep completed. Logs -> ${LOG_DIR}"
