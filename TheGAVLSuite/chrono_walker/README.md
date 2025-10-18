# Chrono-Walker Governance Console

Chrono-Walker fuses Bayesian cadence planning with a Halo-inspired command console. The project ships a shared analytics engine, a CLI for automation, and a web UI that projects Monte Carlo trajectories with neon sci-fi aesthetics.

## Features
- Beta-Bernoulli evidence ingestion backed by a CSV ledger.
- Monte Carlo forecasting with optimistic/neutral/pessimistic drift profiles.
- Cadence solver that estimates events required to enter a target belief band.
- Band enforcement scanner that highlights periods drifting outside guardrails.
- Neon UI with tabbed workflows (ledger, forecast, cadence planner, band guard).

## Repository Layout
```
backend/
  chrono_walker/   # Core analytics, ledger helpers, configuration
  cli.py           # CLI entry point
  server.py        # HTTP server serving API + static UI
frontend/
  index.html       # Halo-inspired console
  styles.css       # Styling + CRT overlay effects
  app.js           # UI logic and API hooks
data/
  chrono_evidence_ledger.csv  # Sample ledger seeded with one entry
requirements.txt   # Python dependencies (numpy)
```

## Setup
1. (Optional) create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## CLI Usage
Run any command via the backend package:
```bash
python -m backend.cli --help
```
Examples:
```bash
# Append a new evidence record
python -m backend.cli add-evidence --field "wetware×AI×quantum" --kind benchmark \
  --strength 0.8 --outcome 0.9 --title "VQC denoising"

# Generate a short-horizon forecast
python -m backend.cli forecast --periods 4 --events_per_period 3 --runs 500

# Solve cadence to reach 0.7 belief in 6 periods
python -m backend.cli plan-cadence --band_low 0.7 --periods 6

# Validate cadence stays within [0.55, 0.8]
python -m backend.cli enforce-band --low 0.55 --high 0.8
```

## Web Console
Launch the server and open the console:
```bash
python -m backend.server --host 127.0.0.1 --port 8765
```
Navigate to `http://127.0.0.1:8765` to access the UI. The console includes tabs for:
- **Ledger**: add evidence and review the latest signals.
- **Forecast**: trigger Monte Carlo projections and inspect percentile bands.
- **Cadence Planner**: compute events per period needed to cross a belief threshold.
- **Band Guard**: test whether the current cadence keeps belief inside a guard band.

The UI interacts with JSON endpoints exposed under `/api/*` (see `backend/server.py`).

## Data & Storage
- Default ledger path: `data/chrono_evidence_ledger.csv`.
- The server and CLI create the file on first write if it is missing.
- Adjust the path by passing `--ledger` to CLI commands or modifying `DEFAULT_LEDGER_PATH` in `backend/chrono_walker/config.py`.

## Next Steps
- Add automated tests (e.g., `pytest`) covering core analytics.
- Integrate lightweight charting (e.g., WebGL radar) for the forecast tab.
- Package as a Python module or container for easier deployment.
