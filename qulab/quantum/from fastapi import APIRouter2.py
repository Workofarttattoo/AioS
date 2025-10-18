from fastapi import APIRouter
from pydantic import BaseModel
import csv, os, time, math, random

router = APIRouter()
LEDGER = "evidence_ledger.csv"

class Evidence(BaseModel):
    metric: str = "fidelity"
    value: float

@router.post("/add")
def add_ev(ev: Evidence):
    exists = os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","metric","value"])
        if not exists: w.writeheader()
        w.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metric": ev.metric, "value": ev.value})
    return {"ok": True}

class ForecastReq(BaseModel):
    alpha0: float = 2.0
    beta0: float = 2.0
    periods: int = 12
    events_per_period: int = 3
    event_strength: float = 0.6
    outcome_mean: float = 0.6
    outcome_std: float = 0.15
    profile: str = "neutral"
    runs: int = 1000

@router.post("/forecast")
def forecast(req: ForecastReq):
    # Beta-Bernoulli MC trajectory
    drift = {"optimistic": +0.01, "neutral": 0.0, "pessimistic": -0.01}.get(req.profile, 0.0)
    trajs = []
    for _ in range(req.runs):
        a, b = req.alpha0, req.beta0
        means = []
        for _p in range(req.periods):
            for _e in range(req.events_per_period):
                o = min(1, max(0, random.gauss(req.outcome_mean+drift, req.outcome_std)))
                s = min(1, max(0, req.event_strength))
                a += s*o; b += s*(1-o)
            means.append(a/(a+b))
        trajs.append(means)
    # summarize
    import statistics
    cols = list(zip(*trajs))
    mean = [statistics.fmean(c) for c in cols]
    lo = [sorted(c)[int(0.05*len(c))] for c in cols]
    hi = [sorted(c)[int(0.95*len(c))-1] for c in cols]
    return {"trajectory_means": mean, "lo": lo, "hi": hi}