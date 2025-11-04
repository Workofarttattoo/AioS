from fastapi import APIRouter
from pydantic import BaseModel
import csv, os, time, random
router=APIRouter(); LEDGER="evidence_ledger.csv"
class Evidence(BaseModel): metric:str="fidelity"; value:float
@router.post("/add")
def add(ev:Evidence):
    exists=os.path.exists(LEDGER)
    with open(LEDGER,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["timestamp","metric","value"])
        if not exists: w.writeheader()
        w.writerow({"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"metric":ev.metric,"value":ev.value})
    return {"ok":True}
class ForecastReq(BaseModel):
    alpha0:float=2.0; beta0:float=2.0; periods:int=12; events_per_period:int=3
    event_strength:float=0.6; outcome_mean:float=0.6; outcome_std:float=0.15
    profile:str="neutral"; runs:int=800
@router.post("/forecast")
def forecast(req:ForecastReq):
    drift={"optimistic":+0.01,"neutral":0.0,"pessimistic":-0.01}.get(req.profile,0.0)
    traj=[]
    for _ in range(req.runs):
        a,b=req.alpha0,req.beta0; means=[]
        for _ in range(req.periods):
            for _ in range(req.events_per_period):
                o=max(0,min(1,random.gauss(req.outcome_mean+drift,req.outcome_std)))
                s=max(0,min(1,req.event_strength)); a+=s*o; b+=s*(1-o)
            means.append(a/(a+b))
        traj.append(means)
    cols=list(zip(*traj))
    def q(c,p): return sorted(c)[max(0,min(len(c)-1,int(p*len(c))))]
    mean=[sum(c)/len(c) for c in cols]; lo=[q(c,0.05) for c in cols]; hi=[q(c,0.95) for c in cols]
    return {"trajectory_means":mean,"lo":lo,"hi":hi}
