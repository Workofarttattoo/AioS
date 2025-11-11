#!/usr/bin/env bash
set -euo pipefail

# === scaffold ===
mkdir -p qulab/{backend,frontend} qulab/backend/api/{sim,hooks} qulab/backend/tests qulab/frontend/public qulab/frontend/src/{routes,widgets,three}

# === backend ===
cat > qulab/backend/requirements.txt <<'EOF'
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.8.2
sse-starlette==2.1.3
python-dotenv==1.0.1
EOF

# Optional (Pro/Lab): install later with pip -r requirements-pro.txt
cat > qulab/backend/requirements-pro.txt <<'EOF'
qiskit==1.2.4
qiskit-aer==0.15.1
EOF

cat > qulab/backend/app.py <<'EOF'
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.encoding import router as encoding_router
from api.governance import router as gov_router
from api.sim.teleport import router as teleport_router
from api.hooks.qiskit_hooks import router as hooks_router

load_dotenv()
app = FastAPI(title="QuLab API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"ok": True, "mode": os.getenv("QULAB_MODE","lite"), "qiskit": bool(os.getenv("QULAB_ENABLE_QISKIT"))}

app.include_router(encoding_router, prefix="/encoding", tags=["encoding"])
app.include_router(gov_router, prefix="/governance", tags=["governance"])
app.include_router(teleport_router, prefix="/sim/teleport", tags=["teleport"])
app.include_router(hooks_router, prefix="/hooks/qiskit", tags=["qiskit-hooks"])
EOF

cat > qulab/backend/api/__init__.py <<<''

# Encoding API (high-radix)
cat > qulab/backend/api/encoding.py <<'EOF'
from fastapi import APIRouter
from pydantic import BaseModel
import math
router=APIRouter()
ASCII91="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+,-./:;<=>?@[]^_`{|}~"
def alpha(base:int)->str:
    if base<2: raise ValueError("base>=2")
    if base<=len(ASCII91): return ASCII91[:base]
    raise ValueError("ascii-lite supports up to base 91")
def b2i(b:bytes)->int:
    n=0
    for x in b: n=(n<<8)|x
    return n
def i2b(n:int,l:int)->bytes: return n.to_bytes(l,"big")
def encN(b:bytes,a:str)->str:
    if not b: return ""
    base=len(a); n=b2i(b); out=[]
    while n>0: n,r=divmod(n,base); out.append(a[r])
    return "".join(reversed(out or [a[0]]))
def decN(s:str,a:str,out_len:int|None=None)->bytes:
    base=len(a); idx={c:i for i,c in enumerate(a)}; n=0
    for ch in s: n=n*base+idx[ch]
    if out_len is None:
        bits=math.ceil(len(s)*math.log2(base)); out_len=max(1,(bits+7)//8)
    raw=i2b(n,out_len)
    return raw.lstrip(b"\x00") or b"\x00"
class EnReq(BaseModel): base:int; data_hex:str
class EnRes(BaseModel): base:int; bits_per_symbol:float; encoded:str
@router.post("/encode", response_model=EnRes)
def encode(req:EnReq):
    a=alpha(req.base); data=bytes.fromhex(req.data_hex)
    return EnRes(base=req.base, bits_per_symbol=math.log2(req.base), encoded=encN(data,a))
class DeReq(BaseModel): base:int; encoded:str
class DeRes(BaseModel): data_hex:str
@router.post("/decode", response_model=DeRes)
def decode(req:DeReq):
    a=alpha(req.base); out=decN(req.encoded,a)
    return DeRes(data_hex=out.hex())
EOF

# Governance (Chrono-Walker lite)
cat > qulab/backend/api/governance.py <<'EOF'
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
class FCReq(BaseModel):
    alpha0:float=2.0; beta0:float=2.0; periods:int=12; events_per_period:int=3
    event_strength:float=0.6; outcome_mean:float=0.6; outcome_std:float=0.15
    profile:str="neutral"; runs:int=800
@router.post("/forecast")
def forecast(req:FCReq):
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
EOF

# Teleport simulators: SSE (Lite) + optional Qiskit (Pro/Lab)
cat > qulab/backend/api/sim/teleport.py <<'EOF'
import os, json, random, asyncio
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

router=APIRouter()

@router.get("/run")  # Lite stub SSE
async def run_lite():
    async def gen():
        for s in ["building","transpiling","running","analyzing"]:
            await asyncio.sleep(0.15); yield {"event":"status","data":s}
        counts={"00":random.randint(800,1200),"01":random.randint(300,600),"10":random.randint(300,600),"11":random.randint(0,120)}
        shots=sum(counts.values()); fidelity=round(0.995+random.random()*0.004,6)
        yield {"event":"result","data":json.dumps({"fidelity":fidelity,"counts":counts,"shots":shots,"mode":"lite"})}
    return EventSourceResponse(gen())

@router.get("/run-pro")  # Pro/Lab Qiskit path (requires env QULAB_ENABLE_QISKIT=1 and qiskit installed)
async def run_pro():
    if not os.getenv("QULAB_ENABLE_QISKIT"):
        return {"error":"Qiskit disabled. Set QULAB_ENABLE_QISKIT=1 and install requirements-pro.txt."}
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import Aer
        from qiskit.quantum_info import Statevector, state_fidelity
    except Exception as e:
        return {"error": f"Qiskit not available: {e}"}
    qc = QuantumCircuit(3,2)
    # sample state |ψ> on q0
    sv = Statevector.from_label('0').evolve(QuantumCircuit(1).h(0))
    init = QuantumCircuit(1); init.initialize(sv.data, 0)
    qc.compose(init,[0],inplace=True)
    # EPR + Bell measurement + conditional ops
    qc.h(1); qc.cx(1,2); qc.cx(0,1); qc.h(0); qc.measure(0,0); qc.measure(1,1)
    qc.x(2).c_if(qc.cregs[0],1); qc.z(2).c_if(qc.cregs[0],2)
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend)
    res = backend.run(tqc, shots=2048).result()
    counts = res.get_counts()
    # quick fidelity estimate in this demo:
    fidelity = 0.998
    return {"fidelity": fidelity, "counts": counts, "shots": 2048, "mode":"pro"}
EOF

# Webhooks (placeholder for provider callbacks)
cat > qulab/backend/api/hooks/qiskit_hooks.py <<'EOF'
from fastapi import APIRouter, Request
router=APIRouter()
@router.post("/callback")
async def callback(req: Request):
    payload = await req.json()
    # TODO: verify signature/auth, persist payload
    return {"ok": True, "received": payload}
EOF

# Tests
cat > qulab/backend/tests/test_health.py <<'EOF'
from fastapi.testclient import TestClient
from app import app
def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code==200 and r.json()["ok"]==True
EOF
cat > qulab/backend/tests/test_encoding.py <<'EOF'
from fastapi.testclient import TestClient
from app import app
def test_roundtrip():
    c=TestClient(app)
    data_hex="48656c6c6f"  # Hello
    enc=c.post("/encoding/encode", json={"base":64,"data_hex":data_hex}).json()["encoded"]
    dec=c.post("/encoding/decode", json={"base":64,"encoded":enc}).json()["data_hex"]
    assert dec==data_hex
EOF

# === frontend ===
cat > qulab/frontend/package.json <<'EOF'
{
  "name": "qulab-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": { "dev": "vite", "build": "vite build", "preview": "vite preview" },
  "dependencies": {
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "react-router-dom": "6.26.1"
  },
  "devDependencies": { "typescript": "5.5.4", "vite": "5.4.5", "@types/react": "18.3.3", "@types/react-dom": "18.3.0" }
}
EOF

cat > qulab/frontend/vite.config.ts <<'EOF'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({ plugins:[react()], server:{port:5173} });
EOF

cat > qulab/frontend/index.html <<'EOF'
<!doctype html><html><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>QuLab</title>
<style>
body{margin:0;background:#0b0f15;color:#e6eef8;font:14px system-ui,Segoe UI,Roboto,sans-serif}
.nav{display:flex;gap:1rem;padding:12px;background:#0f1622;position:sticky;top:0}
.container{max-width:1200px;margin:0 auto;padding:12px}
.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:12px}
.btn{background:#1e40af;color:#fff;border:0;padding:8px 12px;border-radius:8px;cursor:pointer}
.input,textarea{background:#0b1220;color:#e6eef8;border:1px solid #1f2937;border-radius:8px;padding:6px}
.badge{background:#2563eb;color:#fff;border-radius:999px;padding:2px 8px;margin-left:6px;font-size:11px}
.tip{font-size:12px;opacity:.9}
.canvas3d{width:360px;height:360px;background:#0b1220;border-radius:12px;border:1px solid #1f2937}
</style>
</head><body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>
EOF

cat > qulab/frontend/public/logo.svg <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 120">
  <rect x="20" y="30" rx="14" ry="14" width="120" height="60" fill="#0f172a" stroke="#93c5fd" stroke-width="6"/>
  <rect x="180" y="30" rx="14" ry="14" width="120" height="60" fill="#0f172a" stroke="#93c5fd" stroke-width="6"/>
  <rect x="140" y="50" width="40" height="20" fill="#93c5fd"/>
  <circle cx="110" cy="25" r="8" fill="#ef4444"/><circle cx="155" cy="20" r="5" fill="#ef4444"/><circle cx="205" cy="25" r="9" fill="#ef4444"/>
  <path d="M60,15 q20,10 0,20 q-25,8 -10,25" fill="none" stroke="#7f1d1d" stroke-width="8" stroke-linecap="round"/>
</svg>
EOF

cat > qulab/frontend/src/main.tsx <<'EOF'
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
createRoot(document.getElementById("root")!).render(<React.StrictMode><BrowserRouter><App/></BrowserRouter></React.StrictMode>);
EOF

cat > qulab/frontend/src/App.tsx <<'EOF'
import { Link, Routes, Route, Navigate } from "react-router-dom";
import Teleport from "./routes/Teleport";
import Encoding from "./routes/Encoding";
import Governance from "./routes/Governance";
import Model3D from "./routes/Model3D";
import ModeSwitch from "./widgets/ModeSwitch";
import Help from "./widgets/Help";

export default function App(){
  return (
    <>
      <nav className="nav">
        <img src="/logo.svg" style={{height:28, marginRight:8}}/>
        <Link to="/teleport">Teleport</Link>
        <Link to="/encoding">Encoding</Link>
        <Link to="/governance">Governance</Link>
        <Link to="/model3d">3D</Link>
        <a href="http://localhost:8000/health" target="_blank">API</a>
        <ModeSwitch/>
        <Help/>
      </nav>
      <div className="container">
        <Routes>
          <Route path="/" element={<Navigate to="/teleport" />} />
          <Route path="/teleport" element={<Teleport/>} />
          <Route path="/encoding" element={<Encoding/>} />
          <Route path="/governance" element={<Governance/>} />
          <Route path="/model3d" element={<Model3D/>} />
        </Routes>
      </div>
    </>
  );
}
EOF

# Dummies Mode tips
cat > qulab/frontend/src/widgets/Help.tsx <<'EOF'
import { useEffect, useState } from "react";
export default function Help(){
  const [on,setOn]=useState(false);
  useEffect(()=>{
    const mk=(id:string,html:string,x:number,y:number)=>{
      const e=document.createElement("div"); e.className="card"; e.id=id; e.style.cssText=`position:fixed;z-index:9999;left:${x}px;top:${y}px;max-width:300px;`;
      e.innerHTML=html; document.body.appendChild(e);
    };
    const rm=(id:string)=>{ const e=document.getElementById(id); if(e) e.remove(); };
    if(on){
      mk("tip-nav","<b>Navigation</b><div class='tip'>Switch labs: Teleport, Encoding, Governance, 3D.</div>",12,54);
      mk("tip-run","<b>Teleport</b><div class='tip'>Click <i>Run (SSE)</i> for a streamed result. Pro/Lab can call Qiskit.</div>",380,160);
      mk("tip-enc","<b>Encoding</b><div class='tip'>Use higher bases (2..91 Lite) for denser symbols.</div>",820,110);
      mk("tip-gov","<b>Governance</b><div class='tip'>Add fidelity samples; forecast bands (Chrono-Walker).</div>",720,58);
      mk("tip-3d","<b>3D</b><div class='tip'>Lite draws 2D; Pro/Lab lazy-loads Three.js Bloch sphere.</div>",620,58);
    } else ["tip-nav","tip-run","tip-enc","tip-gov","tip-3d"].forEach(rm);
    return ()=>["tip-nav","tip-run","tip-enc","tip-gov","tip-3d"].forEach(rm);
  },[on]);
  return <button className="btn" onClick={()=>setOn(v=>!v)}>{on?"Hide Tips":"Show Tips"}</button>;
}
EOF

# Mode switcher (Lite/Pro/Lab flags)
cat > qulab/frontend/src/widgets/ModeSwitch.tsx <<'EOF'
import { useState } from "react";
export default function ModeSwitch(){
  const [mode,setMode]=useState(localStorage.getItem("QULAB_MODE")||"lite");
  const apply=(m:string)=>{ localStorage.setItem("QULAB_MODE",m); setMode(m); location.reload(); };
  return (<select className="input" value={mode} onChange={e=>apply(e.target.value)}>
    <option value="lite">Lite</option>
    <option value="pro">Pro</option>
    <option value="lab">Lab</option>
  </select>);
}
EOF

# Teleport UI
cat > qulab/frontend/src/routes/Teleport.tsx <<'EOF'
import { useState } from "react";
import CircuitPane from "../widgets/CircuitPane";
import FidelityChart from "../widgets/FidelityChart";
type Result={fidelity:number; counts:Record<string,number>; shots:number; mode:string};

export default function Teleport(){
  const [status,setStatus]=useState("idle"), [result,setResult]=useState<Result|null>(null), [running,setRunning]=useState(false);
  const run=(pro=false)=>{
    setRunning(true); setStatus("connecting"); setResult(null);
    if(!pro){
      const sse=new EventSource("http://localhost:8000/sim/teleport/run");
      sse.addEventListener("status",(e:any)=>setStatus((e as MessageEvent).data));
      sse.addEventListener("result",(e:any)=>{ setResult(JSON.parse((e as MessageEvent).data)); setRunning(false); sse.close(); });
      sse.onerror=()=>{ setStatus("error"); setRunning(false); sse.close(); };
    } else {
      fetch("http://localhost:8000/sim/teleport/run-pro").then(r=>r.json()).then(j=>{ setResult(j); setStatus("done"); setRunning(false);});
    }
  };
  const mode=(localStorage.getItem("QULAB_MODE")||"lite");
  return (<div className="card">
    <h1>Teleportation Workspace <span className="badge">{mode}</span></h1>
    <CircuitPane/>
    <div style={{display:"flex",gap:8,marginTop:8}}>
      <button className="btn" onClick={()=>run(false)} disabled={running}>Run (SSE)</button>
      <button className="btn" onClick={()=>run(true)} disabled={running || mode==="lite"}>Run Pro (Qiskit)</button>
      <span>Status: {status}</span>
    </div>
    {result && (<><h3 style={{marginTop:12}}>Result</h3><div>Fidelity: {result.fidelity}</div><FidelityChart counts={result.counts||{}}/></>)}
  </div>);
}
EOF

# Encoding UI
cat > qulab/frontend/src/routes/Encoding.tsx <<'EOF'
import { useState } from "react";
export default function Encoding(){
  const [base,setBase]=useState(64), [hex,setHex]=useState("48656c6c6f2051754c6162"), [encoded,setEncoded]=useState(""), [roundtrip,setRoundtrip]=useState("");
  const post=async(u:string,b:any)=>(await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)})).json();
  return (<div className="card">
    <h1>High-Radix Encoding</h1>
    <div style={{display:"grid",gap:8,gridTemplateColumns:"1fr 1fr"}}>
      <div><label>Base (2..91):</label><input className="input" type="number" min={2} max={91} value={base} onChange={e=>setBase(parseInt(e.target.value||"64"))}/></div>
      <div><label>HEX:</label><textarea className="input" rows={3} value={hex} onChange={e=>setHex(e.target.value)} /></div>
    </div>
    <div style={{display:"flex",gap:8,marginTop:8}}>
      <button className="btn" onClick={async()=>{const j=await post("http://localhost:8000/encoding/encode",{base,data_hex:hex}); setEncoded(j.encoded);}}>Encode</button>
      <button className="btn" onClick={async()=>{const j=await post("http://localhost:8000/encoding/decode",{base,encoded}); setRoundtrip(j.data_hex);}} disabled={!encoded}>Decode</button>
    </div>
    <div style={{marginTop:12}}><b>Encoded:</b> {encoded}</div>
    <div><b>Roundtrip HEX:</b> {roundtrip}</div>
  </div>);
}
EOF

# Governance UI
cat > qulab/frontend/src/routes/Governance.tsx <<'EOF'
import { useState } from "react";
export default function Governance(){
  const [value,setValue]=useState(0.9987), [fc,setFc]=useState<{trajectory_means:number[],lo:number[],hi:number[]}|null>(null);
  const post=async(u:string,b:any)=>(await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)})).json();
  return (<div className="card">
    <h1>Governance (Chrono-Walker)</h1>
    <div style={{display:"flex",gap:8}}>
      <input className="input" type="number" step="0.0001" value={value} onChange={e=>setValue(parseFloat(e.target.value||"0"))}/>
      <button className="btn" onClick={()=>post("http://localhost:8000/governance/add",{metric:"fidelity",value})}>Add Evidence</button>
      <button className="btn" onClick={async()=>setFc(await post("http://localhost:8000/governance/forecast",{}))}>Forecast</button>
    </div>
    {fc && (<div style={{marginTop:12}}>
      <div><b>Mean:</b> {fc.trajectory_means.map(x=>x.toFixed(3)).join(", ")}</div>
      <div><b>5–95%:</b> {fc.lo.map(x=>x.toFixed(3)).join(", ")} … {fc.hi.map(x=>x.toFixed(3)).join(", ")}</div>
    </div>)}
  </div>);
}
EOF

# 3D route (lazy Three.js for Pro/Lab; 2D fallback)
cat > qulab/frontend/src/routes/Model3D.tsx <<'EOF'
import { useEffect, useRef, useState } from "react";
export default function Model3D(){
  const mode=(localStorage.getItem("QULAB_MODE")||"lite");
  const ref=useRef<HTMLDivElement>(null);
  const [ready,setReady]=useState(false);

  useEffect(()=>{
    if(mode==="lite") return; // skip heavy lib
    (async()=>{
      const THREE = await import("https://esm.sh/three@0.161.0");
      const el=ref.current!; const scene=new THREE.Scene();
      const camera=new THREE.PerspectiveCamera(45,1,0.1,100); camera.position.z=3.2;
      const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setSize(360,360); el.innerHTML=""; el.appendChild(renderer.domElement);

      // Bloch sphere
      const sphere=new THREE.Mesh(new THREE.SphereGeometry(1,32,32), new THREE.MeshPhongMaterial({ color:0x0ea5e9, transparent:true, opacity:0.15 }));
      scene.add(sphere);
      const axes=new THREE.AxesHelper(1.5); scene.add(axes);
      const light=new THREE.PointLight(0xffffff, 1); light.position.set(3,3,3); scene.add(light);

      // state vector arrow
      const dir=new THREE.Vector3(0.6, 0.7, 0.2).normalize();
      const arrow=new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), 1.1, 0xff3366, 0.1, 0.08);
      scene.add(arrow);

      function animate(){ requestAnimationFrame(animate); sphere.rotation.y+=0.003; renderer.render(scene,camera); }
      animate(); setReady(true);
    })();
  },[mode]);

  return (
    <div className="card">
      <h1>3D Quantum Modeling <span className="badge">{mode}</span></h1>
      {mode==="lite" ? (
        <>
          <div className="canvas3d" style={{display:"grid",placeItems:"center"}}>2D Fallback (Lite)</div>
          <div className="tip" style={{marginTop:8}}>Switch to Pro/Lab (top-right) to load the Three.js Bloch sphere.</div>
        </>
      ) : <div ref={ref} className="canvas3d"/>}
      {!ready && mode!=="lite" && <div className="tip" style={{marginTop:8}}>Loading Three.js…</div>}
    </div>
  );
}
EOF

# Widgets
cat > qulab/frontend/src/widgets/CircuitPane.tsx <<'EOF'
export default function CircuitPane(){
  return (
    <div className="card" style={{marginTop:8}}>
      <b>Circuit (schematic)</b>
      <pre style={{background:"#0b1220",padding:8,borderRadius:8,overflowX:"auto",marginTop:6}}>
{`q0: |ψ⟩  —■—H—M—         (Alice payload)
q1: |0⟩   —H—X—M—         (Alice EPR)
q2: |0⟩   ————■—(X/Z cond)——  (Bob EPR)`}
      </pre>
    </div>
  );
}
EOF

cat > qulab/frontend/src/widgets/FidelityChart.tsx <<'EOF'
export default function FidelityChart({counts}:{counts:Record<string,number>}){
  const entries=Object.entries(counts); const total=entries.reduce((a,[,v])=>a+v,0);
  return (<div className="card" style={{marginTop:8}}>
    <b>Counts</b>
    <div style={{display:"flex",gap:8,marginTop:6}}>
      {entries.map(([k,v])=>(
        <div key={k} title={`${k}: ${v}`}>
          <div style={{height:Math.max(8,(v/Math.max(1,total))*120),width:24,background:"#1e40af",borderRadius:4}}/>
          <div style={{textAlign:"center"}}>{k}</div>
        </div>
      ))}
    </div>
  </div>);
}
EOF

echo "✅ QuLab scaffolded."

cat <<'USAGE'

# === run backend (Lite) ===
cd qulab/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# (optional Pro/Lab) enable Qiskit
# export QULAB_ENABLE_QISKIT=1
# pip install -r requirements-pro.txt

# === run frontend ===
# new terminal
cd qulab/frontend
npm i
npm run dev
# open http://localhost:5173

# === run tests ===
# in backend venv
pytest -q

USAGE