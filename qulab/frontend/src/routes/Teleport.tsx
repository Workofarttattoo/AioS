import { useState } from "react";
import CircuitPane from "../widgets/CircuitPane";
import FidelityChart from "../widgets/FidelityChart";
type Result={fidelity:number; counts:Record<string,number>; shots:number};

export default function Teleport(){
  const [status,setStatus]=useState("idle"); const [result,setResult]=useState<Result|null>(null); const [running,setRunning]=useState(false);
  const run=()=>{
    setRunning(true); setStatus("connecting"); setResult(null);
    const sse=new EventSource("http://localhost:8000/sim/teleport/run");
    sse.addEventListener("status",(e:any)=>setStatus((e as MessageEvent).data));
    sse.addEventListener("result",(e:any)=>{ setResult(JSON.parse((e as MessageEvent).data)); setRunning(false); sse.close(); });
    sse.onerror=()=>{ setStatus("error"); setRunning(false); sse.close(); };
  };
  return (<div className="card">
    <h1>Teleportation Workspace <span className="badge">Lite</span></h1>
    <p>Build → Run → Inspect fidelity and counts. (Pro/Lab: Qiskit, tomography.)</p>
    <CircuitPane/>
    <div style={{display:"flex",gap:8,marginTop:8}}>
      <button className="btn" onClick={run} disabled={running}>{running?"Running...":"Run (SSE)"}</button>
      <span>Status: {status}</span>
    </div>
    {result && (<><h3 style={{marginTop:12}}>Result</h3><div>Fidelity: {result.fidelity}</div><FidelityChart counts={result.counts}/></>)}
  </div>);
}
