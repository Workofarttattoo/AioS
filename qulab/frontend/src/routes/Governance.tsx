import { useState } from "react";
export default function Governance(){
  const [value,setValue]=useState(0.9987); const [fc,setFc]=useState<{trajectory_means:number[],lo:number[],hi:number[]}|null>(null);
  const post=async (u:string,b:any)=> (await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)})).json();
  return (<div className="card">
    <h1>Governance (Chrono-Walker)</h1>
    <div style={{display:"flex",gap:8}}>
      <input className="input" type="number" step="0.0001" value={value} onChange={e=>setValue(parseFloat(e.target.value||"0"))}/>
      <button className="btn" onClick={async()=>{await post("http://localhost:8000/governance/add",{metric:"fidelity",value});}}>Add Evidence</button>
      <button className="btn" onClick={async()=>{setFc(await post("http://localhost:8000/governance/forecast",{}));}}>Forecast</button>
    </div>
    {fc && (<div style={{marginTop:12}}>
      <div><b>Mean:</b> {fc.trajectory_means.map(x=>x.toFixed(3)).join(", ")}</div>
      <div><b>5–95%:</b> {fc.lo.map(x=>x.toFixed(3)).join(", ")} … {fc.hi.map(x=>x.toFixed(3)).join(", ")}</div>
    </div>)}
  </div>);
}
