import { useState } from "react";
export default function Encoding(){
  const [base,setBase]=useState(64); const [hex,setHex]=useState("48656c6c6f2051754c6162"); const [encoded,setEncoded]=useState(""); const [roundtrip,setRoundtrip]=useState("");
  const post=async (url:string, body:any)=> (await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
  return (<div className="card">
    <h1>High-Radix Encoding</h1>
    <div style={{display:"grid",gap:8,gridTemplateColumns:"1fr 1fr"}}>
      <div><label>Base (2..91): </label><input className="input" type="number" min={2} max={91} value={base} onChange={e=>setBase(parseInt(e.target.value||"64"))}/></div>
      <div><label>HEX:</label><textarea className="input" rows={3} value={hex} onChange={e=>setHex(e.target.value)} /></div>
    </div>
    <div style={{display:"flex",gap:8,marginTop:8}}>
      <button className="btn" onClick={async()=>{ const j=await post("http://localhost:8000/encoding/encode",{base,data_hex:hex}); setEncoded(j.encoded);}}>Encode</button>
      <button className="btn" onClick={async()=>{ const j=await post("http://localhost:8000/encoding/decode",{base,encoded}); setRoundtrip(j.data_hex);}} disabled={!encoded}>Decode</button>
    </div>
    <div style={{marginTop:12}}><b>Encoded:</b> {encoded}</div>
    <div><b>Roundtrip HEX:</b> {roundtrip}</div>
  </div>);
}
