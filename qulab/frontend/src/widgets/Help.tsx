import { useEffect, useState } from "react";

/** Simple Dummies Mode: click goggles to toggle popups pointing at key UI. */
export default function Help(){
  const [on,setOn]=useState(false);
  useEffect(()=>{
    const pop=(id:string, html:string, x:number, y:number)=>{
      const el=document.createElement("div"); el.className="pop"; el.id=id; el.style.left=x+"px"; el.style.top=y+"px";
      el.innerHTML=html; document.body.appendChild(el);
    };
    const kill=(id:string)=>{ const e=document.getElementById(id); if(e) e.remove(); };
    if(on){
      pop("pop-nav", `<h4>Navigation</h4><div class='tip'>Teleport, Encoding, Governance, Field. Click to switch labs.</div>`, 12, 54);
      pop("pop-run", `<h4>Run Teleport</h4><div class='tip'>Press <b>Run (SSE)</b> to stream a result. In Pro/Lab, this calls Qiskit.</div>`, 320, 180);
      pop("pop-enc", `<h4>High-Radix</h4><div class='tip'>Pick a base (2..91 in Lite). Encode/Decode to test "fatter" signals.</div>`, 820, 110);
      pop("pop-gov", `<h4>Governance <span class='badge'>Chrono-Walker</span></h4><div class='tip'>Log fidelity, forecast bands, plan cadence.</div>`, 720, 58);
      pop("pop-field", `<h4>Field Maintenance <span class='badge'>Noise Cancellation</span></h4><div class='tip'>Channel form while modeling the opposite of noise to cancel it out and maintain a field.</div>`, 920, 58);
    } else {
      ["pop-nav","pop-run","pop-enc","pop-gov","pop-field"].forEach(kill);
    }
    return ()=>{ ["pop-nav","pop-run","pop-enc","pop-gov","pop-field"].forEach(kill); };
  },[on]);

  return (
    <button className="btn" title="Dummies Mode: on/off" onClick={()=>setOn(v=>!v)}>
      <span style={{marginRight:6}}>🕶️</span>{on?"Hide Tips":"Show Tips"}
    </button>
  );
}
