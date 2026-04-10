export default function CircuitPane(){
  return (
    <div className="card" style={{marginTop:12}}>
      <b>Circuit</b>
      <div style={{opacity:0.8, marginTop:6}}>
        Lite mode uses a stub editor. Pro/Lab lazy-loads a real sketcher & Qiskit wiring.
      </div>
      <pre style={{background:"#0b1220", padding:8, borderRadius:8, overflowX:"auto"}}>
{`q0: |ψ⟩  —■—H—M—  (Alice payload)
q1: |0⟩  —H—X—M—  (Alice EPR)
q2: |0⟩  ————■—(X/Z cond)—— (Bob EPR)`}
      </pre>
    </div>
  );
}