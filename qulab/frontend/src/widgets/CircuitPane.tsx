export default function CircuitPane(){
  return (
    <div className="card" style={{marginTop:8}}>
      <b>Circuit (schematic)</b>
      <pre style={{background:"#0b1220",padding:8,borderRadius:8,overflowX:"auto",marginTop:6}}>
{`q0: |ψ⟩  —■—H—M—     (Alice payload)
q1: |0⟩   —H—X—M—     (Alice EPR)
q2: |0⟩   ————■—(X/Z cond)——  (Bob EPR)`}
      </pre>
    </div>
  );
}
