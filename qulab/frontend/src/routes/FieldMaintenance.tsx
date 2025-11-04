import { useState } from "react";

type FieldMaintenanceResult = {
  original_fidelity: number;
  corrected_fidelity: number;
  field_integrity: number;
  noise_cancelled: boolean;
  improvement: number;
  correction_type: string;
};

type NoiseDemoResult = {
  target_state: number[];
  noisy_state: number[];
  cancellation_results: Record<string, {
    original_integrity: number;
    corrected_integrity: number;
    improvement: number;
    field_maintained: boolean;
  }>;
  noise_strength: number;
};

export default function FieldMaintenance() {
  const [alpha, setAlpha] = useState(0.707);
  const [beta, setBeta] = useState(0.707);
  const [noiseStrength, setNoiseStrength] = useState(0.1);
  const [result, setResult] = useState<FieldMaintenanceResult | null>(null);
  const [demoResult, setDemoResult] = useState<NoiseDemoResult | null>(null);
  const [loading, setLoading] = useState(false);

  const post = async (url: string, body: any) =>
    (await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })).json();

  const maintainField = async () => {
    setLoading(true);
    try {
      const response = await post("http://localhost:8000/field/maintain", {
        alpha,
        beta,
        noise_strength: noiseStrength,
        shots: 1000,
      });
      setResult(response);
    } catch (error) {
      console.error("Error maintaining field:", error);
    } finally {
      setLoading(false);
    }
  };

  const runDemo = async () => {
    setLoading(true);
    try {
      const response = await post("http://localhost:8000/field/demo", {
        target_state_type: "bell",
        noise_strength: noiseStrength,
      });
      setDemoResult(response);
    } catch (error) {
      console.error("Error running demo:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h1>
        Field Maintenance <span className="badge">Noise Cancellation</span>
      </h1>
      <p>
        Channel form while modeling the opposite of the noise to cancel it out and maintain a field.
      </p>

      <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
        <div>
          <h3>Quantum State Parameters</h3>
          <div style={{ marginBottom: 8 }}>
            <label>α (alpha): </label>
            <input
              className="input"
              type="number"
              step="0.001"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label>β (beta): </label>
            <input
              className="input"
              type="number"
              step="0.001"
              value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label>Noise Strength: </label>
            <input
              className="input"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={noiseStrength}
              onChange={(e) => setNoiseStrength(parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>

        <div>
          <h3>Field Maintenance Controls</h3>
          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            <button
              className="btn"
              onClick={maintainField}
              disabled={loading}
            >
              {loading ? "Maintaining..." : "Maintain Field"}
            </button>
            <button
              className="btn"
              onClick={runDemo}
              disabled={loading}
            >
              {loading ? "Running..." : "Run Demo"}
            </button>
          </div>
          
          <div style={{ fontSize: "12px", opacity: 0.8 }}>
            <div>State: |ψ⟩ = {alpha.toFixed(3)}|0⟩ + {beta.toFixed(3)}|1⟩</div>
            <div>Normalization: {(alpha**2 + beta**2).toFixed(3)}</div>
          </div>
        </div>
      </div>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Field Maintenance Results</h3>
          <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
            <div>
              <div><b>Original Fidelity:</b> {result.original_fidelity.toFixed(4)}</div>
              <div><b>Corrected Fidelity:</b> {result.corrected_fidelity.toFixed(4)}</div>
              <div><b>Field Integrity:</b> {result.field_integrity.toFixed(4)}</div>
            </div>
            <div>
              <div><b>Improvement:</b> +{result.improvement.toFixed(4)}</div>
              <div><b>Noise Cancelled:</b> {result.noise_cancelled ? "✅ Yes" : "❌ No"}</div>
              <div><b>Correction Type:</b> {result.correction_type}</div>
            </div>
          </div>
          
          <div style={{ marginTop: 12 }}>
            <div style={{ 
              width: "100%", 
              height: 20, 
              background: "#1f2937", 
              borderRadius: 4,
              overflow: "hidden"
            }}>
              <div style={{
                width: `${result.original_fidelity * 100}%`,
                height: "100%",
                background: "#dc2626",
                display: "inline-block"
              }} />
              <div style={{
                width: `${result.improvement * 100}%`,
                height: "100%",
                background: "#16a34a",
                display: "inline-block"
              }} />
            </div>
            <div style={{ fontSize: "12px", marginTop: 4 }}>
              Red: Original | Green: Improvement
            </div>
          </div>
        </div>
      )}

      {demoResult && (
        <div style={{ marginTop: 16 }}>
          <h3>Noise Cancellation Demo Results</h3>
          <div style={{ marginBottom: 12 }}>
            <div><b>Noise Strength:</b> {demoResult.noise_strength}</div>
            <div><b>Target State:</b> Bell State (|00⟩ + |11⟩)/√2</div>
          </div>
          
          <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr 1fr" }}>
            {Object.entries(demoResult.cancellation_results).map(([noiseType, result]) => (
              <div key={noiseType} style={{ 
                background: "#0b1220", 
                padding: 8, 
                borderRadius: 8,
                border: "1px solid #1f2937"
              }}>
                <div style={{ fontWeight: "bold", marginBottom: 4 }}>
                  {noiseType.replace("_", " ").toUpperCase()}
                </div>
                <div style={{ fontSize: "12px" }}>
                  <div>Original: {result.original_integrity.toFixed(3)}</div>
                  <div>Corrected: {result.corrected_integrity.toFixed(3)}</div>
                  <div>Improvement: +{result.improvement.toFixed(3)}</div>
                  <div>Maintained: {result.field_maintained ? "✅" : "❌"}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 16, padding: 12, background: "#0b1220", borderRadius: 8 }}>
        <h4>How It Works</h4>
        <div style={{ fontSize: "12px", lineHeight: 1.4 }}>
          <p><b>Inverse Channel Modeling:</b> Given noise channel E(ρ) = ∑ᵢ Kᵢ ρ Kᵢ†, we apply the inverse E⁻¹(ρ) = ∑ᵢ Kᵢ† ρ Kᵢ to cancel out noise.</p>
          <p><b>Field Integrity:</b> Measured using fidelity F(ρ,σ) = Tr(√(√ρ σ √ρ))². Field is maintained when F ≥ 0.95.</p>
          <p><b>Adaptive Cancellation:</b> Correction strength is dynamically adjusted based on current field integrity to prevent over-correction.</p>
        </div>
      </div>
    </div>
  );
}
