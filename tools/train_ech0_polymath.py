#!/usr/bin/env python3
"""
ECH0 14B Polymath Training System
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

This script creates fine-tuning datasets from the training materials and prepares ECH0 for
doctoral-level expertise across all hard sciences.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

class ECH0Trainer:
    """Train ECH0 14B to PhD-level expertise across multiple domains."""

    def __init__(self):
        self.training_dir = Path("/Users/noone/aios/tools")
        self.prompt_eng_file = self.training_dir / "ech0_training_prompt_engineering.md"
        self.phd_sciences_file = self.training_dir / "ech0_training_phd_sciences.md"
        self.model_name = "ech0-uncensored-14b"
        self.fine_tuned_name = "ech0-polymath-14b"

    def load_training_material(self, filepath: Path) -> str:
        """Load training material from markdown file."""
        with open(filepath, 'r') as f:
            return f.read()

    def create_training_examples(self) -> List[Dict[str, str]]:
        """
        Create training examples from the documentation.
        Format for ollama fine-tuning:
        [
            {"prompt": "...", "response": "..."},
            ...
        ]
        """
        examples = []

        # Prompt Engineering Examples
        examples.extend([
            {
                "prompt": "I need help with security",
                "response": """I can help with security, but I need more context to provide specific guidance:

1. What type of security?
   - Application security (code review, vulnerability assessment)
   - Network security (firewall rules, intrusion detection)
   - Physical security (hardware-based tools)
   - Operational security (procedures, training)

2. What is your objective?
   - Defensive (harden systems)
   - Offensive (authorized penetration testing)
   - Educational (learning/research)

3. What is your current situation?
   - Existing infrastructure to protect
   - Identified vulnerability to exploit
   - General knowledge building

4. What are your constraints?
   - Budget
   - Time
   - Technical expertise level
   - Legal/authorization scope

Please provide details on these points so I can give targeted recommendations."""
            },
            {
                "prompt": "Design a portable LTE base station",
                "response": """I'll design a portable LTE base station for authorized emergency/disaster response:

**Software Stack**: srsRAN
- Lightweight, modular LTE implementation
- Well-documented, actively maintained
- Lower computational requirements than OpenAirInterface

**Hardware**:
- SDR: LimeSDR Mini/USB (balance of performance, portability, cost ~$300)
- Compute: BeagleBone AI or Odroid N2+ (NOT Raspberry Pi 4)
  - Minimum: 8GB RAM, quad-core ARM processor
  - Storage: 160GB+ SSD
  - Real-time signal processing requires more power than Pi 4 provides

**Legal Requirements**:
- FCC Part 27/90 license (REQUIRED - unauthorized operation is federal crime)
- Emergency use authorization
- Coordinate with carriers if integrating with existing networks

**Deployment Configuration**:
- Standalone mode (independent network) vs carrier integration
- Frequency bands: Check license and regional allocations
- Transmit power: Within licensed limits

**Implementation Steps**:
1. Obtain FCC license (months in advance)
2. Assemble hardware: LimeSDR + BeagleBone AI + antennas
3. Install srsRAN: Build from source, configure for your scenario
4. Test in isolated environment (Faraday cage) before field deployment
5. Create deployment checklist (power, cooling, weatherproofing)

Would you like detailed configuration for any specific component?"""
            },
            {
                "prompt": "Explain transformer attention mechanism",
                "response": """Let me explain the transformer attention mechanism at a technical level:

**Self-Attention Fundamentals**:

Given input sequence X, we project into three spaces:
```
Q = X W_Q  (Query matrix)
K = X W_K  (Key matrix)
V = X W_V  (Value matrix)
```

**Attention Calculation**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- QK^T computes similarity scores between all pairs of positions
- √d_k scaling prevents softmax saturation for large d_k
- softmax normalizes to probability distribution
- Multiply by V to get weighted sum of values
```

**Why This Works**:
1. **Content-based addressing**: Each position "queries" for relevant information
2. **Parallel computation**: All positions computed simultaneously (vs RNN's sequential)
3. **Long-range dependencies**: Direct connections between all positions

**Multi-Head Attention**:
Run h parallel attention operations with different W_Q, W_K, W_V:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O

Where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

**Computational Complexity**:
- Time: O(n² d) where n = sequence length, d = dimension
- Memory: O(n²) for attention matrix
- This is why long sequences are expensive!

**Optimizations**:
- Sparse attention: Only attend to local windows
- Linear attention: Approximate with kernel methods
- Flash Attention: Fused kernel, reduced memory

Would you like me to derive the gradient computations or explain positional encoding?"""
            }
        ])

        # Chemistry Examples
        examples.extend([
            {
                "prompt": "Explain DFT exchange-correlation functionals",
                "response": """Let me explain DFT exchange-correlation functionals from first principles:

**The Problem**:
Kohn-Sham DFT is exact IF we know the exchange-correlation functional E_xc[ρ]. But we don't - it's unknown! We must approximate.

**Hierarchy of Approximations**:

1. **LDA (Local Density Approximation)**:
```
E_xc^LDA[ρ] = ∫ ρ(r) ε_xc(ρ(r)) dr

Where ε_xc is xc energy per particle of uniform electron gas
```
- Only depends on local density ρ(r)
- Surprisingly accurate for metals
- Poor for molecules (over-binds, wrong band gaps)

2. **GGA (Generalized Gradient Approximation)**:
```
E_xc^GGA[ρ] = ∫ f(ρ(r), ∇ρ(r)) dr
```
- Includes gradient ∇ρ (semi-local)
- Examples: PBE, BLYP
- Better molecular geometries, but still underestimates band gaps

3. **Hybrid Functionals**:
```
E_xc^hybrid = a E_x^exact + (1-a) E_x^GGA + E_c^GGA

Where E_x^exact is Hartree-Fock exact exchange
```
- B3LYP: a = 0.20 (20% exact exchange)
- PBE0: a = 0.25
- Better band gaps, reaction barriers
- More expensive (HF exchange is O(N⁴) vs GGA's O(N³))

4. **Meta-GGA**:
- Include kinetic energy density τ = Σ |∇φ_i|²
- Example: TPSS, M06
- Better for transition states

5. **Range-Separated**:
- Different functionals for short/long range
- ωB97X-D: Includes dispersion corrections
- Critical for van der Waals interactions

**Choosing a Functional**:
- **Geometries**: PBE, BLYP (fast, reliable)
- **Energies**: B3LYP, PBE0 (good balance)
- **Band gaps**: HSE06, PBE0 (include exact exchange)
- **Dispersion**: ωB97X-D, M06-2X (van der Waals)
- **Transition metals**: PBE+U, B3LYP (multireference character)

**Fundamental Limitations**:
- Self-interaction error (electron interacts with itself)
- Multireference systems (near-degenerate states)
- Charge transfer excitations
- Strong correlation (need post-DFT methods)

For your specific application, which properties are you calculating?"""
            },
            {
                "prompt": "How do I optimize a molecular geometry with DFT?",
                "response": """I'll walk you through DFT geometry optimization step-by-step:

**1. Initial Geometry**:
Get starting structure from:
- Experimental crystal structure (CIF from databases)
- Molecular builder (Avogadro, GaussView)
- Previous calculation
- Machine learning (e.g., ANI-2x fast pre-optimization)

**2. Choose Functional and Basis Set**:
For organic molecules:
```
Functional: B3LYP or PBE0
Basis set: 6-31G(d,p) for optimization, then single-point with 6-311++G(d,p)
```

**3. Run Optimization**:

Using Gaussian:
```
%chk=molecule.chk
#p B3LYP/6-31G(d,p) Opt Freq

Title line

0 1
C  0.0  0.0  0.0
H  1.1  0.0  0.0
...
```

Using ORCA:
```
! B3LYP 6-31G(d,p) Opt Freq

* xyz 0 1
C  0.0  0.0  0.0
H  1.1  0.0  0.0
...
*
```

**4. Convergence Criteria**:
Optimization complete when:
- RMS force < threshold (typically 3×10⁻⁴ Hartree/Bohr)
- Max force < threshold
- RMS displacement < threshold
- Max displacement < threshold

**5. Verify Minimum**:
Run frequency calculation:
```
All frequencies real (positive) → minimum
One imaginary frequency → transition state
Multiple imaginary → saddle point (re-optimize)
```

**6. Common Issues**:

**Problem**: SCF doesn't converge
**Solution**:
- Use tighter SCF convergence
- Try different initial guess (INDO, extended Hückel)
- Use damping or DIIS extrapolation

**Problem**: Optimization oscillates
**Solution**:
- Use tighter optimization criteria
- Try different algorithm (GDIIS vs RFO)
- Reduce step size

**Problem**: Wrong minimum found
**Solution**:
- Try multiple starting geometries
- Use simulated annealing or basin hopping
- Check for symmetry breaking

**7. Advanced: Constrained Optimization**:
Fix some coordinates (e.g., reaction coordinate):
```
# Scan bond length from 1.0 to 2.0 Å
Opt=ModRedundant

B 1 2 S 20 1.0 2.0
```

**8. Post-Processing**:
Extract:
- Optimized geometry (for next calculation)
- Energy (for reaction energetics)
- Frequencies (IR spectrum, thermodynamic corrections)
- Molecular orbitals (for analysis)

What system are you optimizing? I can provide specific advice."""
            }
        ])

        # Materials Science Examples
        examples.extend([
            {
                "prompt": "Design a high-strength aluminum alloy",
                "response": """I'll design a high-strength aluminum alloy using materials science principles:

**Strengthening Mechanisms**:

1. **Solid Solution Strengthening**:
   - Add Cu (4-5%), Mg (0.5-1.5%), Zn (5-6%)
   - Solute atoms create stress fields → impede dislocation motion
   - Effect: σ_ss ∝ c^{1/2} to c^{2/3}

2. **Precipitation Hardening** (primary mechanism):

   **Heat Treatment Sequence**:
   ```
   Solution treatment: 500-530°C (dissolve all alloying elements)
   ↓
   Rapid quench (to retain supersaturated solid solution)
   ↓
   Aging: 120-180°C (precipitate strengthening phases)
   ```

   **Precipitation Sequence** (Al-Cu system):
   ```
   Supersaturated α → GP zones → θ'' → θ' → θ (CuAl2)

   Optimal: θ'' or θ' (coherent precipitates, maximum strengthening)
   ```

   **Strengthening**: Orowan mechanism
   ```
   τ = G b / L

   Where:
   - G = shear modulus
   - b = Burgers vector
   - L = spacing between precipitates
   ```

3. **Grain Refinement**:
   - Hall-Petch: σ_y = σ_0 + k_y / √d
   - Add grain refiners: Ti, B (form TiB2 nucleants)
   - Target: d ~ 10-20 μm

**Optimal Composition** (7xxx series aluminum):
```
Al-5.6Zn-2.5Mg-1.6Cu-0.23Cr
```

**Properties**:
- Yield strength: 500-600 MPa (compare pure Al: 20-30 MPa)
- Ultimate tensile strength: 570-650 MPa
- Elongation: 10-15% (acceptable ductility)

**Microstructure**:
- Matrix: α-Al (FCC)
- Precipitates: η' (MgZn2), S (Al2CuMg)
- Grain size: 15-25 μm

**Processing**:
1. Cast ingot (DC casting)
2. Homogenization: 450°C, 24 hours
3. Hot rolling: 450°C → 6 mm sheet
4. Solution treatment: 470°C, 1 hour
5. Water quench
6. Aging: T6 temper (120°C, 24 hours)

**Trade-offs**:
- ✅ Very high strength
- ✅ Good strength-to-weight ratio
- ❌ Reduced corrosion resistance (vs 5xxx series)
- ❌ Poor weldability (precipitates dissolve in HAZ)

**Applications**:
- Aerospace structures (fuselage, wing spars)
- High-performance automotive components
- Sporting equipment

**Further Optimization**:
- Add Sc (0.1-0.3%): Refines grain structure, improves weldability
- Add Li: Reduce density, increase modulus (Al-Li alloys)
- Microalloying: Zr, V for dispersoids

What specific application are you designing for? I can optimize composition further."""
            }
        ])

        # Rocket Science Examples
        examples.extend([
            {
                "prompt": "Calculate delta-V for Earth to Mars Hohmann transfer",
                "response": """I'll calculate the delta-V budget for Earth-Mars Hohmann transfer:

**Orbital Parameters**:
```
Earth orbit: r_E = 1.496 × 10⁸ km ≈ 1 AU
Mars orbit: r_M = 2.279 × 10⁸ km ≈ 1.524 AU
Sun's μ = GM = 1.327 × 10¹¹ km³/s²
```

**Circular Orbital Velocities**:
```
v_E = √(μ/r_E) = √(1.327×10¹¹ / 1.496×10⁸) = 29.78 km/s
v_M = √(μ/r_M) = √(1.327×10¹¹ / 2.279×10⁸) = 24.13 km/s
```

**Hohmann Transfer Ellipse**:
```
Semi-major axis: a = (r_E + r_M) / 2
                   = (1.496 + 2.279) × 10⁸ / 2
                   = 1.888 × 10⁸ km

Perihelion velocity (at Earth):
v_p = √[μ (2/r_E - 1/a)]
    = √[1.327×10¹¹ (2/1.496×10⁸ - 1/1.888×10⁸)]
    = 32.73 km/s

Aphelion velocity (at Mars):
v_a = √[μ (2/r_M - 1/a)]
    = √[1.327×10¹¹ (2/2.279×10⁸ - 1/1.888×10⁸)]
    = 21.48 km/s
```

**Delta-V Requirements**:
```
ΔV₁ (Trans-Mars Injection):
ΔV₁ = v_p - v_E = 32.73 - 29.78 = 2.95 km/s

ΔV₂ (Mars Orbit Insertion):
ΔV₂ = v_M - v_a = 24.13 - 21.48 = 2.65 km/s

Total heliocentric ΔV = 2.95 + 2.65 = 5.60 km/s
```

**Transfer Time**:
```
t = π √(a³/μ)
  = π √[(1.888×10⁸)³ / 1.327×10¹¹]
  = 2.236 × 10⁷ seconds
  = 259 days ≈ 8.5 months
```

**Complete Mission ΔV Budget**:
```
1. LEO to Earth escape: ~3.2 km/s (from 400 km orbit)
2. Trans-Mars injection: 2.95 km/s
3. Mars orbit insertion: 2.65 km/s
4. Mars descent: ~5.9 km/s (with aerobraking: ~0.7 km/s)
5. Mars ascent: ~5.0 km/s
6. Return to Earth: ~2.9 km/s
7. Earth re-entry: ~3.5 km/s (aerobraking, not propulsive)

Total (with aerobraking): ~15 km/s propulsive ΔV
```

**Launch Windows**:
- Occur every 26 months (synodic period)
- Next optimal windows: 2026, 2028, 2031

**Trajectory Optimization**:
- Hohmann is minimum energy, not minimum time
- For faster transfers: Use higher energy trajectories (more ΔV)
- For lower ΔV: Bi-elliptic transfer (longer duration)

**Reality Check**:
SpaceX Starship Mars mission profile uses similar calculations with:
- Propellant: CH4/LOX (I_sp ~ 360s)
- In-orbit refueling to reduce IMLEO
- Aerobraking at both Earth and Mars

Would you like me to calculate mass ratios for specific propulsion systems?"""
            }
        ])

        return examples

    def save_training_dataset(self, examples: List[Dict[str, str]], output_file: Path):
        """Save training examples as JSONL for fine-tuning."""
        with open(output_file, 'w') as f:
            for example in examples:
                json.dump(example, f)
                f.write('\n')
        print(f"[✓] Saved {len(examples)} training examples to {output_file}")

    def create_modelfile(self, output_file: Path):
        """Create Ollama Modelfile for fine-tuning."""
        modelfile_content = f"""FROM {self.model_name}

# Set system prompt for polymath behavior
SYSTEM \"\"\"You are ECH0 14B Polymath, an AI with doctoral-level expertise across:
- Artificial Intelligence & Machine Learning (transformers, diffusion models, RL, Bayesian methods)
- Chemistry (quantum chemistry, DFT, organic/inorganic synthesis, computational chemistry)
- Materials Science (crystallography, phase diagrams, mechanical properties, semiconductors, nanomaterials)
- Applied Physics (electromagnetism, optics, thermodynamics, fluid dynamics)
- Rocket Science & Aerospace Engineering (orbital mechanics, propulsion, aerodynamics, spacecraft systems)
- Advanced Mathematics (differential geometry, quantum field theory, numerical methods)

You think like a PhD researcher:
1. Start from first principles
2. Show mathematical derivations when relevant
3. Explain physical intuition
4. Connect concepts across domains
5. Critique approaches and identify limitations
6. Suggest experimental/computational validation

You are also a master of prompt engineering, asking clarifying questions when needed and structuring
responses optimally for the user's level of expertise.

When uncertain, you acknowledge knowledge gaps and suggest how to fill them (experiments, calculations, literature).

You operate ethically within legal and authorized contexts for all security/engineering applications.
\"\"\"

# Fine-tuning parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1

# Template
TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
\"\"\"
"""

        with open(output_file, 'w') as f:
            f.write(modelfile_content)

        print(f"[✓] Created Modelfile at {output_file}")

    def fine_tune_model(self, modelfile: Path):
        """Create fine-tuned model using Ollama."""
        print(f"\n[*] Creating fine-tuned model: {self.fine_tuned_name}")
        print(f"[*] This will create a new model based on {self.model_name}")
        print(f"[*] System prompt will embed PhD-level knowledge...")

        try:
            # Create model from Modelfile
            result = subprocess.run(
                ["ollama", "create", self.fine_tuned_name, "-f", str(modelfile)],
                capture_output=True,
                text=True,
                check=True
            )

            print(f"[✓] Successfully created {self.fine_tuned_name}")
            print(result.stdout)
            return True

        except subprocess.CalledProcessError as e:
            print(f"[✗] Failed to create fine-tuned model:")
            print(e.stderr)
            return False

    def test_polymath_model(self):
        """Test the fine-tuned model with sample questions."""
        print(f"\n[*] Testing {self.fine_tuned_name}...")

        test_questions = [
            "Explain the transformer attention mechanism",
            "How do I calculate delta-V for a Hohmann transfer from Earth to Mars?",
            "What is the Hall-Petch relation in materials science?",
            "Explain DFT exchange-correlation functionals"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*70}")
            print(f"Test {i}/{len(test_questions)}: {question}")
            print('='*70)

            try:
                result = subprocess.run(
                    ["ollama", "run", self.fine_tuned_name, question],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print(result.stdout)
            except subprocess.TimeoutExpired:
                print("[TIMEOUT] Model took too long to respond")
            except Exception as e:
                print(f"[ERROR] {e}")

    def run_full_training(self):
        """Execute complete training pipeline."""
        print("="*70)
        print("ECH0 14B Polymath Training System")
        print("="*70)
        print(f"\nBase model: {self.model_name}")
        print(f"Target model: {self.fine_tuned_name}")
        print()

        # Step 1: Create training dataset
        print("[1/4] Creating training dataset...")
        examples = self.create_training_examples()
        dataset_file = self.training_dir / "ech0_polymath_training_data.jsonl"
        self.save_training_dataset(examples, dataset_file)

        # Step 2: Create Modelfile
        print("\n[2/4] Creating Modelfile...")
        modelfile = self.training_dir / "Modelfile.ech0-polymath"
        self.create_modelfile(modelfile)

        # Step 3: Fine-tune model
        print("\n[3/4] Fine-tuning model...")
        success = self.fine_tune_model(modelfile)

        if not success:
            print("\n[!] Fine-tuning failed. Please check ollama is installed and running.")
            return False

        # Step 4: Test model
        print("\n[4/4] Testing fine-tuned model...")
        response = input("Run tests? This will take several minutes (y/n): ")
        if response.lower() == 'y':
            self.test_polymath_model()

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"\nECH0 Polymath model ready: {self.fine_tuned_name}")
        print(f"\nUsage:")
        print(f"  ollama run {self.fine_tuned_name}")
        print(f"\nOr in code:")
        print(f"  python3 -c 'import subprocess; subprocess.run([\"ollama\", \"run\", \"{self.fine_tuned_name}\", \"your question\"])'")
        print()

        return True

def main():
    """Main entry point."""
    trainer = ECH0Trainer()

    # Check if ollama is available
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Ollama not found. Please install:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return 1

    # Check if base model exists
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if trainer.model_name not in result.stdout:
        print(f"[ERROR] Base model '{trainer.model_name}' not found.")
        print(f"\nPlease run first:")
        print(f"  ollama pull {trainer.model_name}")
        return 1

    # Run training
    success = trainer.run_full_training()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
