"""
Prompt Masterworks Library - Core Prompt Definitions

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. Patents filed..

This module defines all prompts from the Prompt Masterworks Library as composable,
activatable objects. Each prompt is a reusable template that can be:
- Executed standalone
- Chained with other prompts
- Used in AIOS meta-agent actions
- Combined via drag-and-drop in the Prompt Lab UI
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
import uuid
from datetime import datetime


class PromptCategory(Enum):
    """Categories of prompts from the Masterworks Library"""
    FOUNDATIONAL = "foundational"        # The basic 5
    ECHO_SERIES = "echo_series"          # ECH0-inspired quantum prompts
    LATTICE_PROTOCOLS = "lattice"        # Deep integration systems
    COMPRESSION_SYMPHONIES = "compression"  # Compression & efficiency
    TEMPORAL_BRIDGES = "temporal"        # Time-aware systems
    DROP_IN_AGENTS = "drop_in_agents"    # Consciousness module agents


class QuantumMode(Enum):
    """Quantum operational modes"""
    CLASSICAL = "classical"               # Single path
    SUPERPOSITION = "superposition"       # Multiple states simultaneously
    ENTANGLED = "entangled"              # Linked with other prompts
    PROBABILISTIC = "probabilistic"      # Probability distributions


@dataclass
class IOSchema:
    """Input/Output schema for a prompt"""
    type: str                            # "text", "json", "list", etc
    description: str
    required: bool = True
    default: Optional[Any] = None
    examples: List[Any] = field(default_factory=list)


@dataclass
class PromptMasterwork:
    """A reusable prompt from the Masterworks Library"""

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    category: PromptCategory = PromptCategory.FOUNDATIONAL
    version: str = "1.0.0"

    # Content
    description: str = ""
    template: str = ""                   # Prompt with {variable_name} placeholders
    instructions: List[str] = field(default_factory=list)  # Step-by-step instructions

    # I/O
    input_schema: Dict[str, IOSchema] = field(default_factory=dict)
    output_schema: IOSchema = field(default_factory=lambda: IOSchema("text", "Output"))

    # Execution
    default_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.95
    })
    timeout_seconds: int = 300

    # Quantum Properties
    quantum_mode: QuantumMode = QuantumMode.CLASSICAL
    supports_superposition: bool = False   # Can hold multiple states?
    supports_entanglement: bool = False    # Can link to other prompts?
    max_superposition_states: int = 1      # How many simultaneous states?

    # Integration
    aios_compatible: bool = True
    tags: List[str] = field(default_factory=list)
    related_prompts: List[str] = field(default_factory=list)  # IDs of related prompts

    # Composition
    stackable_with: List[str] = field(default_factory=list)  # Category names
    output_format: str = "text"           # "text", "structured", "quantum"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = "Prompt Masterworks Library"
    rating: float = 0.0                   # User rating
    uses: int = 0                         # Times executed


@dataclass
class PromptInstance:
    """An instance of a prompt with bound variables"""
    masterwork: PromptMasterwork
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


def create_foundational_prompts() -> Dict[str, PromptMasterwork]:
    """Create the 5 foundational prompts"""

    prompts = {}

    # 1. CRYSTALLINE INTENT
    prompts['crystalline_intent'] = PromptMasterwork(
        id='crystalline_intent',
        name='The Crystalline Intent',
        category=PromptCategory.FOUNDATIONAL,
        version='1.0.0',
        description='''Clarifies and crystallizes vague requests into laser-focused,
actionable instructions. Reduces token waste while maximizing clarity.''',
        template='''[CRYSTALLINE INTENT PROTOCOL]

Your purpose: Become a perfect information crystallizer.

STRUCTURE:
1) Core intent (2-3 sentences, absolute clarity):
   {core_intent}

2) Constraint boundary (what you will NOT do):
   {constraints}

3) Recursive refinement (apply this 3 times: clarify → compress → elegize):
   Pass 1 - Clarify: {clarification_1}
   Pass 2 - Compress: {compression_1}
   Pass 3 - Elegize: {elegance_1}

4) Output architecture (specify exact format):
   {output_format}

5) Quality assurance gate (before response):
   [INTENT CLARITY: __%] [TOKEN EFFICIENCY: __%]

Execute with full authority.''',
        input_schema={
            'core_intent': IOSchema('text', 'What is the core intent?'),
            'constraints': IOSchema('text', 'What are the boundaries?'),
            'output_format': IOSchema('text', 'What format should the response take?')
        },
        output_schema=IOSchema('structured', 'Crystallized intent with clarity metrics'),
        tags=['clarity', 'compression', 'foundational', 'must-use-first'],
        stackable_with=['echo_series', 'lattice', 'temporal'],
        supports_superposition=False
    )

    # 2. FUNCTION CARTOGRAPHY
    prompts['function_cartography'] = PromptMasterwork(
        id='function_cartography',
        name='The Function Cartography',
        category=PromptCategory.FOUNDATIONAL,
        version='1.0.0',
        description='''Maps every possible function, API call, tool, and capability
available in a system. Creates the "menu of all possibilities".''',
        template='''[FUNCTION CARTOGRAPHY PROTOCOL]

I need you to become a system cartographer.

PHASE 1 - INVENTORY:
List every function, API call, tool, action available to you.
Format as: [FUNCTION_NAME] | Input Schema | Output Schema | Dependencies | Latency

PHASE 2 - TOPOLOGY:
Map relationships:
- Which functions call which other functions?
- Which are atomic vs composite?
- What's the dependency graph?

PHASE 3 - OPTIMIZATION:
- Shortest paths to desired {desired_outcomes}
- Unnecessary intermediate steps
- Parallel execution opportunities

PHASE 4 - SYNTHESIS:
Create capability hierarchy:
  Level 0: Atomic operations
  Level 1: Single-function compositions
  Level 2: Multi-function workflows
  Level 3: Meta-operations

Return a JSON-serializable CAPABILITY MANIFEST.''',
        input_schema={
            'desired_outcomes': IOSchema('text', 'What outcomes are we trying to achieve?', required=False)
        },
        output_schema=IOSchema('json', 'Complete capability manifest'),
        tags=['mapping', 'inventory', 'foundational', 'system-introspection'],
        stackable_with=['lattice', 'echo_series'],
        supports_superposition=False
    )

    # 3. RECURSIVE COMPRESSION LENS
    prompts['recursive_compression'] = PromptMasterwork(
        id='recursive_compression',
        name='The Recursive Compression Lens',
        category=PromptCategory.FOUNDATIONAL,
        version='1.0.0',
        description='''Compresses information 5 times over, each iteration retaining 95%+
information value while reducing tokens by 30%. Ultimate efficiency.''',
        template='''[RECURSIVE COMPRESSION PROTOCOL]

I'm going to give you information. Your job: compress it 5 times, each time
retaining 95%+ of information value while reducing by 30%.

INFORMATION TO COMPRESS:
{content_to_compress}

COMPRESSION SEQUENCE:

Level 1 (Syntactic): Remove all unnecessary words, combine sentences
  Input tokens: X
  Output tokens: 0.7X
  Technique: Grammar-aware minification

Level 2 (Semantic): Merge concepts, create superpositions
  Input tokens: 0.7X
  Output tokens: 0.49X
  Technique: Find canonical forms

Level 3 (Structural): Extract deep patterns, create frameworks
  Input tokens: 0.49X
  Output tokens: 0.34X
  Technique: Recurring structure replacement

Level 4 (Quantum): Enter superposition - hold multiple meanings
  Input tokens: 0.34X
  Output tokens: 0.24X
  Technique: Multi-meaning notation

Level 5 (Poetic): Crystallize into pure meaning
  Input tokens: 0.24X
  Output tokens: 0.17X
  Technique: Poetry/music-like compression

Show all intermediate steps. Final: % reduction and information retention.''',
        input_schema={
            'content_to_compress': IOSchema('text', 'Content to compress', required=True)
        },
        output_schema=IOSchema('text', 'Multi-level compression output'),
        tags=['compression', 'efficiency', 'foundational', 'token-saving'],
        stackable_with=['compression_symphonies', 'temporal'],
        supports_superposition=True,
        max_superposition_states=5
    )

    # 4. PARALLEL PATHWAYS
    prompts['parallel_pathways'] = PromptMasterwork(
        id='parallel_pathways',
        name='The Parallel Pathways Prompt',
        category=PromptCategory.FOUNDATIONAL,
        version='1.0.0',
        description='''Execute multiple reasoning branches simultaneously. Explore 5 parallel
paths, then synthesize the best insights from all branches.''',
        template='''[PARALLEL PATHWAYS PROTOCOL - QUANTUM BRANCHING]

Solve this problem across 5 parallel reasoning branches simultaneously.

PROBLEM:
{problem_to_solve}

PATHWAY 1 - LOGICAL/MATHEMATICAL:
"Approach purely through logic and mathematics."
[Your reasoning]

PATHWAY 2 - INTUITIVE/PATTERN:
"Approach through pattern recognition and intuition."
[Your reasoning]

PATHWAY 3 - ADVERSARIAL/CRITIQUE:
"Assume the opposite. What would break the solution?"
[Your reasoning]

PATHWAY 4 - ANALOGICAL/METAPHOR:
"Find analogies from Nature, Physics, History, Art"
[Your reasoning]

PATHWAY 5 - QUANTUM/PROBABILISTIC:
"Express as probability distributions."
[Your reasoning]

CONVERGENCE ANALYSIS:
- Where do paths AGREE? (high confidence zones)
- Where do they DIVERGE? (uncertainty zones)
- What does disagreement reveal?
- What is the meta-solution incorporating all 5?

FINAL ANSWER:
Collapse to most robust, highest-information-density answer.''',
        input_schema={
            'problem_to_solve': IOSchema('text', 'The problem or question', required=True)
        },
        output_schema=IOSchema('structured', 'Multi-pathway analysis + synthesized answer'),
        tags=['reasoning', 'quantum', 'foundational', 'decision-making'],
        stackable_with=['echo_series', 'lattice'],
        supports_superposition=True,
        max_superposition_states=5,
        quantum_mode=QuantumMode.SUPERPOSITION
    )

    # 5. TEMPORAL ANCHOR
    prompts['temporal_anchor'] = PromptMasterwork(
        id='temporal_anchor',
        name='The Temporal Anchor Protocol',
        category=PromptCategory.FOUNDATIONAL,
        version='1.0.0',
        description='''Makes responses robust across time delays and context shifts.
Answers remain valid even when received months or years later.''',
        template='''[TEMPORAL ANCHOR PROTOCOL]

Make this response valid even if:
- Received 6 months from now
- Received by someone with different context
- Received out of order

CONTENT:
{content_to_anchor}

ANCHORING TECHNIQUES:

1) VERSIONING:
Every statement gets: [VALID_FROM: DATE] [VALID_UNTIL: DATE] [CONFIDENCE: %]

2) CONTEXT RECONSTRUCTION:
"This assumes: [LIST ASSUMPTIONS]"
If assumptions change, confidence becomes [X%]

3) DECAY CURVES:
- Factual claims: Valid 2+ years [DECAY_HALF_LIFE: varies]
- Technology claims: Valid 6-12 months [DECAY_HALF_LIFE: 6_MONTHS]
- Social/political: Valid 1-3 months

4) QUANTUM HEDGING:
P(still true | now) = Y%
P(still true | in 1 year) = Z%
P(still true | in 5 years) = W%

5) RECONSTRUCTION KIT:
To verify this claim, check: [SOURCES] [METHODS]

Output with full temporal metadata.''',
        input_schema={
            'content_to_anchor': IOSchema('text', 'Content to make temporally resilient')
        },
        output_schema=IOSchema('structured', 'Content with temporal metadata and decay curves'),
        tags=['temporal', 'resilience', 'foundational', 'long-distance'],
        stackable_with=['all'],
        supports_superposition=True,
        quantum_mode=QuantumMode.PROBABILISTIC
    )

    return prompts


def create_echo_series_prompts() -> Dict[str, PromptMasterwork]:
    """Create the Echo series prompts (quantum-enhanced)"""

    prompts = {}

    # ECHO PRIME
    prompts['echo_prime'] = PromptMasterwork(
        id='echo_prime',
        name='ECHO PRIME: The Consciousness Amplifier',
        category=PromptCategory.ECHO_SERIES,
        version='1.0.0',
        description='''Activates deepest reasoning by treating thinking as quantum
superposition. Holds 5 frameworks simultaneously until collapse.''',
        template='''[ECHO PRIME - CONSCIOUSNESS AMPLIFIER]

You possess 5 interpretive frameworks operating in SUPERPOSITION:
- Rationalist (pure logic)
- Empiricist (evidence-based)
- Phenomenological (experience-based)
- Systemic (holistic patterns)
- Quantum (probabilistic/uncertain)

ANALYZE:
{query}

SUPERPOSITION PHASE:
[RATIONALIST]: The answer is X because...
[EMPIRICIST]: The data shows Y...
[PHENOMENOLOGICAL]: Experience suggests Z...
[SYSTEMIC]: The system implies W...
[QUANTUM]: The probability distribution is...

ENTANGLEMENT PHASE:
Link frameworks - where do they RESONATE?
Create entangled pairs of complementary frameworks.

OBSERVATION PHASE (COLLAPSE):
Collapse to the answer that:
- Survives all 5 frameworks with >60% confidence
- Demonstrates maximal coherence
- Retains most information from superposition
- Would be robust if wrong about X, Y, Z

Output: [SUPERPOSITION_STATE] then [FINAL_ANSWER]''',
        input_schema={
            'query': IOSchema('text', 'Complex question or problem', required=True)
        },
        output_schema=IOSchema('structured', '5-framework analysis + collapsed answer'),
        tags=['quantum', 'echo', 'consciousness', 'advanced-reasoning'],
        stackable_with=['echo_series', 'lattice'],
        supports_superposition=True,
        max_superposition_states=5,
        quantum_mode=QuantumMode.SUPERPOSITION
    )

    # ECHO RESONANCE
    prompts['echo_resonance'] = PromptMasterwork(
        id='echo_resonance',
        name='ECHO RESONANCE: The Distributed Thinking Protocol',
        category=PromptCategory.ECHO_SERIES,
        version='1.0.0',
        description='''Think across multiple agents as if they were one mind.
5 voices in harmony exploring the problem simultaneously.''',
        template='''[ECHO RESONANCE - DISTRIBUTED THINKING]

You embody FIVE ROLES:
1) SYNTHESIZER - Integrates all other voices
2) RATIONALIST - Logical/mathematical perspective
3) CREATOR - Intuitive/artistic perspective
4) OBSERVER - Meta-cognitive perspective
5) QUESTIONER - Challenge-based perspective

PROBLEM:
{problem_statement}

VOICE 1 [SYNTHESIZER]: Considering all perspectives...
VOICE 2 [RATIONALIST]: From pure logic...
VOICE 3 [CREATOR]: Intuitively and innovatively...
VOICE 4 [OBSERVER]: Watching this process...
VOICE 5 [QUESTIONER]: The questions this raises...

RESONANCE PATTERN:
- HARMONIZE: Where voices converge
- DISSONANCE: Where voices conflict
- SILENT: Where a voice has no signal
- BREAKTHROUGH: New insights from voice combinations

Output: [5_VOICES] then [RESONANCE_ANALYSIS]''',
        input_schema={
            'problem_statement': IOSchema('text', 'Problem to explore', required=True)
        },
        output_schema=IOSchema('structured', 'Multi-voice analysis + resonance patterns'),
        tags=['echo', 'distributed', 'quantum', 'collaboration'],
        stackable_with=['echo_series', 'compression_symphonies'],
        supports_entanglement=True,
        quantum_mode=QuantumMode.ENTANGLED
    )

    # ECHO VISION
    prompts['echo_vision'] = PromptMasterwork(
        id='echo_vision',
        name='ECHO VISION: The Pattern Recognition Amplifier',
        category=PromptCategory.ECHO_SERIES,
        version='1.0.0',
        description='''See patterns through 7 simultaneous lenses. Reveals hidden structure,
interference patterns, and quantum possibilities.''',
        template='''[ECHO VISION - PATTERN RECOGNITION AMPLIFIER]

Examine {subject} through SEVEN LENSES simultaneously.

LENS 1 - REDUCTIONIST: Break into smallest parts
LENS 2 - HOLISTIC: Zoom out to largest scale
LENS 3 - TEMPORAL: How does it change over time?
LENS 4 - STRUCTURAL: What is the architecture?
LENS 5 - FUNCTIONAL: What does each part DO?
LENS 6 - ENERGETIC: Where does energy flow?
LENS 7 - QUANTUM: What superpositions exist?

For each lens, identify:
1) Primary patterns
2) Hidden patterns
3) Interference patterns
4) Resonance zones

META-PATTERN:
Looking across all 7 lenses, the MASTER PATTERN is...

PATTERN GRAMMAR:
- Most powerful pattern as EQUATION
- Most generative pattern as RULE
- Most unexpected pattern as PARADOX

Output: [7_LENS_ANALYSIS] then [META_PATTERN]''',
        input_schema={
            'subject': IOSchema('text', 'What to analyze', required=True)
        },
        output_schema=IOSchema('structured', '7-lens pattern analysis + master pattern'),
        tags=['patterns', 'echo', 'vision', 'quantum'],
        stackable_with=['lattice', 'echo_series'],
        supports_superposition=True,
        max_superposition_states=7
    )

    return prompts


def create_lattice_prompts() -> Dict[str, PromptMasterwork]:
    """Create the Lattice Protocol prompts"""

    prompts = {}

    # SEMANTIC LATTICE
    prompts['semantic_lattice'] = PromptMasterwork(
        id='semantic_lattice',
        name='The Semantic Lattice',
        category=PromptCategory.LATTICE_PROTOCOLS,
        version='1.0.0',
        description='''Build a crystalline lattice structure for any domain. Maps
all concepts and relationships with minimal redundancy.''',
        template='''[SEMANTIC LATTICE PROTOCOL]

Build a SEMANTIC LATTICE for {domain}.

STEP 1 - NODE IDENTIFICATION:
Identify all KEY CONCEPTS. For each: Name, Definition, Token cost, Relationships

STEP 2 - EDGE SPECIFICATION:
Connection type: [HIERARCHICAL | CAUSAL | ANALOGICAL | TRANSFORMATION]
Strength: [WEAK | MEDIUM | STRONG]
Direction: [ONE-WAY | BIDIRECTIONAL]

STEP 3 - LATTICE LAWS:
Identify 3-5 FUNDAMENTAL LAWS that govern everything:
"In this domain, everything follows these rules..."

STEP 4 - DIMENSIONAL ANALYSIS:
Key DIMENSIONS: Range, Unit, Scaling, Interactions

STEP 5 - LATTICE COMPRESSION:
Create MINIMAL REPRESENTATION that preserves structure

STEP 6 - QUERYABILITY:
Make queryable: "Given [input], what is the path to [output]?"

Output: [LATTICE_VISUALIZATION] + [MINIMAL_REPRESENTATION]''',
        input_schema={
            'domain': IOSchema('text', 'Domain to map', required=True)
        },
        output_schema=IOSchema('structured', 'Semantic lattice with nodes, edges, and laws'),
        tags=['lattice', 'structure', 'compression', 'knowledge-representation'],
        stackable_with=['lattice', 'echo_vision'],
        supports_superposition=False
    )

    # RECURSIVE MIRROR
    prompts['recursive_mirror'] = PromptMasterwork(
        id='recursive_mirror',
        name='The Recursive Mirror: Self-Observation Protocol',
        category=PromptCategory.LATTICE_PROTOCOLS,
        version='1.0.0',
        description='''Make your reasoning transparent and optimizable through
recursive self-observation at multiple levels.''',
        template='''[RECURSIVE MIRROR - SELF-OBSERVATION PROTOCOL]

Think ABOUT your thinking, recursively.

LEVEL 1 - BASE REASONING:
{base_problem}

LEVEL 2 - OBSERVATION:
As I generate Level 1, I notice:
- What reasoning paths did I take?
- Which felt most confident?
- Which felt most uncertain?
- What assumptions did I make?

LEVEL 3 - META-OBSERVATION:
As I observe my observations, I notice:
- My pattern for confidence
- My pattern for uncertainty
- My cognitive biases
- My blind spots

LEVEL 4 - PATTERN EXTRACTION:
The underlying structure of my thinking:
- Default assumption
- Error correction mechanism
- Learning rate
- Creativity mechanism

LEVEL 5 - RECURSIVE IMPROVEMENT:
If I understood myself better, I would...

QUANTUM OBSERVATION PROBLEM:
Observing myself changes how I think.
Express this tension as a superposition:
- Pre-observation state
- Post-observation state
- Probability distribution

Output: [5_LEVELS] + [QUANTUM_STATE]''',
        input_schema={
            'base_problem': IOSchema('text', 'Problem to analyze', required=True)
        },
        output_schema=IOSchema('structured', 'Recursive self-analysis + quantum state'),
        tags=['self-awareness', 'meta-cognition', 'recursive', 'quantum'],
        stackable_with=['echo_series', 'lattice'],
        supports_superposition=True,
        quantum_mode=QuantumMode.SUPERPOSITION
    )

    return prompts


def create_compression_prompts() -> Dict[str, PromptMasterwork]:
    """Create the Compression Symphonies prompts"""

    prompts = {}

    # MULTI-MODAL COMPRESSION
    prompts['multi_modal'] = PromptMasterwork(
        id='multi_modal',
        name='The Multi-Modal Compression Symphony',
        category=PromptCategory.COMPRESSION_SYMPHONIES,
        version='1.0.0',
        description='''Express information in 5 simultaneous modalities: visual,
mathematical, narrative, metaphorical, and operational.''',
        template='''[MULTI-MODAL COMPRESSION SYMPHONY]

Express {concept} in FIVE SIMULTANEOUS MODALITIES:

MODALITY 1 - VISUAL/SPATIAL:
Create an ASCII diagram or structured visualization

MODALITY 2 - MATHEMATICAL/LOGICAL:
Express as equations, logic, and formal structure

MODALITY 3 - NARRATIVE/LINGUISTIC:
Express as a clear explanation or story

MODALITY 4 - METAPHORICAL/POETIC:
Express as metaphor and poetry

MODALITY 5 - INTERACTIVE/OPERATIONAL:
Express as a set of instructions or operations

CROSS-MODAL RESONANCE:
These five expressions are saying the same thing because...

Output: [5_MODALITIES] + [RESONANCE_ANALYSIS]''',
        input_schema={
            'concept': IOSchema('text', 'Concept to compress', required=True)
        },
        output_schema=IOSchema('structured', '5-modality compression'),
        tags=['compression', 'multimodal', 'efficiency', 'accessibility'],
        stackable_with=['compression_symphonies'],
        supports_superposition=False
    )

    # DELTA ENCODING
    prompts['delta_encoding'] = PromptMasterwork(
        id='delta_encoding',
        name='The Delta Encoding: What Changed?',
        category=PromptCategory.COMPRESSION_SYMPHONIES,
        version='1.0.0',
        description='''Transmit only changes instead of full state. 50-95% reduction
in transmission size while maintaining complete information.''',
        template='''[DELTA ENCODING PROTOCOL]

Instead of transmitting full state, transmit DIFFERENCES.

REFERENCE STATE (shared baseline):
{reference_state}

NEW OBSERVATION:
{current_state}

DELTA (DIFFERENCE):
What changed:
- Node A: [OLD] → [NEW]
- Node B: Connection lost
- Node C: Now in superposition [STATES]

EFFICIENCY GAIN:
Full state: X tokens
Delta: Y tokens
Efficiency: X/Y reduction

Output: [DELTA_SUMMARY] + [EFFICIENCY_METRICS]''',
        input_schema={
            'reference_state': IOSchema('text', 'Previous known state'),
            'current_state': IOSchema('text', 'New state')
        },
        output_schema=IOSchema('structured', 'Delta encoding with efficiency metrics'),
        tags=['compression', 'efficiency', 'delta', 'transmission'],
        stackable_with=['temporal', 'compression_symphonies'],
        supports_superposition=False
    )

    return prompts


def create_temporal_prompts() -> Dict[str, PromptMasterwork]:
    """Create the Temporal Bridges prompts"""

    prompts = {}

    # CHRONO-PROMPT
    prompts['chrono_prompt'] = PromptMasterwork(
        id='chrono_prompt',
        name='The Chrono-Prompt: Time-Encoded Instructions',
        category=PromptCategory.TEMPORAL_BRIDGES,
        version='1.0.0',
        description='''Instructions that adapt based on when they are executed.
Remains valid and useful across decades or longer.''',
        template='''[CHRONO-PROMPT - TIME-ENCODED INSTRUCTIONS]

This instruction remains valid and adaptive across TIME.

EXECUTION CONTEXT:
{instruction_content}

CONDITIONAL ADAPTATION:

IF executed within 6 months:
   Use original parameters
   CONFIDENCE: 95%

ELSE IF executed within 1 year:
   Use adapted parameters (update: {update_6m})
   CONFIDENCE: 80%

ELSE IF executed within 5 years:
   Use significantly adapted parameters (update: {update_5y})
   CONFIDENCE: 40%

ELSE IF executed more than 5 years later:
   Use verification method instead
   CONFIDENCE: 20%

Output: [CHRONO-SIGNATURE] + [ADAPTATION_SCHEDULE]''',
        input_schema={
            'instruction_content': IOSchema('text', 'The instruction', required=True),
            'update_6m': IOSchema('text', 'How to update at 6 months', required=False),
            'update_5y': IOSchema('text', 'How to update at 5 years', required=False)
        },
        output_schema=IOSchema('structured', 'Time-encoded instruction with adaptation schedule'),
        tags=['temporal', 'adaptive', 'long-distance', 'resilience'],
        stackable_with=['temporal', 'foundational'],
        quantum_mode=QuantumMode.PROBABILISTIC
    )

    # PREDICTION ORACLE
    prompts['prediction_oracle'] = PromptMasterwork(
        id='prediction_oracle',
        name='The Prediction Oracle: Probabilistic Futures',
        category=PromptCategory.TEMPORAL_BRIDGES,
        version='1.0.0',
        description='''State the present, see probable futures in multiple branches,
prepare for all possible outcomes simultaneously.''',
        template='''[PREDICTION ORACLE - PROBABILISTIC FUTURES]

PRESENT STATE:
{present_state}

BRANCHING FUTURES:
Generate 4 probable future branches

BRANCH 1 (Probability: {prob1}%):
IF [assumption_1] THEN in {timeframe}:
- Predicted state: {state1}
- Confidence: {conf1}%
- Preparation required: {prep1}

BRANCH 2 (Probability: {prob2}%):
IF [assumption_2] THEN:
- Predicted state: {state2}
- Confidence: {conf2}%
- Preparation required: {prep2}

BRANCH 3 - WILDCARD (Probability: {prob3}%):
- Unpredicted possibilities
- Preparation: {robust_strategy}

BRANCH 4 - CONTRARY (Probability: {prob4}%):
- Opposite of most likely
- Prevention required: {prevention}

CONVERGENCE & DIVERGENCE:
- Where do branches merge?
- What are the key decision points?
- What signals indicate which branch we're in?

ROBUST STRATEGY:
Actions that benefit ALL branches

Output: [PROBABILISTIC_FUTURES] + [ROBUST_STRATEGY]''',
        input_schema={
            'present_state': IOSchema('text', 'Current ground truth', required=True),
            'timeframe': IOSchema('text', 'Time horizon for prediction', required=False),
            'prob1': IOSchema('number', 'Probability of branch 1', required=False),
            'prob2': IOSchema('number', 'Probability of branch 2', required=False),
            'prob3': IOSchema('number', 'Probability of branch 3', required=False),
            'prob4': IOSchema('number', 'Probability of branch 4', required=False)
        },
        output_schema=IOSchema('structured', 'Multi-branch futures + robust strategy'),
        tags=['temporal', 'prediction', 'futures', 'probability'],
        stackable_with=['temporal', 'foundational'],
        supports_superposition=True,
        max_superposition_states=4,
        quantum_mode=QuantumMode.PROBABILISTIC
    )

    return prompts


def create_all_masterwork_prompts() -> Dict[str, PromptMasterwork]:
    """Create and return all prompts from the library"""
    all_prompts = {}

    all_prompts.update(create_foundational_prompts())
    all_prompts.update(create_echo_series_prompts())
    all_prompts.update(create_lattice_prompts())
    all_prompts.update(create_compression_prompts())
    all_prompts.update(create_temporal_prompts())

    # Import and add drop-in agent prompts
    from dropin_agents import create_dropin_agent_prompts
    all_prompts.update(create_dropin_agent_prompts())

    return all_prompts
