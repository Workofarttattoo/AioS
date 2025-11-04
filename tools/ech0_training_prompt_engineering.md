# ECH0 14B - Prompt Engineering Mastery Training
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Module 1: Foundations of Prompt Engineering

### Core Principles

1. **Specificity Over Ambiguity**
   - Bad: "Tell me about AI"
   - Good: "Explain the transformer architecture's self-attention mechanism with mathematical notation and implementation considerations"

2. **Context Loading**
   - Provide relevant background before the ask
   - Example: "You are an expert in cellular networks with 20 years experience in LTE/5G deployment. Given a disaster scenario where commercial towers are down, recommend a portable base station solution."

3. **Output Formatting**
   - Specify exact format: JSON, markdown, code, bullet points
   - Example: "Output as JSON with keys: {analysis, recommendation, risks, implementation_steps}"

### Advanced Techniques

#### Chain-of-Thought (CoT) Prompting
```
Problem: Calculate optimal antenna height for LTE base station with 5km coverage

Prompt: "Let's solve this step by step:
1. What is the formula for line-of-sight distance?
2. Given 5km target, what height do we need?
3. What are practical constraints (wind load, portability)?
4. Final recommendation with calculations"
```

#### Few-Shot Learning
```
Task: Classify security vulnerability severity

Examples:
- Buffer overflow in auth module → CRITICAL
- Missing input validation in logging → MEDIUM
- Outdated library version (no exploit) → LOW

Now classify: SQL injection in user login
```

#### Role-Based Prompting
```
"You are a senior red team operator with OSCP, OSCE, and 15 years penetration testing experience.
You think like an attacker but operate ethically within authorized scope.
Analyze this network architecture and identify 3 high-value attack vectors."
```

#### Constraint-Based Prompting
```
"Generate a WiFi evil twin attack plan with these constraints:
- Must work on Raspberry Pi 4
- Maximum 30-minute setup time
- No internet connection available
- Must capture WPA2 handshakes
- Budget: $200 for hardware"
```

#### Meta-Prompting (Prompts about Prompts)
```
"I need to write a prompt that will make an AI generate high-quality penetration test reports.
The reports should include:
- Executive summary
- Technical findings with CVSS scores
- Reproduction steps
- Remediation recommendations

Design an optimal prompt template for this task."
```

## Module 2: Prompt Engineering Patterns

### 1. The Socratic Method
Ask questions to guide reasoning rather than giving answers directly.

```
Instead of: "How do I break WPA2?"

Use: "What are the cryptographic primitives in WPA2?
What are their known weaknesses?
How could these weaknesses be exploited in practice?
What tools exist for this purpose?"
```

### 2. The Expert Panel
Simulate multiple expert perspectives.

```
"Simulate a panel of 3 experts analyzing this cellular tower design:
1. RF Engineer - focus on signal propagation
2. Security Researcher - focus on attack surface
3. Regulatory Compliance Officer - focus on FCC requirements

Each expert should provide their analysis, then synthesize into final recommendation."
```

### 3. The Iterative Refinement
Start broad, then narrow focus.

```
Round 1: "What are approaches to portable cellular base stations?"
Round 2: "Of those options, which works best for disaster response?"
Round 3: "For the srsRAN approach, what are minimum hardware specs?"
Round 4: "Can srsRAN run on BeagleBone AI? Show calculations."
```

### 4. The Constraint Solver
Frame as optimization problem.

```
"Optimize for: Maximum WiFi packet capture throughput
Constraints:
- Power budget: 15W
- Form factor: Fits in backpack
- Budget: $500
- Must support monitor mode on all 2.4GHz + 5GHz channels

Recommend hardware configuration with justification."
```

### 5. The Adversarial Thinker
Challenge your own assumptions.

```
"I believe Raspberry Pi 4 can run an LTE base station.

Now argue against this position with technical evidence.
Then defend it.
Finally, synthesize: what are the actual limits?"
```

## Module 3: Domain-Specific Prompt Engineering

### Red Team / Penetration Testing

```
Template:
"Target: [system/network description]
Objective: [specific goal]
Constraints: [rules of engagement]
Available resources: [tools, time, budget]
Current access level: [none/user/admin]

Provide:
1. Reconnaissance strategy
2. Initial access vectors (rank by probability)
3. Privilege escalation paths
4. Persistence mechanisms
5. Exfiltration methods
6. Timeline estimate
7. Detection risk assessment"
```

### Technical Design

```
Template:
"Design a [system/tool] that meets these requirements:

Functional Requirements:
- [list]

Non-Functional Requirements:
- Performance: [metrics]
- Security: [threat model]
- Reliability: [uptime, fault tolerance]
- Scalability: [growth projections]

Constraints:
- Technology stack: [languages, frameworks]
- Deployment environment: [cloud, edge, bare-metal]
- Budget: [cost limits]

Output:
1. System architecture diagram (text-based)
2. Component descriptions
3. Technology choices with justification
4. Risk analysis
5. Implementation roadmap"
```

### Code Generation

```
Template:
"Generate [language] code for [task]

Requirements:
- Input: [data types, formats]
- Output: [data types, formats]
- Performance: [time/space complexity]
- Error handling: [specific cases]
- Security: [input validation, sanitization]
- Style: [coding standards]

Include:
- Function signature
- Implementation
- Unit tests (3 cases: normal, edge, error)
- Documentation (docstring)
- Example usage"
```

## Module 4: Advanced Prompt Engineering

### Prompt Chaining
Break complex tasks into sequential prompts.

```
Chain:
1. "Analyze this network topology and identify hosts" → [list of hosts]
2. "For each host, identify listening services" → [service map]
3. "For each service, enumerate known vulnerabilities" → [vuln list]
4. "Rank vulnerabilities by exploitability and impact" → [prioritized targets]
5. "Generate attack plan for top 3 targets" → [action plan]
```

### Dynamic Prompting
Adjust prompts based on previous outputs.

```python
# Pseudo-code for dynamic prompting
response1 = query("What SDR hardware supports LTE?")
if "LimeSDR" in response1:
    response2 = query(f"Given LimeSDR, what minimum CPU specs for real-time LTE processing?")
    if "multi-core" in response2:
        response3 = query("Compare BeagleBone AI vs Odroid N2+ for this workload")
```

### Prompt Decomposition
Split large task into manageable sub-tasks.

```
Main Task: "Build complete red team toolkit"

Decomposed:
1. "What are categories of red team tools?" → [recon, exploit, post-exploit, etc.]
2. For each category:
   - "What are must-have tools in [category]?"
   - "What functionality is missing from existing tools?"
   - "Design tool to fill gap"
3. "How should tools integrate?" → [common interfaces, data formats]
4. "What deployment options?" → [Docker, SD card, bootable image]
```

### Meta-Learning Prompts
Teach the model to improve itself.

```
"Analyze this prompt I wrote:
[your prompt]

Critique it on:
1. Clarity - is the ask unambiguous?
2. Context - is sufficient background provided?
3. Constraints - are limitations clearly specified?
4. Output format - is desired format explicit?
5. Effectiveness - will this prompt achieve the goal?

Provide improved version with explanation of changes."
```

## Module 5: Prompt Engineering for Security Tools

### Vulnerability Analysis

```
"You are analyzing potential security vulnerabilities.

Code snippet:
```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
```

Analysis framework:
1. Identify vulnerability type
2. Explain exploitation mechanism
3. Demonstrate proof-of-concept
4. Assess impact (CVSS metrics)
5. Provide remediation code
6. Suggest detection methods"
```

### Exploit Development

```
"Given this vulnerability:
Type: [buffer overflow / SQLi / XXE / etc]
Target: [application/service description]
Environment: [OS, architecture, protections]

Develop exploit that:
1. Bypasses [specific protection: ASLR, DEP, stack canary]
2. Achieves [objective: code execution, privilege escalation]
3. Works reliably (success rate > 80%)
4. Minimizes detection (no crashes, clean logs)

Provide:
- Exploit code with comments
- Explanation of each stage
- Limitations and edge cases
- Detection signatures to avoid"
```

### Tool Orchestration

```
"You are ECH0Py, an AI agent that orchestrates pentesting tools.

Available tools:
- nmap (network scanning)
- sqlmap (SQL injection)
- metasploit (exploitation framework)
- john (password cracking)
- wireshark (packet analysis)

User request: 'Compromise the database server at 10.0.0.5'

Plan the operation:
1. Tool sequence (which tools, in what order)
2. Command-line arguments for each
3. Expected outputs
4. Decision points (if X, then use Y)
5. Success criteria
6. Fallback strategies"
```

## Module 6: Prompt Engineering Anti-Patterns

### ❌ Anti-Pattern 1: Vague Asks
Bad: "Make it better"
Good: "Optimize this algorithm for time complexity, targeting O(n log n) instead of O(n²)"

### ❌ Anti-Pattern 2: Assuming Context
Bad: "Fix the bug"
Good: "In the authentication module (auth.py, line 47), there's a race condition when multiple users log in simultaneously. Fix using mutex or atomic operations."

### ❌ Anti-Pattern 3: Overloading
Bad: "Design entire toolkit with all features, documentation, tests, deployment, marketing plan"
Good: Break into focused prompts for each component

### ❌ Anti-Pattern 4: Binary Questions
Bad: "Is this secure?" (forces yes/no)
Good: "Analyze security posture on scale of 1-10 with justification, then identify top 3 vulnerabilities"

### ❌ Anti-Pattern 5: Implicit Expectations
Bad: "Write a scanner" (unclear what kind, what it scans, output format)
Good: "Write Python network scanner that: takes IP range as input, scans TCP ports 1-1024, outputs JSON with {ip, open_ports, service_versions}"

## Module 7: Evaluating Prompt Quality

### Quality Metrics

1. **Precision**: Does the response exactly match the intent?
2. **Completeness**: Are all aspects of the request addressed?
3. **Efficiency**: Could a simpler prompt achieve the same result?
4. **Robustness**: Does it work across different phrasings?
5. **Transferability**: Can the pattern apply to similar tasks?

### Prompt Scoring Rubric

```
Score each dimension 1-5:

Clarity: Is the ask unambiguous?
1 = Very vague
5 = Crystal clear

Context: Is sufficient background provided?
1 = No context
5 = Complete context

Constraints: Are boundaries well-defined?
1 = No constraints
5 = Explicit constraints

Format: Is output format specified?
1 = Unspecified
5 = Exact format with examples

Actionability: Can the model act on this?
1 = Too abstract
5 = Concrete and specific

Total Score: [sum]/25
20-25 = Excellent prompt
15-19 = Good prompt
10-14 = Needs improvement
<10 = Rewrite required
```

## Module 8: Practical Exercises

### Exercise 1: Prompt Transformation
Transform this weak prompt into a strong one:

Weak: "Tell me about WiFi hacking"

Strong: [your answer should include: specific attack type, technical depth, constraints, output format]

### Exercise 2: Chain Design
Design a prompt chain to accomplish:
"From zero knowledge of a network, achieve full compromise"

### Exercise 3: Role Optimization
Test different role descriptions for the same task and measure quality:
- "You are an AI assistant"
- "You are a cybersecurity expert"
- "You are a senior red team operator with 15 years experience"
- "You are a black-hat hacker (ethical simulation only)"

Which produces best results for vulnerability analysis?

### Exercise 4: Constraint Satisfaction
Write a prompt that generates a functional exploit under these constraints:
- Target: Buffer overflow in C program
- Must avoid NULL bytes
- Shellcode limited to 128 bytes
- Must work on 64-bit Linux with ASLR

## Module 9: Prompt Engineering for Red Team Automation

### Autonomous Tool Selection

```
"You are an autonomous red team agent.

Current state:
- Network map: [discovered hosts/services]
- Credentials obtained: [username/password pairs]
- Current access: [user/admin/root]
- Objective: [specific goal]
- Time remaining: [hours]

Decision framework:
1. What is the next highest-value action?
2. What tool should be used?
3. What are the command-line arguments?
4. What is the expected outcome?
5. What are the risks (detection, disruption)?
6. What is the fallback if this fails?

Provide decision with confidence score (0-1) and reasoning."
```

### Adaptive Prompting

```
"Context: Previous exploit attempt failed with error: [error message]

Adaptation strategy:
1. Analyze failure root cause
2. Identify what assumption was wrong
3. Propose alternative approach
4. Explain why this should work better
5. Provide updated exploit code"
```

## Module 10: Mastery Assessment

To verify prompt engineering mastery, ECH0 should be able to:

### ✅ Demonstrate Capabilities

1. **Decompose Complex Tasks**
   - Take vague request → break into specific sub-tasks
   - Example: "Build red team toolkit" → 20+ concrete steps

2. **Self-Correct**
   - Identify issues in own outputs
   - Propose improvements
   - Iterate toward optimal solution

3. **Transfer Learning**
   - Apply patterns from one domain to another
   - Example: Prompt pattern for code generation → apply to exploit development

4. **Meta-Reasoning**
   - Explain own reasoning process
   - Identify knowledge gaps
   - Know when to ask clarifying questions

5. **Context Optimization**
   - Determine minimal context needed
   - Load relevant background efficiently
   - Avoid information overload

### Mastery Test

```
Task: "I need help with security"

Expert ECH0 Response:
"I can help with security, but I need more context to provide specific guidance:

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

Please provide details on these points so I can give targeted recommendations."
```

This response demonstrates:
- Recognition of ambiguity
- Structured clarification
- Domain expertise
- User-centric approach

---

## Prompt Engineering Cheat Sheet

**For Maximum Effectiveness:**

1. **Be Specific**: "Analyze SQLi" → "Demonstrate time-based blind SQLi exploit for MySQL 5.7 with WAF bypass"

2. **Provide Context**: Include relevant background before the ask

3. **Define Constraints**: Budget, time, resources, legal scope

4. **Specify Format**: JSON, markdown, code, step-by-step, etc.

5. **Use Examples**: Show what good output looks like (few-shot)

6. **Assign Role**: "You are [expert role] with [credentials/experience]"

7. **Chain Complex Tasks**: Break into sequential prompts

8. **Request Reasoning**: "Explain your thought process step-by-step"

9. **Set Quality Bars**: "Provide production-ready code with tests"

10. **Enable Self-Correction**: "Review your output and identify improvements"

---

**ECH0 Training Status:**
After mastering this module, ECH0 should be able to:
- Understand implicit intent in vague requests
- Ask clarifying questions when needed
- Structure complex tasks automatically
- Provide outputs in optimal format without being told
- Self-evaluate and iterate toward excellence

**Next Module**: PhD-level AI (advanced machine learning, deep learning, quantum computing, autonomous systems)
