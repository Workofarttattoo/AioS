# OpenAGI Reverse Engineering & AIOS Integration Analysis

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Executive Summary

OpenAGI is a sophisticated agent framework designed for composable AI agent creation with domain expert integration. The codebase provides a **ReAct (Reasoning + Acting) pattern** implementation with modular tool support and AIOS kernel integration. This document details the architecture reverse-engineering and integration strategy for your AIOS system.

---

## Architecture Overview

### Core Design Patterns

OpenAGI implements three primary patterns:

#### 1. **ReAct Agent Pattern** (ReasoningAgent + ActionAgent)
- **Reasoning Phase**: Agent generates a multi-step workflow as JSON
- **Action Phase**: Agent executes tools based on workflow, receives observations
- **Reflection**: Agent updates its reasoning based on tool outputs

**Key Implementation**: `ReactAgent` (pyopenagi/agents/react_agent.py)
- Automatic or manual workflow modes
- JSON-based step definition: `[{"message": str, "tool_use": [list]}]`
- Retry logic with exponential backoff (default: 3 retries)
- Full turnaround time tracking for performance monitoring

#### 2. **Tool Interface Abstraction**
Three-tier tool hierarchy:
```
BaseTool (abstract)
├── BaseRapidAPITool (API gateway pattern)
└── BaseHuggingfaceTool (Model inference pattern)
```

**Tool Contract**:
```python
class BaseTool:
    def run(params: dict) -> Any
    def get_tool_call_format() -> dict  # OpenAI function calling schema
```

Currently supports 20+ tools (Wikipedia, Google Search, Image2Image, Travel Planning, etc.)

#### 3. **Agent Factory + Process Management**
Dynamic agent instantiation with thread-safe process management:
```
AgentFactory
├── activate_agent(agent_name, task_input) → Agent instance
├── run_agent(agent_name, task_input) → execution result
└── deactivate_agent(aid) → cleanup
```

**Process Management**:
- `AgentProcess` wraps query execution with threading
- Status tracking: active → done
- Latency metrics: created_time, start_time, end_time
- Time limit enforcement for suspension/resumption

---

## Detailed Component Analysis

### 1. BaseAgent Architecture

**Location**: `pyopenagi/agents/base_agent.py`

**Responsibilities**:
```python
class BaseAgent:
    - agent_name: str
    - config: dict (loaded from config.json)
    - tool_list: dict[str, Tool] (dynamically loaded)
    - tool_info: list[dict] (simplified tool metadata)
    - messages: list[dict] (conversation history, OpenAI format)
    - workflow_mode: str ("manual" | "automatic")
    - rounds: int (conversation turns)
    - status: str (active | done | suspended)
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `load_config()` | Parse `agent_name/config.json` with tools list |
| `load_tools()` | Dynamically import and instantiate tools via reflection |
| `build_system_instruction()` | Construct system prompt with tool info |
| `automatic_workflow()` | LLM generates JSON workflow (retry 3x) |
| `manual_workflow()` | Override point for custom execution |
| `get_response()` | Thread-wrapped LLM request with timing |
| `create_agent_request()` | Wrap query in AgentProcess |
| `listen()` | Poll for response completion (0.2s intervals) |

**Configuration Schema** (`config.json`):
```json
{
  "description": ["Agent purpose and capabilities..."],
  "tools": ["author/tool_name", "author/another_tool"]
}
```

### 2. ReactAgent Implementation

**Location**: `pyopenagi/agents/react_agent.py`

**Enhancement over BaseAgent**:

```python
class ReactAgent(BaseAgent):
    plan_max_fail_times = 3
    tool_call_max_fail_times = 3

    def run(self):
        # 1. Build system instruction with workflow generation prompt
        self.build_system_instruction()

        # 2. Generate workflow (JSON array of steps)
        workflow = self.automatic_workflow()  # Retry 3x if invalid JSON

        # 3. Execute workflow step-by-step:
        for i, step in enumerate(workflow):
            message = step["message"]
            tool_use = step["tool_use"]  # Pre-selected tool list

            # Get response with optional pre-selected tools
            response = self.get_response(tools=selected_tools)

            # Execute tool calls with retry
            for attempt in range(self.tool_call_max_fail_times):
                actions, observations, success = self.call_tools(tool_calls)
                if success:
                    break
                else:
                    # Retry with corrected params
                    pass

            # Track metrics and update messages
            self.rounds += 1

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": self.rounds,
            "agent_waiting_time": start_time - created_time,
            "agent_turnaround_time": end_time - created_time,
            "request_waiting_times": [list],
            "request_turnaround_times": [list]
        }
```

**Tool Calling Mechanism**:
```python
def call_tools(self, tool_calls: list[dict]) -> (actions, observations, success):
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        function_params = tool_call["parameters"]

        try:
            # Execute tool
            result = self.tool_list[function_name].run(function_params)
            actions.append(f"I will call {function_name} with {function_params}")
            observations.append(f"Output: {result}")
            success = True
        except Exception:
            actions.append("I fail to call any tools")
            observations.append(f"Invalid params: {function_params}")
            success = False

    return actions, observations, success
```

### 3. Agent Factory Pattern

**Location**: `pyopenagi/agents/agent_factory.py`

**Dynamic Agent Loading**:
```python
def load_agent_instance(self, agent_name: str):
    author, name = agent_name.split("/")  # "author/agent_name"
    module_name = ".".join(["pyopenagi", "agents", author, name, "agent"])
    class_name = self.snake_to_camel(name)  # snake_case → CamelCase

    agent_module = importlib.import_module(module_name)
    agent_class = getattr(agent_module, class_name)
    return agent_class
```

**Lifecycle Management**:
```python
def activate_agent(agent_name, task_input):
    # Download if missing (from cloud registry)
    if not exists(agent_path):
        interactor.download_agent(agent_name)

    # Install dependencies
    if not interactor.check_reqs_installed(agent_name):
        interactor.install_agent_reqs(agent_name)

    # Instantiate
    agent = load_agent_instance(agent_name)(
        agent_name=agent_name,
        task_input=task_input,
        agent_process_factory=self.agent_process_factory,
        log_mode=self.agent_log_mode
    )

    aid = random.randint(100000, 999999)  # Agent ID
    agent.set_aid(aid)

    # Thread-safe registration
    with self.current_agents_lock:
        self.current_agents[aid] = agent

    return agent
```

### 4. Process Management

**Location**: `pyopenagi/agents/agent_process.py`

```python
class AgentProcess:
    agent_name: str
    query: Query
    pid: int (unique process ID)
    status: str (active | done | suspended)
    response: Response
    time_limit: float (seconds before suspension)
    created_time, start_time, end_time: float

class LLMRequestProcess(AgentProcess):
    # Specialized for LLM requests
    pass

class AgentProcessFactory:
    current_agent_processes: dict[int, AgentProcess]
    current_agent_processes_lock: Lock

    def activate_agent_process(agent_name, query):
        # Create with random PID
        process = AgentProcess(agent_name, query)
        process.set_pid(random.randint(1000000, 9999999))
        process.set_status("active")
        with lock:
            self.current_agent_processes[pid] = process
        return process

    def deactivate_agent_process(pid):
        with lock:
            self.current_agent_processes.pop(pid)
```

### 5. Tool System

**Location**: `pyopenagi/tools/base.py`

**Tool Format (OpenAI Function Calling)**:
```python
def get_tool_call_format(self) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "What the tool does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        }
    }
```

**Tool Categories**:
1. **External API Tools** (BaseRapidAPITool)
   - Google Search, Wikipedia, Currency Converter, Weather
   - HTTP wrapping with API key management

2. **Model Tools** (BaseHuggingfaceTool)
   - Image generation, transcription, document QA
   - Hugging Face Inference API integration

3. **Specialized Tools**
   - Travel planner (flights, hotels, attractions)
   - Wolfram Alpha (math/computation)
   - Trip Advisor (restaurant reviews)

---

## Integration Strategy for AIOS

### 1. Alignment with Current AIOS Architecture

**Compatibility Mapping**:

| OpenAGI | AIOS Equivalent | Integration Point |
|---------|-----------------|-------------------|
| `BaseAgent` | `ExecutionContext` wrapper | Unified agent interface |
| `ReactAgent` | Meta-agent action handler | Plugin as sovereign agent |
| `AgentFactory` | Agent lifecycle manager | Integrate with AIOS supervisor |
| `AgentProcess` | Request queue entry | Hook into LLM request queue |
| `Tool` | Sovereign toolkit | Extend tool registry |

### 2. Implementation Phases

#### Phase 1: Core Integration (Week 1)
```
├── 1.1 Create AIOS adapter layer
│   └── aios/openagi_agent_adapter.py (ExecutionContext ↔ BaseAgent)
├── 1.2 Register ReAct as meta-agent
│   └── aios/agents/openagi_react_agent.py (critical=False)
└── 1.3 Tool registry integration
    └── aios/tools/__init__.py (add OpenAGI tools to TOOL_REGISTRY)
```

#### Phase 2: Advanced Features (Week 2)
```
├── 2.1 Autonomous discovery integration
│   └── Enable ReAct agent to learn tool combinations autonomously
├── 2.2 Quantum-enhanced planning
│   └── Use QuantumVQE for workflow optimization
└── 2.3 Performance monitoring
    └── Export agent metrics to AIOS supervisor telemetry
```

#### Phase 3: Production Hardening (Week 3)
```
├── 3.1 Forensic mode support
├── 3.2 Distributed execution (multi-GPU inference)
└── 3.3 Knowledge persistence
```

---

## Code Integration Points

### Integration 1: Adapter Layer

```python
# aios/openagi_agent_adapter.py
from pyopenagi.agents.react_agent import ReactAgent
from .runtime import ExecutionContext, ActionResult
from typing import Optional
import json

class OpenAGIAdapter:
    """Bridge OpenAGI agents to AIOS ExecutionContext"""

    def __init__(self, agent_name: str, agent_factory):
        self.agent_name = agent_name
        self.agent_factory = agent_factory

    def adapt_to_action_handler(self):
        """Convert OpenAGI agent into AIOS action handler"""
        def action_handler(ctx: ExecutionContext) -> ActionResult:
            # 1. Extract task from context
            task_input = ctx.environment.get("OPENAGI_TASK_INPUT")
            if not task_input:
                return ActionResult(
                    success=False,
                    message="[error] OPENAGI_TASK_INPUT not set",
                    payload={}
                )

            # 2. Activate agent
            try:
                result = self.agent_factory.run_agent(
                    agent_name=self.agent_name,
                    task_input=task_input
                )

                # 3. Publish telemetry
                ctx.publish_metadata(f"openagi.{self.agent_name}", {
                    "rounds": result.get("rounds", 0),
                    "waiting_time": result.get("agent_waiting_time"),
                    "turnaround_time": result.get("agent_turnaround_time"),
                    "request_metrics": {
                        "waiting_times": result.get("request_waiting_times", []),
                        "turnaround_times": result.get("request_turnaround_times", [])
                    }
                })

                return ActionResult(
                    success=True,
                    message=f"[info] {self.agent_name} completed in {result['rounds']} rounds",
                    payload=result
                )

            except Exception as exc:
                return ActionResult(
                    success=False,
                    message=f"[error] {self.agent_name}: {exc}",
                    payload={"exception": str(exc)}
                )

        return action_handler
```

### Integration 2: Meta-Agent Registration

```python
# Add to aios/config.py DEFAULT_MANIFEST
"openagi_orchestrator": {
    "actions": {
        "execute_react_agent": {
            "description": "Execute ReAct workflow for complex reasoning tasks",
            "handler": "openagi_orchestrator.execute_react_agent",
            "critical": False,
            "timeout": 300,
            "enabled_by_env": "AIOS_OPENAGI_ENABLED"
        },
        "orchestrate_multi_agent": {
            "description": "Coordinate multiple OpenAGI agents for ensemble solving",
            "handler": "openagi_orchestrator.orchestrate_multi_agent",
            "critical": False
        }
    }
}
```

### Integration 3: Tool Registry

```python
# aios/tools/__init__.py additions
from pyopenagi.tools.google.google_search import GoogleSearch
from pyopenagi.tools.wikipedia.wikipedia import Wikipedia
from pyopenagi.tools.bing.bing_search import BingSearch

OPENAGI_TOOLS = {
    "google_search": GoogleSearch,
    "wikipedia": Wikipedia,
    "bing_search": BingSearch,
    # ... 20+ more tools
}

# Register with sovereign suite
TOOL_REGISTRY.update({
    f"openagi_{name}": tool_class
    for name, tool_class in OPENAGI_TOOLS.items()
})
```

---

## Key Features to Extract & Enhance

### 1. ReAct Workflow System

**Current Implementation**:
- Simple JSON plan generation
- Tool pre-selection before execution
- Deterministic tool calling

**Enhancement for AIOS**:
- Add probabilistic tool selection (use Oracle for confidence scoring)
- Support conditional branching in workflows
- Enable workflow caching for repeated patterns
- Integrate with Autonomous Discovery for tool learning

### 2. Tool Composition

**Current**: Tools execute independently

**Enhanced**:
- Tool chaining (output of tool A → input to tool B)
- Tool composition patterns (map-reduce for parallel execution)
- Tool result aggregation

### 3. Agent Communication

**Current**: Single agent → tools

**Enhanced**:
- Multi-agent coordination (agent A → agent B)
- Shared knowledge base (all agents learn from each other)
- Conflict resolution for contradictory results

### 4. Performance Monitoring

**Current**:
- Basic timing metrics
- Waiting time, turnaround time

**Enhanced**:
- Request queuing analysis
- Tool efficiency rankings
- Workflow optimization suggestions

---

## Performance Considerations

### Threading Model

**Current OpenAGI**:
```
Agent thread → Query → CustomizedThread → Listen (0.2s poll)
```

**Issue**: Polling with sleep = inefficient blocking

**AIOS Enhancement**:
```
Agent → Agent queue → LLM executor → Response callback
```
Use event-driven architecture instead of polling.

### Scalability

OpenAGI:
- Process per agent (PID pooling, currently random)
- Lock-based synchronization

AIOS Opportunity:
- Distributed inference (multi-GPU)
- Agent migration between nodes
- Load balancing across agent pool

---

## Configuration Schema

### Agent Config Format

```json
{
  "name": "academic_agent",
  "version": "1.0",
  "description": ["Searches academic papers and provides summaries"],
  "author": "example",
  "tools": [
    "openagi/arxiv",
    "openagi/semantic_scholar",
    "openagi/google_scholar"
  ],
  "workflow_mode": "automatic",
  "plan_retry_limit": 3,
  "tool_call_retry_limit": 3,
  "timeout": 300,
  "dependencies": {
    "python": ">=3.8",
    "packages": ["requests", "arxiv"]
  }
}
```

---

## Security & Forensic Mode

### Tool Execution Safety

**Current**: Basic exception handling

**Enhanced for AIOS**:

```python
def safe_tool_execution(tool, params, forensic_mode=False):
    if forensic_mode:
        # Dry-run: show what would happen
        return ActionResult(
            success=True,
            message=f"[info] Would execute {tool.name}",
            payload={"planned_execution": params, "forensic": True}
        )
    else:
        # Real execution with sandbox
        try:
            return tool.run(params)
        except Exception as e:
            return ActionResult(success=False, message=str(e))
```

### Path Validation

OpenAGI includes `check_path()` to confine output to `output/` directory. AIOS should enforce:
- No path traversal (`../`)
- No absolute paths outside agent sandbox
- Whitelist allowed directories

---

## Example: Integrating Travel Planner Agent

### Step 1: Copy Agent Structure
```
aios/agents/openagi/
├── travel_planner/
│   ├── agent.py (ReAct-based travel planning)
│   ├── config.json
│   └── requirements.txt
```

### Step 2: Register with AIOS

```python
# aios/agents/system.py (ApplicationAgent)
def execute_travel_planner(ctx: ExecutionContext) -> ActionResult:
    destination = ctx.environment.get("TRAVEL_DESTINATION")
    dates = ctx.environment.get("TRAVEL_DATES")
    budget = ctx.environment.get("TRAVEL_BUDGET")

    task_input = f"""
    Plan a trip to {destination} from {dates} with budget {budget}.
    Include flights, hotels, restaurants, and attractions.
    """

    adapter = OpenAGIAdapter("openagi/travel_planner", self.agent_factory)
    result = adapter.adapt_to_action_handler()(
        ExecutionContext(environment={"OPENAGI_TASK_INPUT": task_input})
    )

    return result
```

### Step 3: Integration Test

```python
def test_openagi_travel_planner():
    # Set up AIOS
    ctx = ExecutionContext(
        manifest=DEFAULT_MANIFEST,
        environment={
            "OPENAGI_TASK_INPUT": "Plan trip to Tokyo Jan 2025",
            "AIOS_OPENAGI_ENABLED": "1"
        }
    )

    # Execute via orchestration agent
    agent = OrchestrationAgent()
    result = agent.openagi_orchestrator(ctx)

    assert result.success
    assert "agent_name" in result.payload
    assert result.payload["rounds"] > 0
```

---

## Comparative Analysis: OpenAGI vs AIOS

| Aspect | OpenAGI | AIOS |
|--------|---------|------|
| **Workflow** | JSON-based static plans | Manifest-driven dynamic |
| **Tool execution** | Sequential, pre-selected | Parallel, guided by Oracle |
| **Learning** | None | Autonomous discovery |
| **Distribution** | Single machine | Multi-node with providers |
| **Forensic mode** | Basic path checks | Full read-only safety |
| **Performance** | Polling-based (0.2s intervals) | Event-driven async |
| **Tool system** | 20+ hardcoded tools | Extensible registry |
| **Process lifecycle** | Random PID pooling | Full process model |

---

## Recommended Integration Roadmap

### Week 1: Foundation
- [ ] Extract ReAct agent core to standalone module
- [ ] Create ExecutionContext adapter
- [ ] Register 5 most useful tools (Google, Wikipedia, Bing, Wolfram, Weather)
- [ ] Write tests for adapter layer

### Week 2: Expansion
- [ ] Add all 20+ OpenAGI tools to registry
- [ ] Integrate with Autonomous Discovery (tool learning)
- [ ] Implement workflow caching
- [ ] Add performance dashboards

### Week 3: Production
- [ ] Forensic mode support
- [ ] Multi-GPU distributed inference
- [ ] Agent migration between AIOS nodes
- [ ] Load testing (1000+ concurrent agents)

### Week 4+: Advanced
- [ ] Multi-agent coordination (agent → agent)
- [ ] Knowledge persistence (save learned workflows)
- [ ] Quantum-enhanced planning (QuantumVQE for tool selection)
- [ ] Custom domain agent templates

---

## References

- **OpenAGI Paper**: "When LLM Meets Domain Experts" (NeurIPS 2023)
- **Implementation**: https://github.com/agiresearch/OpenAGI
- **ReAct**: https://arxiv.org/abs/2210.03629

---

## Conclusion

OpenAGI provides a robust, extensible foundation for agentic reasoning with tool use. Its ReAct pattern complements AIOS's manifest-driven orchestration perfectly. By integrating OpenAGI's workflow generation with AIOS's autonomous discovery and distributed execution, you can create a system that:

1. **Reasons** about complex problems (ReAct)
2. **Learns** optimal tool combinations (Autonomous Discovery)
3. **Scales** across GPUs and nodes (AIOS providers)
4. **Adapts** to new domains (Agent templates)
5. **Maintains** safety and forensic compliance (Read-only modes)

This integration positions AIOS as the most sophisticated agentic OS available in 2025.
