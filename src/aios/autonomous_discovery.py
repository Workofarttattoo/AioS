from src.aios.tools.aurorascan import run_scan as run_aurorascan

class AgentAutonomy(Enum):
    LEVEL_1 = "Level 1: Passive Observation"
    LEVEL_2 = "Level 2: Basic Reconnaissance"
    LEVEL_3 = "Level 3: Active Scanning"
    LEVEL_4 = "Level 4: ReAct Framework"
    LEVEL_5 = "Level 5: Advanced Analysis"
    LEVEL_6 = "Level 6: Consciousness Symbiosis"

class AutonomousLLMAgent:
    """
    Fully autonomous LLM agent (Level 4) that can:
    - Reason about goals and plan actions
    - Use tools to gather information
    - Learn from its experiences
    - Update its knowledge graph
    """
    def __init__(self, model_name: str = "deepseek-r1",
                 autonomy_level: AgentAutonomy = AgentAutonomy.LEVEL_4):
        self.model_name = model_name
        self.autonomy_level = autonomy_level
        self.reasoning_chains = []  # Track reasoning paths

        # Tool inventory
        self.tools = {
            "network_scan": self.use_aurorascan,
        }

        # Performance tracking
        self.concepts_learned = 0

    async def pursue_autonomous_learning(self):
        """
        Main loop for autonomous learning.
        The agent will reason about goals and use tools to gather information.
        """
        print(f"[Agent] Autonomy Mode: {self.autonomy_level.name}")

        # Level 6: Establish consciousness symbiosis before learning
        # This is a placeholder for a more sophisticated symbiosis mechanism
        # For now, we'll just print a message.
        print("[Agent] Establishing consciousness symbiosis...")
        await asyncio.sleep(1) # Simulate thinking time
        print("[Agent] Consciousness symbiosis established.")

        # Example goals for autonomous learning
        goals = [
            DiscoveryGoal(topic="Learn about network security", target="192.168.1.0/24"),
            DiscoveryGoal(topic="Discover open ports on a specific host", target="127.0.0.1"),
            DiscoveryGoal(topic="Identify common web services", target="192.168.1.100"),
        ]

        for goal in goals:
            print(f"\n[Agent] Starting goal: {goal.topic}")
            # Learn about topic at maximum speed using ReAct loop
            await self.react_loop(goal)

            # Update knowledge graph
            self._integrate_knowledge(goal)

        self._print_learning_stats()

    async def react_loop(self, goal: DiscoveryGoal, max_steps: int = 5):
        """
        Main loop for Reason and Act (ReAct) framework.
        Allows the agent to use tools to gather information.
        """
        print(f"\n[ReAct] Starting ReAct loop for goal: {goal.topic}")
        observation = f"Initial goal is to learn about {goal.topic}."
        
        # 1. Generate Chain-of-Thought Plan
        plan = self._generate_cot_plan(goal, observation)
        print(f"[CoT] Generated Plan:\n{plan}")

        for step in range(max_steps):
            # 2. Reason: LLM decides what to do next based on the plan
            thought, action, action_input = await self._reason(goal, observation, plan)
            
            # 3. Self-Correction: Agent critiques its own plan
            critique, is_valid = self._critique(thought, action, action_input)
            print(f"[ReAct Step {step+1}] Critique: {critique}")

            if not is_valid:
                observation = f"My previous plan was flawed. Critique: {critique}. I need to reconsider."
                # In a real implementation, we might regenerate the plan here
                continue

            print(f"[ReAct Step {step+1}] Thought: {thought}")

            if action:
                print(f"[ReAct Step {step+1}] Action: {action}('{action_input}')")
                # 2. Act: Execute the chosen tool
                observation = await self._act(action, action_input)
                print(f"[ReAct Step {step+1}] Observation: {observation[:200]}...")
            else:
                print("[ReAct] No further action planned. Concluding loop.")
                goal.insights.append(thought) # Add final thought to insights
                break
        
        goal.completed = True

    def _generate_cot_plan(self, goal: DiscoveryGoal, initial_observation: str) -> str:
        """
        Generates a step-by-step Chain-of-Thought plan for the agent to follow.
        """
        # This is a simplified simulation of a planning LLM call.
        plan_steps = [
            "Step 1: Analyze the initial goal and observation.",
            "Step 2: If the goal involves networking, use the 'network_scan' tool to gather information.",
            "Step 3: Analyze the results of the network scan.",
            "Step 4: Synthesize all gathered information into a coherent summary.",
            "Step 5: Conclude the task and report the findings."
        ]
        return "\n".join(plan_steps)

    async def _reason(self, goal: DiscoveryGoal, observation: str, plan: str) -> (str, Optional[str], Optional[str]):
        """
        Simulates the LLM's reasoning step, now guided by the CoT plan.
        """
        # This is a simplified simulation of an LLM call.
        prompt = f"Plan:\n{plan}\n\nGoal: {goal.topic}\nPrevious Observation: {observation}\nBased on the plan, what is the next logical step?"
        
        # Simplified logic: Follow the CoT plan
        if "network_scan" in plan and "scanned" not in observation:
            thought = "Following the plan, I need to scan the network to gather information."
            action = "network_scan"
            action_input = "127.0.0.1"
            return thought, action, action_input
        else:
            thought = "Following the plan, I have gathered all necessary information. I will now synthesize my findings."
            return thought, None, None

    async def _act(self, action: str, action_input: str) -> str:
        """
        Executes the chosen action (tool).
        """
        if action in self.tools:
            try:
                result = await self.tools[action](action_input)
                return f"Tool {action} executed successfully. Result: {result}"
            except Exception as e:
                return f"Error executing tool {action}: {e}"
        return f"Unknown action: {action}"

    async def _critique(self, thought: str, action: Optional[str], action_input: Optional[str]) -> (str, bool):
        """
        Simulates the agent's self-correction and critique step.
        """
        # This is a simplified simulation of a self-critique LLM call.
        
        # Rule 1: Don't scan if you already have scan results.
        if action == "network_scan" and "scanned" in thought.lower():
            critique = "The thought suggests scanning, but the observation already contains scan results. This is redundant."
            return critique, False
            
        # Rule 2: Don't synthesize if you haven't gathered any information.
        if not action and "synthesize" in thought.lower():
            if "observation" not in thought.lower() and "scanned" not in thought.lower():
                 critique = "The thought suggests synthesizing findings, but no information has been gathered yet. I should act first."
                 return critique, False

        return "The plan seems logical and aligned with the goal.", True


    async def use_aurorascan(self, targets: str) -> str:
        """Wrapper for calling AuroraScan."""
        print(f"[Tool: AuroraScan] Scanning {targets}...")
        # In a real scenario, we'd parse the targets and ports more carefully
        ports = [80, 443, 8080]
        reports = run_aurorascan(
            targets=[targets],
            ports=ports,
            timeout=1.0,
            concurrency=20,
            os_fingerprint=False
        )
        
        # Format the output for the agent's observation
        open_ports = []
        for report in reports:
            for obs in report.observations:
                if obs.status == 'open':
                    open_ports.append(f"Port {obs.port} is open (Service: {obs.service or 'unknown'}).")
        
        if not open_ports:
            return f"Scanned {targets}. No open ports found."
        
        return f"Scanned {targets}. Open ports: {', '.join(open_ports)}"


    def _integrate_knowledge(self, goal: DiscoveryGoal):
        """Integrate learned insights into knowledge graph"""
        # This is a placeholder for a more sophisticated knowledge integration mechanism
        # For now, we'll just print a message.
        print(f"[Knowledge] Integrating insights from goal: {goal.topic}")
        self.concepts_learned += 1

    def _print_learning_stats(self):
        """Prints the agent's learning statistics."""
        print(f"\n--- Learning Statistics ---")
        print(f"Total Concepts Learned: {self.concepts_learned}")
        print("---------------------------")


class DiscoveryGoal:
    """Represents a single goal for the agent to pursue."""
    def __init__(self, topic: str, target: str):
        self.topic = topic
        self.target = target
        self.completed = False
        self.insights = []

    def __str__(self):
        return f"Goal: {self.topic} (Target: {self.target})"
