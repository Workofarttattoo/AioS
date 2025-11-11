"""
Example demonstrating how to integrate autonomous discovery with AgentaOS meta-agents.

Shows integration patterns for:
1. Security agent - autonomous threat pattern learning
2. Scalability agent - autonomous resource optimization
3. Orchestration agent - autonomous policy learning
"""

import sys
import asyncio
from pathlib import Path

# Add AgentaOS to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous_discovery import (
    AutonomousLLMAgent,
    AgentAutonomy,
    create_autonomous_discovery_action,
    check_autonomous_discovery_dependencies
)
from runtime import ExecutionContext, ActionResult
from config import DEFAULT_MANIFEST


async def example_security_autonomous_learning():
    """
    Example 1: Security agent with autonomous threat pattern learning.
    The agent autonomously learns about new threat vectors.
    """
    print("═" * 70)
    print("EXAMPLE 1: Security Agent - Autonomous Threat Learning")
    print("═" * 70)

    # Create execution context
    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_AUTONOMOUS_DISCOVERY': '1'
    }

    async def security_threat_research_handler(ctx: ExecutionContext) -> ActionResult:
        """
        Security agent action with autonomous threat research.
        The agent learns about emerging threats autonomously.
        """
        print("\n[SecurityAgent] Initiating autonomous threat research...")

        # Create autonomous discovery action
        mission = "ransomware attack vectors cloud infrastructure vulnerabilities"
        discovery_action = create_autonomous_discovery_action(
            mission=mission,
            duration_hours=0.5  # 30 minutes
        )

        # Let agent autonomously research
        knowledge = await discovery_action()

        # Integrate discovered knowledge into security metadata
        ctx.publish_metadata('security.autonomous_research', knowledge)
        ctx.publish_metadata('security.threat_patterns', {
            'total_patterns_discovered': knowledge['stats']['total_concepts'],
            'confidence': knowledge['stats']['average_confidence'],
            'learning_rate': knowledge['stats']['learning_rate']
        })

        print(f"\n[SecurityAgent] Research complete!")
        print(f"  Threat patterns discovered: {knowledge['stats']['total_concepts']}")
        print(f"  Average confidence: {knowledge['stats']['average_confidence']:.2%}")
        print(f"  Learning rate: {knowledge['stats']['learning_rate']:.2f} concepts/sec")

        return ActionResult(
            success=True,
            message="[info] Autonomous threat research completed",
            payload=knowledge['stats']
        )

    # Execute
    result = await security_threat_research_handler(ctx)
    print(f"\n✓ Result: {result.message}")
    print(f"  Knowledge graph nodes: {len(ctx.metadata.get('security.autonomous_research', {}).get('nodes', {}))}")
    print()


async def example_scalability_autonomous_optimization():
    """
    Example 2: Scalability agent with autonomous resource optimization.
    The agent learns optimal scaling strategies.
    """
    print("═" * 70)
    print("EXAMPLE 2: Scalability Agent - Autonomous Resource Optimization")
    print("═" * 70)

    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_AUTONOMOUS_DISCOVERY': '1'
    }

    async def scalability_optimization_handler(ctx: ExecutionContext) -> ActionResult:
        """
        Scalability agent learns optimal resource allocation strategies.
        """
        print("\n[ScalabilityAgent] Starting autonomous optimization learning...")

        # Agent autonomously learns about scaling strategies
        agent = AutonomousLLMAgent(
            model_name="deepseek-r1",
            autonomy_level=AgentAutonomy.LEVEL_4
        )

        mission = "Kubernetes autoscaling patterns distributed system load balancing"
        agent.set_mission(mission, duration_hours=0.3)

        await agent.pursue_autonomous_learning()

        # Extract optimization strategies
        knowledge = agent.export_knowledge_graph()

        # Apply learned strategies to scaling decisions
        scaling_strategies = []
        for concept, node_data in knowledge['nodes'].items():
            if node_data['confidence'] > 0.85:
                scaling_strategies.append({
                    'strategy': concept,
                    'confidence': node_data['confidence'],
                    'discovered_at': node_data['discovered_at']
                })

        ctx.publish_metadata('scalability.learned_strategies', scaling_strategies)
        ctx.publish_metadata('scalability.knowledge_graph', knowledge)

        print(f"\n[ScalabilityAgent] Optimization learning complete!")
        print(f"  High-confidence strategies: {len(scaling_strategies)}")
        print(f"  Total concepts learned: {knowledge['stats']['total_concepts']}")
        print(f"\n  Top 3 strategies:")
        for i, strategy in enumerate(sorted(scaling_strategies, key=lambda x: x['confidence'], reverse=True)[:3], 1):
            print(f"    {i}. {strategy['strategy']} (confidence: {strategy['confidence']:.2%})")

        return ActionResult(
            success=True,
            message="[info] Autonomous optimization learning completed",
            payload={'strategies_count': len(scaling_strategies)}
        )

    # Execute
    result = await scalability_optimization_handler(ctx)
    print(f"\n✓ Result: {result.message}")
    print()


async def example_orchestration_autonomous_policy():
    """
    Example 3: Orchestration agent with autonomous policy learning.
    The agent learns best practices for orchestration.
    """
    print("═" * 70)
    print("EXAMPLE 3: Orchestration Agent - Autonomous Policy Learning")
    print("═" * 70)

    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_AUTONOMOUS_DISCOVERY': '1'
    }

    async def orchestration_policy_handler(ctx: ExecutionContext) -> ActionResult:
        """
        Orchestration agent learns optimal coordination policies.
        """
        print("\n[OrchestrationAgent] Learning orchestration policies...")

        # Create discovery action for policy learning
        mission = "distributed system coordination patterns microservices orchestration best practices"
        discovery = create_autonomous_discovery_action(
            mission=mission,
            duration_hours=0.4
        )

        knowledge = await discovery()

        # Extract high-confidence policies
        policies = []
        for concept, node_data in knowledge['nodes'].items():
            if node_data['confidence'] > 0.80:
                policies.append({
                    'policy': concept,
                    'confidence': node_data['confidence'],
                    'connections': len(node_data.get('children', []))
                })

        ctx.publish_metadata('orchestration.learned_policies', policies)
        ctx.publish_metadata('orchestration.knowledge_graph', knowledge)

        print(f"\n[OrchestrationAgent] Policy learning complete!")
        print(f"  Policies discovered: {len(policies)}")
        print(f"  Learning rate: {knowledge['stats']['learning_rate']:.2f} concepts/sec")
        print(f"\n  Top policies by confidence:")
        for i, policy in enumerate(sorted(policies, key=lambda x: x['confidence'], reverse=True)[:5], 1):
            print(f"    {i}. {policy['policy']} (confidence: {policy['confidence']:.2%}, connections: {policy['connections']})")

        return ActionResult(
            success=True,
            message="[info] Autonomous policy learning completed",
            payload={'policies_count': len(policies)}
        )

    # Execute
    result = await orchestration_policy_handler(ctx)
    print(f"\n✓ Result: {result.message}")
    print()


async def example_continuous_autonomous_learning():
    """
    Example 4: Continuous autonomous learning across agent lifecycle.
    Demonstrates how agents can continuously learn and adapt.
    """
    print("═" * 70)
    print("EXAMPLE 4: Continuous Autonomous Learning")
    print("═" * 70)

    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_AUTONOMOUS_DISCOVERY': '1',
        'AGENTA_CONTINUOUS_LEARNING': '1'
    }

    # Create persistent learning agent
    agent = AutonomousLLMAgent(
        model_name="deepseek-r1",
        autonomy_level=AgentAutonomy.LEVEL_4
    )

    print("\n[System] Continuous learning agent initialized")
    print("[System] Agent will autonomously learn and adapt over time")

    # Learning cycle 1: Initial knowledge acquisition
    print("\n--- Learning Cycle 1: Initial Acquisition ---")
    agent.set_mission(
        "system performance optimization monitoring",
        duration_hours=0.2
    )
    await agent.pursue_autonomous_learning()

    cycle1_stats = agent.export_knowledge_graph()['stats']
    print(f"  Concepts learned: {cycle1_stats['total_concepts']}")
    print(f"  Learning rate: {cycle1_stats['learning_rate']:.2f} concepts/sec")

    # Learning cycle 2: Agent autonomously expands based on what it learned
    print("\n--- Learning Cycle 2: Autonomous Expansion ---")
    # Agent continues learning autonomously (in real implementation, it would
    # self-direct based on discovered concepts)
    agent.set_mission(
        "performance observability distributed tracing",
        duration_hours=0.2
    )
    await agent.pursue_autonomous_learning()

    cycle2_stats = agent.export_knowledge_graph()['stats']
    print(f"  Total concepts: {cycle2_stats['total_concepts']}")
    print(f"  Average confidence: {cycle2_stats['average_confidence']:.2%}")

    # Agent has built comprehensive knowledge graph
    knowledge = agent.export_knowledge_graph()
    ctx.publish_metadata('continuous_learning.knowledge_graph', knowledge)

    print(f"\n✓ Continuous learning cycles complete")
    print(f"  Final knowledge graph: {knowledge['stats']['total_concepts']} nodes")
    print(f"  Average confidence: {knowledge['stats']['average_confidence']:.2%}")
    print()


async def main():
    """Run all autonomous discovery examples."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   AgentaOS Autonomous Discovery - Integration Examples          ║")
    print("║   Demonstrating Level 4 Autonomy with Meta-Agents               ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Check dependencies
    deps = check_autonomous_discovery_dependencies()
    print("Dependency Check:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")

    if not all(deps.values()):
        print("\n✗ Missing required dependencies")
        print("  Install with: pip install torch numpy")
        return

    print("\n" + "="*70)
    print("RUNNING AUTONOMOUS DISCOVERY EXAMPLES")
    print("="*70 + "\n")

    # Run examples
    try:
        await example_security_autonomous_learning()
    except Exception as e:
        print(f"✗ Security example failed: {e}\n")

    try:
        await example_scalability_autonomous_optimization()
    except Exception as e:
        print(f"✗ Scalability example failed: {e}\n")

    try:
        await example_orchestration_autonomous_policy()
    except Exception as e:
        print(f"✗ Orchestration example failed: {e}\n")

    try:
        await example_continuous_autonomous_learning()
    except Exception as e:
        print(f"✗ Continuous learning example failed: {e}\n")

    print("="*70)
    print("All autonomous discovery examples completed!")
    print("="*70)
    print()
    print("Key Takeaways:")
    print("  • Agents can operate at Level 4 autonomy (fully self-directed)")
    print("  • Knowledge graphs build automatically from missions")
    print("  • Learning rates achieve superhuman speed with distributed inference")
    print("  • Integration with AgentaOS metadata system is seamless")
    print("  • Continuous learning enables adaptive behavior over time")
    print()
    print("Next Steps:")
    print("  • Integrate into production meta-agent handlers")
    print("  • Configure distributed GPU infrastructure for maximum speed")
    print("  • Set up knowledge graph persistence for long-term learning")
    print("  • Implement human-in-the-loop for critical decisions (Level 1-3)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
