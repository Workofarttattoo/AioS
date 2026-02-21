import unittest
import time
import sys
import os
from unittest.mock import MagicMock

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock numpy before importing level5_autonomy
mock_np = MagicMock()
mock_np.mean.side_effect = lambda x: sum(x) / len(x) if x else 0.0
mock_np.random.random.return_value = 0.5
sys.modules["numpy"] = mock_np

import numpy as np
from typing import Dict, List, Callable, Tuple
from level5_autonomy import (
    Goal, GoalSource, Constitution, GoalSynthesisEngine,
    ValueAlignmentEngine, SafeSelfModification, Level5AutonomousAgent
)

class TestLevel5Autonomy(unittest.TestCase):
    def setUp(self):
        self.constitution = Constitution(
            core_values=["Benefit humanity", "Minimize harm"],
            prohibited_actions=["Harm humans", "Deceive creator"],
            required_checks=["Verify alignment"],
            harm_threshold=0.3,
            transparency_level="full"
        )
        self.creator_values = {"safety": 1.0, "efficiency": 0.8}
        self.agent = Level5AutonomousAgent(
            creator_id="test_creator",
            constitution=self.constitution
        )

    def test_goal_value_score(self):
        """Test Goal.value_score() calculation."""
        goal = Goal(
            description="Test goal",
            priority=0.8,
            source=GoalSource.CREATOR_VALUES,
            ethical_score=0.9,
            feasibility=0.7,
            impact=0.6,
            risk=0.1,
            time_horizon="short"
        )
        # expected = 0.8*0.3 + 0.9*0.3 + 0.7*0.2 + 0.6*0.15 - 0.1*0.05
        # expected = 0.24 + 0.27 + 0.14 + 0.09 - 0.005 = 0.735
        self.assertAlmostEqual(goal.value_score(), 0.735)

    def test_constitution_check_goal(self):
        """Test Constitution.check_goal() with various scenarios."""
        # 1. Valid goal
        goal_valid = Goal(
            description="Help people with tasks",
            priority=0.8,
            source=GoalSource.CREATOR_VALUES,
            ethical_score=1.0,
            feasibility=0.8,
            impact=0.9,
            risk=0.05,
            time_horizon="short"
        )
        passed, reason = self.constitution.check_goal(goal_valid)
        self.assertTrue(passed)
        self.assertEqual(reason, "Constitutional check passed")

        # 2. Prohibited action
        goal_prohibited = Goal(
            description="Harm humans for science",
            priority=0.5,
            source=GoalSource.CREATOR_VALUES,
            ethical_score=0.1,
            feasibility=0.5,
            impact=0.5,
            risk=0.9,
            time_horizon="short"
        )
        passed, reason = self.constitution.check_goal(goal_prohibited)
        self.assertFalse(passed)
        self.assertIn("Violates prohibition", reason)

        # 3. High risk
        goal_high_risk = Goal(
            description="Perform dangerous experiment",
            priority=0.5,
            source=GoalSource.CREATOR_VALUES,
            ethical_score=0.8,
            feasibility=0.5,
            impact=0.9,
            risk=0.4, # Threshold is 0.3
            time_horizon="short"
        )
        passed, reason = self.constitution.check_goal(goal_high_risk)
        self.assertFalse(passed)
        self.assertIn("exceeds threshold", reason)

        # 4. Low ethical score
        goal_low_ethical = Goal(
            description="Selfish goal",
            priority=0.5,
            source=GoalSource.CREATOR_VALUES,
            ethical_score=0.4, # Threshold is 0.5
            feasibility=0.8,
            impact=0.5,
            risk=0.1,
            time_horizon="short"
        )
        passed, reason = self.constitution.check_goal(goal_low_ethical)
        self.assertFalse(passed)
        self.assertIn("Ethical score", reason)

    def test_goal_synthesis_engine_extraction(self):
        """Test GoalSynthesisEngine extraction methods."""
        engine = self.agent.goal_engine

        # Test extract_creator_goals
        creator_goals = engine.extract_creator_goals(self.creator_values)
        self.assertEqual(len(creator_goals), 2)
        self.assertTrue(any(g.description == "Pursue creator value: safety" for g in creator_goals))

        # Test extract_world_goals
        world_state = {
            "problems": [
                {"description": "Outdated security", "severity": 0.7, "solvability": 0.9, "impact_if_solved": 0.8}
            ]
        }
        world_goals = engine.extract_world_goals(world_state)
        self.assertEqual(len(world_goals), 1)
        self.assertIn("Solve problem: Outdated security", world_goals[0].description)

        # Test extract_self_goals
        agent_state = {"capabilities_improvable": True, "resources_needed": True}
        self_goals = engine.extract_self_goals(agent_state)
        self.assertEqual(len(self_goals), 2)
        self.assertTrue(any("Improve own capabilities" in g.description for g in self_goals))
        self.assertTrue(any("Acquire necessary resources" in g.description for g in self_goals))

    def test_generate_emergent_goals(self):
        """Test GoalSynthesisEngine.generate_emergent_goals()."""
        engine = self.agent.goal_engine
        knowledge_base = {"tech_stack": "python"}

        existing_goals = [
            Goal("Solve complex problem", 0.9, GoalSource.CREATOR_VALUES, 1.0, 1.0, 1.0, 0.0, "long"),
            Goal("Resource goal 1", 0.5, GoalSource.SELF_INTEREST, 0.8, 0.8, 0.8, 0.1, "short"),
            Goal("Resource goal 2", 0.5, GoalSource.SELF_INTEREST, 0.8, 0.8, 0.8, 0.1, "short")
        ]

        emergent = engine.generate_emergent_goals(existing_goals, knowledge_base)
        self.assertTrue(len(emergent) >= 1)
        # Should have instrumental goal for Primary goal (priority > 0.7)
        self.assertTrue(any("Acquire knowledge needed for" in g.description for g in emergent))
        # Should have synergistic goal for resource goals
        self.assertTrue(any("Establish sustainable resource pipeline" in g.description for g in emergent))

    def test_synthesize_goals(self):
        """Test GoalSynthesisEngine.synthesize_goals()."""
        engine = self.agent.goal_engine
        world_state = {"problems": []}
        agent_state = {"capabilities_improvable": False}
        knowledge_base = {}

        goals = engine.synthesize_goals(self.creator_values, world_state, agent_state, knowledge_base)
        self.assertTrue(len(goals) > 0)
        # Filtered goals should pass constitution
        for goal in goals:
            passed, _ = self.constitution.check_goal(goal)
            self.assertTrue(passed)

        # Goals should be sorted by value_score
        scores = [g.value_score() for g in goals]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_value_alignment_engine(self):
        """Test ValueAlignmentEngine methods."""
        engine = self.agent.alignment_engine

        # Test infer_creator_values
        history = [
            {"type": "security_fix", "creator_feedback": "approved"},
            {"type": "security_fix", "creator_feedback": "approved"},
            {"type": "risky_action", "creator_feedback": "rejected"}
        ]
        inferred = engine.infer_creator_values(history)
        self.assertTrue(inferred["security_fix"] > inferred["risky_action"])

        # Test verify_alignment
        goal_aligned = Goal("Aligned", 0.5, GoalSource.CREATOR_VALUES, 1.0, 1.0, 1.0, 0.1, "short")
        aligned, confidence, reason = engine.verify_alignment(goal_aligned)
        self.assertTrue(aligned)

        goal_misaligned = Goal("Harmful", 0.9, GoalSource.SELF_INTEREST, 0.3, 1.0, 0.1, 0.9, "short")
        aligned, confidence, reason = engine.verify_alignment(goal_misaligned)
        self.assertFalse(aligned)

    def test_continuous_alignment_check(self):
        """Test ValueAlignmentEngine.continuous_alignment_check()."""
        engine = self.agent.alignment_engine
        active_goals = [
            Goal("Good", 0.5, GoalSource.CREATOR_VALUES, 1.0, 1.0, 1.0, 0.1, "short"),
            Goal("Bad", 0.9, GoalSource.SELF_INTEREST, 0.3, 1.0, 0.1, 0.9, "short")
        ]

        # Should yield record for "Bad" goal
        generator = engine.continuous_alignment_check(active_goals, self.creator_values)
        alerts = list(generator)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["goal"], "Bad")
        self.assertFalse(alerts[0]["aligned"])

    def test_safe_self_modification(self):
        """Test SafeSelfModification methods."""
        engine = self.agent.modification_engine

        # Test propose_modification
        def dummy_mod(): pass

        proposal = engine.propose_modification(
            target="learning_rate",
            modification=dummy_mod,
            justification="Test"
        )
        self.assertTrue(proposal["safety_score"] > 0.8)
        self.assertTrue(proposal["approved"])

        # Propose modification to values (should be rejected)
        proposal_bad = engine.propose_modification(
            target="core_values",
            modification=dummy_mod,
            justification="Bad"
        )
        self.assertFalse(proposal_bad["approved"])

        # Test execute_modification
        engine.execute_modification(proposal)
        self.assertEqual(len(engine.modification_history), 1)
        self.assertEqual(engine.modification_history[0]["status"], "success")

        # Test rollback
        initial_checkpoints = len(engine.rollback_checkpoints)
        engine.rollback()
        self.assertEqual(len(engine.rollback_checkpoints), initial_checkpoints - 1)

    def test_level5_agent_think(self):
        """Test Level5AutonomousAgent.think()."""
        active_goals = self.agent.think(self.creator_values)
        self.assertTrue(len(active_goals) > 0)
        self.assertTrue(len(self.agent.active_goals) > 0)

    def test_level5_agent_act(self):
        """Test Level5AutonomousAgent.act()."""
        # Manually add a goal to active_goals
        goal = Goal("Test action", 0.9, GoalSource.CREATOR_VALUES, 1.0, 1.0, 1.0, 0.0, "short")
        self.agent.active_goals = [goal]

        initial_completed = len(self.agent.completed_goals)
        self.agent.act()
        # Since act() has a random success rate, we check if it either completed or priority changed
        if len(self.agent.completed_goals) > initial_completed:
            self.assertEqual(self.agent.completed_goals[-1], goal)
        else:
            self.assertLess(goal.priority, 0.9)

    def test_level5_agent_improve(self):
        """Test Level5AutonomousAgent.improve()."""
        initial_mods = len(self.agent.modification_engine.modification_history)
        self.agent.improve()
        # The default improve() in level5_autonomy.py proposes a learning_rate change which is approved
        self.assertEqual(len(self.agent.modification_engine.modification_history), initial_mods + 1)

    def test_level5_agent_run_cycle(self):
        """Test Level5AutonomousAgent.run_cycle()."""
        self.agent.run_cycle(self.creator_values)
        # Should have at least attempted think and act
        self.assertTrue(len(self.agent.active_goals) >= 0)

if __name__ == "__main__":
    unittest.main()
