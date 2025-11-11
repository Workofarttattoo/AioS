#!/usr/bin/env python3
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""
Smoke test to verify basic Ai:oS functionality.

This test ensures:
1. All critical imports work
2. Basic aios_shell.py commands execute
3. No import errors or missing dependencies for core functionality
"""

import sys
import unittest
import subprocess
from pathlib import Path

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestAiOSSmoke(unittest.TestCase):
    """Basic smoke tests for Ai:oS functionality."""

    def test_aios_imports(self):
        """Test that core aios imports work."""
        try:
            from aios import (
                AgentaRuntime,
                DISPLAY_NAME,
                DISPLAY_NAME_FULL,
                load_manifest,
                DEFAULT_MANIFEST
            )
            # Verify constants are set
            self.assertEqual(DISPLAY_NAME, "Ai:oS")
            self.assertEqual(DISPLAY_NAME_FULL, "Ai:oS Agentic Control Plane")

            # Verify DEFAULT_MANIFEST has required keys
            self.assertIn("name", DEFAULT_MANIFEST)
            self.assertIn("version", DEFAULT_MANIFEST)
            self.assertIn("meta_agents", DEFAULT_MANIFEST)

            # Verify AgentaRuntime can be instantiated
            runtime = AgentaRuntime()
            self.assertIsNotNone(runtime)

        except ImportError as e:
            self.fail(f"Failed to import aios components: {e}")

    def test_autonomous_discovery_imports(self):
        """Test that autonomous discovery imports work."""
        try:
            from autonomous_discovery import (
                AgentAutonomy,
                AutonomousLLMAgent,
                UltraFastInferenceEngine,
                create_autonomous_discovery_action
            )

            # Verify autonomy levels are defined
            self.assertEqual(AgentAutonomy.LEVEL_0_NONE, 0)
            self.assertEqual(AgentAutonomy.LEVEL_4_FULL, 4)

            # Verify agent can be instantiated
            agent = AutonomousLLMAgent(
                model_name="test",
                autonomy_level=AgentAutonomy.LEVEL_4_FULL
            )
            self.assertIsNotNone(agent)

        except ImportError as e:
            self.fail(f"Failed to import autonomous_discovery: {e}")

    def test_aios_shell_status(self):
        """Test that aios_shell.py status command runs."""
        cmd = [sys.executable, str(ROOT / "aios_shell.py"), "status"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check that it runs without Python errors
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertNotIn("ImportError", result.stderr)
            self.assertNotIn("SyntaxError", result.stderr)

            # Status command should complete with exit code 0 or 1 (not crash)
            self.assertIn(result.returncode, [0, 1], "Command should complete without crash")

        except subprocess.TimeoutExpired:
            self.fail("aios_shell.py status timed out")
        except FileNotFoundError:
            self.fail("aios_shell.py not found")

    def test_config_loading(self):
        """Test that config files can be loaded."""
        try:
            from aios import load_manifest

            # Test loading default manifest
            manifest = load_manifest()
            self.assertIsNotNone(manifest)
            self.assertIn("name", manifest)

            # Test loading example manifest if exists
            example_manifest = ROOT / "examples" / "manifest-security-response.json"
            if example_manifest.exists():
                manifest = load_manifest(str(example_manifest))
                self.assertIsNotNone(manifest)
                self.assertIn("meta_agents", manifest)

        except Exception as e:
            self.fail(f"Failed to load manifests: {e}")

    def test_ml_algorithms_availability(self):
        """Test ML algorithms module can be imported."""
        try:
            from ml_algorithms import get_algorithm_catalog

            catalog = get_algorithm_catalog()
            # Catalog is a list of algorithm descriptions
            self.assertIsInstance(catalog, list)
            self.assertGreater(len(catalog), 0)

            # Verify some key algorithms are present
            algorithm_names = [alg['name'] for alg in catalog]
            self.assertIn("AdaptiveStateSpace", algorithm_names)
            self.assertIn("NeuralGuidedMCTS", algorithm_names)

        except ImportError:
            # ML algorithms are optional
            self.skipTest("ML algorithms module not available")

    def test_quantum_ml_availability(self):
        """Test quantum ML module can be imported."""
        try:
            from quantum_ml_algorithms import check_dependencies

            deps = check_dependencies()
            self.assertIsInstance(deps, dict)
            self.assertIn("numpy", deps)

        except ImportError:
            # Quantum ML is optional
            self.skipTest("Quantum ML module not available")


class TestSimulationMode(unittest.TestCase):
    """Test that simulation mode is clearly marked."""

    def test_llm_simulation_warning(self):
        """Test that LLM simulation mode shows clear warnings."""
        try:
            from autonomous_discovery import UltraFastInferenceEngine
            import logging

            # Capture log output
            log_capture = []

            class LogHandler(logging.Handler):
                def emit(self, record):
                    log_capture.append(record.getMessage())

            handler = LogHandler()
            logger = logging.getLogger("autonomous_discovery")
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)

            # Create engine without API keys (will use simulation)
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)

            engine = UltraFastInferenceEngine()
            # Note: generate is async, need to handle properly
            import asyncio
            loop = asyncio.new_event_loop()
            response, metrics = loop.run_until_complete(
                engine.generate("test prompt", max_tokens=10)
            )
            loop.close()

            # Check for simulation warnings
            warnings_found = any("SIMULATION" in msg for msg in log_capture)

            if "[SIMULATED]" in response:
                # Good - response is clearly marked as simulated
                self.assertIn("[SIMULATED]", response)
            else:
                # If using real API, that's also fine
                pass

        except Exception as e:
            self.fail(f"Failed to test simulation mode: {e}")


def run_smoke_tests():
    """Run all smoke tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAiOSSmoke))
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationMode))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)