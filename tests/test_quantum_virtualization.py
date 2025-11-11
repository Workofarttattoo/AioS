"""
Tests for Quantum-backed Virtualization Layer integration.
"""

import unittest
import sys
from pathlib import Path

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aios.runtime import ProductionRuntime, ActionResult  # type: ignore


class TestQuantumVirtualization(unittest.TestCase):
    def setUp(self):
        self.runtime = ProductionRuntime(
            environment={
                "AGENTA_VIRTUALIZATION_BACKEND": "quantum",
                "AGENTA_QUANTUM_QUBITS": "30",
            }
        )

    def test_virtualization_inspect_provisions_domain(self):
        """Inspect should report quantum backend and provision an OS domain."""
        result = self.runtime.execute_action("scalability.virtualization_inspect", retry=False)
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        payload = result.payload

        self.assertEqual(payload.get("backend"), "quantum")
        self.assertEqual(int(payload.get("qubits", 0)), 30)
        self.assertIn("domain", payload)
        self.assertIn("domains", payload)
        self.assertGreaterEqual(len(payload.get("domains", [])), 1)

        domain = payload["domain"]
        self.assertEqual(domain.get("name"), "aios-os")
        self.assertEqual(int(domain.get("qubits", 0)), 30)

    def test_virtualization_domains_lists_names(self):
        """Domains action should return list of domain names."""
        # Ensure a domain exists via inspect
        self.runtime.execute_action("scalability.virtualization_inspect", retry=False)
        # List domains
        result = self.runtime.execute_action("scalability.virtualization_domains", retry=False)
        self.assertTrue(result.success)
        payload = result.payload
        names = payload.get("domains", [])
        self.assertIsInstance(names, list)
        self.assertIn("aios-os", names)


if __name__ == "__main__":
    unittest.main()


