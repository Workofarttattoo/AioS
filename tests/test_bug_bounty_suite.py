import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_llm_brain import LocalLLMBrain
from bug_bounty_scanner import BugBountyScanner, ScopeEnforcer
from apex_hunter import ApexHunter

class TestBugBountySuite(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_scope_enforcement(self):
        enforcer = ScopeEnforcer(allowed_targets=["example.com", "*.test.com"], blocked_targets=["evil.com"])

        self.assertTrue(enforcer.is_allowed("example.com"))
        self.assertTrue(enforcer.is_allowed("sub.test.com"))
        self.assertFalse(enforcer.is_allowed("evil.com"))
        self.assertFalse(enforcer.is_allowed("other.com"))
        self.assertTrue(enforcer.is_allowed("127.0.0.1")) # Default allowed for now

    @patch('ollama.generate')
    def test_llm_brain_reasoning(self, mock_generate):
        mock_generate.return_value = {'response': '{"key": "value"}'}
        brain = LocalLLMBrain()

        result = self.loop.run_until_complete(brain.reason("test", json_mode=True))
        self.assertEqual(result.get("key"), "value")

    @patch('asyncio.create_subprocess_shell')
    def test_scanner_tool_execution(self, mock_shell):
        # Mocking subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"error")
        mock_process.returncode = 0
        mock_shell.return_value = mock_process

        scanner = BugBountyScanner()
        result = self.loop.run_until_complete(scanner._run_tool("ls"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stdout"], "output")

    @patch('local_llm_brain.LocalLLMBrain.reason', new_callable=AsyncMock)
    @patch('bug_bounty_scanner.BugBountyScanner.run_nmap', new_callable=AsyncMock)
    def test_apex_hunter_workflow(self, mock_nmap, mock_reason):
        mock_reason.side_effect = [
            {"tools": ["nmap"], "objectives": ["scan"]}, # Strategy
            {"vulnerability": "none", "severity": "LOW"} # Final Analysis
        ]
        mock_nmap.return_value = {"status": "success", "stdout": "nmap output"}

        hunter = ApexHunter(allowed_targets=["target.com"])
        # Mock analyze_target
        hunter.brain.analyze_target = AsyncMock(return_value={"target": "target.com"})

        result = self.loop.run_until_complete(hunter.hunt("target.com"))

        self.assertEqual(result["target"], "target.com")
        self.assertTrue(any(f["tool"] == "nmap" for f in result["findings"]))
        self.assertEqual(result["report"]["severity"], "LOW")

if __name__ == '__main__':
    unittest.main()
