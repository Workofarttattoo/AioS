import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_authorization import (
    AgentAuthorizationManager,
    ApprovalStatus,
    ApprovalDecision
)

class TestAgentAuthorizationStatus(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.auth_dir_patcher = patch('agent_authorization.AGENT_AUTHORIZATION_DIR', Path(self.temp_dir.name))
        self.auth_dir_patcher.start()
        # We need to ensure the directory exists because the patcher might be too late if it's already used in __init__
        Path(self.temp_dir.name).mkdir(parents=True, exist_ok=True)
        self.manager = AgentAuthorizationManager()

    def tearDown(self):
        self.auth_dir_patcher.stop()
        self.temp_dir.cleanup()

    def test_get_action_status_missing_id(self):
        """Test that get_action_status returns None for an unknown action_id."""
        status = self.manager.get_action_status("non_existent_id")
        self.assertIsNone(status)

    def test_get_action_status_approved(self):
        """Test that get_action_status returns 'approved'."""
        action_id = "test_approved"
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.APPROVED,
            admin_user="admin"
        )
        self.manager.decisions[action_id] = decision
        status = self.manager.get_action_status(action_id)
        self.assertEqual(status, "approved")

    def test_get_action_status_rejected(self):
        """Test that get_action_status returns 'rejected'."""
        action_id = "test_rejected"
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.REJECTED,
            admin_user="admin"
        )
        self.manager.decisions[action_id] = decision
        status = self.manager.get_action_status(action_id)
        self.assertEqual(status, "rejected")

    def test_get_action_status_revoked(self):
        """Test that get_action_status returns 'revoked'."""
        action_id = "test_revoked"
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.REVOKED,
            admin_user="admin"
        )
        self.manager.decisions[action_id] = decision
        status = self.manager.get_action_status(action_id)
        self.assertEqual(status, "revoked")

    def test_get_action_status_expired(self):
        """Test that get_action_status returns 'expired'."""
        action_id = "test_expired"
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.EXPIRED,
            admin_user="admin"
        )
        self.manager.decisions[action_id] = decision
        status = self.manager.get_action_status(action_id)
        self.assertEqual(status, "expired")

    def test_get_action_status_pending(self):
        """Test that get_action_status returns 'pending' if it's in decisions."""
        action_id = "test_pending"
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.PENDING,
            admin_user="admin"
        )
        self.manager.decisions[action_id] = decision
        status = self.manager.get_action_status(action_id)
        self.assertEqual(status, "pending")

if __name__ == '__main__':
    unittest.main()
