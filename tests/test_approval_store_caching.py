
import unittest
import json
import shutil
import time
from pathlib import Path
from openagi_approval_workflow import (
    ApprovalStore, ApprovalRequest, ApprovalDecision,
    ApprovalRequirement, ActionSensitivity, ApprovalStatus
)

class TestApprovalStoreCaching(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_caching")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        self.store = ApprovalStore(storage_path=self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_and_get_request(self):
        req = ApprovalRequest(
            action_path="test.action",
            action_name="Test Action",
            description="Test Description",
            requirement=ApprovalRequirement(ActionSensitivity.MEDIUM)
        )
        self.store.save_request(req)

        # Should be in cache
        cached_req = self.store.get_request(req.request_id)
        self.assertEqual(cached_req.request_id, req.request_id)
        self.assertEqual(cached_req.action_path, "test.action")

        # Verify it's the same object (or at least equal)
        self.assertEqual(cached_req, req)

    def test_cache_initialization(self):
        # Manually write to file
        req = ApprovalRequest(request_id="manual_id", action_path="manual.path")
        with open(self.store.requests_file, 'a') as f:
            f.write(json.dumps(req.to_dict()) + '\n')

        # New store instance should load it
        new_store = ApprovalStore(storage_path=self.test_dir)
        loaded_req = new_store.get_request("manual_id")
        self.assertIsNotNone(loaded_req)
        self.assertEqual(loaded_req.request_id, "manual_id")
        self.assertEqual(loaded_req.action_path, "manual.path")

    def test_deduplication_latest_wins(self):
        req_id = "same_id"
        req1 = ApprovalRequest(request_id=req_id, action_path="path1", status=ApprovalStatus.PENDING)
        req2 = ApprovalRequest(request_id=req_id, action_path="path2", status=ApprovalStatus.APPROVED)

        self.store.save_request(req1)
        self.store.save_request(req2)

        loaded_req = self.store.get_request(req_id)
        self.assertEqual(loaded_req.action_path, "path2")
        self.assertEqual(loaded_req.status, ApprovalStatus.APPROVED)

    def test_get_pending_requests(self):
        req1 = ApprovalRequest(request_id="id1", status=ApprovalStatus.PENDING)
        req2 = ApprovalRequest(request_id="id2", status=ApprovalStatus.APPROVED)
        req3 = ApprovalRequest(request_id="id3", status=ApprovalStatus.PENDING)

        self.store.save_request(req1)
        self.store.save_request(req2)
        self.store.save_request(req3)

        pending = self.store.get_pending_requests()
        self.assertEqual(len(pending), 2)
        ids = [r.request_id for r in pending]
        self.assertIn("id1", ids)
        self.assertIn("id3", ids)
        self.assertNotIn("id2", ids)

    def test_get_decision(self):
        decision = ApprovalDecision(
            request_id="req123",
            approver_id="user1",
            approved=True,
            reason="Verified"
        )
        self.store.save_decision(decision)

        loaded_decision = self.store.get_decision("req123")
        self.assertIsNotNone(loaded_decision)
        self.assertEqual(loaded_decision.approver_id, "user1")
        self.assertTrue(loaded_decision.approved)

if __name__ == "__main__":
    unittest.main()
