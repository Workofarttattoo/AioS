
import time
import os
import shutil
from pathlib import Path
import json
from openagi_approval_workflow import ApprovalStore, ApprovalRequest, ActionSensitivity, ApprovalRequirement, ApprovalStatus

def benchmark_get_request():
    test_dir = Path("benchmark_storage")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    store = ApprovalStore(storage_path=test_dir)

    num_requests = 1000
    request_ids = []

    print(f"Populating {num_requests} requests...")
    for i in range(num_requests):
        req = ApprovalRequest(
            action_path=f"test.action.{i}",
            action_name=f"Test Action {i}",
            description=f"Description {i}",
            requirement=ApprovalRequirement(ActionSensitivity.MEDIUM)
        )
        store.save_request(req)
        request_ids.append(req.request_id)

        # Add some duplicate entries to simulate updates
        if i % 10 == 0:
            store.save_request(req)

    # Measure get_request
    num_queries = 500
    start_time = time.time()
    for i in range(num_queries):
        rid = request_ids[i % num_requests]
        store.get_request(rid)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time taken for {num_queries} queries: {duration:.4f} seconds")
    print(f"Average time per query: {duration/num_queries:.6f} seconds")

    if test_dir.exists():
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    benchmark_get_request()
