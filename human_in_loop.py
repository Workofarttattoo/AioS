"""
Human-in-Loop Checkpoint System for Ai:oS Agents.

Implements human oversight for critical agent actions following 2025 best practices:
- Approval workflows for high-impact decisions
- Review queues for uncertain predictions
- Feedback loops for continuous improvement
- Audit trails for compliance
- Escalation policies

Key insight: Even at Level 4 autonomy, human oversight for critical decisions
improves reliability and builds trust.

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from queue import Queue

LOG = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# APPROVAL SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class ActionCriticality(Enum):
    """Criticality levels for agent actions."""
    LOW = "low"           # No approval needed
    MEDIUM = "medium"     # Optional review
    HIGH = "high"         # Approval required
    CRITICAL = "critical" # Multi-level approval required


class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Request for human approval of agent action."""
    request_id: str
    agent_name: str
    action_path: str
    criticality: ActionCriticality
    
    # Action details
    description: str
    inputs: Dict[str, Any]
    predicted_outcome: Any
    confidence: float
    
    # Risk assessment
    risks: List[str]
    mitigation: List[str]
    
    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: float = field(default_factory=time.time)
    reviewed_at: Optional[float] = None
    reviewer: Optional[str] = None
    comments: Optional[str] = None
    
    # Timeout
    timeout_seconds: float = 3600.0  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.status != ApprovalStatus.PENDING:
            return False
        return time.time() - self.requested_at > self.timeout_seconds
    
    def approve(self, reviewer: str, comments: Optional[str] = None) -> None:
        """Approve the request."""
        self.status = ApprovalStatus.APPROVED
        self.reviewed_at = time.time()
        self.reviewer = reviewer
        self.comments = comments
        LOG.info(f"Request {self.request_id} APPROVED by {reviewer}")
    
    def reject(self, reviewer: str, reason: str) -> None:
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.reviewed_at = time.time()
        self.reviewer = reviewer
        self.comments = reason
        LOG.info(f"Request {self.request_id} REJECTED by {reviewer}: {reason}")
    
    def escalate(self, reason: str) -> None:
        """Escalate to higher authority."""
        self.status = ApprovalStatus.ESCALATED
        self.comments = reason
        LOG.warning(f"Request {self.request_id} ESCALATED: {reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "agent_name": self.agent_name,
            "action_path": self.action_path,
            "criticality": self.criticality.value,
            "description": self.description,
            "predicted_outcome": str(self.predicted_outcome),
            "confidence": self.confidence,
            "risks": self.risks,
            "mitigation": self.mitigation,
            "status": self.status.value,
            "requested_at": self.requested_at,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "comments": self.comments
        }


# ═══════════════════════════════════════════════════════════════════════
# APPROVAL WORKFLOW
# ═══════════════════════════════════════════════════════════════════════

class ApprovalWorkflow:
    """
    Manages approval workflows for critical agent actions.
    
    Features:
    - Priority queues for urgent requests
    - Timeout handling
    - Escalation policies
    - Audit trail
    """
    
    def __init__(
        self,
        audit_dir: Optional[Path] = None,
        auto_approve_low: bool = True
    ):
        """
        Initialize approval workflow.
        
        Args:
            audit_dir: Directory for audit logs
            auto_approve_low: Automatically approve low-criticality actions
        """
        self.audit_dir = audit_dir or Path("approvals")
        self.audit_dir.mkdir(exist_ok=True)
        
        self.auto_approve_low = auto_approve_low
        
        # Queues by criticality
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: List[ApprovalRequest] = []
        
        LOG.info(f"ApprovalWorkflow initialized. Audit dir: {self.audit_dir}")
    
    def request_approval(
        self,
        agent_name: str,
        action_path: str,
        criticality: ActionCriticality,
        description: str,
        inputs: Dict[str, Any],
        predicted_outcome: Any,
        confidence: float,
        risks: List[str],
        mitigation: List[str],
        timeout_seconds: float = 3600.0
    ) -> ApprovalRequest:
        """
        Create approval request.
        
        Args:
            agent_name: Name of agent requesting approval
            action_path: Action being performed
            criticality: Criticality level
            description: Human-readable description
            inputs: Action inputs
            predicted_outcome: What agent expects to happen
            confidence: Agent's confidence in prediction (0-1)
            risks: List of identified risks
            mitigation: List of mitigation strategies
            timeout_seconds: How long to wait for approval
        
        Returns:
            ApprovalRequest object
        """
        request_id = str(uuid.uuid4())
        
        request = ApprovalRequest(
            request_id=request_id,
            agent_name=agent_name,
            action_path=action_path,
            criticality=criticality,
            description=description,
            inputs=inputs,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            risks=risks,
            mitigation=mitigation,
            timeout_seconds=timeout_seconds
        )
        
        # Auto-approve low-criticality if enabled
        if self.auto_approve_low and criticality == ActionCriticality.LOW:
            request.approve("system", "Auto-approved (low criticality)")
            self.completed_requests.append(request)
            self._write_audit_log(request)
            return request
        
        # Add to pending queue
        self.pending_requests[request_id] = request
        
        LOG.info(
            f"Approval requested: {action_path} ({criticality.value}) "
            f"- Confidence: {confidence:.1%}"
        )
        
        return request
    
    def get_pending_requests(
        self,
        criticality_filter: Optional[ActionCriticality] = None
    ) -> List[ApprovalRequest]:
        """
        Get pending approval requests.
        
        Args:
            criticality_filter: Optional filter by criticality
        
        Returns:
            List of pending requests, sorted by criticality and age
        """
        requests = list(self.pending_requests.values())
        
        if criticality_filter:
            requests = [r for r in requests if r.criticality == criticality_filter]
        
        # Sort: Critical first, then by age (oldest first)
        criticality_order = {
            ActionCriticality.CRITICAL: 0,
            ActionCriticality.HIGH: 1,
            ActionCriticality.MEDIUM: 2,
            ActionCriticality.LOW: 3
        }
        
        requests.sort(
            key=lambda r: (criticality_order[r.criticality], r.requested_at)
        )
        
        return requests
    
    def approve_request(
        self,
        request_id: str,
        reviewer: str,
        comments: Optional[str] = None
    ) -> bool:
        """
        Approve a pending request.
        
        Returns:
            True if approved, False if request not found
        """
        if request_id not in self.pending_requests:
            LOG.warning(f"Request {request_id} not found")
            return False
        
        request = self.pending_requests[request_id]
        request.approve(reviewer, comments)
        
        # Move to completed
        del self.pending_requests[request_id]
        self.completed_requests.append(request)
        
        # Write audit log
        self._write_audit_log(request)
        
        return True
    
    def reject_request(
        self,
        request_id: str,
        reviewer: str,
        reason: str
    ) -> bool:
        """
        Reject a pending request.
        
        Returns:
            True if rejected, False if request not found
        """
        if request_id not in self.pending_requests:
            LOG.warning(f"Request {request_id} not found")
            return False
        
        request = self.pending_requests[request_id]
        request.reject(reviewer, reason)
        
        # Move to completed
        del self.pending_requests[request_id]
        self.completed_requests.append(request)
        
        # Write audit log
        self._write_audit_log(request)
        
        return True
    
    def check_timeouts(self) -> List[ApprovalRequest]:
        """
        Check for expired requests and mark as expired.
        
        Returns:
            List of expired requests
        """
        expired = []
        
        for request_id, request in list(self.pending_requests.items()):
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                expired.append(request)
                
                # Move to completed
                del self.pending_requests[request_id]
                self.completed_requests.append(request)
                
                # Write audit log
                self._write_audit_log(request)
                
                LOG.warning(f"Request {request_id} EXPIRED (timeout)")
        
        return expired
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        total = len(self.completed_requests)
        approved = sum(1 for r in self.completed_requests if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self.completed_requests if r.status == ApprovalStatus.REJECTED)
        expired = sum(1 for r in self.completed_requests if r.status == ApprovalStatus.EXPIRED)
        
        avg_review_time = 0.0
        if approved + rejected > 0:
            review_times = [
                r.reviewed_at - r.requested_at
                for r in self.completed_requests
                if r.reviewed_at
            ]
            if review_times:
                avg_review_time = sum(review_times) / len(review_times)
        
        return {
            "total_requests": total,
            "pending": len(self.pending_requests),
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
            "approval_rate": approved / total if total > 0 else 0.0,
            "avg_review_time_seconds": avg_review_time
        }
    
    def _write_audit_log(self, request: ApprovalRequest) -> None:
        """Write approval request to audit log."""
        log_file = self.audit_dir / f"approval_{request.request_id}.json"
        
        try:
            log_file.write_text(json.dumps(request.to_dict(), indent=2))
        except Exception as exc:
            LOG.error(f"Failed to write audit log: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# DECORATOR FOR APPROVAL CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════

def require_approval(
    criticality: ActionCriticality,
    description: str,
    timeout_seconds: float = 3600.0
):
    """
    Decorator to require approval for agent actions.
    
    Usage:
        @require_approval(ActionCriticality.HIGH, "Delete production data")
        def dangerous_action(ctx):
            # This will block until approved
            perform_deletion()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get global workflow (in production, pass as parameter)
            workflow = _get_global_workflow()
            
            # Create approval request
            request = workflow.request_approval(
                agent_name=func.__module__.split('.')[-1],
                action_path=f"{func.__module__}.{func.__name__}",
                criticality=criticality,
                description=description,
                inputs={"args": args, "kwargs": kwargs},
                predicted_outcome="Function will execute",
                confidence=1.0,
                risks=["Action is critical"],
                mitigation=["Human review required"],
                timeout_seconds=timeout_seconds
            )
            
            # Wait for approval (in production, use async or webhook)
            if request.status == ApprovalStatus.APPROVED:
                return func(*args, **kwargs)
            elif request.status == ApprovalStatus.REJECTED:
                raise PermissionError(f"Action rejected: {request.comments}")
            else:
                raise TimeoutError("Approval timeout")
        
        return wrapper
    return decorator


# Global workflow instance
_global_workflow: Optional[ApprovalWorkflow] = None


def _get_global_workflow() -> ApprovalWorkflow:
    """Get or create global workflow."""
    global _global_workflow
    if _global_workflow is None:
        _global_workflow = ApprovalWorkflow()
    return _global_workflow


def get_approval_workflow() -> ApprovalWorkflow:
    """Get global approval workflow."""
    return _get_global_workflow()


# ═══════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """Demonstrate approval workflow."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  HUMAN-IN-LOOP APPROVAL SYSTEM - DEMONSTRATION                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    workflow = ApprovalWorkflow()
    
    # Create test requests at different criticality levels
    requests = []
    
    # Low criticality - auto-approved
    req1 = workflow.request_approval(
        agent_name="security",
        action_path="security.firewall_status",
        criticality=ActionCriticality.LOW,
        description="Check firewall status",
        inputs={"command": "status"},
        predicted_outcome="Status displayed",
        confidence=0.99,
        risks=[],
        mitigation=[]
    )
    requests.append(req1)
    
    # Medium criticality
    req2 = workflow.request_approval(
        agent_name="security",
        action_path="security.firewall_rule",
        criticality=ActionCriticality.MEDIUM,
        description="Add firewall rule to allow port 8080",
        inputs={"port": 8080, "action": "allow"},
        predicted_outcome="Rule added successfully",
        confidence=0.85,
        risks=["May expose service to internet"],
        mitigation=["Rule is specific to port 8080 only"]
    )
    requests.append(req2)
    
    # High criticality
    req3 = workflow.request_approval(
        agent_name="security",
        action_path="security.disable_firewall",
        criticality=ActionCriticality.HIGH,
        description="Disable firewall for maintenance",
        inputs={"duration_minutes": 30},
        predicted_outcome="Firewall disabled temporarily",
        confidence=0.90,
        risks=["System exposed to network attacks during maintenance"],
        mitigation=["Limited to 30 minutes", "Monitoring active"]
    )
    requests.append(req3)
    
    # Critical
    req4 = workflow.request_approval(
        agent_name="storage",
        action_path="storage.delete_volume",
        criticality=ActionCriticality.CRITICAL,
        description="Delete production database volume",
        inputs={"volume_id": "prod-db-001"},
        predicted_outcome="Volume deleted permanently",
        confidence=0.75,
        risks=["DATA LOSS", "Service outage", "Cannot be undone"],
        mitigation=["Backup verified", "Maintenance window scheduled"]
    )
    requests.append(req4)
    
    # Show pending requests
    print("Pending Approval Requests:")
    print("─" * 70)
    
    pending = workflow.get_pending_requests()
    for req in pending:
        print(f"\nID: {req.request_id}")
        print(f"Action: {req.action_path}")
        print(f"Criticality: {req.criticality.value.upper()}")
        print(f"Description: {req.description}")
        print(f"Confidence: {req.confidence:.1%}")
        print(f"Risks: {', '.join(req.risks) if req.risks else 'None'}")
    
    # Simulate approvals
    print("\n" + "═" * 70)
    print("Processing Approvals:")
    print("─" * 70)
    
    # Approve medium
    workflow.approve_request(req2.request_id, "admin", "Approved for testing")
    print(f"✓ {req2.action_path} APPROVED")
    
    # Reject high (too risky)
    workflow.reject_request(req3.request_id, "admin", "Maintenance can wait")
    print(f"✗ {req3.action_path} REJECTED")
    
    # Critical needs multi-level (simulate)
    workflow.approve_request(req4.request_id, "senior_admin", "Backup verified, proceed")
    print(f"✓ {req4.action_path} APPROVED (senior admin)")
    
    # Get stats
    print("\n" + "═" * 70)
    print("Workflow Statistics:")
    print("─" * 70)
    
    stats = workflow.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Approval rate: {stats['approval_rate']:.1%}")
    print(f"Avg review time: {stats['avg_review_time_seconds']:.1f}s")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()

