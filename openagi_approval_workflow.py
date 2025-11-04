"""
OpenAGI Approval Workflow System

Implements human-in-loop approval workflows for sensitive actions with comprehensive
audit trails, decision tracking, and integration with AIOS manifest system.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from uuid import uuid4

try:
    from aios.runtime import ExecutionContext, ActionResult
except Exception:
    ExecutionContext = Any
    ActionResult = Any

LOG = logging.getLogger(__name__)


class ActionSensitivity(Enum):
    """Sensitivity level for actions requiring approval."""
    CRITICAL = "critical"  # System-level mutations (firewall, kernels, boot)
    HIGH = "high"  # Data deletion, access control changes
    MEDIUM = "medium"  # Configuration changes affecting multiple services
    LOW = "low"  # No approval needed
    NONE = "none"  # Auto-approved


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"  # Waiting for approval
    APPROVED = "approved"  # Approved by authorized user
    DENIED = "denied"  # Explicitly denied
    EXPIRED = "expired"  # Request timed out
    REVOKED = "revoked"  # Approval was revoked


@dataclass
class ApprovalRequirement:
    """
    Defines what makes an action sensitive and requires approval.

    Attributes:
        sensitivity_level: How sensitive is this action
        requires_reason: Whether approver must provide reasoning
        requires_two_factor: Whether 2FA is needed for approval
        auto_approve_if: Optional function to auto-approve
        expires_in_seconds: How long until request expires (0 = no expiry)
        affected_systems: List of systems this action affects
    """
    sensitivity_level: ActionSensitivity
    requires_reason: bool = True
    requires_two_factor: bool = False
    auto_approve_if: Optional[Callable[[Dict[str, Any]], bool]] = None
    expires_in_seconds: int = 3600  # 1 hour
    affected_systems: List[str] = field(default_factory=list)

    def should_auto_approve(self, action_context: Dict[str, Any]) -> bool:
        """Check if action should auto-approve based on context."""
        if self.auto_approve_if is None:
            return False
        try:
            return self.auto_approve_if(action_context)
        except Exception as e:
            LOG.error(f"Error in auto-approve check: {e}")
            return False


@dataclass
class ApprovalRequest:
    """
    Request for approval of a sensitive action.

    Attributes:
        request_id: Unique identifier for this approval request
        action_path: Path to the action (e.g., "security.firewall")
        action_name: Human-readable action name
        description: Description of what the action will do
        requirement: ApprovalRequirement defining sensitivity
        context: Additional context data for the action
        requester_id: User who requested the action
        requested_at: Timestamp when requested
        expires_at: When this request expires
        status: Current approval status
        created_at: When this request was created
    """
    request_id: str = field(default_factory=lambda: str(uuid4()))
    action_path: str = ""
    action_name: str = ""
    description: str = ""
    requirement: ApprovalRequirement = field(default_factory=lambda: ApprovalRequirement(ActionSensitivity.MEDIUM))
    context: Dict[str, Any] = field(default_factory=dict)
    requester_id: str = ""
    requested_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=time.time)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Calculate expiry time."""
        if self.requirement.expires_in_seconds > 0:
            self.expires_at = self.requested_at + self.requirement.expires_in_seconds

    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.requirement.expires_in_seconds <= 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['requirement'] = {
            'sensitivity_level': self.requirement.sensitivity_level.value,
            'requires_reason': self.requirement.requires_reason,
            'requires_two_factor': self.requirement.requires_two_factor,
            'expires_in_seconds': self.requirement.expires_in_seconds,
            'affected_systems': self.requirement.affected_systems,
        }
        data['status'] = self.status.value
        return data


@dataclass
class ApprovalDecision:
    """
    Decision on an approval request.

    Attributes:
        request_id: ID of the request being decided
        approver_id: User who made the decision
        approved: Whether the action was approved
        reason: Reason for approval/denial
        two_factor_verified: Whether 2FA was completed
        decided_at: Timestamp of decision
        execution_allowed_until: When approval expires (if approved)
    """
    request_id: str = ""
    approver_id: str = ""
    approved: bool = False
    reason: str = ""
    two_factor_verified: bool = False
    decided_at: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_allowed_until: float = field(default_factory=lambda: time.time() + 3600)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ApprovalStore:
    """
    Persistent storage for approval requests and decisions.

    Stores requests and decisions in JSON files for audit trail.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize approval storage.

        Args:
            storage_path: Path for storing approval data (default: ~/.aios/approvals)
        """
        self.storage_path = storage_path or Path.home() / ".aios" / "approvals"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.requests_file = self.storage_path / "requests.jsonl"
        self.decisions_file = self.storage_path / "decisions.jsonl"
        self.audit_file = self.storage_path / "audit_trail.jsonl"

    def save_request(self, request: ApprovalRequest) -> None:
        """Save approval request to storage."""
        try:
            with open(self.requests_file, 'a') as f:
                f.write(json.dumps(request.to_dict()) + '\n')

            # Log to audit trail
            self._log_audit(
                event="request_created",
                request_id=request.request_id,
                data={"action_path": request.action_path, "sensitivity": request.requirement.sensitivity_level.value}
            )
        except Exception as e:
            LOG.error(f"Error saving approval request: {e}")

    def save_decision(self, decision: ApprovalDecision) -> None:
        """Save approval decision to storage."""
        try:
            with open(self.decisions_file, 'a') as f:
                f.write(json.dumps(decision.to_dict()) + '\n')

            # Log to audit trail
            status = "approved" if decision.approved else "denied"
            self._log_audit(
                event=f"decision_{status}",
                request_id=decision.request_id,
                data={"approver": decision.approver_id, "reason": decision.reason}
            )
        except Exception as e:
            LOG.error(f"Error saving approval decision: {e}")

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Retrieve approval request by ID (returns most recent entry)."""
        try:
            if not self.requests_file.exists():
                return None

            latest_request = None

            with open(self.requests_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get('request_id') == request_id:
                        # Reconstruct ApprovalRequirement from stored data
                        req_data = data.get('requirement', {})
                        requirement = ApprovalRequirement(
                            sensitivity_level=ActionSensitivity(req_data.get('sensitivity_level', 'medium')),
                            requires_reason=req_data.get('requires_reason', True),
                            requires_two_factor=req_data.get('requires_two_factor', False),
                            expires_in_seconds=req_data.get('expires_in_seconds', 3600),
                            affected_systems=req_data.get('affected_systems', [])
                        )

                        # Reconstruct ApprovalRequest
                        req = ApprovalRequest(
                            request_id=data['request_id'],
                            action_path=data['action_path'],
                            action_name=data['action_name'],
                            description=data['description'],
                            requirement=requirement,
                            context=data.get('context', {}),
                            requester_id=data['requester_id'],
                            requested_at=data['requested_at'],
                            status=ApprovalStatus(data['status']),
                            created_at=data['created_at']
                        )
                        req.expires_at = data['expires_at']
                        latest_request = req  # Keep the latest one

            return latest_request
        except Exception as e:
            LOG.error(f"Error retrieving approval request: {e}")
            return None

    def get_decision(self, request_id: str) -> Optional[ApprovalDecision]:
        """Retrieve approval decision for a request (returns most recent)."""
        try:
            if not self.decisions_file.exists():
                return None

            latest_decision = None

            with open(self.decisions_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get('request_id') == request_id:
                        latest_decision = ApprovalDecision(
                            request_id=data['request_id'],
                            approver_id=data['approver_id'],
                            approved=data['approved'],
                            reason=data['reason'],
                            two_factor_verified=data.get('two_factor_verified', False),
                            decided_at=data['decided_at'],
                            execution_allowed_until=data['execution_allowed_until']
                        )
            return latest_decision
        except Exception as e:
            LOG.error(f"Error retrieving approval decision: {e}")
            return None

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests (deduplicated by ID)."""
        requests_by_id = {}
        try:
            if not self.requests_file.exists():
                return []

            with open(self.requests_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    # Keep the latest entry for each request_id (always overwrite)
                    request_id = data.get('request_id')

                    if data.get('status') == ApprovalStatus.PENDING.value:
                        # Reconstruct requirement
                        req_data = data.get('requirement', {})
                        requirement = ApprovalRequirement(
                            sensitivity_level=ActionSensitivity(req_data.get('sensitivity_level', 'medium')),
                            requires_reason=req_data.get('requires_reason', True),
                            requires_two_factor=req_data.get('requires_two_factor', False),
                            expires_in_seconds=req_data.get('expires_in_seconds', 3600),
                            affected_systems=req_data.get('affected_systems', [])
                        )

                        req = ApprovalRequest(
                            request_id=request_id,
                            action_path=data['action_path'],
                            action_name=data['action_name'],
                            description=data['description'],
                            requirement=requirement,
                            context=data.get('context', {}),
                            requester_id=data['requester_id'],
                            requested_at=data['requested_at'],
                            status=ApprovalStatus.PENDING,
                            created_at=data['created_at']
                        )
                        req.expires_at = data['expires_at']
                        requests_by_id[request_id] = req
                    else:
                        # If we see a newer status (non-pending), remove from pending
                        requests_by_id.pop(request_id, None)

        except Exception as e:
            LOG.error(f"Error getting pending requests: {e}")
        return list(requests_by_id.values())

    def get_audit_trail(self, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail entries."""
        entries = []
        try:
            if not self.audit_file.exists():
                return entries

            with open(self.audit_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if request_id is None or data.get('request_id') == request_id:
                        entries.append(data)
        except Exception as e:
            LOG.error(f"Error retrieving audit trail: {e}")
        return entries

    def _log_audit(self, event: str, request_id: str, data: Dict[str, Any]) -> None:
        """Log event to audit trail."""
        try:
            with open(self.audit_file, 'a') as f:
                audit_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'event': event,
                    'request_id': request_id,
                    'data': data
                }
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            LOG.error(f"Error logging audit event: {e}")


class ApprovalWorkflowManager:
    """
    Manages approval workflow for sensitive actions.

    Coordinates:
    1. Creating approval requests for sensitive actions
    2. Tracking approval status
    3. Enforcing approval requirements before execution
    4. Maintaining audit trail
    5. Supporting auto-approval rules
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize approval workflow manager."""
        self.storage = ApprovalStore(storage_path)
        self.sensitivity_rules: Dict[str, ApprovalRequirement] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default sensitivity rules for common actions."""
        # Critical system actions
        self.register_sensitivity_rule(
            "kernel.*",
            ApprovalRequirement(
                ActionSensitivity.CRITICAL,
                requires_reason=True,
                requires_two_factor=True,
                affected_systems=["kernel", "boot", "system"]
            )
        )
        self.register_sensitivity_rule(
            "security.firewall",
            ApprovalRequirement(
                ActionSensitivity.CRITICAL,
                requires_reason=True,
                requires_two_factor=True,
                affected_systems=["security", "network"]
            )
        )
        self.register_sensitivity_rule(
            "storage.delete*",
            ApprovalRequirement(
                ActionSensitivity.HIGH,
                requires_reason=True,
                affected_systems=["storage", "data"]
            )
        )
        self.register_sensitivity_rule(
            "user.*",
            ApprovalRequirement(
                ActionSensitivity.HIGH,
                requires_reason=True,
                affected_systems=["user", "security"]
            )
        )

    def register_sensitivity_rule(
        self,
        action_pattern: str,
        requirement: ApprovalRequirement
    ) -> None:
        """
        Register sensitivity requirement for action pattern.

        Args:
            action_pattern: Pattern like "security.*" or "kernel.process_management"
            requirement: ApprovalRequirement for matching actions
        """
        self.sensitivity_rules[action_pattern] = requirement

    def get_sensitivity_requirement(self, action_path: str) -> ApprovalRequirement:
        """
        Get sensitivity requirement for an action.

        Args:
            action_path: Path to the action (e.g., "security.firewall")

        Returns:
            ApprovalRequirement if action is sensitive, else None requirement
        """
        for pattern, requirement in self.sensitivity_rules.items():
            if self._pattern_matches(action_path, pattern):
                return requirement

        # Default: no approval needed
        return ApprovalRequirement(ActionSensitivity.NONE)

    def create_approval_request(
        self,
        action_path: str,
        action_name: str,
        description: str,
        context: Dict[str, Any],
        requester_id: str = "system"
    ) -> Optional[ApprovalRequest]:
        """
        Create approval request for a sensitive action.

        Args:
            action_path: Path to the action
            action_name: Human-readable action name
            description: Description of what the action does
            context: Additional context data
            requester_id: User requesting the action

        Returns:
            ApprovalRequest if action requires approval, else None
        """
        requirement = self.get_sensitivity_requirement(action_path)

        # Auto-approve if no approval needed or auto-approve condition met
        if requirement.sensitivity_level == ActionSensitivity.NONE:
            return None

        if requirement.should_auto_approve(context):
            LOG.info(f"[info] Auto-approving action {action_path}")
            return None

        # Create approval request
        request = ApprovalRequest(
            action_path=action_path,
            action_name=action_name,
            description=description,
            requirement=requirement,
            context=context,
            requester_id=requester_id
        )

        # Save to storage
        self.storage.save_request(request)

        return request

    def submit_approval_decision(
        self,
        request_id: str,
        approved: bool,
        approver_id: str,
        reason: str = "",
        two_factor_verified: bool = False
    ) -> bool:
        """
        Submit an approval decision.

        Args:
            request_id: ID of the approval request
            approved: Whether to approve or deny
            approver_id: User making the decision
            reason: Reason for decision
            two_factor_verified: Whether 2FA was completed

        Returns:
            True if decision was recorded, False otherwise
        """
        request = self.storage.get_request(request_id)
        if not request:
            LOG.error(f"Approval request not found: {request_id}")
            return False

        # Create decision
        decision = ApprovalDecision(
            request_id=request_id,
            approver_id=approver_id,
            approved=approved,
            reason=reason,
            two_factor_verified=two_factor_verified,
            execution_allowed_until=time.time() + 3600  # 1 hour execution window
        )

        # Update request status before saving
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED

        # Save both decision and updated request
        self.storage.save_decision(decision)
        self.storage.save_request(request)

        LOG.info(f"[info] Approval decision: {request_id} = {approved} by {approver_id}")

        return True

    def can_execute_action(self, action_path: str, request_id: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if action can be executed.

        Args:
            action_path: Path to the action
            request_id: Approval request ID if action requires approval

        Returns:
            Tuple of (can_execute, reason)
        """
        requirement = self.get_sensitivity_requirement(action_path)

        # No approval needed
        if requirement.sensitivity_level == ActionSensitivity.NONE:
            return True, "No approval required"

        # Approval required but no request ID provided
        if not request_id:
            return False, "Approval required but no request ID provided"

        # Get approval request
        request = self.storage.get_request(request_id)
        if not request:
            return False, f"Approval request not found: {request_id}"

        # Check if expired
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            self.storage.save_request(request)
            return False, "Approval request has expired"

        # Get decision
        decision = self.storage.get_decision(request_id)
        if not decision:
            return False, "Approval pending"

        # Check if denied
        if not decision.approved:
            return False, "Approval was denied"

        # Check if execution window has expired
        if time.time() > decision.execution_allowed_until:
            return False, "Execution window has expired"

        # Check 2FA requirement
        if requirement.requires_two_factor and not decision.two_factor_verified:
            return False, "Two-factor authentication not verified"

        return True, "Approval granted"

    def get_approval_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get status of an approval request.

        Args:
            request_id: ID of approval request

        Returns:
            Dictionary with approval status information
        """
        request = self.storage.get_request(request_id)
        if not request:
            return {"status": "not_found"}

        decision = self.storage.get_decision(request_id)

        return {
            "request_id": request_id,
            "action_path": request.action_path,
            "status": request.status.value,
            "created_at": request.created_at,
            "expires_at": datetime.fromtimestamp(request.expires_at).isoformat() if request.expires_at else None,
            "is_expired": request.is_expired(),
            "decision": decision.to_dict() if decision else None
        }

    def get_audit_trail(self, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for approval requests."""
        return self.storage.get_audit_trail(request_id)

    def _pattern_matches(self, action_path: str, pattern: str) -> bool:
        """Check if action path matches pattern (with * wildcard support)."""
        import fnmatch
        return fnmatch.fnmatch(action_path, pattern)

    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get statistics about approval requests."""
        request_by_id = {}  # Use dict to deduplicate by request_id
        decisions_approved = 0
        decisions_denied = 0

        try:
            if self.storage.requests_file.exists():
                with open(self.storage.requests_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Keep the latest entry for each request_id
                            request_by_id[data.get('request_id')] = data

            if self.storage.decisions_file.exists():
                with open(self.storage.decisions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            decision = json.loads(line)
                            if decision.get('approved'):
                                decisions_approved += 1
                            else:
                                decisions_denied += 1
        except Exception as e:
            LOG.error(f"Error getting statistics: {e}")

        # Count unique requests and pending
        requests = list(request_by_id.values())
        total_requests = len(requests)
        critical_requests = sum(1 for r in requests if r.get('requirement', {}).get('sensitivity_level') == 'critical')
        pending_requests = sum(1 for r in requests if r.get('status') == 'pending')

        return {
            "total_requests": total_requests,
            "critical_requests": critical_requests,
            "pending_requests": pending_requests,
            "approved_decisions": decisions_approved,
            "denied_decisions": decisions_denied,
            "approval_rate": decisions_approved / (decisions_approved + decisions_denied) if (decisions_approved + decisions_denied) > 0 else 0.0
        }
