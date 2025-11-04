"""
Agent Authorization and Approval Framework

Implements fail-safe mechanisms for Level 5-6 autonomous agents to prevent
rogue actions, ensure human oversight of destructive operations, and maintain
cryptographic audit trails.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import json
import hashlib
import time
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading
import queue

logger = logging.getLogger(__name__)

# Authorization configuration
AGENT_AUTHORIZATION_DIR = Path('/Users/noone/aios/.agent_authorizations')
AGENT_AUTHORIZATION_DIR.mkdir(exist_ok=True)


class ActionRiskLevel(Enum):
    """Risk classification for agent actions."""
    SAFE = "safe"  # Read-only operations, no side effects
    LOW = "low"  # Minor state changes, easily reversible
    MEDIUM = "medium"  # Significant changes, reversible with effort
    HIGH = "high"  # Critical changes, difficult to reverse
    CRITICAL = "critical"  # Destructive operations (delete, format, etc)


class ApprovalStatus(Enum):
    """Status of action approval requests."""
    PENDING = "pending"  # Awaiting admin decision
    APPROVED = "approved"  # Admin approved
    REJECTED = "rejected"  # Admin rejected
    EXPIRED = "expired"  # Approval window closed
    REVOKED = "revoked"  # Previously approved action revoked


class ActionType(Enum):
    """Types of agent actions requiring special handling."""
    READ = "read"  # Read operations
    WRITE = "write"  # Write new files
    MODIFY = "modify"  # Modify existing files
    DELETE = "delete"  # Delete files/directories
    EXECUTE = "execute"  # Execute system commands
    NETWORK = "network"  # Network operations
    MOUNT = "mount"  # Mount/unmount volumes
    PRIVILEGE = "privilege"  # Change permissions/ownership
    REBOOT = "reboot"  # System reboot


@dataclass
class ActionRequest:
    """Request for agent to perform an action requiring approval."""
    action_id: str
    agent_name: str
    action_type: ActionType
    risk_level: ActionRiskLevel
    description: str
    target_path: Optional[str] = None
    command: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    approval_deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'agent_name': self.agent_name,
            'action_type': self.action_type.value,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'target_path': self.target_path,
            'command': self.command,
            'timestamp': self.timestamp,
            'approval_deadline': self.approval_deadline,
            'metadata': self.metadata
        }

    def to_hash(self) -> str:
        """Create cryptographic hash of action for audit trail."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ApprovalDecision:
    """Admin decision on action approval request."""
    action_id: str
    status: ApprovalStatus
    admin_user: str
    decision_timestamp: float = field(default_factory=time.time)
    reason: str = ""
    signature: str = ""  # Digital signature for authenticity
    revocation_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'status': self.status.value,
            'admin_user': self.admin_user,
            'decision_timestamp': self.decision_timestamp,
            'reason': self.reason,
            'signature': self.signature,
            'revocation_timestamp': self.revocation_timestamp
        }


class RiskAssessmentEngine:
    """Evaluates risk level of agent actions."""

    # Risk classification rules
    SAFE_OPERATIONS = {
        ActionType.READ: ['*.md', '*.txt', '*.json', '*.py'],
        ActionType.EXECUTE: ['python', 'grep', 'find', 'cat'],
    }

    CRITICAL_OPERATIONS = {
        ActionType.DELETE: ['*', '/'],  # All deletes are critical
        ActionType.REBOOT: ['*'],  # All reboots are critical
        ActionType.MOUNT: ['/'],  # Root mounts are critical
    }

    PROTECTED_PATHS = [
        '/Users/noone/.ssh',
        '/Users/noone/.aws',
        '/Users/noone/.env',
        '/etc',
        '/root',
        '/sys',
        '/proc',
    ]

    @staticmethod
    def assess_risk(action: ActionRequest) -> ActionRiskLevel:
        """Determine risk level of action."""

        # Critical operations always require approval
        if action.action_type == ActionType.DELETE:
            return ActionRiskLevel.CRITICAL

        if action.action_type == ActionType.REBOOT:
            return ActionRiskLevel.CRITICAL

        if action.action_type == ActionType.PRIVILEGE:
            return ActionRiskLevel.HIGH

        if action.action_type == ActionType.MOUNT:
            return ActionRiskLevel.HIGH

        # Check protected paths
        if action.target_path:
            for protected in RiskAssessmentEngine.PROTECTED_PATHS:
                if action.target_path.startswith(protected):
                    return ActionRiskLevel.CRITICAL

        # Read operations are safe
        if action.action_type == ActionType.READ:
            return ActionRiskLevel.SAFE

        # Network operations are medium risk
        if action.action_type == ActionType.NETWORK:
            return ActionRiskLevel.MEDIUM

        # Write/Modify are low to medium
        if action.action_type in (ActionType.WRITE, ActionType.MODIFY):
            return ActionRiskLevel.MEDIUM

        return ActionRiskLevel.LOW

    @staticmethod
    def requires_approval(risk_level: ActionRiskLevel) -> bool:
        """Determine if action requires admin approval."""
        return risk_level in (
            ActionRiskLevel.MEDIUM,
            ActionRiskLevel.HIGH,
            ActionRiskLevel.CRITICAL
        )


class AgentAuthorizationManager:
    """Manages agent authorization and approval workflows."""

    def __init__(self, admin_users: Optional[List[str]] = None):
        """
        Initialize authorization manager.

        Args:
            admin_users: List of authorized admin users (default: ['admin'])
        """
        self.admin_users = admin_users or ['admin']
        self.pending_requests: Dict[str, ActionRequest] = {}
        self.decisions: Dict[str, ApprovalDecision] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.request_queue = queue.Queue()
        self._load_persisted_state()

    def request_approval(self, action: ActionRequest) -> Tuple[bool, str]:
        """
        Request admin approval for an action.

        Args:
            action: ActionRequest to be approved

        Returns:
            (approved: bool, request_id: str)
        """
        # Assess risk
        risk_level = RiskAssessmentEngine.assess_risk(action)
        action.risk_level = risk_level

        # If not high risk, auto-approve for trusted agents
        if not RiskAssessmentEngine.requires_approval(risk_level):
            decision = ApprovalDecision(
                action_id=action.action_id,
                status=ApprovalStatus.APPROVED,
                admin_user='system',
                reason=f'Auto-approved: {risk_level.value} risk'
            )
            self.decisions[action.action_id] = decision
            self._log_action('auto_approved', action, decision)
            return (True, action.action_id)

        # High risk actions require explicit approval
        action.approval_deadline = time.time() + 3600  # 1 hour window
        self.pending_requests[action.action_id] = action
        self.request_queue.put(action)

        self._log_action('approval_requested', action)

        logger.warning(
            f"[AUTHORIZATION] Action requires approval: {action.action_id}\n"
            f"  Agent: {action.agent_name}\n"
            f"  Type: {action.action_type.value}\n"
            f"  Risk: {risk_level.value}\n"
            f"  Description: {action.description}\n"
            f"  Deadline: {action.approval_deadline}"
        )

        return (False, action.action_id)

    def get_pending_requests(self, admin_user: str) -> List[Dict[str, Any]]:
        """Get list of pending approval requests for admin review."""
        if admin_user not in self.admin_users:
            raise PermissionError(f"User {admin_user} is not an admin")

        requests = []
        for action_id, action in self.pending_requests.items():
            # Check if not expired
            if action.approval_deadline and time.time() > action.approval_deadline:
                decision = ApprovalDecision(
                    action_id=action_id,
                    status=ApprovalStatus.EXPIRED,
                    admin_user='system',
                    reason='Approval window expired'
                )
                self.decisions[action_id] = decision
                continue

            requests.append({
                'action_id': action.action_id,
                'agent_name': action.agent_name,
                'action_type': action.action_type.value,
                'risk_level': action.risk_level.value,
                'description': action.description,
                'target_path': action.target_path,
                'timestamp': datetime.fromtimestamp(action.timestamp).isoformat(),
                'deadline': datetime.fromtimestamp(action.approval_deadline).isoformat()
                    if action.approval_deadline else None,
                'hash': action.to_hash()
            })

        return requests

    def approve_action(
        self,
        action_id: str,
        admin_user: str,
        signature: str = "",
        reason: str = "Approved by admin"
    ) -> bool:
        """
        Admin approves an action.

        Args:
            action_id: ID of action to approve
            admin_user: Admin user making decision
            signature: Digital signature for audit
            reason: Reason for approval

        Returns:
            True if approved, False if not found/already decided
        """
        if admin_user not in self.admin_users:
            logger.error(f"Unauthorized approval attempt by {admin_user}")
            return False

        if action_id not in self.pending_requests:
            logger.error(f"Action {action_id} not found or already decided")
            return False

        action = self.pending_requests[action_id]
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.APPROVED,
            admin_user=admin_user,
            reason=reason,
            signature=signature
        )

        self.decisions[action_id] = decision
        del self.pending_requests[action_id]

        self._log_action('action_approved', action, decision)

        logger.info(
            f"[AUTHORIZATION] Action approved: {action_id}\n"
            f"  Approved by: {admin_user}\n"
            f"  Reason: {reason}"
        )

        return True

    def reject_action(
        self,
        action_id: str,
        admin_user: str,
        reason: str = "Rejected by admin"
    ) -> bool:
        """
        Admin rejects an action.

        Args:
            action_id: ID of action to reject
            admin_user: Admin user making decision
            reason: Reason for rejection

        Returns:
            True if rejected, False if not found
        """
        if admin_user not in self.admin_users:
            logger.error(f"Unauthorized rejection attempt by {admin_user}")
            return False

        if action_id not in self.pending_requests:
            logger.error(f"Action {action_id} not found or already decided")
            return False

        action = self.pending_requests[action_id]
        decision = ApprovalDecision(
            action_id=action_id,
            status=ApprovalStatus.REJECTED,
            admin_user=admin_user,
            reason=reason
        )

        self.decisions[action_id] = decision
        del self.pending_requests[action_id]

        self._log_action('action_rejected', action, decision)

        logger.warning(
            f"[AUTHORIZATION] Action rejected: {action_id}\n"
            f"  Rejected by: {admin_user}\n"
            f"  Reason: {reason}"
        )

        return True

    def revoke_approval(
        self,
        action_id: str,
        admin_user: str,
        reason: str = "Approval revoked"
    ) -> bool:
        """
        Revoke a previously approved action.

        Args:
            action_id: ID of approved action to revoke
            admin_user: Admin user revoking approval
            reason: Reason for revocation

        Returns:
            True if revoked, False if not found
        """
        if admin_user not in self.admin_users:
            logger.error(f"Unauthorized revocation attempt by {admin_user}")
            return False

        if action_id not in self.decisions:
            logger.error(f"Action {action_id} not found")
            return False

        decision = self.decisions[action_id]
        if decision.status != ApprovalStatus.APPROVED:
            logger.error(f"Cannot revoke non-approved action")
            return False

        decision.status = ApprovalStatus.REVOKED
        decision.revocation_timestamp = time.time()

        self._log_action('approval_revoked', action_id=action_id, decision=decision)

        logger.warning(
            f"[AUTHORIZATION] Approval revoked: {action_id}\n"
            f"  Revoked by: {admin_user}\n"
            f"  Reason: {reason}"
        )

        return True

    def is_approved(self, action_id: str) -> bool:
        """Check if action is approved and still valid."""
        if action_id not in self.decisions:
            return False

        decision = self.decisions[action_id]
        return decision.status == ApprovalStatus.APPROVED

    def get_action_status(self, action_id: str) -> Optional[str]:
        """Get current status of action."""
        if action_id not in self.decisions:
            return None
        return self.decisions[action_id].status.value

    def _log_action(
        self,
        event_type: str,
        action: Optional[ActionRequest] = None,
        decision: Optional[ApprovalDecision] = None,
        action_id: Optional[str] = None
    ) -> None:
        """Log action to audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'action_id': action.action_id if action else action_id,
            'action_data': action.to_dict() if action else None,
            'decision_data': decision.to_dict() if decision else None,
            'sequence_number': len(self.audit_log)
        }

        # Add hash chain for cryptographic integrity
        if self.audit_log:
            prev_entry = self.audit_log[-1]
            prev_hash = hashlib.sha256(
                json.dumps(prev_entry, sort_keys=True).encode()
            ).hexdigest()
            log_entry['previous_hash'] = prev_hash

        self.audit_log.append(log_entry)
        self._save_audit_log()

    def _save_audit_log(self) -> None:
        """Persist audit log to disk."""
        audit_file = AGENT_AUTHORIZATION_DIR / 'audit_log.jsonl'
        with open(audit_file, 'a') as f:
            if self.audit_log:
                latest = self.audit_log[-1]
                f.write(json.dumps(latest) + '\n')

    def _load_persisted_state(self) -> None:
        """Load previously saved decisions and audit log."""
        audit_file = AGENT_AUTHORIZATION_DIR / 'audit_log.jsonl'
        if audit_file.exists():
            with open(audit_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.audit_log.append(entry)

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete cryptographic audit trail."""
        return self.audit_log

    def verify_audit_trail(self) -> bool:
        """Verify integrity of audit trail using hash chain."""
        for i, entry in enumerate(self.audit_log[1:], 1):
            prev_entry = self.audit_log[i - 1]
            prev_hash = hashlib.sha256(
                json.dumps(prev_entry, sort_keys=True).encode()
            ).hexdigest()
            if entry.get('previous_hash') != prev_hash:
                logger.error(f"Audit trail integrity check failed at entry {i}")
                return False
        return True


class AgentActionBlocker:
    """
    Blocks destructive agent actions that lack proper authorization.

    Integrated with agent execution to prevent rogue operations.
    """

    def __init__(self, auth_manager: AgentAuthorizationManager):
        self.auth_manager = auth_manager

    def can_execute(self, action_id: str) -> Tuple[bool, str]:
        """
        Check if action can be executed.

        Returns:
            (can_execute: bool, reason: str)
        """
        if self.auth_manager.is_approved(action_id):
            return (True, "Action approved")

        status = self.auth_manager.get_action_status(action_id)
        if status == ApprovalStatus.REJECTED.value:
            return (False, "Action was rejected by admin")

        if status == ApprovalStatus.EXPIRED.value:
            return (False, "Approval window expired")

        if status == ApprovalStatus.REVOKED.value:
            return (False, "Approval was revoked")

        return (False, "Action requires admin approval")

    def block_if_unauthorized(self, action_id: str) -> None:
        """
        Raise exception if action is unauthorized (block execution).

        Raises:
            PermissionError: If action is not approved
        """
        can_execute, reason = self.can_execute(action_id)
        if not can_execute:
            logger.error(f"[AUTHORIZATION] BLOCKED: {reason}")
            raise PermissionError(f"Action blocked: {reason}")


# Global singleton instance
_auth_manager: Optional[AgentAuthorizationManager] = None


def get_authorization_manager() -> AgentAuthorizationManager:
    """Get or create global authorization manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AgentAuthorizationManager(admin_users=['admin', 'joshua'])
    return _auth_manager


def create_action_request(
    agent_name: str,
    action_type: ActionType,
    description: str,
    target_path: Optional[str] = None,
    command: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ActionRequest:
    """
    Create and request approval for an agent action.

    Args:
        agent_name: Name of agent requesting action
        action_type: Type of action (read, write, delete, etc)
        description: Human-readable description
        target_path: Path to target resource
        command: Command to execute (if applicable)
        metadata: Additional metadata

    Returns:
        ActionRequest instance
    """
    action = ActionRequest(
        action_id=str(uuid.uuid4()),
        agent_name=agent_name,
        action_type=action_type,
        risk_level=ActionRiskLevel.SAFE,  # Will be set by authorization manager
        description=description,
        target_path=target_path,
        command=command,
        metadata=metadata or {}
    )

    auth_manager = get_authorization_manager()
    approved, action_id = auth_manager.request_approval(action)

    if not approved:
        logger.warning(
            f"Action {action_id} requires admin approval before execution"
        )

    return action


# Decorators for easy integration

def require_approval(action_type: ActionType, risk_level: Optional[ActionRiskLevel] = None):
    """
    Decorator to require approval for agent methods.

    Usage:
        @require_approval(ActionType.DELETE, ActionRiskLevel.CRITICAL)
        def delete_file(self, path: str):
            ...
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            action_id = kwargs.get('action_id', str(uuid.uuid4()))
            action = ActionRequest(
                action_id=action_id,
                agent_name=getattr(self, 'name', 'unknown_agent'),
                action_type=action_type,
                risk_level=risk_level or ActionRiskLevel.MEDIUM,
                description=f"Executing {func.__name__}"
            )

            auth_manager = get_authorization_manager()
            approved, _ = auth_manager.request_approval(action)

            if not approved and (risk_level == ActionRiskLevel.CRITICAL or risk_level == ActionRiskLevel.HIGH):
                raise PermissionError(
                    f"Action requires admin approval: {action_id}\n"
                    f"Please contact admin to review pending request"
                )

            return func(self, *args, **kwargs)

        return wrapper
    return decorator
