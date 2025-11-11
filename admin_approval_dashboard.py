"""
Admin Approval Dashboard and CLI

Provides admin interface for reviewing and approving/rejecting agent actions.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import json
import sys
import argparse
from datetime import datetime
from typing import Optional
from pathlib import Path

from agent_authorization import (
    get_authorization_manager,
    ActionType,
    ApprovalStatus
)


class AdminDashboard:
    """Interactive admin approval dashboard."""

    def __init__(self):
        self.auth_manager = get_authorization_manager()
        self.admin_user = self._get_admin_user()

    @staticmethod
    def _get_admin_user() -> str:
        """Get current admin user from environment or prompt."""
        import os
        admin = os.environ.get('ADMIN_USER', 'admin')
        if admin not in ['admin', 'joshua']:
            raise PermissionError(f"User {admin} is not authorized as admin")
        return admin

    def print_header(self):
        """Print dashboard header."""
        print("\n" + "=" * 80)
        print("AGENT AUTHORIZATION ADMIN DASHBOARD".center(80))
        print("=" * 80 + "\n")

    def show_pending_requests(self):
        """Display all pending approval requests."""
        requests = self.auth_manager.get_pending_requests(self.admin_user)

        if not requests:
            print("✓ No pending approval requests\n")
            return

        print(f"📋 PENDING REQUESTS: {len(requests)}")
        print("-" * 80)

        for i, req in enumerate(requests, 1):
            risk_color = self._color_risk(req['risk_level'])
            print(f"\n{i}. [{req['action_id'][:8]}...] {req['agent_name']}")
            print(f"   Type: {req['action_type']} | Risk: {risk_color}{req['risk_level']}\033[0m")
            print(f"   Description: {req['description']}")
            if req['target_path']:
                print(f"   Target: {req['target_path']}")
            print(f"   Requested: {req['timestamp']}")
            print(f"   Deadline:  {req['deadline']}")
            print(f"   Hash: {req['hash'][:16]}...")

        print("\n" + "-" * 80)

    def show_recent_decisions(self, limit: int = 10):
        """Display recent admin decisions."""
        audit_trail = self.auth_manager.get_audit_trail()

        if not audit_trail:
            print("✓ No audit trail entries\n")
            return

        # Filter to decision entries
        decisions = [
            e for e in audit_trail[-limit:]
            if e.get('event_type') in ['action_approved', 'action_rejected', 'approval_revoked']
        ]

        if not decisions:
            print("✓ No recent decisions\n")
            return

        print(f"📋 RECENT DECISIONS: {len(decisions)}")
        print("-" * 80)

        for entry in decisions:
            decision_data = entry.get('decision_data', {})
            timestamp = entry.get('timestamp', 'unknown')
            status = decision_data.get('status', 'unknown')

            status_symbol = {
                'approved': '✓',
                'rejected': '✗',
                'revoked': '⊗'
            }.get(status, '?')

            print(f"{status_symbol} {entry['action_id'][:8]}... | "
                  f"{decision_data.get('admin_user', 'unknown')} | "
                  f"{status} | {timestamp}")

        print("\n" + "-" * 80)

    def approve_action(self, action_id: str, reason: str = ""):
        """Approve pending action."""
        if self.auth_manager.approve_action(
            action_id,
            self.admin_user,
            reason=reason or "Approved by admin"
        ):
            print(f"✓ Action {action_id[:8]}... APPROVED\n")
            return True
        else:
            print(f"✗ Failed to approve action {action_id[:8]}...\n")
            return False

    def reject_action(self, action_id: str, reason: str = ""):
        """Reject pending action."""
        if self.auth_manager.reject_action(
            action_id,
            self.admin_user,
            reason=reason or "Rejected by admin"
        ):
            print(f"✓ Action {action_id[:8]}... REJECTED\n")
            return True
        else:
            print(f"✗ Failed to reject action {action_id[:8]}...\n")
            return False

    def revoke_approval(self, action_id: str, reason: str = ""):
        """Revoke previously approved action."""
        if self.auth_manager.revoke_approval(
            action_id,
            self.admin_user,
            reason=reason or "Revoked by admin"
        ):
            print(f"✓ Approval {action_id[:8]}... REVOKED\n")
            return True
        else:
            print(f"✗ Failed to revoke approval {action_id[:8]}...\n")
            return False

    def show_audit_trail(self, limit: int = 50):
        """Display cryptographic audit trail."""
        trail = self.auth_manager.get_audit_trail()

        if not trail:
            print("✓ Audit trail is empty\n")
            return

        print(f"🔐 AUDIT TRAIL: {len(trail)} entries (showing last {limit})")
        print("-" * 80)

        for entry in trail[-limit:]:
            timestamp = entry.get('timestamp', 'unknown')
            event_type = entry.get('event_type', 'unknown')
            action_id = entry.get('action_id', 'unknown')[:8]
            hash_val = entry.get('previous_hash', 'none')[:16] if entry.get('previous_hash') else 'NONE'

            print(f"[{timestamp}] {event_type:20} | {action_id}... | ↳{hash_val}...")

        # Verify integrity
        print("\n" + "-" * 80)
        if self.auth_manager.verify_audit_trail():
            print("✓ Audit trail integrity: VERIFIED\n")
        else:
            print("✗ Audit trail integrity: FAILED - Tampering detected!\n")

    def verify_audit_integrity(self):
        """Verify integrity of entire audit trail."""
        print("Verifying audit trail integrity...")
        if self.auth_manager.verify_audit_trail():
            print("✓ All entries verified - no tampering detected\n")
            return True
        else:
            print("✗ Integrity verification FAILED - audit log may have been tampered with!\n")
            return False

    @staticmethod
    def _color_risk(risk_level: str) -> str:
        """Return colored risk level string."""
        colors = {
            'safe': '\033[92m',      # Green
            'low': '\033[92m',       # Green
            'medium': '\033[93m',    # Yellow
            'high': '\033[91m',      # Red
            'critical': '\033[91m'   # Red
        }
        return colors.get(risk_level, '\033[0m') + risk_level

    def interactive_mode(self):
        """Run interactive approval dashboard."""
        self.print_header()

        while True:
            print("\n📊 ADMIN MENU")
            print("-" * 80)
            print("1. Show pending approval requests")
            print("2. Show recent decisions")
            print("3. Show audit trail")
            print("4. Approve action")
            print("5. Reject action")
            print("6. Revoke approval")
            print("7. Verify audit integrity")
            print("8. Export audit log")
            print("0. Exit")
            print("-" * 80)

            choice = input("\nSelect option [0-8]: ").strip()

            if choice == '1':
                self.show_pending_requests()
            elif choice == '2':
                limit = input("How many recent decisions? [10]: ").strip() or "10"
                self.show_recent_decisions(int(limit))
            elif choice == '3':
                limit = input("How many audit entries? [50]: ").strip() or "50"
                self.show_audit_trail(int(limit))
            elif choice == '4':
                action_id = input("Action ID to approve: ").strip()
                reason = input("Reason (optional): ").strip()
                self.approve_action(action_id, reason)
            elif choice == '5':
                action_id = input("Action ID to reject: ").strip()
                reason = input("Reason (optional): ").strip()
                self.reject_action(action_id, reason)
            elif choice == '6':
                action_id = input("Action ID to revoke: ").strip()
                reason = input("Reason (optional): ").strip()
                self.revoke_approval(action_id, reason)
            elif choice == '7':
                self.verify_audit_integrity()
            elif choice == '8':
                self._export_audit_log()
            elif choice == '0':
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("Invalid choice")

    @staticmethod
    def _export_audit_log():
        """Export audit log to file."""
        auth_manager = get_authorization_manager()
        trail = auth_manager.get_audit_trail()

        filename = f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(trail, f, indent=2)

        print(f"✓ Audit log exported to {filename}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Agent Authorization Admin Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python admin_approval_dashboard.py

  # Show pending requests
  python admin_approval_dashboard.py --show-pending

  # Approve action
  python admin_approval_dashboard.py --approve 12345678-1234-5678-1234-567812345678

  # Reject action
  python admin_approval_dashboard.py --reject 12345678-1234-5678-1234-567812345678

  # Show audit trail
  python admin_approval_dashboard.py --audit --limit 100

  # Verify integrity
  python admin_approval_dashboard.py --verify
        """
    )

    parser.add_argument('--show-pending', action='store_true',
                        help='Show pending approval requests')
    parser.add_argument('--show-recent', type=int, metavar='N',
                        help='Show N recent decisions')
    parser.add_argument('--approve', metavar='ACTION_ID',
                        help='Approve action')
    parser.add_argument('--reject', metavar='ACTION_ID',
                        help='Reject action')
    parser.add_argument('--revoke', metavar='ACTION_ID',
                        help='Revoke approval')
    parser.add_argument('--audit', action='store_true',
                        help='Show audit trail')
    parser.add_argument('--limit', type=int, default=50,
                        help='Limit number of entries (default: 50)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify audit trail integrity')
    parser.add_argument('--export', action='store_true',
                        help='Export audit log to JSON file')
    parser.add_argument('--reason', metavar='REASON',
                        help='Reason for decision')

    args = parser.parse_args()
    dashboard = AdminDashboard()

    # If no specific action, run interactive mode
    if not any([args.show_pending, args.show_recent, args.approve,
                args.reject, args.revoke, args.audit, args.verify, args.export]):
        dashboard.interactive_mode()
        return

    # Handle specific commands
    if args.show_pending:
        dashboard.print_header()
        dashboard.show_pending_requests()

    if args.show_recent:
        dashboard.print_header()
        dashboard.show_recent_decisions(args.show_recent)

    if args.approve:
        dashboard.approve_action(args.approve, args.reason or "")

    if args.reject:
        dashboard.reject_action(args.reject, args.reason or "")

    if args.revoke:
        dashboard.revoke_approval(args.revoke, args.reason or "")

    if args.audit:
        dashboard.print_header()
        dashboard.show_audit_trail(args.limit)

    if args.verify:
        dashboard.print_header()
        dashboard.verify_audit_integrity()

    if args.export:
        dashboard._export_audit_log()


if __name__ == '__main__':
    main()
