#!/usr/bin/env python3
"""
Quantum Teleportation Agent for aios
=====================================

Secure integration of patent-pending quantum teleportation protocols
into aios meta-agent framework.

SECURITY:
- User authentication required
- One-by-one execution (mutex locked)
- All operations logged
- Can be globally disabled
- Emergency stop capability

Usage in aios manifest:
{
    "meta_agents": {
        "quantum_teleportation": {
            "actions": {
                "analyze_feasibility": {...},
                "design_protocol": {...},
                "execute_teleportation": {...}
            }
        }
    }
}

Author: Claude Code
License: PROPRIETARY - Patent Pending
"""

import os
import sys
from typing import Dict, Any, Optional

# Add Claude Teleport Brainstorm to path
brainstorm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Claude Teleport Brainstorm")
if os.path.exists(brainstorm_path):
    sys.path.insert(0, brainstorm_path)

try:
    from quantum_teleportation_secure import SecureQuantumTeleportation, ExecutionRecord
    QUANTUM_TELEPORTATION_AVAILABLE = True
except ImportError:
    QUANTUM_TELEPORTATION_AVAILABLE = False
    print("[warn] Quantum teleportation module not available")


# ============================================================================
# AGENT ACTIONS
# ============================================================================

# Global instance (singleton pattern for security)
_quantum_system: Optional[SecureQuantumTeleportation] = None


def get_quantum_system() -> Optional[SecureQuantumTeleportation]:
    """Get or create quantum teleportation system."""
    global _quantum_system

    if not QUANTUM_TELEPORTATION_AVAILABLE:
        return None

    if _quantum_system is None:
        _quantum_system = SecureQuantumTeleportation()

    return _quantum_system


def quantum_teleportation_status(ctx) -> Dict[str, Any]:
    """
    Get quantum teleportation system status.

    Safe action - no execution, just status check.
    """
    from .runtime import ActionResult

    system = get_quantum_system()

    if system is None:
        return ActionResult(
            success=False,
            message="[error] Quantum teleportation not available",
            payload={"available": False}
        )

    status = system.get_status()
    status["available"] = True
    status["patent_status"] = "PROVISIONAL FILED"

    ctx.publish_metadata("quantum_teleportation.status", status)

    return ActionResult(
        success=True,
        message=f"[info] System {'ENABLED' if status['enabled'] else 'DISABLED'}",
        payload=status
    )


def quantum_teleportation_login(ctx) -> Dict[str, Any]:
    """
    Login to quantum teleportation system.

    Environment variables:
    - QUANTUM_USERNAME: Username
    - QUANTUM_PASSWORD: Password
    """
    from .runtime import ActionResult

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    username = ctx.environment.get("QUANTUM_USERNAME")
    password = ctx.environment.get("QUANTUM_PASSWORD")

    if not username or not password:
        return ActionResult(
            success=False,
            message="[error] QUANTUM_USERNAME and QUANTUM_PASSWORD required",
            payload={}
        )

    if system.login(username, password):
        ctx.publish_metadata("quantum_teleportation.user", username)
        return ActionResult(
            success=True,
            message=f"[info] User {username} authenticated",
            payload={"username": username}
        )
    else:
        return ActionResult(
            success=False,
            message="[error] Authentication failed",
            payload={}
        )


def quantum_teleportation_enable(ctx) -> Dict[str, Any]:
    """
    Enable quantum teleportation system.

    REQUIRES: Admin password in QUANTUM_ADMIN_PASSWORD

    SECURITY:
    - Only admin can enable
    - Requires authentication first
    - All operations logged
    """
    from .runtime import ActionResult

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    admin_password = ctx.environment.get("QUANTUM_ADMIN_PASSWORD")

    if not admin_password:
        return ActionResult(
            success=False,
            message="[error] QUANTUM_ADMIN_PASSWORD required",
            payload={}
        )

    if system.enable_system(admin_password):
        ctx.publish_metadata("quantum_teleportation.enabled", True)
        return ActionResult(
            success=True,
            message="[warn] Quantum teleportation system ENABLED",
            payload={"enabled": True}
        )
    else:
        return ActionResult(
            success=False,
            message="[error] Failed to enable system - invalid admin password",
            payload={}
        )


def quantum_teleportation_execute(ctx) -> Dict[str, Any]:
    """
    Execute quantum teleportation protocol.

    SAFETY FEATURES:
    - One-by-one execution (mutex locked)
    - User authentication required
    - System must be enabled
    - All operations logged

    Environment variables:
    - QUANTUM_PROTOCOL: Protocol name (HQT, DBO, INTEGRATED)
    - QUANTUM_PARTICLES: Number of particles
    - QUANTUM_PARAMS: JSON parameters (optional)

    Returns execution record with results.
    """
    from .runtime import ActionResult
    import json

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    # Get parameters
    protocol = ctx.environment.get("QUANTUM_PROTOCOL", "HQT")
    num_particles = int(ctx.environment.get("QUANTUM_PARTICLES", "1000"))
    params_str = ctx.environment.get("QUANTUM_PARAMS", "{}")

    try:
        parameters = json.loads(params_str)
    except json.JSONDecodeError:
        return ActionResult(
            success=False,
            message="[error] Invalid QUANTUM_PARAMS JSON",
            payload={}
        )

    # Execute with safety checks
    record = system.execute_teleportation(
        protocol=protocol,
        num_particles=num_particles,
        parameters=parameters
    )

    if record is None:
        return ActionResult(
            success=False,
            message="[error] Execution failed - check logs",
            payload={}
        )

    # Publish metadata
    ctx.publish_metadata("quantum_teleportation.last_execution", {
        "id": record.id,
        "protocol": record.protocol,
        "num_particles": record.num_particles,
        "status": record.status,
        "result": record.result
    })

    if record.status == "completed":
        return ActionResult(
            success=True,
            message=f"[info] Execution {record.id} completed successfully",
            payload=record.result
        )
    else:
        return ActionResult(
            success=False,
            message=f"[error] Execution {record.id} failed: {record.error}",
            payload={"error": record.error}
        )


def quantum_teleportation_disable(ctx) -> Dict[str, Any]:
    """
    Disable quantum teleportation system.

    Safe action - can be called by any authenticated user.
    """
    from .runtime import ActionResult

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    system.disable_system()

    ctx.publish_metadata("quantum_teleportation.enabled", False)

    return ActionResult(
        success=True,
        message="[info] Quantum teleportation system disabled",
        payload={"enabled": False}
    )


def quantum_teleportation_emergency_stop(ctx) -> Dict[str, Any]:
    """
    Emergency stop - immediately halt all operations.

    CRITICAL ACTION:
    - Stops any running execution
    - Disables system
    - Requires re-enable to use again
    """
    from .runtime import ActionResult

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    system.emergency_stop()

    ctx.publish_metadata("quantum_teleportation.emergency_stop", True)

    return ActionResult(
        success=True,
        message="[warn] EMERGENCY STOP activated",
        payload={"emergency_stop": True}
    )


def quantum_teleportation_history(ctx) -> Dict[str, Any]:
    """
    Get execution history.

    Returns recent execution records.
    """
    from .runtime import ActionResult

    system = get_quantum_system()
    if system is None:
        return ActionResult(success=False, message="[error] System not available")

    limit = int(ctx.environment.get("QUANTUM_HISTORY_LIMIT", "10"))
    history = system.get_history(limit=limit)

    ctx.publish_metadata("quantum_teleportation.history", [
        {
            "id": r.id,
            "username": r.username,
            "protocol": r.protocol,
            "num_particles": r.num_particles,
            "status": r.status,
            "start_time": r.start_time,
            "end_time": r.end_time
        }
        for r in history
    ])

    return ActionResult(
        success=True,
        message=f"[info] Retrieved {len(history)} execution records",
        payload={"count": len(history)}
    )


# ============================================================================
# AGENT REGISTRATION
# ============================================================================

QUANTUM_TELEPORTATION_ACTIONS = {
    "status": quantum_teleportation_status,
    "login": quantum_teleportation_login,
    "enable": quantum_teleportation_enable,
    "execute": quantum_teleportation_execute,
    "disable": quantum_teleportation_disable,
    "emergency_stop": quantum_teleportation_emergency_stop,
    "history": quantum_teleportation_history,
}


def register_quantum_teleportation_agent():
    """
    Register quantum teleportation agent with aios.

    Call this during aios initialization to add quantum teleportation
    capabilities to the meta-agent framework.
    """
    if not QUANTUM_TELEPORTATION_AVAILABLE:
        print("[warn] Quantum teleportation agent not registered (module unavailable)")
        return False

    print("[info] Quantum teleportation agent registered")
    print("[info] Patent status: PROVISIONAL FILED")
    print("[warn] System DISABLED by default - requires user activation")

    return True


# ============================================================================
# CLI FOR TESTING
# ============================================================================

def main():
    """CLI for testing quantum teleportation agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Teleportation Agent")
    parser.add_argument('action', choices=['status', 'enable', 'execute', 'disable', 'emergency'])
    parser.add_argument('--username', help="Username")
    parser.add_argument('--password', help="Password")
    parser.add_argument('--admin-password', help="Admin password")
    parser.add_argument('--protocol', choices=['HQT', 'DBO', 'INTEGRATED'], default='HQT')
    parser.add_argument('--particles', type=int, default=1000)

    args = parser.parse_args()

    system = get_quantum_system()
    if system is None:
        print("[error] Quantum teleportation not available")
        return 1

    if args.action == 'status':
        status = system.get_status()
        print(json.dumps(status, indent=2))

    elif args.action == 'enable':
        if not args.admin_password:
            print("[error] --admin-password required")
            return 1
        system.enable_system(args.admin_password)

    elif args.action == 'execute':
        if not args.username or not args.password:
            print("[error] --username and --password required")
            return 1

        system.login(args.username, args.password)
        record = system.execute_teleportation(args.protocol, args.particles, {})

        if record:
            print(json.dumps(asdict(record), indent=2))
        else:
            return 1

    elif args.action == 'disable':
        system.disable_system()

    elif args.action == 'emergency':
        system.emergency_stop()

    return 0


if __name__ == "__main__":
    import sys
    import json
    from dataclasses import asdict
    sys.exit(main())
