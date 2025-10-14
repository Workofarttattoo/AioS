"""
AgentaOS Self-Update System
Autonomous system that updates itself daily in the early morning hours.

Features:
- Scheduled updates (3 AM default)
- Git-based version control
- Dependency management
- Safe rollback capability
- Zero-downtime updates
- Self-healing on failures
"""

import asyncio
import subprocess
import os
import sys
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# ═══════════════════════════════════════════════════════════════════════
# UPDATE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UpdateConfig:
    """Configuration for self-update system."""
    update_hour: int = 3  # 3 AM
    update_minute: int = 0
    check_interval_hours: int = 24
    auto_update_enabled: bool = True
    backup_before_update: bool = True
    max_backups: int = 5
    git_remote: str = "origin"
    git_branch: str = "main"
    update_log_path: str = "logs/self_update.log"
    require_confirmation: bool = False  # False = fully autonomous


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    timestamp: float
    version_before: str
    version_after: str
    changes: List[str]
    errors: List[str] = field(default_factory=list)
    rollback_performed: bool = False
    duration_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# SELF-UPDATE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class SelfUpdateEngine:
    """
    Autonomous self-update system for AgentaOS.
    Updates itself from git repository every day at configured time.
    """

    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        self.base_path = Path(__file__).parent
        self.backup_dir = self.base_path / "backups"
        self.log_file = Path(self.config.update_log_path)
        self.running = False
        self.last_update = None
        self.update_history = []

        # Ensure directories exist
        self.backup_dir.mkdir(exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def start_scheduler(self):
        """Start the autonomous update scheduler."""
        self.running = True
        print(f"[SelfUpdate] Scheduler started")
        print(f"[SelfUpdate] Updates scheduled for {self.config.update_hour:02d}:{self.config.update_minute:02d} daily")

        while self.running:
            # Calculate time until next update
            next_update = self._calculate_next_update_time()
            wait_seconds = (next_update - datetime.now()).total_seconds()

            if wait_seconds > 0:
                print(f"[SelfUpdate] Next update in {wait_seconds/3600:.1f} hours at {next_update}")
                await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour max
            else:
                # Time for update!
                if self.config.auto_update_enabled:
                    print(f"[SelfUpdate] Initiating scheduled update at {datetime.now()}")
                    await self.perform_update()
                    self.last_update = datetime.now()
                else:
                    print(f"[SelfUpdate] Auto-update disabled, skipping")

                # Wait until next day
                await asyncio.sleep(3600)  # Check again in an hour

    def stop_scheduler(self):
        """Stop the scheduler."""
        self.running = False
        print("[SelfUpdate] Scheduler stopped")

    def _calculate_next_update_time(self) -> datetime:
        """Calculate next scheduled update time."""
        now = datetime.now()
        target = now.replace(
            hour=self.config.update_hour,
            minute=self.config.update_minute,
            second=0,
            microsecond=0
        )

        if target <= now:
            # Already passed today, schedule for tomorrow
            target += timedelta(days=1)

        return target

    async def perform_update(self) -> UpdateResult:
        """
        Perform complete system update.

        Steps:
        1. Get current version
        2. Create backup
        3. Fetch latest changes
        4. Update code
        5. Update dependencies
        6. Run tests
        7. Rollback if failure
        """
        start_time = time.time()
        self._log("═" * 70)
        self._log(f"UPDATE STARTED at {datetime.now()}")
        self._log("═" * 70)

        # Get current version
        version_before = await self._get_current_version()
        self._log(f"Current version: {version_before}")

        # Create backup
        if self.config.backup_before_update:
            backup_path = await self._create_backup()
            self._log(f"Backup created: {backup_path}")
        else:
            backup_path = None

        changes = []
        errors = []
        rollback_performed = False

        try:
            # Check for updates
            updates_available, remote_changes = await self._check_for_updates()

            if not updates_available:
                self._log("No updates available")
                return UpdateResult(
                    success=True,
                    timestamp=time.time(),
                    version_before=version_before,
                    version_after=version_before,
                    changes=["No updates available"],
                    duration_seconds=time.time() - start_time
                )

            self._log(f"Found {len(remote_changes)} changes to apply")

            # Pull latest changes
            pull_success = await self._git_pull()
            if not pull_success:
                raise Exception("Git pull failed")

            changes.extend(remote_changes)

            # Update dependencies
            deps_updated = await self._update_dependencies()
            if deps_updated:
                changes.append("Dependencies updated")

            # Run health checks
            health_ok = await self._run_health_checks()
            if not health_ok:
                raise Exception("Health checks failed after update")

            # Get new version
            version_after = await self._get_current_version()
            self._log(f"Updated to version: {version_after}")

            # Clean old backups
            await self._cleanup_old_backups()

            duration = time.time() - start_time
            self._log(f"UPDATE COMPLETED successfully in {duration:.2f}s")

            result = UpdateResult(
                success=True,
                timestamp=time.time(),
                version_before=version_before,
                version_after=version_after,
                changes=changes,
                duration_seconds=duration
            )

            self.update_history.append(result)
            return result

        except Exception as e:
            error_msg = f"Update failed: {e}"
            self._log(f"ERROR: {error_msg}")
            errors.append(error_msg)

            # Attempt rollback
            if backup_path:
                self._log("Attempting rollback...")
                rollback_success = await self._rollback_from_backup(backup_path)
                rollback_performed = rollback_success

                if rollback_success:
                    self._log("Rollback successful")
                else:
                    self._log("CRITICAL: Rollback failed!")
                    errors.append("Rollback failed")

            duration = time.time() - start_time

            result = UpdateResult(
                success=False,
                timestamp=time.time(),
                version_before=version_before,
                version_after=version_before,
                changes=changes,
                errors=errors,
                rollback_performed=rollback_performed,
                duration_seconds=duration
            )

            self.update_history.append(result)
            return result

    async def _get_current_version(self) -> str:
        """Get current git version."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    async def _create_backup(self) -> Path:
        """Create backup of current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        # Create backup archive
        shutil.make_archive(
            str(backup_path),
            'gztar',
            self.base_path,
            '.'
        )

        return Path(str(backup_path) + '.tar.gz')

    async def _check_for_updates(self) -> Tuple[bool, List[str]]:
        """Check if updates are available."""
        try:
            # Fetch from remote
            subprocess.run(
                ["git", "fetch", self.config.git_remote],
                cwd=self.base_path,
                capture_output=True,
                check=True
            )

            # Check for differences
            result = subprocess.run(
                ["git", "log", "--oneline", f"HEAD..{self.config.git_remote}/{self.config.git_branch}"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                check=True
            )

            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return len(commits) > 0, commits

        except Exception as e:
            self._log(f"Error checking for updates: {e}")
            return False, []

    async def _git_pull(self) -> bool:
        """Pull latest changes from git."""
        try:
            result = subprocess.run(
                ["git", "pull", self.config.git_remote, self.config.git_branch],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                check=True
            )

            self._log(f"Git pull output: {result.stdout}")
            return True

        except Exception as e:
            self._log(f"Git pull failed: {e}")
            return False

    async def _update_dependencies(self) -> bool:
        """Update Python dependencies."""
        requirements_file = self.base_path / "requirements.txt"

        if not requirements_file.exists():
            return False

        try:
            # Check if requirements changed
            result = subprocess.run(
                ["git", "diff", "HEAD@{1}", "HEAD", "--", "requirements.txt"],
                cwd=self.base_path,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                self._log("No dependency changes")
                return False

            # Update dependencies
            self._log("Updating dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--upgrade"],
                cwd=self.base_path,
                capture_output=True,
                check=True
            )

            self._log("Dependencies updated successfully")
            return True

        except Exception as e:
            self._log(f"Dependency update failed: {e}")
            return False

    async def _run_health_checks(self) -> bool:
        """Run system health checks after update."""
        try:
            # Try to import core modules
            sys.path.insert(0, str(self.base_path))

            # Import critical modules
            import runtime
            import config

            # Run basic validation
            if not hasattr(config, 'DEFAULT_MANIFEST'):
                return False

            self._log("Health checks passed")
            return True

        except Exception as e:
            self._log(f"Health check failed: {e}")
            return False

    async def _rollback_from_backup(self, backup_path: Path) -> bool:
        """Rollback system from backup."""
        try:
            # Extract backup
            shutil.unpack_archive(str(backup_path), self.base_path)

            # Verify rollback
            health_ok = await self._run_health_checks()

            return health_ok

        except Exception as e:
            self._log(f"Rollback failed: {e}")
            return False

    async def _cleanup_old_backups(self):
        """Remove old backups, keep only max_backups most recent."""
        try:
            backups = sorted(self.backup_dir.glob("backup_*.tar.gz"))

            if len(backups) > self.config.max_backups:
                to_remove = backups[:-self.config.max_backups]
                for backup in to_remove:
                    backup.unlink()
                    self._log(f"Removed old backup: {backup.name}")

        except Exception as e:
            self._log(f"Backup cleanup failed: {e}")

    def _log(self, message: str):
        """Write to update log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"

        print(f"[SelfUpdate] {message}")

        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception:
            pass  # Fail silently on log errors

    def get_update_history(self) -> List[UpdateResult]:
        """Get history of updates."""
        return self.update_history

    def get_status(self) -> Dict[str, Any]:
        """Get current status of update system."""
        return {
            'running': self.running,
            'auto_update_enabled': self.config.auto_update_enabled,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'next_update': self._calculate_next_update_time().isoformat(),
            'update_count': len(self.update_history),
            'successful_updates': len([u for u in self.update_history if u.success]),
            'failed_updates': len([u for u in self.update_history if not u.success])
        }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION WITH AGENTAOS
# ═══════════════════════════════════════════════════════════════════════

async def start_self_update_daemon():
    """
    Start the self-update daemon.
    Call this from aios boot sequence.
    """
    config = UpdateConfig(
        update_hour=3,  # 3 AM
        update_minute=0,
        auto_update_enabled=True,
        backup_before_update=True,
        max_backups=5
    )

    engine = SelfUpdateEngine(config)

    print("[AgentaOS] Starting self-update daemon...")
    print(f"[AgentaOS] Daily updates at {config.update_hour:02d}:{config.update_minute:02d}")

    # Start scheduler in background
    asyncio.create_task(engine.start_scheduler())

    return engine


def check_self_update_dependencies() -> Dict[str, bool]:
    """Check if git and required tools are available."""
    deps = {}

    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        deps['git'] = True
    except Exception:
        deps['git'] = False

    try:
        import shutil as _shutil  # Check if shutil available
        deps['shutil'] = True
    except ImportError:
        deps['shutil'] = False

    return deps
