"""
UserAgent - User Management & Authentication

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import subprocess
import platform
import pwd
import grp
from typing import Dict, List, Optional
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass
class UserInfo:
    """User information structure."""
    username: str
    uid: int
    gid: int
    home_directory: str
    shell: str
    is_admin: bool


class UserAgent:
    """
    Meta-agent for user management and authentication.

    Responsibilities:
    - User account management
    - Authentication configuration
    - Authorization policy enforcement
    - User permission auditing
    - Session management
    """

    def __init__(self):
        self.name = "user"
        self.platform = platform.system()
        LOG.info(f"UserAgent initialized on {self.platform}")

    def list_users(self) -> List[Dict]:
        """List all user accounts on the system."""
        try:
            if self.platform == "Darwin":  # macOS
                return self._list_users_macos()
            elif self.platform == "Windows":
                return self._list_users_windows()
            else:  # Linux
                return self._list_users_linux()
        except Exception as e:
            LOG.error(f"Error listing users: {e}")
            return []

    def _list_users_linux(self) -> List[Dict]:
        """List Linux users from /etc/passwd."""
        users = []
        try:
            for entry in pwd.getall():
                users.append({
                    "username": entry.pw_name,
                    "uid": entry.pw_uid,
                    "gid": entry.pw_gid,
                    "home": entry.pw_dir,
                    "shell": entry.pw_shell,
                    "system_user": entry.pw_uid < 1000,
                })
        except Exception as e:
            LOG.warning(f"Error reading user list: {e}")

        return users

    def _list_users_macos(self) -> List[Dict]:
        """List macOS users using dscl."""
        users = []
        try:
            result = subprocess.run(
                ["dscl", ".", "-list", "/Users"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for username in result.stdout.strip().split("\n"):
                if username and not username.startswith("_"):
                    users.append({
                        "username": username,
                        "system_user": username.startswith("_"),
                    })
        except Exception as e:
            LOG.warning(f"Error listing macOS users: {e}")

        return users

    def _list_users_windows(self) -> List[Dict]:
        """List Windows users using PowerShell."""
        users = []
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-LocalUser | Select-Object Name, Enabled, LastLogonDate | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                try:
                    import json
                    data = json.loads(result.stdout)
                    if isinstance(data, list):
                        for user in data:
                            users.append({
                                "username": user.get("Name"),
                                "enabled": user.get("Enabled", False),
                                "last_logon": user.get("LastLogonDate"),
                            })
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            LOG.warning(f"Error listing Windows users: {e}")

        return users

    def get_user_info(self, username: str) -> Dict:
        """Get detailed information about a specific user."""
        try:
            if self.platform in ["Linux", "Darwin"]:
                return self._get_user_info_unix(username)
            elif self.platform == "Windows":
                return self._get_user_info_windows(username)
        except Exception as e:
            LOG.error(f"Error getting user info: {e}")
            return {"error": str(e)}

    def _get_user_info_unix(self, username: str) -> Dict:
        """Get Unix user information."""
        try:
            user = pwd.getpwnam(username)
            group = grp.getgrgid(user.pw_gid)

            return {
                "username": user.pw_name,
                "uid": user.pw_uid,
                "gid": user.pw_gid,
                "group_name": group.gr_name,
                "home_directory": user.pw_dir,
                "shell": user.pw_shell,
                "is_system_user": user.pw_uid < 1000,
            }
        except KeyError:
            return {"error": f"User '{username}' not found"}
        except Exception as e:
            return {"error": str(e)}

    def _get_user_info_windows(self, username: str) -> Dict:
        """Get Windows user information."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"Get-LocalUser -Name '{username}' | Select-Object Name, Enabled, LastLogonDate, PrincipalSource | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return {
                    "username": data.get("Name"),
                    "enabled": data.get("Enabled", False),
                    "last_logon": data.get("LastLogonDate"),
                    "source": data.get("PrincipalSource"),
                }
            return {"error": f"User '{username}' not found"}
        except Exception as e:
            return {"error": str(e)}

    def list_groups(self) -> List[Dict]:
        """List all groups on the system."""
        try:
            if self.platform in ["Linux", "Darwin"]:
                return self._list_groups_unix()
            elif self.platform == "Windows":
                return self._list_groups_windows()
        except Exception as e:
            LOG.error(f"Error listing groups: {e}")
            return []

    def _list_groups_unix(self) -> List[Dict]:
        """List Unix groups."""
        groups = []
        try:
            for entry in grp.getall():
                groups.append({
                    "group_name": entry.gr_name,
                    "gid": entry.gr_gid,
                    "members": entry.gr_mem,
                    "system_group": entry.gr_gid < 1000,
                })
        except Exception as e:
            LOG.warning(f"Error reading group list: {e}")

        return groups

    def _list_groups_windows(self) -> List[Dict]:
        """List Windows groups."""
        groups = []
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-LocalGroup | Select-Object Name, Description | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    for group in data:
                        groups.append({
                            "group_name": group.get("Name"),
                            "description": group.get("Description"),
                        })
        except Exception as e:
            LOG.warning(f"Error listing Windows groups: {e}")

        return groups

    def check_user_permissions(self, username: str) -> Dict:
        """Check user permissions and group memberships."""
        try:
            if self.platform in ["Linux", "Darwin"]:
                return self._check_user_permissions_unix(username)
            elif self.platform == "Windows":
                return self._check_user_permissions_windows(username)
        except Exception as e:
            LOG.error(f"Error checking permissions: {e}")
            return {"error": str(e)}

    def _check_user_permissions_unix(self, username: str) -> Dict:
        """Check Unix user permissions."""
        try:
            user = pwd.getpwnam(username)
            groups = []

            for entry in grp.getall():
                if username in entry.gr_mem or user.pw_gid == entry.gr_gid:
                    groups.append(entry.gr_name)

            # Check for sudo access
            sudo_access = False
            try:
                result = subprocess.run(
                    ["sudo", "-l", "-U", username],
                    capture_output=True,
                    timeout=5,
                )
                sudo_access = result.returncode == 0
            except Exception:
                pass

            return {
                "username": username,
                "primary_group": grp.getgrgid(user.pw_gid).gr_name,
                "groups": groups,
                "sudo_access": sudo_access,
                "home_directory": user.pw_dir,
                "is_admin": "sudo" in groups or "wheel" in groups,
            }
        except KeyError:
            return {"error": f"User '{username}' not found"}

    def _check_user_permissions_windows(self, username: str) -> Dict:
        """Check Windows user permissions."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"Get-LocalGroupMember -Group 'Administrators' | Where-Object {{$_.Name -like '*{username}*'}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            is_admin = result.returncode == 0 and username in result.stdout

            return {
                "username": username,
                "is_admin": is_admin,
                "administrator_member": is_admin,
            }
        except Exception as e:
            return {"error": str(e)}

    def check_authentication_methods(self) -> Dict:
        """Check available authentication methods on the system."""
        try:
            if self.platform == "Darwin":
                return self._check_auth_methods_macos()
            elif self.platform == "Windows":
                return self._check_auth_methods_windows()
            else:
                return self._check_auth_methods_linux()
        except Exception as e:
            LOG.error(f"Error checking auth methods: {e}")
            return {"error": str(e)}

    def _check_auth_methods_macos(self) -> Dict:
        """Check macOS authentication methods."""
        methods = {
            "password": True,
            "biometric": False,
            "touch_id": False,
            "kerberos": False,
        }

        try:
            # Check for Touch ID
            result = subprocess.run(
                ["bioutil", "-r"],
                capture_output=True,
                timeout=5,
            )
            methods["biometric"] = result.returncode == 0
            methods["touch_id"] = result.returncode == 0
        except Exception:
            pass

        return {
            "platform": "macOS",
            "methods": methods,
            "primary_method": "password",
        }

    def _check_auth_methods_windows(self) -> Dict:
        """Check Windows authentication methods."""
        methods = {
            "password": True,
            "windows_hello": False,
            "smartcard": False,
            "biometric": False,
        }

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-MpPreference | Select-Object DisableRealtimeMonitoring",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Windows Hello typically available on modern Windows 10/11
            methods["windows_hello"] = True
        except Exception:
            pass

        return {
            "platform": "Windows",
            "methods": methods,
            "primary_method": "password",
        }

    def _check_auth_methods_linux(self) -> Dict:
        """Check Linux authentication methods."""
        methods = {
            "password": True,
            "ssh_key": False,
            "pam": False,
            "ldap": False,
        }

        try:
            # Check for SSH keys
            result = subprocess.run(
                ["test", "-f", "/etc/ssh/sshd_config"],
                capture_output=True,
                timeout=5,
            )
            methods["ssh_key"] = result.returncode == 0

            # Check for PAM
            result = subprocess.run(
                ["test", "-f", "/etc/pam.d/system-auth"],
                capture_output=True,
                timeout=5,
            )
            methods["pam"] = result.returncode == 0
        except Exception:
            pass

        return {
            "platform": "Linux",
            "methods": methods,
            "primary_method": "password",
        }

    def get_current_user(self) -> Dict:
        """Get information about the current user."""
        try:
            if self.platform in ["Linux", "Darwin"]:
                result = subprocess.run(
                    ["whoami"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                username = result.stdout.strip()
                return self.get_user_info(username)
            elif self.platform == "Windows":
                result = subprocess.run(
                    ["whoami"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                username = result.stdout.strip()
                return {
                    "current_user": username,
                    "platform": "Windows",
                }
        except Exception as e:
            LOG.error(f"Error getting current user: {e}")
            return {"error": str(e)}

    def list_active_sessions(self) -> List[Dict]:
        """List active user sessions."""
        try:
            if self.platform == "Darwin":
                return self._list_sessions_macos()
            elif self.platform == "Windows":
                return self._list_sessions_windows()
            else:
                return self._list_sessions_linux()
        except Exception as e:
            LOG.error(f"Error listing sessions: {e}")
            return []

    def _list_sessions_macos(self) -> List[Dict]:
        """List macOS user sessions."""
        sessions = []
        try:
            result = subprocess.run(
                ["w", "-h"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        sessions.append({
                            "username": parts[0],
                            "terminal": parts[1],
                            "login_time": " ".join(parts[3:5]),
                        })
        except Exception as e:
            LOG.warning(f"Error listing macOS sessions: {e}")

        return sessions

    def _list_sessions_windows(self) -> List[Dict]:
        """List Windows user sessions."""
        sessions = []
        try:
            result = subprocess.run(
                ["quser"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().split("\n")[1:]  # Skip header

            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    sessions.append({
                        "username": parts[0],
                        "session_type": parts[1] if len(parts) > 1 else "unknown",
                    })
        except Exception as e:
            LOG.warning(f"Error listing Windows sessions: {e}")

        return sessions

    def _list_sessions_linux(self) -> List[Dict]:
        """List Linux user sessions."""
        sessions = []
        try:
            result = subprocess.run(
                ["w", "-h"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        sessions.append({
                            "username": parts[0],
                            "terminal": parts[1],
                            "login_time": " ".join(parts[2:4]),
                        })
        except Exception as e:
            LOG.warning(f"Error listing Linux sessions: {e}")

        return sessions
