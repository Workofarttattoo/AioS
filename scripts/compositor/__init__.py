"""
Wayland compositor bootstrap helpers for AgentaOS.

This module prepares a minimal Wayland session, launches the GTK/Qt dashboard
viewer, and falls back to a curses dashboard in headless environments.  The
actual rendering logic lives in ``compositor_main.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from aios.gui.bus import resolve_endpoint


def _detect_compositor_binary(preferred: Optional[str] = None) -> Optional[str]:
  """Locate a usable Wayland compositor binary (sway or cage)."""

  candidates = [preferred] if preferred else []
  candidates.extend(["sway", "cage"])
  for candidate in candidates:
    if candidate and shutil.which(candidate):
      return candidate
  return None


def _launch_external_compositor(binary: str) -> subprocess.Popen:
  """
  Spawn the external compositor in the foreground.

  The worker relies on sway/cage defaults so no additional configuration
  files are required.  The temporary runtime directory prevents accidental
  interference with an existing session.
  """

  runtime_dir = Path(tempfile.mkdtemp(prefix="agentaos-wl-"))
  env = os.environ.copy()
  env.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
  env.setdefault("XDG_SESSION_TYPE", "wayland")

  command = [binary]
  if binary.endswith("sway"):
    command.extend(["--unsupported-gpu", "--config", "/dev/null"])

  return subprocess.Popen(command, env=env)


def run_dashboard(endpoint: str, *, headless: bool = False) -> int:
  """Invoke the compositor main application with the requested mode."""

  from . import compositor_main

  args = ["--endpoint", endpoint]
  if headless:
    args.append("--headless")
  return compositor_main.main(args)


def launch_wayland_session(
  endpoint: str,
  *,
  headless: bool = False,
  compositor: Optional[str] = None,
) -> int:
  """
  Launch the dashboard compositor.

  When running on a headless builder or when no Wayland compositor is
  available, the fallback curses dashboard is used automatically.  The caller
  receives the exit code from the compositor application.
  """

  config = resolve_endpoint(endpoint)
  endpoint_uri = config.as_uri()

  if headless:
    return run_dashboard(endpoint_uri, headless=True)

  binary = _detect_compositor_binary(compositor)
  if not binary:
    print("[warn] No Wayland compositor detected; switching to curses dashboard.")
    return run_dashboard(endpoint_uri, headless=True)

  try:
    proc = _launch_external_compositor(binary)
  except Exception as exc:
    print(f"[warn] Unable to launch compositor '{binary}': {exc}. Falling back to curses.")
    return run_dashboard(endpoint_uri, headless=True)

  try:
    return run_dashboard(endpoint_uri, headless=False)
  finally:
    proc.terminate()
    try:
      proc.wait(timeout=5)
    except Exception:
      proc.kill()

