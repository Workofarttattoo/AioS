#!/usr/bin/env python3
"""
Bootstrap Sovereign Security Toolkit command wrappers.

The script creates a `bin/` directory inside the repository and copies or
symlinks the lightweight CLI wrappers so they can be added to PATH.  On Windows
it also generates `.cmd` launchers that invoke the wrappers via the active
Python interpreter.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "bin"
WRAPPERS: Iterable[str] = (
  "cipherspear",
  "skybreaker",
  "mythickey",
  "spectratrace",
  "nemesishydra",
  "obsidianhunt",
  "vectorflux",
)


def ensure_bin_dir() -> None:
  BIN_DIR.mkdir(exist_ok=True)


def install_wrapper(name: str) -> None:
  source = REPO_ROOT / name
  if not source.exists():
    print(f"[warn] Wrapper '{name}' not found; skipping.")
    return

  if os.name == "nt":
    destination = BIN_DIR / f"{name}.py"
    shutil.copy2(source, destination)
    launcher = BIN_DIR / f"{name}.cmd"
    launcher.write_text(
      "@echo off\r\n"
      f"\"{sys.executable}\" \"%~dp0\\{name}.py\" %*\r\n",
      encoding="utf-8",
    )
    return

  destination = BIN_DIR / name
  if destination.exists() or destination.is_symlink():
    destination.unlink()
  try:
    destination.symlink_to(source)
  except OSError:
    shutil.copy2(source, destination)


def main() -> int:
  ensure_bin_dir()
  for wrapper in WRAPPERS:
    install_wrapper(wrapper)

  print(f"[info] Sovereign toolkit wrappers installed under {BIN_DIR}.")
  if os.name == "nt":
    print("[info] Add the directory to your PATH and invoke the `.cmd` launchers.")
  else:
    print("[info] Add the directory to your PATH to run commands directly.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
