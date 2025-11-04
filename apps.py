"""
Application supervisor utilities for AgentaOS.

This module provides dataclasses and an asyncio-based scheduler that can launch
multiple tools concurrently, capture their output, and enforce simple restart
policies.  It intentionally keeps everything in-memory so forensic mode can
disable execution without mutating host state.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


MAX_CAPTURE_BYTES = 8192
DEFAULT_CONCURRENCY = 10


class AppConfigurationError(RuntimeError):
    """Raised when the application supervisor configuration is invalid."""


@dataclass
class AppSpec:
    name: str
    mode: str = "process"  # process|docker|podman
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    image: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    restart: str = "never"  # never|on-failure|always
    max_restarts: Optional[int] = None
    volumes: List[str] = field(default_factory=list)
    nice: Optional[int] = None
    docker_cpus: Optional[str] = None
    docker_memory: Optional[str] = None

    def normalised_command(self) -> List[str]:
        if self.mode != "process":
            raise AppConfigurationError("normalised_command is only valid for process mode.")
        if not self.command:
            raise AppConfigurationError(f"App '{self.name}' missing command.")
        command = list(self.command)
        if self.args:
            command.extend(self.args)
        return command


def _normalise_command(value: Optional[Iterable[str] | str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return shlex.split(value)
    return list(value)


def _normalise_args(value: Optional[Iterable[str] | str]) -> Optional[List[str]]:
    return _normalise_command(value)


def load_app_specs(config_path: Optional[str]) -> List[AppSpec]:
    if not config_path:
        return []
    path = Path(config_path).expanduser()
    if not path.exists():
        raise AppConfigurationError(f"App config '{path}' does not exist.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AppConfigurationError(f"Failed to parse app config '{path}': {exc}") from exc
    if not isinstance(data, list):
        raise AppConfigurationError("App config must be a list of application definitions.")

    specs: List[AppSpec] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise AppConfigurationError("Each application entry must be an object.")
        name = entry.get("name")
        if not name:
            raise AppConfigurationError("Application entry missing 'name'.")
        mode = (entry.get("mode") or "process").lower()
        command = _normalise_command(entry.get("command"))
        args = _normalise_args(entry.get("args"))
        image = entry.get("image")
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise AppConfigurationError(f"Application '{name}' has invalid 'env' field.")
        restart = (entry.get("restart") or "never").lower()
        max_restarts = entry.get("max_restarts")
        volumes = entry.get("volumes") or []
        if not isinstance(volumes, list):
            raise AppConfigurationError(f"Application '{name}' has invalid 'volumes' field.")
        nice = entry.get("nice")
        docker_cpus = entry.get("docker_cpus")
        docker_memory = entry.get("docker_memory")
        if nice is not None:
            try:
                nice = int(nice)
            except (TypeError, ValueError):
                raise AppConfigurationError(f"Application '{name}' has invalid nice value '{nice}'.")

        spec = AppSpec(
            name=name,
            mode=mode,
            command=command,
            args=args,
            image=image,
            env={str(k): str(v) for k, v in env.items()},
            cwd=entry.get("cwd"),
            restart=restart,
            max_restarts=max_restarts,
            volumes=[str(v) for v in volumes],
            nice=nice,
            docker_cpus=str(docker_cpus) if docker_cpus is not None else None,
            docker_memory=str(docker_memory) if docker_memory is not None else None,
        )
        specs.append(spec)
    return specs


class SupervisorScheduler:
    """
    In-memory supervisor that launches applications with concurrency limits.

    Records stdout/stderr (truncated), exit codes, and restart counts.  When
    forensic mode is enabled, execution is skipped and results are marked as
    such.
    """

    def __init__(
        self,
        specs: List[AppSpec],
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        base_env: Optional[Dict[str, str]] = None,
        forensic_mode: bool = False,
    ) -> None:
        self.specs = specs
        self.concurrency = max(1, concurrency) if specs else 1
        self.base_env = base_env or os.environ.copy()
        self.forensic_mode = forensic_mode

    async def run(self) -> Dict[str, object]:
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [asyncio.create_task(self._run_spec(spec, sem)) for spec in self.specs]
        records_nested = await asyncio.gather(*tasks, return_exceptions=False)
        records = [record for sublist in records_nested for record in sublist]

        summary = {
            "total_specs": len(self.specs),
            "total_runs": len(records),
            "completed": sum(1 for rec in records if rec.get("status") == "completed"),
            "failed": sum(1 for rec in records if rec.get("status") == "failed"),
            "errors": sum(1 for rec in records if rec.get("status") == "error"),
            "skipped": sum(1 for rec in records if rec.get("status") == "skipped"),
        }
        total_specs = summary["total_specs"]
        completed = summary["completed"]
        summary["success_ratio"] = (completed / total_specs) if total_specs else None
        return {"results": records, "summary": summary}

    async def _run_spec(self, spec: AppSpec, sem: asyncio.Semaphore) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        max_restarts = spec.max_restarts
        if max_restarts is None:
            if spec.restart == "always":
                max_restarts = 1
            elif spec.restart == "on-failure":
                max_restarts = 1
            else:
                max_restarts = 0

        attempt = 0
        while True:
            record = {
                "name": spec.name,
                "mode": spec.mode,
                "attempt": attempt,
            }
            if self.forensic_mode:
                record.update({"status": "skipped", "reason": "forensic_mode"})
                results.append(record)
                break

            async with sem:
                try:
                    command = self._build_command(spec)
                except AppConfigurationError as exc:
                    record.update({"status": "error", "message": str(exc)})
                    results.append(record)
                    break

                if command:
                    record["command"] = " ".join(command)

                env = self.base_env.copy()
                env.update(spec.env)
                preexec = None
                if spec.mode == "process" and spec.nice is not None and os.name != "nt":
                    def _set_nice(value: int):
                        def inner():
                            try:
                                os.nice(value)
                            except OSError:
                                pass
                        return inner
                    preexec = _set_nice(spec.nice)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=spec.cwd,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        preexec_fn=preexec,
                    )
                    record["pid"] = proc.pid
                    stdout, stderr = await proc.communicate()
                    record.update(
                        {
                            "returncode": proc.returncode,
                            "stdout": stdout.decode("utf-8", errors="ignore")[:MAX_CAPTURE_BYTES],
                            "stderr": stderr.decode("utf-8", errors="ignore")[:MAX_CAPTURE_BYTES],
                        }
                    )
                    if proc.returncode == 0:
                        record["status"] = "completed"
                    else:
                        record["status"] = "failed"
                except FileNotFoundError as exc:
                    record.update({"status": "error", "message": f"Command not found: {exc}"})
                except Exception as exc:
                    record.update({"status": "error", "message": str(exc)})

            results.append(record)

            should_restart = False
            if record["status"] == "completed":
                should_restart = spec.restart == "always"
            elif record["status"] in {"failed", "error"}:
                should_restart = spec.restart in {"on-failure", "always"}

            if not should_restart or attempt >= max_restarts:
                break

            attempt += 1

        return results

    def _build_command(self, spec: AppSpec) -> List[str]:
        mode = spec.mode
        if mode == "process":
            return spec.normalised_command()
        if mode in {"docker", "podman"}:
            binary = shutil.which(mode)
            if not binary:
                raise AppConfigurationError(f"{mode} binary not found on PATH.")
            if not spec.image:
                raise AppConfigurationError(f"App '{spec.name}' missing container image.")
            command = [binary, "run", "--rm"]
            for key, value in spec.env.items():
                command.extend(["-e", f"{key}={value}"])
            for volume in spec.volumes:
                command.extend(["-v", volume])
            if spec.docker_cpus:
                command.extend(["--cpus", spec.docker_cpus])
            if spec.docker_memory:
                command.extend(["--memory", spec.docker_memory])
            command.append(spec.image)
            if spec.args:
                command.extend(spec.args)
            return command
        raise AppConfigurationError(f"Unsupported mode '{mode}' for app '{spec.name}'.")
