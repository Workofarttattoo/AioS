"""
Lightweight IPC bus used by the AgentaOS compositor.

The bus intentionally avoids heavyweight dependencies (ZeroMQ, DBus) so it can
operate inside constrained live-build environments.  It prefers Unix-domain
sockets when available and falls back to TCP loopback streams on platforms
without AF_UNIX support (e.g. Windows).  Messages are newline-delimited JSON
documents.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

from .schema import DashboardDescriptor


DEFAULT_ENDPOINT = Path(tempfile.gettempdir()) / "agentaos-display.sock"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 59483


EndpointType = Union[str, Path, Tuple[str, int]]


@dataclass(frozen=True)
class IPCConfig:
  """Resolved endpoint configuration."""

  mode: str
  path: Optional[Path] = None
  host: Optional[str] = None
  port: Optional[int] = None

  def as_uri(self) -> str:
    if self.mode == "unix" and self.path:
      return f"unix://{self.path}"
    if self.mode == "tcp" and self.host is not None and self.port is not None:
      return f"tcp://{self.host}:{self.port}"
    raise ValueError("Incomplete IPC configuration.")


def resolve_endpoint(endpoint: Optional[EndpointType] = None) -> IPCConfig:
  """
  Normalise endpoint input into an IPCConfig instance.

  Supported forms:
    - ``None``: uses the platform default (Unix socket or TCP loopback).
    - ``Path`` or ``str`` pointing to a filesystem location: treated as Unix socket.
    - ``'unix:///tmp/socket'``: explicit Unix socket URI.
    - ``'tcp://127.0.0.1:5001'`` or ``'127.0.0.1:5001'``: TCP loopback.
    - ``(host, port)`` tuple: TCP loopback.
  """

  if endpoint is None:
    if hasattr(socket, "AF_UNIX") and os.name != "nt":
      return IPCConfig(mode="unix", path=DEFAULT_ENDPOINT)
    return IPCConfig(mode="tcp", host=DEFAULT_TCP_HOST, port=DEFAULT_TCP_PORT)

  if isinstance(endpoint, Path):
    return IPCConfig(mode="unix", path=endpoint)

  if isinstance(endpoint, tuple):
    host, port = endpoint
    return IPCConfig(mode="tcp", host=host, port=port)

  # Interpret string inputs.
  if isinstance(endpoint, str):
    if endpoint.startswith("unix://"):
      return IPCConfig(mode="unix", path=Path(endpoint[len("unix://") :]))
    if endpoint.startswith("tcp://"):
      host_port = endpoint[len("tcp://") :]
      host, port_str = host_port.split(":", maxsplit=1)
      return IPCConfig(mode="tcp", host=host, port=int(port_str))
    if os.path.sep in endpoint or endpoint.startswith("/"):
      return IPCConfig(mode="unix", path=Path(endpoint))
    if ":" in endpoint:
      host, port_str = endpoint.split(":", maxsplit=1)
      return IPCConfig(mode="tcp", host=host, port=int(port_str))

  raise ValueError(f"Unsupported endpoint value: {endpoint!r}")


class SchemaPublisher:
  """Broadcast dashboard descriptors to subscribed compositors."""

  def __init__(
    self,
    endpoint: Optional[EndpointType] = None,
    *,
    backlog: int = 5,
    on_command: Optional[Callable[[Dict[str, object]], None]] = None,
  ) -> None:
    self.config = resolve_endpoint(endpoint)
    self.backlog = backlog
    self.on_command = on_command
    self._server_socket: Optional[socket.socket] = None
    self._clients: set[socket.socket] = set()
    self._clients_lock = threading.Lock()
    self._thread: Optional[threading.Thread] = None
    self._running = threading.Event()
    self._client_threads: set[threading.Thread] = set()
    self._client_threads_lock = threading.Lock()

  @property
  def uri(self) -> str:
    return self.config.as_uri()

  def start(self) -> None:
    if self._thread and self._thread.is_alive():
      return
    self._running.set()
    self._server_socket = self._create_socket()
    self._thread = threading.Thread(target=self._serve, name="SchemaPublisher", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._running.clear()
    if self._server_socket:
      try:
        self._server_socket.close()
      finally:
        self._server_socket = None

    with self._clients_lock:
      for client in list(self._clients):
        try:
          client.close()
        finally:
          self._clients.discard(client)

    with self._client_threads_lock:
      for thread in list(self._client_threads):
        thread.join(timeout=0.5)
        self._client_threads.discard(thread)

    if self.config.mode == "unix" and self.config.path:
      try:
        os.unlink(self.config.path)
      except OSError:
        pass

    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=0.5)
      self._thread = None

  def publish(self, descriptor: Union[DashboardDescriptor, Dict[str, object]]) -> None:
    data = descriptor.as_payload() if isinstance(descriptor, DashboardDescriptor) else descriptor
    payload = json.dumps(data) + "\n"
    message = payload.encode("utf-8")

    dead_clients: list[socket.socket] = []
    with self._clients_lock:
      for client in self._clients:
        try:
          client.sendall(message)
        except OSError:
          dead_clients.append(client)
      for client in dead_clients:
        try:
          client.close()
        finally:
          self._clients.discard(client)

  # Internal helpers -----------------------------------------------------

  def _create_socket(self) -> socket.socket:
    if self.config.mode == "unix":
      if not self.config.path:
        raise ValueError("Unix socket configuration requires a path.")
      sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      try:
        if self.config.path.exists():
          os.unlink(self.config.path)
      except OSError:
        pass
      sock.bind(str(self.config.path))
    else:
      host = self.config.host or DEFAULT_TCP_HOST
      port = self.config.port or DEFAULT_TCP_PORT
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      sock.bind((host, port))
      # Update stored port if ephemeral requested.
      if self.config.port in (None, 0):
        _, resolved_port = sock.getsockname()
        object.__setattr__(self.config, "port", resolved_port)
    sock.listen(self.backlog)
    sock.settimeout(0.5)
    return sock

  def _serve(self) -> None:
    assert self._server_socket is not None
    while self._running.is_set():
      try:
        client, _ = self._server_socket.accept()
        client.setblocking(True)
        with self._clients_lock:
          self._clients.add(client)
        handler = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
        with self._client_threads_lock:
          self._client_threads.add(handler)
        handler.start()
      except socket.timeout:
        continue
      except OSError:
        break

  def _handle_client(self, client: socket.socket) -> None:
    file = client.makefile("r", encoding="utf-8")
    try:
      while self._running.is_set():
        line = file.readline()
        if not line:
          break
        try:
          message = json.loads(line)
        except json.JSONDecodeError:
          continue
        if self.on_command:
          try:
            self.on_command(message)
          except Exception:
            continue
    finally:
      file.close()
      with self._clients_lock:
        if client in self._clients:
          try:
            client.close()
          finally:
            self._clients.discard(client)
      with self._client_threads_lock:
        self._client_threads.discard(threading.current_thread())


class SchemaSubscriber:
  """Client that listens for dashboard descriptors."""

  def __init__(self, endpoint: Optional[EndpointType] = None, *, reconnect_delay: float = 1.0) -> None:
    self.config = resolve_endpoint(endpoint)
    self.reconnect_delay = reconnect_delay
    self._socket: Optional[socket.socket] = None
    self._file = None
    self._send_lock = threading.Lock()

  def connect(self) -> None:
    while True:
      try:
        self._socket = self._create_socket()
        if self.config.mode == "unix":
          assert self.config.path is not None
          self._socket.connect(str(self.config.path))
        else:
          assert self.config.host is not None and self.config.port is not None
          self._socket.connect((self.config.host, self.config.port))
        self._file = self._socket.makefile("r", encoding="utf-8")
        return
      except (FileNotFoundError, ConnectionRefusedError, OSError):
        time.sleep(self.reconnect_delay)

  def iter_messages(self) -> Iterator[Dict[str, object]]:
    while True:
      if not self._socket or not self._file:
        self.connect()
      assert self._file is not None
      line = self._file.readline()
      if not line:
        self.close()
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        continue

  def close(self) -> None:
    if self._file:
      try:
        self._file.close()
      finally:
        self._file = None
    if self._socket:
      try:
        self._socket.close()
      finally:
        self._socket = None

  def send(self, payload: Dict[str, object]) -> None:
    message = json.dumps(payload) + "\n"
    encoded = message.encode("utf-8")
    while True:
      if not self._socket:
        self.connect()
      if not self._socket:
        time.sleep(self.reconnect_delay)
        continue
      try:
        with self._send_lock:
          self._socket.sendall(encoded)
        return
      except OSError:
        self.close()
        time.sleep(self.reconnect_delay)

  def send_action(self, hook: str, parameters: Optional[Dict[str, object]] = None) -> None:
    payload = {
      "type": "action",
      "hook": hook,
      "parameters": parameters or {},
    }
    self.send(payload)

  def __iter__(self) -> Iterator[Dict[str, object]]:
    return self.iter_messages()

  def __enter__(self) -> "SchemaSubscriber":
    self.connect()
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.close()
