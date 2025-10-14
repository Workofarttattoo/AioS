"""
Minimal compositor client for AgentaOS dashboards.

The client listens for dashboard schema updates on the IPC bus and renders them
using PyGObject when available.  If the graphical backend is unavailable the
client falls back to a curses dashboard so headless targets still receive a
live summary.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from typing import Dict, Iterable, List, Optional

from aios.gui.bus import SchemaSubscriber
from aios.gui.schema import DashboardDescriptor


def _load_descriptor(payload) -> DashboardDescriptor:
  if hasattr(DashboardDescriptor, "model_validate"):
    return DashboardDescriptor.model_validate(payload)  # type: ignore[attr-defined]
  return DashboardDescriptor.parse_obj(payload)  # type: ignore[attr-defined]


def _format_descriptor(descriptor: DashboardDescriptor) -> str:
  return "\n".join(_format_descriptor_lines(descriptor))


def _format_descriptor_lines(descriptor: DashboardDescriptor) -> List[str]:
  lines: List[str] = [
    f"AgentaOS Dashboard · worker={descriptor.worker} · version={descriptor.version}",
  ]
  if descriptor.annotations:
    for key, value in descriptor.annotations.items():
      lines.append(f"  {key}: {value}")
  if not descriptor.panels:
    lines.append("  (no panels registered)")
    return lines

  for panel in descriptor.panels:
    lines.append("")
    header = f"[{panel.title}]"
    if panel.description:
      header += f" – {panel.description}"
    lines.append(header)
    if panel.metrics:
      for metric in panel.metrics:
        suffix = f" {metric.unit}" if metric.unit else ""
        lines.append(f"  • {metric.label}: {metric.value}{suffix} ({metric.severity})")
    if panel.tables:
      for table in panel.tables:
        lines.append(f"  • Table: {table.title} columns={', '.join(table.columns)}")
        if not table.rows:
          lines.append(f"      {table.empty_state or '(no rows)'}")
        else:
          sample = table.rows[:5]
          for row in sample:
            cells = ", ".join(f"{key}={value}" for key, value in row.cells.items())
            lines.append(f"      {cells}")
          if len(table.rows) > len(sample):
            lines.append(f"      … {len(table.rows) - len(sample)} more rows")
    if panel.actions:
      lines.append("  • Actions:")
      for action in panel.actions:
        lines.append(f"      [{action.id}] {action.label} -> {action.runtime_hook}")
  return lines


def _run_curses_dashboard(subscriber: SchemaSubscriber) -> int:
  import curses

  lock = threading.Lock()
  state_lines = ["Waiting for dashboard schema..."]
  state_actions: List[Dict[str, str]] = []
  stop_event = threading.Event()

  def pump() -> None:
    for message in subscriber.iter_messages():
      descriptor = _load_descriptor(message)
      formatted = _format_descriptor_lines(descriptor)
      actions: List[Dict[str, str]] = []
      for panel in descriptor.panels:
        for action in panel.actions:
          actions.append({"hook": action.runtime_hook, "label": action.label})
      with lock:
        state_lines[:] = formatted
        state_actions[:] = actions
      if stop_event.is_set():
        break

  thread = threading.Thread(target=pump, name="DashboardPump", daemon=True)
  thread.start()

  def draw(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    while not stop_event.is_set():
      stdscr.erase()
      with lock:
        snapshot = list(state_lines)
        actions_snapshot = list(state_actions)
      for idx, line in enumerate(snapshot):
        if idx >= curses.LINES - 2:
          break
        try:
          stdscr.addnstr(idx, 0, line, curses.COLS - 1)
        except curses.error:
          continue
      hint_row = min(len(snapshot) + 1, curses.LINES - 2)
      if actions_snapshot:
        stdscr.addnstr(hint_row, 0, "Actions:", curses.COLS - 1)
        for idx, action in enumerate(actions_snapshot[:9]):
          row = hint_row + idx + 1
          if row >= curses.LINES - 2:
            break
          label = f"  {idx + 1}. {action['label']}"
          try:
            stdscr.addnstr(row, 0, label, curses.COLS - 1)
          except curses.error:
            continue
      stdscr.addnstr(curses.LINES - 2, 0, "Press 'q' to exit dashboard.", curses.COLS - 1)
      stdscr.refresh()
      ch = stdscr.getch()
      if ch in (ord("q"), ord("Q")):
        stop_event.set()
        break
      if ord("1") <= ch <= ord("9"):
        idx = ch - ord("1")
        with lock:
          if 0 <= idx < len(state_actions):
            hook = state_actions[idx]["hook"]
            try:
              subscriber.send_action(hook)
            except Exception:
              pass
      time.sleep(0.1)

  try:
    curses.wrapper(draw)
  finally:
    stop_event.set()
    subscriber.close()
    thread.join(timeout=1.0)
  return 0


def _run_gtk_dashboard(subscriber: SchemaSubscriber) -> int:
  try:
    from gi.repository import GLib, Gtk  # type: ignore
  except Exception:
    print("[warn] PyGObject not available; falling back to curses dashboard.")
    return _run_curses_dashboard(subscriber)

  updates: "queue.Queue[DashboardDescriptor]" = queue.Queue()
  stop_event = threading.Event()

  def pump() -> None:
    for message in subscriber.iter_messages():
      descriptor = _load_descriptor(message)
      updates.put(descriptor)
      if stop_event.is_set():
        break

  thread = threading.Thread(target=pump, name="DashboardPump", daemon=True)
  thread.start()

  class DashboardWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, subscriber: SchemaSubscriber) -> None:
      super().__init__(application=app)
      self.subscriber = subscriber
      self.set_title("AgentaOS Dashboard")
      self.set_default_size(1024, 640)
      root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
      root.set_border_width(12)
      scroller = Gtk.ScrolledWindow()
      self.text_view = Gtk.TextView()
      self.text_view.set_editable(False)
      self.text_view.set_monospace(True)
      scroller.add(self.text_view)
      if hasattr(root, "append"):
        root.append(scroller)
      else:
        root.pack_start(scroller, True, True, 0)
      self.button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
      if hasattr(root, "append"):
        root.append(self.button_box)
      else:
        root.pack_start(self.button_box, False, False, 0)
      self.add(root)
      self.show_all()

    def update_descriptor(self, descriptor: DashboardDescriptor) -> None:
      text = _format_descriptor(descriptor)
      buffer = self.text_view.get_buffer()
      buffer.set_text(text)
      self._render_actions(descriptor)

    def _clear_button_box(self) -> None:
      for child in self._iter_children():
        self.button_box.remove(child)

    def _iter_children(self):
      if hasattr(self.button_box, "get_children"):
        return list(self.button_box.get_children())
      if hasattr(self.button_box, "get_first_child"):
        children = []
        child = self.button_box.get_first_child()
        while child:
          children.append(child)
          child = child.get_next_sibling()
        return children
      return []

    def _append_child(self, widget) -> None:
      if hasattr(self.button_box, "append"):
        self.button_box.append(widget)
      else:
        self.button_box.pack_start(widget, False, False, 0)

    def _render_actions(self, descriptor: DashboardDescriptor) -> None:
      self._clear_button_box()
      actions = []
      for panel in descriptor.panels:
        actions.extend(panel.actions)
      if not actions:
        label = Gtk.Label(label="No actions registered.")
        self._append_child(label)
      else:
        for action in actions:
          button = Gtk.Button(label=action.label)
          button.connect("clicked", self._on_action_clicked, action.runtime_hook)
          self._append_child(button)
      self.button_box.show_all()

    def _on_action_clicked(self, _button, hook: str) -> None:
      try:
        self.subscriber.send_action(hook)
      except Exception as exc:
        print(f"[warn] Failed to dispatch action '{hook}': {exc}")

  class DashboardApplication(Gtk.Application):
    def __init__(self, subscriber: SchemaSubscriber) -> None:
      super().__init__(application_id="com.agentaos.dashboard")
      self.window: DashboardWindow | None = None
      self.subscriber = subscriber

    def do_activate(self) -> None:  # type: ignore[override]
      if not self.window:
        self.window = DashboardWindow(self, self.subscriber)
      self.window.present()
      GLib.timeout_add(200, self._drain_updates)

    def _drain_updates(self) -> bool:
      if stop_event.is_set():
        return False
      try:
        while True:
          descriptor = updates.get_nowait()
          if self.window:
            self.window.update_descriptor(descriptor)
      except queue.Empty:
        pass
      return True

    def do_shutdown(self) -> None:  # type: ignore[override]
      stop_event.set()
      subscriber.close()
      super().do_shutdown()

  app = DashboardApplication(subscriber)
  try:
    return app.run()
  finally:
    stop_event.set()
    subscriber.close()
    thread.join(timeout=1.0)


def main(argv: Iterable[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="AgentaOS compositor client.")
  parser.add_argument("--endpoint", required=True, help="IPC endpoint URI (unix:// or tcp://).")
  parser.add_argument("--headless", action="store_true", help="Force curses fallback.")
  parser.add_argument(
    "--backend",
    choices=["auto", "gtk", "curses"],
    default="auto",
    help="Preferred rendering backend.",
  )
  args = parser.parse_args(list(argv) if argv is not None else None)

  subscriber = SchemaSubscriber(args.endpoint)
  if args.headless:
    return _run_curses_dashboard(subscriber)

  backend = args.backend
  if backend == "auto":
    backend = "gtk"

  if backend == "gtk":
    return _run_gtk_dashboard(subscriber)
  return _run_curses_dashboard(subscriber)


if __name__ == "__main__":
  raise SystemExit(main())
