"""Tkinter GUI for the SkyBreaker wireless audit orchestrator."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox
from typing import Dict, List, Optional

from .skybreaker import (
  AUDIT_PROFILES,
  TOOL_NAME,
  run_analysis,
  run_capture,
)


class SkyBreakerApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("SkyBreaker")
    self.geometry("900x620")
    self.resizable(True, True)

    self._capture_payload: Optional[Dict[str, object]] = None
    self._analysis_payload: Optional[Dict[str, object]] = None
    self._capture_thread: Optional[threading.Thread] = None
    self._capture_running = False
    self._advanced_visible = False

    self._build_ui()

  def _build_ui(self) -> None:
    notebook = ttk.Notebook(self)
    notebook.pack(fill=tk.BOTH, expand=True)

    self.capture_tab = ttk.Frame(notebook, padding=10)
    self.analyze_tab = ttk.Frame(notebook, padding=10)
    notebook.add(self.capture_tab, text="Capture")
    notebook.add(self.analyze_tab, text="Analyze")

    self._build_capture_tab()
    self._build_analyze_tab()

  def _build_capture_tab(self) -> None:
    frame = self.capture_tab

    header = ttk.Frame(frame)
    header.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(header, text="Interface").grid(row=0, column=0, sticky=tk.W)
    self.iface_var = tk.StringVar(value="wlan0")
    ttk.Entry(header, textvariable=self.iface_var, width=16).grid(row=1, column=0, sticky=tk.W)

    ttk.Label(header, text="Profile").grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
    self.profile_var = tk.StringVar(value="audit")
    self.profile_combo = ttk.Combobox(header, textvariable=self.profile_var, values=list(AUDIT_PROFILES.keys()), state="readonly", width=14)
    self.profile_combo.grid(row=1, column=1, sticky=tk.W, padx=(12, 0))
    self.profile_combo.bind("<<ComboboxSelected>>", self._update_profile_description)

    ttk.Label(header, text="Scan file (CSV)").grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
    self.scan_file_var = tk.StringVar()
    scan_entry = ttk.Entry(header, textvariable=self.scan_file_var)
    scan_entry.grid(row=1, column=2, sticky=tk.EW, padx=(12, 0))
    ttk.Button(header, text="Browse", command=self._pick_scan_file).grid(row=1, column=3, padx=(6, 0))

    header.columnconfigure(2, weight=1)

    self.profile_desc = ttk.Label(frame, text=AUDIT_PROFILES.get("audit", ""), foreground="#555")
    self.profile_desc.pack(fill=tk.X)

    self.advanced_button = ttk.Button(frame, text="Show advanced ▼", command=self._toggle_advanced)
    self.advanced_button.pack(anchor=tk.W, pady=(10, 0))

    self.advanced_frame = ttk.Frame(frame)

    ttk.Label(self.advanced_frame, text="Channel filter (0 = all)").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
    self.channel_var = tk.IntVar(value=0)
    ttk.Spinbox(self.advanced_frame, from_=0, to=196, increment=1, textvariable=self.channel_var, width=6).grid(row=1, column=0, sticky=tk.W)

    ttk.Label(self.advanced_frame, text="Notes").grid(row=0, column=1, sticky=tk.W)
    self.notes_text = tk.Text(self.advanced_frame, height=3, width=40)
    self.notes_text.grid(row=1, column=1, sticky=tk.EW)
    self.advanced_frame.columnconfigure(1, weight=1)

    action_row = ttk.Frame(frame)
    action_row.pack(fill=tk.X, pady=10)
    self.capture_button = ttk.Button(action_row, text="Run Capture", command=self._run_capture)
    self.capture_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Export JSON", command=self._export_capture_json).pack(side=tk.RIGHT)

    columns = ("ssid", "bssid", "signal", "channel", "security")
    self.capture_tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
    headings = {
      "ssid": "SSID",
      "bssid": "BSSID",
      "signal": "Signal (dBm)",
      "channel": "Channel",
      "security": "Security",
    }
    for column, title in headings.items():
      self.capture_tree.heading(column, text=title)
    self.capture_tree.column("ssid", width=220)
    self.capture_tree.column("bssid", width=140)
    self.capture_tree.column("signal", width=120, anchor=tk.CENTER)
    self.capture_tree.column("channel", width=80, anchor=tk.CENTER)
    self.capture_tree.column("security", width=120, anchor=tk.CENTER)
    self.capture_tree.pack(fill=tk.BOTH, expand=True)

    self.capture_status = tk.StringVar(value="Ready to capture.")
    ttk.Label(frame, textvariable=self.capture_status, foreground="#555").pack(fill=tk.X, pady=(6, 0))

  def _build_analyze_tab(self) -> None:
    frame = self.analyze_tab

    top = ttk.Frame(frame)
    top.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(top, text="Capture file").grid(row=0, column=0, sticky=tk.W)
    self.analysis_file_var = tk.StringVar()
    ttk.Entry(top, textvariable=self.analysis_file_var).grid(row=1, column=0, sticky=tk.EW, padx=(0, 8))
    ttk.Button(top, text="Browse", command=self._pick_analysis_file).grid(row=1, column=1)
    top.columnconfigure(0, weight=1)

    analysis_actions = ttk.Frame(frame)
    analysis_actions.pack(fill=tk.X, pady=(0, 10))
    ttk.Button(analysis_actions, text="Run Analysis", command=self._run_analysis).pack(side=tk.LEFT)
    ttk.Button(analysis_actions, text="Export JSON", command=self._export_analysis_json).pack(side=tk.RIGHT)

    self.analysis_tree = ttk.Treeview(frame, columns=("metric", "value"), show="headings", height=10)
    self.analysis_tree.heading("metric", text="Metric")
    self.analysis_tree.heading("value", text="Value")
    self.analysis_tree.column("metric", width=240)
    self.analysis_tree.column("value", width=240)
    self.analysis_tree.pack(fill=tk.BOTH, expand=True)

    self.analysis_status = tk.StringVar(value="Awaiting capture file.")
    ttk.Label(frame, textvariable=self.analysis_status, foreground="#555").pack(fill=tk.X, pady=(6, 0))

  def _update_profile_description(self, _event=None) -> None:
    self.profile_desc.configure(text=AUDIT_PROFILES.get(self.profile_var.get(), ""))

  def _pick_scan_file(self) -> None:
    path = filedialog.askopenfilename(title="Select scan CSV", filetypes=[("CSV", "*.csv"), ("All files", "*")])
    if path:
      self.scan_file_var.set(path)

  def _toggle_advanced(self) -> None:
    if self._advanced_visible:
      self.advanced_frame.pack_forget()
      self.advanced_button.configure(text="Show advanced ▼")
    else:
      self.advanced_frame.pack(fill=tk.X, pady=(6, 8))
      self.advanced_button.configure(text="Hide advanced ▲")
    self._advanced_visible = not self._advanced_visible

  def _run_capture(self) -> None:
    if self._capture_running:
      messagebox.showinfo(TOOL_NAME, "Capture already in progress.")
      return

    iface = self.iface_var.get().strip() or "wlan0"
    profile = self.profile_var.get() or "audit"
    channel = int(self.channel_var.get())
    scan_file = self.scan_file_var.get().strip() or None

    self.capture_status.set("Capturing…")
    self._capture_running = True
    self.capture_button.configure(state=tk.DISABLED)

    def worker() -> None:
      try:
        diagnostic, payload = run_capture(
          iface,
          profile,
          channel,
          scan_file=scan_file,
        )
        self.after(0, self._render_capture_results, diagnostic.summary, payload)
      except Exception as exc:  # pylint: disable=broad-except
        self.after(0, self._capture_error, str(exc))

    self._capture_thread = threading.Thread(target=worker, daemon=True)
    self._capture_thread.start()

  def _render_capture_results(self, summary: str, payload: Dict[str, object]) -> None:
    self._capture_payload = payload
    for item in self.capture_tree.get_children():
      self.capture_tree.delete(item)
    networks: List[Dict[str, object]] = payload.get("networks", [])  # type: ignore[assignment]
    for record in networks:
      self.capture_tree.insert(
        "",
        tk.END,
        values=(
          record.get("ssid", "") or "<hidden>",
          record.get("bssid", ""),
          record.get("signal", ""),
          record.get("channel", ""),
          record.get("security", ""),
        ),
      )
    self.capture_status.set(summary)
    self.capture_button.configure(state=tk.NORMAL)
    self._capture_running = False

  def _capture_error(self, message: str) -> None:
    messagebox.showerror(TOOL_NAME, message)
    self.capture_status.set("Capture failed.")
    self.capture_button.configure(state=tk.NORMAL)
    self._capture_running = False

  def _export_capture_json(self) -> None:
    if not self._capture_payload:
      messagebox.showinfo(TOOL_NAME, "Run a capture before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._capture_payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Capture exported to {path}")

  def _pick_analysis_file(self) -> None:
    path = filedialog.askopenfilename(title="Select capture JSON", filetypes=[("JSON", "*.json"), ("All files", "*")])
    if path:
      self.analysis_file_var.set(path)

  def _run_analysis(self) -> None:
    path = self.analysis_file_var.get().strip()
    if not path:
      messagebox.showwarning(TOOL_NAME, "Select a capture JSON file.")
      return
    try:
      diagnostic, payload = run_analysis(path)
    except FileNotFoundError:
      messagebox.showerror(TOOL_NAME, "Capture file not found.")
      return
    except json.JSONDecodeError:
      messagebox.showerror(TOOL_NAME, "Capture file is not valid JSON.")
      return

    self._analysis_payload = payload
    for item in self.analysis_tree.get_children():
      self.analysis_tree.delete(item)
    findings = payload.get("findings", {})
    for key, value in findings.items():
      self.analysis_tree.insert("", tk.END, values=(key.replace("_", " ").title(), value))

    self.analysis_status.set(diagnostic.summary)

  def _export_analysis_json(self) -> None:
    if not self._analysis_payload:
      messagebox.showinfo(TOOL_NAME, "Run an analysis before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._analysis_payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Analysis exported to {path}")


def launch() -> None:
  app = SkyBreakerApp()
  app.mainloop()
