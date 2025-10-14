"""Tkinter GUI for AuroraScan."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import List, Optional, Tuple

from .aurorascan import (
  PROFILE_DESCRIPTIONS,
  PORT_PROFILES,
  run_scan,
  parse_ports,
  parse_targets,
  load_targets_from_file,
  write_json,
  write_zap_targets,
)


class AuroraScanApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("AuroraScan")
    self.geometry("720x520")
    self.resizable(True, True)

    self._scan_thread: Optional[threading.Thread] = None
    self._stop_flag = threading.Event()
    self._progress_queue: queue.Queue = queue.Queue()
    self._reports = []

    self._build_ui()
    self.after(100, self._drain_queue)

  def _build_ui(self) -> None:
    main_frame = ttk.Frame(self, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    target_frame = ttk.LabelFrame(main_frame, text="Targets")
    target_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(target_frame, text="Hostnames / IPs (comma separated)").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    self.targets_entry = ttk.Entry(target_frame)
    self.targets_entry.grid(row=1, column=0, sticky=tk.EW, padx=5)
    target_frame.columnconfigure(0, weight=1)

    file_frame = ttk.Frame(target_frame)
    file_frame.grid(row=2, column=0, sticky=tk.EW, pady=2, padx=5)
    ttk.Label(file_frame, text="Targets file (optional)").pack(side=tk.LEFT)
    self.targets_file_var = tk.StringVar()
    ttk.Entry(file_frame, textvariable=self.targets_file_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(file_frame, text="Browse", command=self._pick_targets_file).pack(side=tk.LEFT)

    opts_frame = ttk.LabelFrame(main_frame, text="Options")
    opts_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(opts_frame, text="Profile").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
    self.profile_var = tk.StringVar(value="recon")
    profile_values = list(PORT_PROFILES.keys())
    self.profile_combo = ttk.Combobox(opts_frame, textvariable=self.profile_var, values=profile_values, state="readonly")
    self.profile_combo.grid(row=1, column=0, padx=5, sticky=tk.W)
    self.profile_combo.bind("<<ComboboxSelected>>", self._update_profile_description)

    ttk.Label(opts_frame, text="Extra ports (optional, e.g. 8080,4000-4010)").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
    self.extra_ports_entry = ttk.Entry(opts_frame)
    self.extra_ports_entry.grid(row=1, column=1, padx=5, sticky=tk.EW)

    ttk.Label(opts_frame, text="Timeout (s)").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
    self.timeout_var = tk.DoubleVar(value=1.5)
    ttk.Spinbox(opts_frame, textvariable=self.timeout_var, from_=0.5, to=10.0, increment=0.5, width=6).grid(row=1, column=2, padx=5)

    ttk.Label(opts_frame, text="Concurrency").grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
    self.concurrent_var = tk.IntVar(value=64)
    ttk.Spinbox(opts_frame, textvariable=self.concurrent_var, from_=1, to=512, increment=8, width=6).grid(row=1, column=3, padx=5)

    opts_frame.columnconfigure(1, weight=1)

    self.profile_desc = ttk.Label(opts_frame, text=PROFILE_DESCRIPTIONS.get("recon", ""), foreground="#555")
    self.profile_desc.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(4, 0))

    action_frame = ttk.Frame(main_frame)
    action_frame.pack(fill=tk.X, padx=5, pady=5)
    self.start_button = ttk.Button(action_frame, text="Start Scan", command=self._start_scan)
    self.start_button.pack(side=tk.LEFT)
    self.stop_button = ttk.Button(action_frame, text="Stop", command=self._stop_scan, state=tk.DISABLED)
    self.stop_button.pack(side=tk.LEFT, padx=(5, 0))
    ttk.Button(action_frame, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT, padx=(0, 5))
    ttk.Button(action_frame, text="Export ZAP URLs", command=self._export_zap).pack(side=tk.RIGHT, padx=(0, 5))

    table_frame = ttk.LabelFrame(main_frame, text="Results")
    table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    columns = ("target", "port", "status", "latency", "banner")
    self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    self.results_tree.heading("target", text="Target")
    self.results_tree.heading("port", text="Port")
    self.results_tree.heading("status", text="Status")
    self.results_tree.heading("latency", text="Latency (ms)")
    self.results_tree.heading("banner", text="Banner")
    self.results_tree.column("target", width=160)
    self.results_tree.column("port", width=60, anchor=tk.CENTER)
    self.results_tree.column("status", width=80, anchor=tk.CENTER)
    self.results_tree.column("latency", width=100, anchor=tk.CENTER)
    self.results_tree.column("banner", width=250)
    self.results_tree.pack(fill=tk.BOTH, expand=True)

  def _update_profile_description(self, _event=None) -> None:
    description = PROFILE_DESCRIPTIONS.get(self.profile_var.get(), "")
    self.profile_desc.configure(text=description)

  def _pick_targets_file(self) -> None:
    path = filedialog.askopenfilename(title="Select targets file")
    if path:
      self.targets_file_var.set(path)

  def _start_scan(self) -> None:
    if self._scan_thread and self._scan_thread.is_alive():
      messagebox.showwarning("Scan running", "A scan is already in progress.")
      return

    targets = parse_targets(self.targets_entry.get())
    extra_targets = load_targets_from_file(self.targets_file_var.get())
    targets.extend(extra_targets)
    if not targets:
      messagebox.showwarning("Targets required", "Enter at least one target or choose a targets file.")
      return

    base_ports = parse_ports(None, self.profile_var.get())
    extra_ports = parse_ports(self.extra_ports_entry.get(), self.profile_var.get()) if self.extra_ports_entry.get() else []
    ports = sorted(set(list(base_ports) + list(extra_ports)))
    if not ports:
      messagebox.showwarning("Invalid ports", "No ports were selected after parsing inputs.")
      return

    self._reports = []
    for item in self.results_tree.get_children():
      self.results_tree.delete(item)

    self._stop_flag = threading.Event()
    self._progress_queue = queue.Queue()

    timeout = float(self.timeout_var.get())
    concurrency = int(self.concurrent_var.get())

    def worker() -> None:
      try:
        reports = run_scan(
          targets,
          ports,
          timeout=timeout,
          concurrency=concurrency,
          progress_queue=self._progress_queue,
          stop_flag=self._stop_flag,
        )
        self._progress_queue.put(("__COMPLETE__", reports))
      except Exception as exc:  # pylint: disable=broad-except
        self._progress_queue.put(("__ERROR__", str(exc)))

    self.start_button.configure(state=tk.DISABLED)
    self.stop_button.configure(state=tk.NORMAL)

    self._scan_thread = threading.Thread(target=worker, daemon=True)
    self._scan_thread.start()

  def _stop_scan(self) -> None:
    if self._stop_flag:
      self._stop_flag.set()
    self.stop_button.configure(state=tk.DISABLED)

  def _drain_queue(self) -> None:
    try:
      while True:
        item = self._progress_queue.get_nowait()
        if isinstance(item, tuple) and item and item[0] == "__COMPLETE__":
          self._reports = item[1]
          self._on_scan_complete()
        elif isinstance(item, tuple) and item and item[0] == "__ERROR__":
          messagebox.showerror("Scan error", item[1])
          self._on_scan_complete()
        else:
          self._append_result(item)  # type: ignore[arg-type]
    except queue.Empty:
      pass
    finally:
      self.after(100, self._drain_queue)

  def _append_result(self, payload: Tuple[str, int, str, float, Optional[str]]) -> None:
    target, port, status, latency, banner = payload
    banner_display = (banner or "")[:80]
    self.results_tree.insert("", tk.END, values=(target, port, status, f"{latency:.2f}", banner_display))

  def _on_scan_complete(self) -> None:
    self.start_button.configure(state=tk.NORMAL)
    self.stop_button.configure(state=tk.DISABLED)
    if self._scan_thread and self._scan_thread.is_alive():
      self._scan_thread.join(timeout=0.1)

  def _export_json(self) -> None:
    if not self._reports:
      messagebox.showinfo("No data", "Run a scan before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if path:
      write_json(self._reports, path, tag="aurorascan-gui")

  def _export_zap(self) -> None:
    if not self._reports:
      messagebox.showinfo("No data", "Run a scan before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
    if path:
      write_zap_targets(self._reports, path, default_scheme="auto")


def launch() -> None:
  app = AuroraScanApp()
  app.mainloop()
