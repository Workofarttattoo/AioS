"""
Tkinter interface for the ObsidianHunt hardening assessor.

The GUI mirrors the CLI workflow: choose an assessment profile, optionally
load supplemental control definitions, run the baseline survey, and export the
resulting remediation manifest for supervisor consumption.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from .obsidianhunt import gather_baseline, load_additional_checks


DEFAULT_PROFILES = ["workstation", "server", "forensic"]


class ObsidianHuntApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("ObsidianHunt")
    self.geometry("720x520")
    self.resizable(True, True)

    self._baseline: Optional[Dict[str, object]] = None

    self._build_ui()

  def _build_ui(self) -> None:
    main_frame = ttk.Frame(self, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Profile selection
    profile_frame = ttk.LabelFrame(main_frame, text="Assessment Options")
    profile_frame.pack(fill=tk.X, padx=4, pady=6)

    ttk.Label(profile_frame, text="Profile").grid(row=0, column=0, padx=4, pady=2, sticky=tk.W)
    self.profile_var = tk.StringVar(value=DEFAULT_PROFILES[0])
    self.profile_combo = ttk.Combobox(
      profile_frame,
      textvariable=self.profile_var,
      values=DEFAULT_PROFILES,
      state="readonly",
      width=20,
    )
    self.profile_combo.grid(row=1, column=0, padx=4, pady=2, sticky=tk.W)

    ttk.Label(profile_frame, text="Additional checks (JSON)").grid(row=0, column=1, padx=4, pady=2, sticky=tk.W)
    self.checks_path_var = tk.StringVar()
    checks_entry = ttk.Entry(profile_frame, textvariable=self.checks_path_var)
    checks_entry.grid(row=1, column=1, padx=4, pady=2, sticky=tk.EW)
    ttk.Button(profile_frame, text="Browse…", command=self._pick_checks_file).grid(row=1, column=2, padx=4, pady=2)

    profile_frame.columnconfigure(1, weight=1)

    # Action buttons
    action_frame = ttk.Frame(main_frame)
    action_frame.pack(fill=tk.X, padx=4, pady=(0, 6))
    ttk.Button(action_frame, text="Run Assessment", command=self._run_assessment).pack(side=tk.LEFT)
    ttk.Button(action_frame, text="Export Manifest", command=self._export_manifest).pack(side=tk.RIGHT)

    # Summary area
    summary_frame = ttk.LabelFrame(main_frame, text="Summary")
    summary_frame.pack(fill=tk.X, padx=4, pady=6)
    self.summary_var = tk.StringVar(value="Run an assessment to populate the baseline results.")
    ttk.Label(summary_frame, textvariable=self.summary_var, foreground="#444").pack(anchor=tk.W, padx=6, pady=4)

    # Results tree
    results_frame = ttk.LabelFrame(main_frame, text="Controls")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 6))
    columns = ("control", "status", "evidence")
    self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
    self.results_tree.heading("control", text="Control")
    self.results_tree.heading("status", text="Status")
    self.results_tree.heading("evidence", text="Evidence")
    self.results_tree.column("control", width=160)
    self.results_tree.column("status", width=80, anchor=tk.CENTER)
    self.results_tree.column("evidence", width=360)
    self.results_tree.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
    self.results_tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

  def _pick_checks_file(self) -> None:
    path = filedialog.askopenfilename(
      title="Select additional checks JSON",
      filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if path:
      self.checks_path_var.set(path)

  def _run_assessment(self) -> None:
    profile = self.profile_var.get().strip() or "workstation"
    checks_path = self.checks_path_var.get().strip() or None

    try:
      extra_checks = load_additional_checks(checks_path)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
      messagebox.showerror("Invalid checks file", f"Failed to parse checks file: {exc}")
      return

    baseline = gather_baseline(profile, extra_checks)
    self._baseline = baseline
    self._render_baseline(baseline)

  def _render_baseline(self, baseline: Dict[str, object]) -> None:
    # Update summary text
    controls = baseline.get("controls", [])
    warnings = baseline.get("warnings", 0)
    summary = (
      f"Profile '{baseline.get('profile')}' on host {baseline.get('hostname')} "
      f"captured {len(controls)} control(s); {warnings} warning(s)."
    )
    self.summary_var.set(summary)

    # Refresh tree contents
    for item in self.results_tree.get_children():
      self.results_tree.delete(item)
    for control in controls:
      control_name = control.get("control", "")
      status = control.get("status", "")
      evidence = control.get("evidence", "")
      tag = "warn" if status != "pass" else "pass"
      self.results_tree.insert("", tk.END, values=(control_name, status, evidence), tags=(tag,))

    self.results_tree.tag_configure("warn", foreground="#b58900")
    self.results_tree.tag_configure("pass", foreground="#2aa198")

  def _export_manifest(self) -> None:
    if not self._baseline:
      messagebox.showwarning("No baseline", "Run an assessment before exporting a manifest.")
      return
    path = filedialog.asksaveasfilename(
      defaultextension=".json",
      filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
      title="Export remediation manifest",
    )
    if not path:
      return
    try:
      payload = {"tool": "ObsidianHunt", **self._baseline}
      Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
      messagebox.showinfo("Export complete", f"Manifest written to {path}")
    except OSError as exc:
      messagebox.showerror("Export failed", str(exc))


def launch() -> None:
  """Entry point used by the CLI when `--gui` is provided."""

  app = ObsidianHuntApp()
  app.mainloop()


__all__ = ["launch", "ObsidianHuntApp"]
