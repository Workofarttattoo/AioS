"""NemesisHydra GUI for authentication rehearsal planning."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List

from .nemesishydra import (
  TOOL_NAME,
  evaluate_targets,
  load_targets,
  load_wordlist,
)

PROFILE_PRESETS = [
  ("Conservative", 8),
  ("Balanced", 12),
  ("Aggressive", 20),
  ("Maximum", 40),
]


class NemesisHydraApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("NemesisHydra")
    self.geometry("900x640")
    self.resizable(True, True)

    self._payload: dict | None = None
    self._worker: threading.Thread | None = None
    self._advanced_visible = False

    self._build_ui()

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=12)
    root.pack(fill=tk.BOTH, expand=True)

    targets_frame = ttk.LabelFrame(root, text="Targets")
    targets_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    t_toolbar = ttk.Frame(targets_frame)
    t_toolbar.pack(fill=tk.X, pady=(4, 6))
    ttk.Button(t_toolbar, text="Load file", command=self._load_targets_file).pack(side=tk.LEFT)
    ttk.Button(t_toolbar, text="Sample", command=self._load_sample_targets).pack(side=tk.LEFT, padx=6)
    ttk.Button(t_toolbar, text="Clear", command=self._clear_targets).pack(side=tk.LEFT)

    self.targets_text = tk.Text(targets_frame, height=8)
    self.targets_text.pack(fill=tk.BOTH, expand=True)
    self._load_sample_targets()

    word_frame = ttk.LabelFrame(root, text="Wordlist")
    word_frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(word_frame, text="Wordlist path").grid(row=0, column=0, sticky=tk.W)
    self.wordlist_var = tk.StringVar()
    ttk.Entry(word_frame, textvariable=self.wordlist_var).grid(row=1, column=0, sticky=tk.EW)
    ttk.Button(word_frame, text="Browse", command=self._pick_wordlist).grid(row=1, column=1, padx=6)

    word_frame.columnconfigure(0, weight=1)

    rate_frame = ttk.Frame(root)
    rate_frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(rate_frame, text="Profile").grid(row=0, column=0, sticky=tk.W)
    self.profile_var = tk.StringVar(value="Balanced")
    self.profile_combo = ttk.Combobox(rate_frame, textvariable=self.profile_var, values=[p[0] for p in PROFILE_PRESETS], state="readonly", width=16)
    self.profile_combo.grid(row=1, column=0, sticky=tk.W)
    self.profile_combo.bind("<<ComboboxSelected>>", self._apply_profile)

    ttk.Label(rate_frame, text="Attempts per minute").grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
    self.rate_var = tk.IntVar(value=12)
    ttk.Spinbox(rate_frame, from_=1, to=120, increment=1, textvariable=self.rate_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=(12, 0))

    self.include_samples_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(rate_frame, text="Include demo wordlist", variable=self.include_samples_var).grid(row=1, column=2, padx=(12, 0))

    rate_frame.columnconfigure(0, weight=1)

    self.advanced_button = ttk.Button(root, text="Show advanced ▼", command=self._toggle_advanced)
    self.advanced_button.pack(anchor=tk.W)

    self.advanced_frame = ttk.LabelFrame(root, text="Advanced")
    ttk.Label(self.advanced_frame, text="Extra targets (comma separated URIs)").grid(row=0, column=0, sticky=tk.W)
    self.extra_targets_var = tk.StringVar()
    ttk.Entry(self.advanced_frame, textvariable=self.extra_targets_var).grid(row=1, column=0, sticky=tk.EW)
    self.advanced_frame.columnconfigure(0, weight=1)

    action_row = ttk.Frame(root)
    action_row.pack(fill=tk.X, pady=10)
    self.run_button = ttk.Button(action_row, text="Plan Rehearsal", command=self._run_assessment)
    self.run_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT)

    columns = ("target", "service", "host", "port", "duration", "risk", "resolution")
    self.result_tree = ttk.Treeview(root, columns=columns, show="headings", height=12)
    headers = {
      "target": "Target",
      "service": "Service",
      "host": "Host",
      "port": "Port",
      "duration": "Est. Minutes",
      "risk": "Throttle",
      "resolution": "Resolution",
    }
    for col, title in headers.items():
      self.result_tree.heading(col, text=title)
    self.result_tree.column("target", width=260)
    self.result_tree.column("service", width=80, anchor=tk.CENTER)
    self.result_tree.column("host", width=140)
    self.result_tree.column("port", width=60, anchor=tk.CENTER)
    self.result_tree.column("duration", width=120, anchor=tk.CENTER)
    self.result_tree.column("risk", width=100, anchor=tk.CENTER)
    self.result_tree.column("resolution", width=120, anchor=tk.CENTER)
    self.result_tree.pack(fill=tk.BOTH, expand=True)

    self.status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=self.status_var, foreground="#555").pack(fill=tk.X, pady=(6, 0))

  def _load_sample_targets(self) -> None:
    self.targets_text.delete("1.0", tk.END)
    targets = load_targets(None, None, True)
    self.targets_text.insert(tk.END, "\n".join(targets))

  def _clear_targets(self) -> None:
    self.targets_text.delete("1.0", tk.END)

  def _load_targets_file(self) -> None:
    path = filedialog.askopenfilename(title="Select targets file", filetypes=[("Text", "*.txt"), ("All files", "*")])
    if not path:
      return
    try:
      content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
      messagebox.showerror(TOOL_NAME, "Targets file not found.")
      return
    self.targets_text.delete("1.0", tk.END)
    self.targets_text.insert(tk.END, content)

  def _pick_wordlist(self) -> None:
    path = filedialog.askopenfilename(title="Select wordlist", filetypes=[("Text", "*.txt"), ("All files", "*")])
    if path:
      self.wordlist_var.set(path)

  def _toggle_advanced(self) -> None:
    if self._advanced_visible:
      self.advanced_frame.pack_forget()
      self.advanced_button.configure(text="Show advanced ▼")
    else:
      self.advanced_frame.pack(fill=tk.X, pady=(6, 10))
      self.advanced_button.configure(text="Hide advanced ▲")
    self._advanced_visible = not self._advanced_visible

  def _collect_targets(self) -> List[str]:
    base = [line.strip() for line in self.targets_text.get("1.0", tk.END).splitlines() if line.strip()]
    if self.include_samples_var.get():
      base.extend(load_targets(None, None, True))
    extra = self.extra_targets_var.get().strip()
    if extra:
      base.extend(item.strip() for item in extra.split(",") if item.strip())
    return list(dict.fromkeys(base))

  def _collect_wordlist(self) -> List[str]:
    wordlist_path = self.wordlist_var.get().strip() or None
    words = load_wordlist(wordlist_path, demo=False)
    if self.include_samples_var.get():
      words.extend(load_wordlist(None, True))
    return list(dict.fromkeys(words))

  def _apply_profile(self, _event=None) -> None:
    label = self.profile_var.get()
    for name, value in PROFILE_PRESETS:
      if name == label:
        self.rate_var.set(value)
        break

  def _run_assessment(self) -> None:
    if self._worker and self._worker.is_alive():
      messagebox.showinfo(TOOL_NAME, "Assessment already running.")
      return

    targets = self._collect_targets()
    if not targets:
      messagebox.showwarning(TOOL_NAME, "Provide at least one target URI.")
      return
    wordlist = self._collect_wordlist()
    if not wordlist:
      messagebox.showwarning(TOOL_NAME, "Wordlist is empty.")
      return

    rate_limit = int(self.rate_var.get())
    self.status_var.set("Planning rehearsal…")
    self.run_button.configure(state=tk.DISABLED)
    for item in self.result_tree.get_children():
      self.result_tree.delete(item)

    def worker() -> None:
      try:
        diagnostic, payload = evaluate_targets(targets, len(wordlist), rate_limit)
        self.after(0, self._render_results, diagnostic.summary, payload)
      except Exception as exc:
        self.after(0, self._handle_error, str(exc))

    self._worker = threading.Thread(target=worker, daemon=True)
    self._worker.start()

  def _render_results(self, summary: str, payload: dict) -> None:
    self._payload = payload
    for item in self.result_tree.get_children():
      self.result_tree.delete(item)
    for target in payload.get("targets", []):
      self.result_tree.insert(
        "",
        tk.END,
        values=(
          target.get("target", ""),
          target.get("service", ""),
          target.get("host", ""),
          target.get("port", ""),
          f"{target.get('estimated_minutes', 0):.2f}",
          target.get("throttle_risk", ""),
          target.get("resolution_status", ""),
        ),
      )
    self.status_var.set(summary)
    self.run_button.configure(state=tk.NORMAL)

  def _handle_error(self, message: str) -> None:
    messagebox.showerror(TOOL_NAME, message)
    self.status_var.set("Assessment failed.")
    self.run_button.configure(state=tk.NORMAL)

  def _export_json(self) -> None:
    if not self._payload:
      messagebox.showinfo(TOOL_NAME, "Run an assessment before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Rehearsal plan exported to {path}")


def launch() -> None:
  app = NemesisHydraApp()
  app.mainloop()
