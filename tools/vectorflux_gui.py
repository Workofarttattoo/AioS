"""VectorFlux GUI for payload staging with guardrails."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict

from .vectorflux import (
  SAMPLE_MODULES,
  TOOL_NAME,
  generate_manifest,
  list_modules,
  load_scenario,
)


class VectorFluxApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("VectorFlux")
    self.geometry("820x580")
    self.resizable(True, True)

    self._payload: Dict[str, object] | None = None
    self._worker: threading.Thread | None = None
    self._advanced_visible = False

    self._build_ui()

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=12)
    root.pack(fill=tk.BOTH, expand=True)

    workspace_frame = ttk.Frame(root)
    workspace_frame.pack(fill=tk.X, pady=(0, 6))

    ttk.Label(workspace_frame, text="Workspace").grid(row=0, column=0, sticky=tk.W)
    self.workspace_var = tk.StringVar(value="vectorflux-workspace")
    ttk.Entry(workspace_frame, textvariable=self.workspace_var).grid(row=1, column=0, sticky=tk.EW)

    ttk.Label(workspace_frame, text="Scenario file").grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
    self.scenario_var = tk.StringVar()
    ttk.Entry(workspace_frame, textvariable=self.scenario_var).grid(row=1, column=1, sticky=tk.EW, padx=(12, 0))
    ttk.Button(workspace_frame, text="Browse", command=self._pick_scenario).grid(row=1, column=2, padx=(6, 0))

    workspace_frame.columnconfigure(0, weight=1)
    workspace_frame.columnconfigure(1, weight=1)

    module_frame = ttk.LabelFrame(root, text="Modules")
    module_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    self.module_list = tk.Listbox(module_frame, height=6)
    for name, descriptor in SAMPLE_MODULES.items():
      self.module_list.insert(tk.END, f"{name} — {descriptor.description}")
    self.module_list.selection_set(0)
    self.module_list.pack(fill=tk.BOTH, expand=True)

    self.advanced_button = ttk.Button(root, text="Show advanced ▼", command=self._toggle_advanced)
    self.advanced_button.pack(anchor=tk.W)

    self.advanced_frame = ttk.LabelFrame(root, text="Constraints")
    ttk.Label(self.advanced_frame, text="Additional constraints (comma separated)").grid(row=0, column=0, sticky=tk.W)
    self.constraints_var = tk.StringVar()
    ttk.Entry(self.advanced_frame, textvariable=self.constraints_var).grid(row=1, column=0, sticky=tk.EW)
    self.advanced_frame.columnconfigure(0, weight=1)

    action_row = ttk.Frame(root)
    action_row.pack(fill=tk.X, pady=10)
    self.run_button = ttk.Button(action_row, text="Stage Module", command=self._stage_module)
    self.run_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT)

    summary_frame = ttk.LabelFrame(root, text="Manifest Summary")
    summary_frame.pack(fill=tk.BOTH, expand=True)
    self.summary_text = tk.Text(summary_frame, height=10, state=tk.DISABLED)
    self.summary_text.pack(fill=tk.BOTH, expand=True)

    self.status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=self.status_var, foreground="#555").pack(fill=tk.X, pady=(6, 0))

  def _pick_scenario(self) -> None:
    path = filedialog.askopenfilename(title="Select scenario JSON", filetypes=[("JSON", "*.json"), ("All files", "*")])
    if path:
      self.scenario_var.set(path)

  def _toggle_advanced(self) -> None:
    if self._advanced_visible:
      self.advanced_frame.pack_forget()
      self.advanced_button.configure(text="Show advanced ▼")
    else:
      self.advanced_frame.pack(fill=tk.X, pady=(6, 10))
      self.advanced_button.configure(text="Hide advanced ▲")
    self._advanced_visible = not self._advanced_visible

  def _selected_module(self) -> str:
    idxs = self.module_list.curselection()
    if not idxs:
      return list(SAMPLE_MODULES.keys())[0]
    entry = self.module_list.get(idxs[0])
    return entry.split(" — ", maxsplit=1)[0]

  def _stage_module(self) -> None:
    if self._worker and self._worker.is_alive():
      messagebox.showinfo(TOOL_NAME, "Module staging already running.")
      return

    workspace = self.workspace_var.get().strip() or "vectorflux-workspace"
    module = self._selected_module()
    scenario_path = self.scenario_var.get().strip() or None
    scenario = load_scenario(scenario_path)

    extra_constraints = [item.strip() for item in self.constraints_var.get().split(",") if item.strip()]
    if extra_constraints:
      scenario = dict(scenario)
      scenario.setdefault("constraints", [])
      scenario["constraints"] = list(dict.fromkeys(list(scenario["constraints"]) + extra_constraints))

    self.status_var.set("Staging module…")
    self.run_button.configure(state=tk.DISABLED)

    def worker() -> None:
      try:
        diagnostic, payload = generate_manifest(workspace, module, scenario)
        self.after(0, self._render_results, diagnostic.summary, payload)
      except ValueError as exc:
        self.after(0, self._handle_error, str(exc))

    self._worker = threading.Thread(target=worker, daemon=True)
    self._worker.start()

  def _render_results(self, summary: str, payload: Dict[str, object]) -> None:
    self._payload = payload
    self.summary_text.configure(state=tk.NORMAL)
    self.summary_text.delete("1.0", tk.END)
    self.summary_text.insert(tk.END, json.dumps(payload, indent=2))
    self.summary_text.configure(state=tk.DISABLED)
    self.status_var.set(summary)
    self.run_button.configure(state=tk.NORMAL)

  def _handle_error(self, message: str) -> None:
    messagebox.showerror(TOOL_NAME, message)
    self.status_var.set("Staging failed.")
    self.run_button.configure(state=tk.NORMAL)

  def _export_json(self) -> None:
    if not self._payload:
      messagebox.showinfo(TOOL_NAME, "Stage a module before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Manifest exported to {path}")


def launch() -> None:
  app = VectorFluxApp()
  app.mainloop()
