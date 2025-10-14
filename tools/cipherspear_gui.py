"""Tkinter interface for the CipherSpear rehearsal engine."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox
from typing import List

from .cipherspear import (
  TOOL_NAME,
  SAMPLE_VECTORS,
  evaluate_vectors,
  load_vectors,
)


TECHNIQUE_PRESETS = {
  "Baseline": [],
  "Boolean / Blind": ["blind", "bool"],
  "Time-based": ["time"],
  "Stacked": ["stacked", "bool", "time"],
}


class CipherSpearApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("CipherSpear")
    self.geometry("820x600")
    self.resizable(True, True)

    self._assessment_thread: threading.Thread | None = None
    self._result_payload: dict | None = None

    self._build_ui()

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=10)
    root.pack(fill=tk.BOTH, expand=True)

    dsn_frame = ttk.LabelFrame(root, text="Connection")
    dsn_frame.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(dsn_frame, text="Target DSN").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
    self.dsn_var = tk.StringVar(value="postgresql://analyst@localhost/sample")
    ttk.Entry(dsn_frame, textvariable=self.dsn_var).grid(row=1, column=0, sticky=tk.EW, padx=5)
    dsn_frame.columnconfigure(0, weight=1)

    options = ttk.LabelFrame(root, text="Techniques")
    options.pack(fill=tk.X, pady=8)

    ttk.Label(options, text="Preset").grid(row=0, column=0, padx=5, pady=4, sticky=tk.W)
    self.preset_var = tk.StringVar(value="Baseline")
    preset_box = ttk.Combobox(options, textvariable=self.preset_var, values=list(TECHNIQUE_PRESETS.keys()), state="readonly")
    preset_box.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
    preset_box.bind("<<ComboboxSelected>>", self._apply_preset)

    ttk.Label(options, text="Additional techniques (comma separated)").grid(row=0, column=1, padx=5, pady=4, sticky=tk.W)
    self.techniques_var = tk.StringVar()
    ttk.Entry(options, textvariable=self.techniques_var).grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

    self.demo_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(options, text="Include sample vectors", variable=self.demo_var).grid(row=1, column=2, padx=5, pady=2)

    options.columnconfigure(1, weight=1)

    vector_frame = ttk.LabelFrame(root, text="Candidate vectors")
    vector_frame.pack(fill=tk.BOTH, expand=True, pady=8)

    toolbar = ttk.Frame(vector_frame)
    toolbar.pack(fill=tk.X)
    ttk.Button(toolbar, text="Load from file", command=self._load_vectors_from_file).pack(side=tk.LEFT)
    ttk.Button(toolbar, text="Clear", command=self._clear_vectors).pack(side=tk.LEFT, padx=5)

    self.vector_text = tk.Text(vector_frame, height=12)
    self.vector_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    self._populate_sample_vectors()

    action_row = ttk.Frame(root)
    action_row.pack(fill=tk.X, pady=5)
    self.run_button = ttk.Button(action_row, text="Run Assessment", command=self._run_assessment)
    self.run_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT)

    result_frame = ttk.LabelFrame(root, text="Findings")
    result_frame.pack(fill=tk.BOTH, expand=True)

    columns = ("vector", "risk", "findings", "recommendation")
    self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=6)
    self.result_tree.heading("vector", text="Vector")
    self.result_tree.heading("risk", text="Risk")
    self.result_tree.heading("findings", text="Findings")
    self.result_tree.heading("recommendation", text="Recommendation")
    self.result_tree.column("vector", width=220)
    self.result_tree.column("risk", width=60, anchor=tk.CENTER)
    self.result_tree.column("findings", width=180)
    self.result_tree.column("recommendation", width=240)
    self.result_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    self.status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=self.status_var, foreground="#555").pack(fill=tk.X, pady=(4, 0))

  def _populate_sample_vectors(self) -> None:
    self.vector_text.delete("1.0", tk.END)
    self.vector_text.insert(tk.END, "\n".join(SAMPLE_VECTORS))

  def _apply_preset(self, _event=None) -> None:
    preset = self.preset_var.get()
    values = TECHNIQUE_PRESETS.get(preset, [])
    self.techniques_var.set(",".join(values))

  def _load_vectors_from_file(self) -> None:
    path = filedialog.askopenfilename(title="Select vectors file")
    if not path:
      return
    content = Path(path).read_text(encoding="utf-8")
    self.vector_text.delete("1.0", tk.END)
    self.vector_text.insert(tk.END, content)
    self.demo_var.set(False)

  def _clear_vectors(self) -> None:
    self.vector_text.delete("1.0", tk.END)

  def _collect_vectors(self) -> List[str]:
    raw = self.vector_text.get("1.0", tk.END).strip()
    vectors = [line.strip() for line in raw.splitlines() if line.strip()]
    if self.demo_var.get():
      vectors.extend(SAMPLE_VECTORS)
    return vectors or list(SAMPLE_VECTORS)

  def _run_assessment(self) -> None:
    if self._assessment_thread and self._assessment_thread.is_alive():
      messagebox.showinfo("Scan running", "An assessment is already running.")
      return

    dsn = self.dsn_var.get().strip()
    if not dsn:
      messagebox.showwarning("Missing DSN", "Provide a DSN for the rehearsal.")
      return

    vectors = self._collect_vectors()
    techniques = [chunk.strip() for chunk in self.techniques_var.get().split(",") if chunk.strip()]
    preset = TECHNIQUE_PRESETS.get(self.preset_var.get(), [])
    techniques = sorted(set(techniques + preset))

    self.status_var.set("Running assessment…")
    self.run_button.configure(state=tk.DISABLED)
    for item in self.result_tree.get_children():
      self.result_tree.delete(item)

    def worker() -> None:
      try:
        diagnostic, payload = evaluate_vectors(dsn, techniques, vectors)
        self.after(0, self._render_results, diagnostic.summary, payload)
      except Exception as exc:  # pylint: disable=broad-except
        self.after(0, self._handle_error, str(exc))

    self._assessment_thread = threading.Thread(target=worker, daemon=True)
    self._assessment_thread.start()

  def _render_results(self, summary: str, payload: dict) -> None:
    self._result_payload = payload
    for assessment in payload.get("assessments", []):
      self.result_tree.insert(
        "",
        tk.END,
        values=(
          assessment.get("vector", "")[:60],
          assessment.get("risk_label", ""),
          ", ".join(assessment.get("findings", []))[:80],
          assessment.get("recommendation", "")[:120],
        ),
      )
    self.status_var.set(summary)
    self.run_button.configure(state=tk.NORMAL)

  def _handle_error(self, message: str) -> None:
    messagebox.showerror("CipherSpear", message)
    self.status_var.set("Assessment failed.")
    self.run_button.configure(state=tk.NORMAL)

  def _export_json(self) -> None:
    if not self._result_payload:
      messagebox.showinfo(TOOL_NAME, "Run an assessment before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._result_payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Exported assessment to {path}")


def launch() -> None:
  app = CipherSpearApp()
  app.mainloop()

