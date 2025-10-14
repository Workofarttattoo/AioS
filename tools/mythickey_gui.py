"""Tkinter GUI for the MythicKey credential analyser."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox
from typing import List

from .mythickey import (
  SAMPLE_HASHES,
  SAMPLE_WORDS,
  TOOL_NAME,
  evaluate_hashes,
  load_hashes,
  load_wordlist,
)

PROFILE_PRESETS = [
  "cpu",
  "gpu-balanced",
  "gpu-max",
]


class MythicKeyApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("MythicKey")
    self.geometry("820x640")
    self.resizable(True, True)

    self._assessment_thread: threading.Thread | None = None
    self._payload: dict | None = None
    self._advanced_visible = False

    self._build_ui()

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=10)
    root.pack(fill=tk.BOTH, expand=True)

    hash_frame = ttk.LabelFrame(root, text="Hash digests")
    hash_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    toolbar = ttk.Frame(hash_frame)
    toolbar.pack(fill=tk.X, pady=(4, 6))
    ttk.Button(toolbar, text="Load file", command=self._load_hash_file).pack(side=tk.LEFT)
    ttk.Button(toolbar, text="Sample", command=self._load_sample_hashes).pack(side=tk.LEFT, padx=6)
    ttk.Button(toolbar, text="Clear", command=self._clear_hashes).pack(side=tk.LEFT)

    self.hash_text = tk.Text(hash_frame, height=10)
    self.hash_text.pack(fill=tk.BOTH, expand=True)
    self._load_sample_hashes()

    options = ttk.LabelFrame(root, text="Wordlist & profile")
    options.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(options, text="Wordlist file").grid(row=0, column=0, sticky=tk.W)
    self.wordlist_var = tk.StringVar()
    ttk.Entry(options, textvariable=self.wordlist_var).grid(row=1, column=0, sticky=tk.EW)
    ttk.Button(options, text="Browse", command=self._pick_wordlist).grid(row=1, column=1, padx=6)

    ttk.Label(options, text="Profile").grid(row=0, column=2, sticky=tk.W)
    self.profile_var = tk.StringVar(value="cpu")
    ttk.Combobox(options, textvariable=self.profile_var, values=PROFILE_PRESETS, state="readonly").grid(row=1, column=2, sticky=tk.W)

    self.include_samples_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(options, text="Include sample hashes/wordlist", variable=self.include_samples_var).grid(row=1, column=3, padx=8)

    options.columnconfigure(0, weight=1)

    self.advanced_button = ttk.Button(root, text="Show advanced ▼", command=self._toggle_advanced)
    self.advanced_button.pack(anchor=tk.W)

    self.advanced_frame = ttk.Frame(root)
    ttk.Label(self.advanced_frame, text="Extra dictionary words (comma separated)").grid(row=0, column=0, sticky=tk.W)
    self.extra_words_var = tk.StringVar()
    ttk.Entry(self.advanced_frame, textvariable=self.extra_words_var).grid(row=1, column=0, sticky=tk.EW)
    self.advanced_frame.columnconfigure(0, weight=1)

    action_row = ttk.Frame(root)
    action_row.pack(fill=tk.X, pady=10)
    self.run_button = ttk.Button(action_row, text="Run Assessment", command=self._run_assessment)
    self.run_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT)

    columns = ("digest", "algorithm", "status", "plaintext", "attempts")
    self.result_tree = ttk.Treeview(root, columns=columns, show="headings", height=10)
    headers = {
      "digest": "Digest",
      "algorithm": "Algorithm",
      "status": "Status",
      "plaintext": "Plaintext",
      "attempts": "Attempts",
    }
    for col, title in headers.items():
      self.result_tree.heading(col, text=title)
    self.result_tree.column("digest", width=180)
    self.result_tree.column("algorithm", width=100, anchor=tk.CENTER)
    self.result_tree.column("status", width=90, anchor=tk.CENTER)
    self.result_tree.column("plaintext", width=180)
    self.result_tree.column("attempts", width=100, anchor=tk.CENTER)
    self.result_tree.pack(fill=tk.BOTH, expand=True)

    self.status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=self.status_var, foreground="#555").pack(fill=tk.X, pady=(6, 0))

  def _load_sample_hashes(self) -> None:
    self.hash_text.delete("1.0", tk.END)
    self.hash_text.insert(tk.END, "\n".join(SAMPLE_HASHES))

  def _clear_hashes(self) -> None:
    self.hash_text.delete("1.0", tk.END)

  def _load_hash_file(self) -> None:
    path = filedialog.askopenfilename(title="Select hash file", filetypes=[("Text", "*.txt"), ("All files", "*")])
    if not path:
      return
    try:
      content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
      messagebox.showerror(TOOL_NAME, "File not found.")
      return
    self.hash_text.delete("1.0", tk.END)
    self.hash_text.insert(tk.END, content)

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

  def _collect_hashes(self) -> List[str]:
    hashes = [line.strip() for line in self.hash_text.get("1.0", tk.END).splitlines() if line.strip()]
    if self.include_samples_var.get():
      hashes.extend(SAMPLE_HASHES)
    return list(dict.fromkeys(hashes))

  def _collect_wordlist(self) -> List[str]:
    paths = self.wordlist_var.get().strip() or None
    words = load_wordlist(paths, demo=False)
    if self.include_samples_var.get():
      words.extend(SAMPLE_WORDS)
    extra_raw = self.extra_words_var.get().strip()
    if extra_raw:
      for chunk in extra_raw.split(","):
        chunk = chunk.strip()
        if chunk:
          words.append(chunk)
    return list(dict.fromkeys(words))

  def _run_assessment(self) -> None:
    if self._assessment_thread and self._assessment_thread.is_alive():
      messagebox.showinfo(TOOL_NAME, "Assessment in progress.")
      return

    hashes = self._collect_hashes()
    if not hashes:
      messagebox.showwarning(TOOL_NAME, "Provide at least one hash digest.")
      return
    wordlist = self._collect_wordlist()
    if not wordlist:
      messagebox.showwarning(TOOL_NAME, "Wordlist is empty.")
      return

    profile = self.profile_var.get().strip().lower() or "cpu"
    self.status_var.set("Running assessment…")
    self.run_button.configure(state=tk.DISABLED)
    for item in self.result_tree.get_children():
      self.result_tree.delete(item)

    def worker() -> None:
      try:
        diagnostic, payload = evaluate_hashes(hashes, wordlist, profile)
        self.after(0, self._render_results, diagnostic.summary, payload)
      except Exception as exc:  # pylint: disable=broad-except
        self.after(0, self._handle_error, str(exc))

    self._assessment_thread = threading.Thread(target=worker, daemon=True)
    self._assessment_thread.start()

  def _render_results(self, summary: str, payload: dict) -> None:
    self._payload = payload
    for item in self.result_tree.get_children():
      self.result_tree.delete(item)
    for entry in payload.get("hashes", []):
      status = "cracked" if entry.get("cracked") else "safe"
      self.result_tree.insert(
        "",
        tk.END,
        values=(
          entry.get("digest_prefix", ""),
          entry.get("algorithm", ""),
          status,
          (entry.get("plaintext") or "")[:64],
          entry.get("attempts", 0),
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
    messagebox.showinfo(TOOL_NAME, f"Assessment exported to {path}")


def launch() -> None:
  app = MythicKeyApp()
  app.mainloop()
