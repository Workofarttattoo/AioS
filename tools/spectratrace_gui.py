"""Tkinter GUI for the SpectraTrace protocol inspector."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List

from .spectratrace import (
  WORKFLOWS,
  TOOL_NAME,
  evaluate_capture,
  generate_heatmap,
  ingest_stream,
  load_capture,
  load_playbook,
)

RECIPE_MODULES: Dict[str, List[Dict[str, object]]] = {
  "enrich": [
    {
      "id": "tls_ja3",
      "label": "TLS JA3 Fingerprints",
      "description": "Capture JA3 hashes for TLS ClientHello packets.",
      "playbook_rules": {},
    },
    {
      "id": "http_metadata",
      "label": "HTTP Metadata",
      "description": "Extract HTTP verbs, hosts, and content types for downstream correlation.",
      "playbook_rules": {"http-verb-trace": "TRACE ", "http-suspicious-host": "Host: localhost"},
    },
  ],
  "detect": [
    {
      "id": "behavioral_score",
      "label": "Behavioral Scoring",
      "description": "Apply heuristics for beaconing, data spikes, and anomalous ports.",
      "playbook_rules": {"suspicious-beacon": "interval", "data-spike": "Content-Length: [1-9][0-9]{5,}"},
    },
    {
      "id": "rule_dsl",
      "label": "Rule DSL",
      "description": "Enable custom analyst rules defined in the inline editor.",
      "playbook_rules": {},
    },
  ],
  "respond": [
    {
      "id": "alert_route",
      "label": "Alert Routing",
      "description": "Tag packets that should raise tickets or pager alerts.",
      "playbook_rules": {"payload-alert": "ALERT:"},
    },
    {
      "id": "annotation_push",
      "label": "Live Annotation Push",
      "description": "Send annotations to collaborative timelines for swarm analysis.",
      "playbook_rules": {},
    },
  ],
}


class SpectraTraceApp(tk.Tk):
  def __init__(self) -> None:
    super().__init__()
    self.title("SpectraTrace")
    self.geometry("960x680")
    self.resizable(True, True)

    self._payload: Dict[str, object] | None = None
    self._worker: threading.Thread | None = None
    self._advanced_visible = False
    self.current_packets: List[Dict[str, object]] = []
    self.current_annotations: Dict[int, Dict[str, object]] = {}
    self.current_insights: List[Dict[str, object]] = []
    self.timeline_packets: List[Dict[str, object]] = []
    self.timeline_zoom = tk.DoubleVar(value=1.0)
    self.selected_recipe: Dict[str, set] = {stage: set() for stage in RECIPE_MODULES}
    self.recipe_vars: Dict[str, Dict[str, tk.BooleanVar]] = {}

    self._build_ui()

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=12)
    root.pack(fill=tk.BOTH, expand=True)

    capture_frame = ttk.LabelFrame(root, text="Capture")
    capture_frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(capture_frame, text="Interface").grid(row=0, column=0, sticky=tk.W)
    self.iface_var = tk.StringVar(value="eth0")
    ttk.Entry(capture_frame, textvariable=self.iface_var, width=16).grid(row=1, column=0, sticky=tk.W)

    ttk.Label(capture_frame, text="Capture file").grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
    self.capture_path_var = tk.StringVar()
    ttk.Entry(capture_frame, textvariable=self.capture_path_var).grid(row=1, column=1, sticky=tk.EW, padx=(12, 0))
    ttk.Button(capture_frame, text="Browse", command=self._pick_capture).grid(row=1, column=2, padx=(6, 0))

    ttk.Label(capture_frame, text="Workflow").grid(row=0, column=3, sticky=tk.W, padx=(12, 0))
    self.workflow_label_to_key = {"None": None}
    for key, cfg in WORKFLOWS.items():
      label = str(cfg.get("label") or key)
      self.workflow_label_to_key[label] = key
    workflow_values = list(self.workflow_label_to_key.keys())
    self.workflow_var = tk.StringVar(value=workflow_values[0])
    self.workflow_combo = ttk.Combobox(
      capture_frame,
      textvariable=self.workflow_var,
      values=workflow_values,
      state="readonly",
      width=20,
    )
    self.workflow_combo.grid(row=1, column=3, sticky=tk.W, padx=(12, 0))
    self.workflow_combo.bind("<<ComboboxSelected>>", self._update_workflow_description)

    capture_frame.columnconfigure(1, weight=1)
    capture_frame.columnconfigure(3, weight=0)

    self.workflow_desc_var = tk.StringVar(value="")
    ttk.Label(root, textvariable=self.workflow_desc_var, foreground="#777").pack(fill=tk.X, pady=(4, 8))
    self._update_workflow_description()

    self.stream_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
      capture_frame,
      text="Stream NDJSON from buffer",
      variable=self.stream_var,
      command=self._toggle_stream_mode,
    ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

    playbook_frame = ttk.Frame(root)
    playbook_frame.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(playbook_frame, text="Playbook").grid(row=0, column=0, sticky=tk.W)
    self.playbook_var = tk.StringVar()
    ttk.Entry(playbook_frame, textvariable=self.playbook_var).grid(row=1, column=0, sticky=tk.EW)
    ttk.Button(playbook_frame, text="Browse", command=self._pick_playbook).grid(row=1, column=1, padx=(6, 0))
    playbook_frame.columnconfigure(0, weight=1)

    self.advanced_button = ttk.Button(root, text="Show advanced ▼", command=self._toggle_advanced)
    self.advanced_button.pack(anchor=tk.W)

    self.advanced_frame = ttk.LabelFrame(root, text="Advanced")
    ttk.Label(self.advanced_frame, text="Max packets").grid(row=0, column=0, sticky=tk.W)
    self.max_packets_var = tk.IntVar(value=2000)
    ttk.Spinbox(
      self.advanced_frame,
      from_=10,
      to=100000,
      increment=100,
      textvariable=self.max_packets_var,
      width=10,
    ).grid(row=1, column=0, sticky=tk.W)

    ttk.Label(self.advanced_frame, text="Inline playbook rules (name:pattern)").grid(row=0, column=1, padx=(12, 0), sticky=tk.W)
    self.inline_playbook = tk.Text(self.advanced_frame, height=4)
    self.inline_playbook.grid(row=1, column=1, sticky=tk.EW)

    ttk.Label(self.advanced_frame, text="Stream buffer (NDJSON)").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
    self.stream_buffer = tk.Text(self.advanced_frame, height=6, state=tk.DISABLED)
    self.stream_buffer.grid(row=3, column=0, columnspan=2, sticky=tk.EW)

    self.advanced_frame.columnconfigure(1, weight=1)

    action_row = ttk.Frame(root)
    action_row.pack(fill=tk.X, pady=10)
    self.run_button = ttk.Button(action_row, text="Run Analysis", command=self._run_analysis)
    self.run_button.pack(side=tk.LEFT)
    ttk.Button(action_row, text="Guided Tour", command=self._show_tour).pack(side=tk.RIGHT)
    ttk.Button(action_row, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT, padx=(0, 6))

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    self.timeline_canvas = self._build_timeline_tab(notebook)
    self.packet_tree = self._build_packets_tab(notebook)
    self.protocol_tree = self._build_protocols_tab(notebook)
    self.heatmap_tree = self._build_heatmap_tab(notebook)
    self.annotations_tree = self._build_annotations_tab(notebook)
    self.alerts_tree = self._build_alerts_tab(notebook)
    self.insights_tree = self._build_insights_tab(notebook)
    self.recipe_preview = self._build_recipe_tab(notebook)
    self.insights_tree = self._build_insights_tab(notebook)

    summary_frame = ttk.LabelFrame(root, text="Summary")
    summary_frame.pack(fill=tk.X, pady=(10, 0))
    self.summary_var = tk.StringVar(value="Ready.")
    ttk.Label(summary_frame, textvariable=self.summary_var, foreground="#555").pack(fill=tk.X)
    self.talkers_var = tk.StringVar(value="")
    ttk.Label(summary_frame, textvariable=self.talkers_var, foreground="#777").pack(fill=tk.X)
    self.helper_var = tk.StringVar(value="Select a packet to view contextual insights.")
    ttk.Label(root, textvariable=self.helper_var, foreground="#666").pack(fill=tk.X, pady=(4, 0))

    self.packet_tree.bind("<<TreeviewSelect>>", self._on_packet_selected)
    self.after(75, self._draw_timeline)
    self._update_recipe_preview()

  def _pick_capture(self) -> None:
    path = filedialog.askopenfilename(
      title="Select capture",
      filetypes=[("PCAP", "*.pcap;*.pcapng"), ("JSON", "*.json;*.jsonl"), ("All files", "*")],
    )
    if path:
      self.capture_path_var.set(path)

  def _pick_playbook(self) -> None:
    path = filedialog.askopenfilename(
      title="Select playbook",
      filetypes=[("JSON/YAML", "*.json;*.yaml;*.yml"), ("All files", "*")],
    )
    if path:
      self.playbook_var.set(path)

  def _toggle_advanced(self) -> None:
    if self._advanced_visible:
      self.advanced_frame.pack_forget()
      self.advanced_button.configure(text="Show advanced ▼")
    else:
      self.advanced_frame.pack(fill=tk.X, pady=(6, 10))
      self.advanced_button.configure(text="Hide advanced ▲")
    self._advanced_visible = not self._advanced_visible

  def _toggle_stream_mode(self) -> None:
    if self.stream_var.get():
      self.stream_buffer.configure(state=tk.NORMAL)
      self.capture_path_var.set("")
    else:
      self.stream_buffer.configure(state=tk.DISABLED)

  def _combined_playbook(self) -> Dict[str, str]:
    playbook_path = self.playbook_var.get().strip() or None
    rules = load_playbook(playbook_path)
    inline = self.inline_playbook.get("1.0", tk.END).strip()
    if inline:
      for line in inline.splitlines():
        if ":" not in line:
          continue
        name, pattern = line.split(":", maxsplit=1)
        name = name.strip()
        pattern = pattern.strip()
        if name and pattern:
          rules[name] = pattern
    return rules

  def _run_analysis(self) -> None:
    if self._worker and self._worker.is_alive():
      messagebox.showinfo(TOOL_NAME, "Analysis already running.")
      return

    workflow_label = self.workflow_var.get()
    workflow_key = self.workflow_label_to_key.get(workflow_label)
    workflow_cfg = WORKFLOWS.get(workflow_key) if workflow_key else None

    max_packets = self.max_packets_var.get()
    if workflow_cfg and workflow_cfg.get("max_packets"):
      max_packets = min(max_packets, int(workflow_cfg["max_packets"]))

    if self.stream_var.get():
      buffer_text = self.stream_buffer.get("1.0", tk.END).strip()
      lines: List[str] = buffer_text.splitlines()
      packets = ingest_stream(lines, max_packets)
      ingest_mode = "stream-buffer"
    else:
      packets = load_capture(self.capture_path_var.get().strip() or None, max_packets)
      ingest_mode = "capture" if self.capture_path_var.get().strip() else "sample"

    playbook = self._combined_playbook()
    recipe_config = self._recipe_configuration()
    self._apply_recipe_to_playbook(playbook, recipe_config)
    self._update_recipe_preview()
    if workflow_cfg:
      for name, pattern in workflow_cfg.get("playbook_rules", {}).items():
        playbook.setdefault(name, pattern)
    iface = self.iface_var.get().strip() or "any"

    self.summary_var.set("Analysing packets…")
    self.talkers_var.set("")
    self.helper_var.set("Analysing packets…")
    self.run_button.configure(state=tk.DISABLED)
    self._clear_tables()

    workflow_meta = None
    if workflow_cfg:
      workflow_meta = {
        "name": workflow_key,
        "label": workflow_cfg.get("label", workflow_key),
        "description": workflow_cfg.get("description", ""),
        "max_packets": workflow_cfg.get("max_packets"),
        "extra_rules": list(workflow_cfg.get("playbook_rules", {}).keys()),
      }

    def worker() -> None:
      try:
        diagnostic, payload = evaluate_capture(
          packets,
          playbook,
          iface,
          ingest_mode=ingest_mode,
          workflow=workflow_key,
          workflow_meta=workflow_meta,
          recipe_config=recipe_config,
        )
        self.after(0, self._render_results, diagnostic.summary, payload)
      except Exception as exc:
        self.after(0, self._handle_error, str(exc))

    self._worker = threading.Thread(target=worker, daemon=True)
    self._worker.start()

  def _render_results(self, summary: str, payload: Dict[str, object]) -> None:
    self._payload = payload
    analysis = payload.get("analysis", {})
    alerts = analysis.get("alerts", [])
    annotations = payload.get("annotations", [])
    ingest = payload.get("ingest", {})
    packets = payload.get("packets", [])
    workflow_info = payload.get("workflow") or {}
    plugin_insights = payload.get("plugin_insights", [])

    self._clear_tables()

    self.current_packets = packets
    self.current_annotations = {
      int(annotation.get("packet_index", -1)): annotation for annotation in annotations if isinstance(annotation, dict)
    }
    self.current_insights = plugin_insights if isinstance(plugin_insights, list) else []

    for entry in packets:
      self.packet_tree.insert(
        "",
        tk.END,
        values=(
          f"{entry.get('timestamp', 0.0):.3f}",
          entry.get("src", ""),
          entry.get("dst", ""),
          entry.get("protocol", ""),
          entry.get("length", 0),
          str(entry.get("info", ""))[:100],
        ),
      )

    for annotation in annotations:
      tags = ", ".join(annotation.get("tags", []))
      self.annotations_tree.insert(
        "",
        tk.END,
        values=(
          annotation.get("packet_index", 0),
          tags,
          annotation.get("summary", ""),
        ),
      )

    for alert in alerts:
      packet = alert.get("packet", {})
      self.alerts_tree.insert(
        "",
        tk.END,
        values=(
          alert.get("rule", ""),
          alert.get("pattern", ""),
          packet.get("src", ""),
          packet.get("dst", ""),
          packet.get("protocol", ""),
          packet.get("info", ""),
        ),
      )

    for insight in self.current_insights:
      summary_line = insight.get("summary", "") or insight.get("details", "")
      self.insights_tree.insert(
        "",
        tk.END,
        values=(
          insight.get("name", "Plugin"),
          insight.get("severity", "info"),
          summary_line,
        ),
      )

    mode = ingest.get("mode", "capture")
    count = ingest.get("count", len(packets))
    workflow_label = workflow_info.get("label") or workflow_info.get("name")
    summary_text = f"{summary} | Mode: {mode} | Packets: {count} | Alerts: {len(alerts)}"
    if workflow_label:
      summary_text += f" | Workflow: {workflow_label}"

    self.timeline_zoom.set(1.0)

    talkers = analysis.get("top_talkers", [])
    if talkers:
      formatted = ", ".join(f"{entry['address']} ({entry['bytes']}B)" for entry in talkers)
      self.talkers_var.set(f"Top talkers: {formatted}")
    else:
      self.talkers_var.set("")

    self._render_protocols(analysis.get("protocol_breakdown", {}))
    recipe_info = payload.get("recipe") or self._recipe_configuration()
    heatmap_data = payload.get("heatmap") or generate_heatmap(packets)
    self._render_heatmap(heatmap_data)
    if heatmap_data:
      peak_bucket = max(heatmap_data, key=lambda entry: entry.get("packet_count", 0))
      summary_text += (
        f" | Peak bucket: {peak_bucket.get('packet_count', 0)} pkt"
        f" between {peak_bucket.get('start', 0.0):.3f}-{peak_bucket.get('end', 0.0):.3f}"
      )

    self.timeline_packets = packets
    self._draw_timeline()

    if recipe_info:
      modules = recipe_info.get("module_count", 0)
      if modules:
        summary_text += f" | Recipe modules: {modules}"
      self._render_recipe_preview(recipe_info)

    self.summary_var.set(summary_text)

    self.helper_var.set("Select a packet to view contextual insights.")
    self.run_button.configure(state=tk.NORMAL)

  def _handle_error(self, message: str) -> None:
    messagebox.showerror(TOOL_NAME, message)
    self.summary_var.set("Analysis failed.")
    self.run_button.configure(state=tk.NORMAL)

  def _update_workflow_description(self, _event=None) -> None:
    label = self.workflow_var.get()
    key = self.workflow_label_to_key.get(label)
    config = WORKFLOWS.get(key) if key else None
    if config:
      description = str(config.get("description", ""))
      self.workflow_desc_var.set(description)
      if config.get("max_packets") and hasattr(self, "helper_var"):
        self.helper_var.set(
          f"Workflow '{config.get('label', key)}' caps packets at {config.get('max_packets')} and enriches playbook hints."
        )
    else:
      self.workflow_desc_var.set("")

  def _show_tour(self) -> None:
    overlay = tk.Toplevel(self)
    overlay.title("SpectraTrace Guided Tour")
    overlay.geometry("520x420")
    overlay.transient(self)
    overlay.grab_set()

    steps = [
      "1. Select a capture file or paste NDJSON into the stream buffer.",
      "2. Choose a workflow preset to load recommended filters and limits.",
      "3. Optionally provide a playbook to flag interesting strings.",
      "4. Click Run Analysis to populate the timeline, tables, and alerts.",
      "5. Use the tabs to inspect packets, annotations, heatmaps, and plugin insights.",
      "6. Select a packet to view contextual tips in the helper bar at the bottom.",
      "7. Export JSON for automated pipelines or generate a Markdown/HTML report via the CLI.",
    ]

    frame = ttk.Frame(overlay, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="Getting Started", font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, pady=(0, 12))
    for step in steps:
      ttk.Label(frame, text=step, wraplength=480, justify=tk.LEFT).pack(anchor=tk.W, pady=4)
    ttk.Button(frame, text="Close", command=overlay.destroy).pack(anchor=tk.E, pady=(18, 0))

  def _on_packet_selected(self, _event=None) -> None:
    selection = self.packet_tree.selection()
    if not selection:
      self.helper_var.set("Select a packet to view contextual insights.")
      return
    index = self.packet_tree.index(selection[0])
    if index >= len(self.current_packets):
      self.helper_var.set("Select a packet to view contextual insights.")
      return
    packet = self.current_packets[index]
    annotation = self.current_annotations.get(index)
    tags = ", ".join(annotation.get("tags", [])) if annotation else "none"
    proto = packet.get("protocol", "")
    summary = (
      f"Packet #{index + 1}: {packet.get('src', '')} → {packet.get('dst', '')} "
      f"{proto} len={packet.get('length', 0)} | Tags: {tags}"
    )
    self.helper_var.set(summary)

  def _export_json(self) -> None:
    if not self._payload:
      messagebox.showinfo(TOOL_NAME, "Run an analysis before exporting.")
      return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if not path:
      return
    Path(path).write_text(json.dumps(self._payload, indent=2), encoding="utf-8")
    messagebox.showinfo(TOOL_NAME, f"Results exported to {path}")

  def _clear_tables(self) -> None:
    trees = [
      getattr(self, attr)
      for attr in (
        "packet_tree",
        "protocol_tree",
        "heatmap_tree",
        "annotations_tree",
        "alerts_tree",
        "insights_tree",
      )
      if hasattr(self, attr)
    ]
    for tree in trees:
      for item in tree.get_children():
        tree.delete(item)
    self.current_packets = []
    self.current_annotations = {}
    self.current_insights = []
    self.timeline_packets = []
    self.timeline_zoom.set(1.0)
    if hasattr(self, "timeline_canvas"):
      self.timeline_canvas.delete("all")
      self._draw_timeline()
    if hasattr(self, "helper_var"):
      self.helper_var.set("Select a packet to view contextual insights.")

  def _build_timeline_tab(self, notebook: ttk.Notebook) -> tk.Canvas:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Timeline")
    canvas = tk.Canvas(frame, height=220, background="#161616", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    controls = ttk.Frame(frame)
    controls.pack(fill=tk.X, pady=6)
    ttk.Label(controls, text="Zoom").pack(side=tk.LEFT)
    ttk.Scale(
      controls,
      from_=0.25,
      to=1.0,
      variable=self.timeline_zoom,
      command=lambda _value: self._draw_timeline(),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
    ttk.Button(controls, text="Reset", command=self._reset_timeline_zoom).pack(side=tk.RIGHT)
    ttk.Label(frame, text="Colours correspond to protocol families; zoom narrows the trailing window.").pack(anchor=tk.W, padx=6)
    canvas.bind("<Configure>", lambda _event: self._draw_timeline())
    return canvas

  def _build_packets_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Packets")
    columns = ("timestamp", "src", "dst", "protocol", "length", "info")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    headings = {
      "timestamp": "Timestamp",
      "src": "Source",
      "dst": "Destination",
      "protocol": "Protocol",
      "length": "Length",
      "info": "Info",
    }
    for col, title in headings.items():
      tree.heading(col, text=title)
    tree.column("timestamp", width=120)
    tree.column("src", width=150)
    tree.column("dst", width=150)
    tree.column("protocol", width=90, anchor=tk.CENTER)
    tree.column("length", width=80, anchor=tk.CENTER)
    tree.column("info", width=320)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_protocols_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Protocols")
    columns = ("protocol", "count", "percent")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("protocol", text="Protocol")
    tree.heading("count", text="Packets")
    tree.heading("percent", text="Percent")
    tree.column("protocol", width=160)
    tree.column("count", width=120, anchor=tk.E)
    tree.column("percent", width=120, anchor=tk.E)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_heatmap_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Heatmap")
    columns = ("index", "start", "end", "packets", "bytes")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("index", text="Bucket")
    tree.heading("start", text="Start")
    tree.heading("end", text="End")
    tree.heading("packets", text="Packets")
    tree.heading("bytes", text="Bytes")
    tree.column("index", width=80, anchor=tk.CENTER)
    tree.column("start", width=140)
    tree.column("end", width=140)
    tree.column("packets", width=120, anchor=tk.E)
    tree.column("bytes", width=120, anchor=tk.E)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_annotations_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Annotations")
    columns = ("index", "tags", "summary")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("index", text="Packet #")
    tree.heading("tags", text="Tags")
    tree.heading("summary", text="Summary")
    tree.column("index", width=100, anchor=tk.CENTER)
    tree.column("tags", width=240)
    tree.column("summary", width=360)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_alerts_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Alerts")
    columns = ("rule", "pattern", "src", "dst", "protocol", "info")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    headers = {
      "rule": "Rule",
      "pattern": "Pattern",
      "src": "Source",
      "dst": "Destination",
      "protocol": "Protocol",
      "info": "Details",
    }
    for col, title in headers.items():
      tree.heading(col, text=title)
    tree.column("rule", width=160)
    tree.column("pattern", width=200)
    tree.column("src", width=150)
    tree.column("dst", width=150)
    tree.column("protocol", width=90, anchor=tk.CENTER)
    tree.column("info", width=300)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_insights_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Plugin Insights")
    columns = ("name", "severity", "summary")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("name", text="Plugin")
    tree.heading("severity", text="Severity")
    tree.heading("summary", text="Summary")
    tree.column("name", width=180)
    tree.column("severity", width=120, anchor=tk.CENTER)
    tree.column("summary", width=420)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _build_recipe_tab(self, notebook: ttk.Notebook) -> tk.Text:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Recipe Builder")

    stages_container = ttk.Frame(frame)
    stages_container.pack(fill=tk.X, padx=6, pady=6)

    self.recipe_vars = {}
    for col, (stage, modules) in enumerate(RECIPE_MODULES.items()):
      lf = ttk.LabelFrame(stages_container, text=stage.title())
      lf.grid(row=0, column=col, sticky=tk.NSEW, padx=6)
      self.recipe_vars[stage] = {}
      for module in modules:
        var = tk.BooleanVar(value=False)
        self.recipe_vars[stage][module["id"]] = var
        ttk.Checkbutton(
          lf,
          text=module["label"],
          variable=var,
          command=lambda s=stage, mid=module["id"]: self._toggle_recipe_module(s, mid),
        ).pack(anchor=tk.W, padx=4, pady=2)
        ttk.Label(lf, text=module.get("description", ""), wraplength=200, foreground="#666").pack(anchor=tk.W, padx=12)

    controls = ttk.Frame(frame)
    controls.pack(fill=tk.X, padx=6)
    ttk.Button(controls, text="Clear Selection", command=self._clear_recipe_selection).pack(side=tk.LEFT)
    ttk.Button(controls, text="Preview JSON", command=self._update_recipe_preview).pack(side=tk.LEFT, padx=6)

    preview = tk.Text(frame, height=8, state=tk.DISABLED)
    preview.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))
    return preview

  def _build_insights_tab(self, notebook: ttk.Notebook) -> ttk.Treeview:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Insights")
    columns = ("name", "severity", "summary")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("name", text="Source")
    tree.heading("severity", text="Severity")
    tree.heading("summary", text="Summary")
    tree.column("name", width=180)
    tree.column("severity", width=100, anchor=tk.CENTER)
    tree.column("summary", width=380)
    tree.pack(fill=tk.BOTH, expand=True)
    return tree

  def _render_protocols(self, breakdown: Dict[str, int]) -> None:
    self.protocol_tree.delete(*self.protocol_tree.get_children())
    total = sum(breakdown.values()) or 1
    for proto, count in sorted(breakdown.items(), key=lambda item: item[1], reverse=True):
      percent = (count / total) * 100
      self.protocol_tree.insert("", tk.END, values=(proto, count, f"{percent:.1f}%"))

  def _render_heatmap(self, heatmap: List[Dict[str, object]]) -> None:
    self.heatmap_tree.delete(*self.heatmap_tree.get_children())
    for bucket in heatmap:
      self.heatmap_tree.insert(
        "",
        tk.END,
        values=(
          bucket.get("index", 0),
          f"{bucket.get('start', 0.0):.3f}",
          f"{bucket.get('end', 0.0):.3f}",
          bucket.get("packet_count", 0),
          bucket.get("bytes", 0),
        ),
      )

  def _toggle_recipe_module(self, stage: str, module_id: str) -> None:
    var = self.recipe_vars.get(stage, {}).get(module_id)
    if var is None:
      return
    if var.get():
      self.selected_recipe.setdefault(stage, set()).add(module_id)
    else:
      self.selected_recipe.setdefault(stage, set()).discard(module_id)
    self._update_recipe_preview()

  def _clear_recipe_selection(self) -> None:
    for stage, modules in self.recipe_vars.items():
      for module_id, var in modules.items():
        var.set(False)
    for stage in self.selected_recipe:
      self.selected_recipe[stage].clear()
    self._update_recipe_preview()

  def _recipe_configuration(self) -> Dict[str, object]:
    stages: List[Dict[str, object]] = []
    module_count = 0
    total_rules = 0
    for stage, module_ids in self.selected_recipe.items():
      modules_meta: List[Dict[str, object]] = []
      for module_id in module_ids:
        module_info = next((m for m in RECIPE_MODULES.get(stage, []) if m["id"] == module_id), None)
        if not module_info:
          continue
        modules_meta.append({
          "id": module_info["id"],
          "label": module_info.get("label"),
          "description": module_info.get("description"),
          "playbook_rules": module_info.get("playbook_rules", {}),
        })
        total_rules += len(module_info.get("playbook_rules", {}))
      if modules_meta:
        module_count += len(modules_meta)
        stages.append({"stage": stage, "modules": modules_meta})
    return {
      "module_count": module_count,
      "stages": stages,
      "rule_count": total_rules,
    }

  def _apply_recipe_to_playbook(self, playbook: Dict[str, str], recipe: Dict[str, object]) -> None:
    for stage, module_ids in self.selected_recipe.items():
      for module_id in module_ids:
        module_info = next((m for m in RECIPE_MODULES.get(stage, []) if m["id"] == module_id), None)
        if not module_info:
          continue
        for rule, pattern in module_info.get("playbook_rules", {}).items():
          playbook.setdefault(rule, pattern)

  def _update_recipe_preview(self) -> None:
    recipe = self._recipe_configuration()
    preview_text = json.dumps(recipe, indent=2)
    self.recipe_preview.configure(state=tk.NORMAL)
    self.recipe_preview.delete("1.0", tk.END)
    self.recipe_preview.insert(tk.END, preview_text)
    self.recipe_preview.configure(state=tk.DISABLED)

  def _render_recipe_preview(self, recipe: Dict[str, object]) -> None:
    preview_text = json.dumps(recipe, indent=2)
    self.recipe_preview.configure(state=tk.NORMAL)
    self.recipe_preview.delete("1.0", tk.END)
    self.recipe_preview.insert(tk.END, preview_text)
    self.recipe_preview.configure(state=tk.DISABLED)

  def _draw_timeline(self) -> None:
    canvas = self.timeline_canvas
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width <= 1 or height <= 1:
      return
    canvas.delete("all")
    packets = self.timeline_packets
    if not packets:
      canvas.create_text(
        width / 2,
        height / 2,
        text="No packets ingested.",
        fill="#888",
      )
      return
    times = [float(pkt.get("timestamp", 0.0)) for pkt in packets]
    min_t = min(times)
    max_t = max(times)
    span = max(max_t - min_t, 1e-6)
    zoom = max(min(self.timeline_zoom.get(), 1.0), 0.25)
    window_span = span * zoom
    window_end = max_t
    window_start = window_end - window_span
    usable_width = width - 32
    visible_packets = [pkt for pkt in packets if window_start <= float(pkt.get("timestamp", 0.0)) <= window_end]
    if not visible_packets:
      visible_packets = packets
      window_start = min_t
      window_end = max_t
      window_span = span
    for pkt in visible_packets:
      proto = str(pkt.get("protocol", "")).upper()
      colour = self._protocol_colour(proto)
      timestamp = float(pkt.get("timestamp", 0.0))
      x = ((timestamp - window_start) / max(window_span, 1e-6)) * usable_width + 16
      canvas.create_line(x, 12, x, height - 24, fill=colour, width=2)
    canvas.create_text(16, height - 12, text=f"{window_start:.3f}", anchor=tk.W, fill="#aaa")
    canvas.create_text(width - 16, height - 12, text=f"{window_end:.3f}", anchor=tk.E, fill="#aaa")

  def _reset_timeline_zoom(self) -> None:
    self.timeline_zoom.set(1.0)
    self._draw_timeline()

  @staticmethod
  def _protocol_colour(protocol: str) -> str:
    palette = {
      "TCP": "#4caf50",
      "UDP": "#03a9f4",
      "HTTP": "#ff9800",
      "TLS": "#9c27b0",
      "DNS": "#ffeb3b",
      "ICMP": "#f44336",
    }
    return palette.get(protocol, "#9e9e9e")


def launch() -> None:
  app = SpectraTraceApp()
  app.mainloop()


__all__ = ["launch", "SpectraTraceApp"]
