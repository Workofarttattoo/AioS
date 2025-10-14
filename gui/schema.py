"""
Dashboard schema descriptors used by the AgentaOS GUI compositor.

The descriptors define a small vocabulary of dashboard primitives so that
meta-agents can announce their capabilities to external viewers without
embedding UI-specific logic.  Downstream compositors only need to understand
these structures to render panels, metrics, tables, and interactive actions.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LayoutHint(BaseModel):
  """Lightweight layout metadata understood by the compositor."""

  placement: Literal["grid", "stack", "tabs"] = "grid"
  columns: Optional[int] = Field(default=None, ge=1)
  rows: Optional[int] = Field(default=None, ge=1)
  stretch: Optional[Literal["horizontal", "vertical", "both"]] = None


class ActionDescriptor(BaseModel):
  """
  Interactive action linked to a runtime hook.

  The compositor maps these descriptors to buttons or menu entries and invokes
  the configured runtime hook via the IPC bus when activated.
  """

  id: str
  label: str
  runtime_hook: str = Field(description="Fully-qualified runtime action path.")
  confirmation: Optional[str] = Field(
    default=None,
    description="Optional confirmation prompt presented to the operator.",
  )
  icon: Optional[str] = Field(default=None, description="Named icon identifier.")


class MetricDescriptor(BaseModel):
  """Key metric rendered inline or within a panel."""

  id: str
  label: str
  value: str
  unit: Optional[str] = None
  severity: Literal["info", "warn", "error"] = "info"


class TableRow(BaseModel):
  """Single row within a table widget."""

  cells: Dict[str, str]


class TableDescriptor(BaseModel):
  """Tabular dataset for lists, logs, or inventories."""

  id: str
  title: str
  columns: List[str]
  rows: List[TableRow] = Field(default_factory=list)
  empty_state: Optional[str] = Field(default=None, description="Message when no rows exist.")


class PanelDescriptor(BaseModel):
  """Collection of widgets grouped under a single heading."""

  id: str
  title: str
  description: Optional[str] = None
  layout: LayoutHint = Field(default_factory=LayoutHint)
  metrics: List[MetricDescriptor] = Field(default_factory=list)
  tables: List[TableDescriptor] = Field(default_factory=list)
  actions: List[ActionDescriptor] = Field(default_factory=list)


class DashboardDescriptor(BaseModel):
  """Complete dashboard description for a meta-agent worker."""

  worker: str = Field(description="Name of the worker/meta-agent emitting the dashboard.")
  version: str = Field(default="1.0")
  panels: List[PanelDescriptor] = Field(default_factory=list)
  annotations: Dict[str, str] = Field(default_factory=dict)

  def as_payload(self) -> Dict[str, object]:
    """
    Render the descriptor as a JSON-serialisable payload.

    Sub-agents can hand this payload directly to ``ctx.publish_metadata`` so it
    persists inside the runtime context and can be broadcast to user interfaces.
    """

    return self.model_dump(mode="json")

