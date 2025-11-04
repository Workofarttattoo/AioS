"""
GUI schema and IPC utilities for AgentaOS compositor integration.

The schema module exposes the Pydantic descriptors that agents use to describe
dashboards.  The bus module provides a lightweight IPC channel so the runtime
can stream schema updates to the external compositor or the curses fallback.
"""

from .schema import (
  ActionDescriptor,
  DashboardDescriptor,
  LayoutHint,
  MetricDescriptor,
  PanelDescriptor,
  TableDescriptor,
  TableRow,
)
from .bus import DEFAULT_ENDPOINT, IPCConfig, SchemaPublisher, SchemaSubscriber

__all__ = [
  "ActionDescriptor",
  "DashboardDescriptor",
  "LayoutHint",
  "MetricDescriptor",
  "PanelDescriptor",
  "TableDescriptor",
  "TableRow",
  "SchemaPublisher",
  "SchemaSubscriber",
  "IPCConfig",
  "DEFAULT_ENDPOINT",
]
