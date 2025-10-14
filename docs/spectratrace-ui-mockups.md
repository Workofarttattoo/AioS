# SpectraTrace UI Mock-ups

## Dockable Telemetry Workspace

```mermaid
flowchart TB
  Header["Global Toolbar<br/>• Profile selector<br/>• Capture controls<br/>• Alert bell"] --> Layout
  subgraph Layout["Dockable Workspace"]
    DirectionLR
    Timeline["Packet Timeline<br/>zoom + scrub"] --> AlertsPanel["Alert Feed<br/>severity badges"]
    AlertsPanel --> DetailsPanel["Detail Drawer<br/>decoded payloads<br/>annotations"]
    Timeline --> Heatmap["Protocol Heatmap<br/>adaptive bins"]
    Heatmap --> Notes["Analyst Notes<br/>collapsible stack"]
  end
  Layout --> Footer["Status Bar<br/>stream health · ingest rate · recipe state"]
```

## Adaptive Recipe Builder

```mermaid
flowchart LR
  Source["Source<br/>Capture / Stream / Tap"] --> Stage1{"Enrich"}
  Stage1 --> Stage2{"Detect"}
  Stage2 --> Stage3{"Respond"}

  Stage1 -->|Options| EnrichA["• TLS JA3 fingerprinting"]
  Stage1 --> EnrichB["• HTTP metadata extraction"]
  Stage2 --> DetectA["• Behavioral scoring"]
  Stage2 --> DetectB["• Custom rule DSL"]
  Stage3 --> RespondA["• Alert routing"]
  Stage3 --> RespondB["• Live annotation push"]

  subgraph Preview["Recipe Preview"]
    JSON["YAML/JSON diff"]
    Test["Dry-run sample packets"]
  end
  Stage3 --> Preview
  Preview --> Publish["Publish to Library"]
```

These wireframes guide the new dockable layout and recipe authoring experience planned for the SpectraTrace refresh.
