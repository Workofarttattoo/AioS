#!/usr/bin/env python3
"""
MCP-compatible tool wrapper for Bug Bounty Suite
"""
import asyncio
import json
import sys
from typing import Dict, List, Optional
from apex_hunter import ApexHunter

class BugBountyMCP:
    def __init__(self, allowed_targets: List[str] = None):
        self.hunter = ApexHunter(allowed_targets=allowed_targets)

    async def list_tools(self):
        return [
            {
                "name": "hunt",
                "description": "Start a full bug bounty hunt on a target",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target domain or IP"}
                    },
                    "required": ["target"]
                }
            },
            {
                "name": "scan_nuclei",
                "description": "Run nuclei vulnerability scanner",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target domain or IP"},
                        "severity": {"type": "string", "description": "Severities to scan for", "default": "critical,high"}
                    },
                    "required": ["target"]
                }
            }
        ]

    async def call_tool(self, name: str, arguments: Dict):
        if name == "hunt":
            target = arguments.get("target")
            return await self.hunter.hunt(target)
        elif name == "scan_nuclei":
            target = arguments.get("target")
            severity = arguments.get("severity", "critical,high")
            return await self.hunter.scanner.run_nuclei(target, severity)
        else:
            raise ValueError(f"Unknown tool: {name}")

# Simple stdio-based MCP server loop (simplified)
async def main():
    mcp = BugBountyMCP()
    # In a real MCP server, we would handle JSON-RPC over stdin/stdout
    # For now, this is a placeholder/demonstration of the interface
    print("Bug Bounty MCP Layer ready.")

if __name__ == "__main__":
    asyncio.run(main())
