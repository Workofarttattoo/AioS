#!/usr/bin/env python3
"""
AI:OS Streaming Server - Lightweight web interface for system telemetry
Adaptive deployment: works on any server (Bluehost, VPS, localhost)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import psutil
import os
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI:OS Streaming Dashboard", version="0.1.0")

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Paths
DASHBOARD_HTML = Path(__file__).parent / "streaming_dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the streaming dashboard"""
    if DASHBOARD_HTML.exists():
        return FileResponse(DASHBOARD_HTML)
    return HTMLResponse("<h1>AI:OS Dashboard</h1><p>Loading...</p>")


@app.get("/healthz")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "service": "AI:OS Streaming Server",
        "version": "0.1.0"
    }


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    try:
        return {
            "runtime_status": "OPERATIONAL",
            "agents_active": "7/7",
            "forensic_mode": os.getenv("AGENTA_FORENSIC_MODE", "0") == "1",
            "uptime_seconds": int(psutil.boot_time()),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
            "memory_percent": psutil.virtual_memory().percent,
            "network_io_kb": sum(psutil.net_io_counters()[:2]) / 1024
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"error": str(e)}


@app.get("/api/agents")
async def get_agents():
    """Get list of active meta-agents"""
    return {
        "agents": [
            {"name": "KernelAgent", "status": "active", "description": "Process management, system initialization"},
            {"name": "SecurityAgent", "status": "active", "description": "Firewall, encryption, sovereign toolkit"},
            {"name": "NetworkingAgent", "status": "active", "description": "Network configuration, DNS, routing"},
            {"name": "StorageAgent", "status": "active", "description": "Volume management, filesystem operations"},
            {"name": "ScalabilityAgent", "status": "active", "description": "Load monitoring, virtualization"},
            {"name": "OrchestrationAgent", "status": "active", "description": "Policy engine, telemetry"},
            {"name": "ApplicationAgent", "status": "active", "description": "Application supervisor"}
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected. Total connections: {len(active_connections)}")

    try:
        while True:
            # Send real-time metrics every second
            metrics = {
                "type": "metrics",
                "timestamp": asyncio.get_event_loop().time(),
                "cpu": round(psutil.cpu_percent(interval=0.1), 1),
                "memory": round(psutil.virtual_memory().used / (1024 * 1024), 0),
                "memory_percent": round(psutil.virtual_memory().percent, 1),
                "network_io": round(sum(psutil.net_io_counters()[:2]) / 1024, 1)
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_event(event: dict):
    """Broadcast event to all connected WebSocket clients"""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(event)
        except Exception:
            disconnected.append(connection)

    # Remove disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 AI:OS Streaming Server starting...")
    logger.info(f"📊 Dashboard available at http://localhost:8080")
    logger.info(f"🔌 WebSocket streaming at ws://localhost:8080/ws")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 AI:OS Streaming Server shutting down...")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "streaming_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
