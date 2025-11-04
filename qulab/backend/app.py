from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.encoding import router as encoding_router
from api.governance import router as gov_router
from api.sim.teleport import router as teleport_router
from api.field_maintenance import router as field_router

app = FastAPI(title="QuLab API (Lite+Dummies)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health") 
def health(): return {"ok": True}

app.include_router(encoding_router, prefix="/encoding", tags=["encoding"])
app.include_router(gov_router, prefix="/governance", tags=["governance"])
app.include_router(teleport_router, prefix="/sim/teleport", tags=["teleport"])
app.include_router(field_router, prefix="/field", tags=["field_maintenance"])
