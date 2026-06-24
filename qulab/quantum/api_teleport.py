from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio, json, random

router = APIRouter()

@router.get("/run")
async def run():
    async def gen():
        for step in ["building","transpiling","running","analyzing"]:
            await asyncio.sleep(0.15)
            yield {"event":"status","data":step}
        # stubbed counts/fidelity; Pro/Lab will call qiskit
        counts = {"00": random.randint(800,1200), "01": random.randint(300,600),
                  "10": random.randint(300,600), "11": random.randint(0,120)}
        total = sum(counts.values())
        fid = round(0.995 + random.random()*0.004, 6)
        yield {"event":"result","data": json.dumps({"fidelity": fid, "counts": counts, "shots": total})}
    return EventSourceResponse(gen())