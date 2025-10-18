from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio, json, random
router=APIRouter()
@router.get("/run")
async def run():
    async def gen():
        for s in ["building","transpiling","running","analyzing"]:
            await asyncio.sleep(0.15); yield {"event":"status","data":s}
        counts={"00":random.randint(800,1200),"01":random.randint(300,600),"10":random.randint(300,600),"11":random.randint(0,120)}
        shots=sum(counts.values()); fidelity=round(0.995+random.random()*0.004,6)
        yield {"event":"result","data":json.dumps({"fidelity":fidelity,"counts":counts,"shots":shots})}
    return EventSourceResponse(gen())
