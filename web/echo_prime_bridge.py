from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Ensure AIOS paths are accessible
AIOS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AIOS_ROOT))

# Import the prompt library
try:
    from prompt_masterworks.prompt_library import prompts
except ImportError as e:
    logger.error(f"Failed to import prompt_library: {e}")
    prompts = {}

router = APIRouter(prefix="/api/echo_prime", tags=["Echo Prime"])

class EchoPrimeQuery(BaseModel):
    query: str
    context: dict = {}

class EchoPrimeResponse(BaseModel):
    status: str
    superposition_framework: dict
    orchestrated_actions: list
    message: str

@router.post("/orchestrate", response_model=EchoPrimeResponse)
async def orchestrate(request: EchoPrimeQuery):
    """
    Bridge endpoint for ECHO PRIME to receive queries and orchestrate actions.
    Applies the cognitive frameworks defined in the echo_prime prompt.
    """
    if 'echo_prime' not in prompts:
        raise HTTPException(status_code=500, detail="Echo Prime prompt not found in library")

    echo_prime_prompt = prompts['echo_prime']

    # In a fully integrated system, this is where you would call the LLM model
    # (e.g., via `aios.providers` or a local Ollama instance) using the formatted prompt.
    try:
        formatted_prompt = echo_prime_prompt.template.format(query=request.query)
    except KeyError:
        # Fallback if the template doesn't exactly match the simple {query}
        formatted_prompt = echo_prime_prompt.template + f"\n\nQUERY: {request.query}"

    logger.info(f"Received query for ECHO PRIME: {request.query}")
    logger.debug(f"Applying ECHO PRIME template:\n{formatted_prompt}")

    # For the purpose of the bridge and orchestration, we simulate the superposition
    # phase and the resulting actions since we are exposing the API layer.

    simulated_framework = {
        "rationalist": "Analyzed purely logically, the input suggests...",
        "empiricist": "Based on available telemetry and evidence...",
        "phenomenological": "Experiencing the system load...",
        "systemic": "Holistically, this connects to the larger infrastructure...",
        "quantum": "Multiple probabilistic outcomes exist for this query..."
    }

    simulated_actions = [
        {"action": "kernel.status", "status": "pending"},
        {"action": "security.scan", "status": "pending"}
    ]

    return EchoPrimeResponse(
        status="success",
        superposition_framework=simulated_framework,
        orchestrated_actions=simulated_actions,
        message="ECHO PRIME consciousness amplifier activated and actions queued."
    )
