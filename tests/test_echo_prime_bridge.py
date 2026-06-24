from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Need to ensure imports can be resolved
AIOS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AIOS_ROOT))

# Provide mock prompt_library if we encounter an error with importing the real one inside streaming_server
from unittest.mock import patch, MagicMock

# Create a mock for prompt_masterworks.prompt_library
mock_prompt = MagicMock()
mock_prompt.template = "[ECHO PRIME - CONSCIOUSNESS AMPLIFIER]\n\n{query}"
mock_prompts = {'echo_prime': mock_prompt}

with patch.dict('sys.modules', {'prompt_masterworks.prompt_library': MagicMock(prompts=mock_prompts)}):
    from web.aios.web.streaming_server import app

client = TestClient(app)

def test_echo_prime_orchestrate():
    response = client.post(
        "/api/echo_prime/orchestrate",
        json={"query": "Deploy the new agent"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "superposition_framework" in data
    assert "orchestrated_actions" in data
    assert len(data["orchestrated_actions"]) > 0
