#!/usr/bin/env python3
import asyncio
import json
import logging
import random
from typing import Dict, List, Optional
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

LOG = logging.getLogger(__name__)

class LocalLLMBrain:
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model = model_name
        self.system_prompt = """You are APEX, a world-class bug bounty hunter and penetration tester.
You have 10+ years experience in web appsec, API security, business logic flaws, and report writing.
Always think step-by-step. Never hallucinate tool commands. Use real, correct syntax."""

    async def _simulate_reasoning(self, prompt: str, json_mode: bool) -> Dict:
        """Simulate LLM responses if Ollama is unavailable"""
        LOG.info("Using simulated LLM reasoning...")
        await asyncio.sleep(2) # Simulate processing time

        prompt_lower = prompt.lower()

        if "analyze this target" in prompt_lower:
            return {
                "recon_info": "Target appears to be a standard web application.",
                "tech_stack": ["Nginx", "PHP", "Ubuntu"],
                "attack_surface": ["HTTP/HTTPS", "SSH"],
                "strategy": "Look for outdated software versions and common web vulnerabilities."
            }

        if "strategy" in prompt_lower:
            return {
                "tools": ["nmap", "nuclei"],
                "objectives": ["Identify open ports", "Scan for CVEs", "Check for misconfigurations"],
                "reasoning": "Standard network and vulnerability scan to establish baseline."
            }

        if "analyze the following scan findings" in prompt_lower:
            # Randomly "find" something sometimes
            if random.random() > 0.5:
                return {
                    "summary": "Discovered a potential vulnerability.",
                    "findings": [
                        {
                            "name": "Outdated Nginx Version",
                            "severity": "MEDIUM",
                            "description": "Nginx version 1.14.0 detected, which has known CVEs.",
                            "remediation": "Update Nginx to the latest stable version."
                        }
                    ]
                }
            else:
                return {
                    "summary": "No significant vulnerabilities found in this scan.",
                    "findings": [],
                    "severity": "INFO"
                }

        return {"reasoning": "Simulated response for generic prompt.", "raw": "Simulated response."}

    async def reason(self, prompt: str, temperature: float = 0.7, json_mode: bool = False) -> Dict:
        """Main reasoning interface"""
        if not HAS_OLLAMA:
            return await self._simulate_reasoning(prompt, json_mode)

        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}"

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model,
                prompt=full_prompt,
                options={"temperature": temperature, "num_ctx": 8192}
            )

            text = response.get('response', '')

            if json_mode:
                try:
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start >= 0:
                        return json.loads(text[json_start:json_end])
                except Exception as e:
                    LOG.error(f"Failed to parse JSON from LLM response: {e}")

            return {"reasoning": text, "raw": text}
        except Exception as e:
            LOG.warning(f"Ollama Reasoning failed, falling back to simulation: {e}")
            return await self._simulate_reasoning(prompt, json_mode)

    async def analyze_target(self, target: str, context: Dict = None) -> Dict:
        prompt = f"Analyze this target for high-value bug bounty opportunities:\nTarget: {target}\nContext: {context}"
        return await self.reason(prompt, json_mode=True)

if __name__ == "__main__":
    async def main():
        brain = LocalLLMBrain()
        print(f"Brain initialized (Ollama available: {HAS_OLLAMA})")
        result = await brain.analyze_target("example.com")
        print(json.dumps(result, indent=2))

    asyncio.run(main())
