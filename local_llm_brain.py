#!/usr/bin/env python3
import asyncio
import json
import logging
from typing import Dict, List, Optional
import ollama

LOG = logging.getLogger(__name__)

class LocalLLMBrain:
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model = model_name
        self.system_prompt = """You are APEX, a world-class bug bounty hunter and penetration tester.
You have 10+ years experience in web appsec, API security, business logic flaws, and report writing.
Always think step-by-step. Never hallucinate tool commands. Use real, correct syntax."""

    async def reason(self, prompt: str, temperature: float = 0.7, json_mode: bool = False) -> Dict:
        """Main reasoning interface"""
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
            LOG.error(f"LLM Reasoning failed: {e}")
            return {"error": str(e), "raw": ""}

    async def analyze_target(self, target: str, context: Dict = None) -> Dict:
        prompt = f"Analyze this target for high-value bug bounty opportunities:\nTarget: {target}\nContext: {context}"
        return await self.reason(prompt, json_mode=True)

if __name__ == "__main__":
    # Quick test
    async def main():
        brain = LocalLLMBrain()
        print(f"Brain initialized with model: {brain.model}")
        # Note: This will fail if ollama server is not running or model is not pulled
        # result = await brain.analyze_target("example.com")
        # print(result)

    asyncio.run(main())
