#!/usr/bin/env python3
import asyncio
import json
import logging
from typing import Dict, List, Optional
from local_llm_brain import LocalLLMBrain
from bug_bounty_scanner import BugBountyScanner, ScopeEnforcer

LOG = logging.getLogger(__name__)

class ApexHunter:
    def __init__(self, model_name: str = "deepseek-r1:8b", allowed_targets: List[str] = None):
        self.brain = LocalLLMBrain(model_name=model_name)
        self.scope = ScopeEnforcer(allowed_targets=allowed_targets)
        self.scanner = BugBountyScanner(scope=self.scope)
        LOG.info("APEX Hunter initialized and ready for deployment.")

    async def _infer_application_context(self, target: str) -> Dict:
        """Use LLM to infer context about the target"""
        return await self.brain.analyze_target(target)

    async def hunt(self, target: str) -> Dict:
        """Main orchestration loop for hunting bugs"""
        LOG.info(f"Starting hunt on {target}")

        if not self.scope.is_allowed(target):
            LOG.warning(f"Target {target} is OUT OF SCOPE!")
            return {"status": "error", "message": "OUT_OF_SCOPE"}

        # 1. Reconnaissance & Context Inference
        context = await self._infer_application_context(target)
        LOG.info(f"Target Context: {context}")

        # 2. Strategy Formulation
        strategy_prompt = f"""Based on the target context: {json.dumps(context)}
Formulate a penetration testing strategy for {target}.
Identify which tools to use (nmap, nuclei, ffuf, sqlmap) and what specifically to look for.
Return the strategy in JSON format with a 'tools' list and 'objectives' list."""

        strategy = await self.brain.reason(strategy_prompt, json_mode=True)
        LOG.info(f"Hunting Strategy: {strategy}")

        results = {
            "target": target,
            "context": context,
            "strategy": strategy,
            "findings": []
        }

        # 3. Execution (Sequential for now, can be parallelized)
        if "nmap" in str(strategy.get("tools", "")).lower():
            LOG.info("Running Nmap scan...")
            nmap_res = await self.scanner.run_nmap(target)
            results["findings"].append({"tool": "nmap", "result": nmap_res})

        if "nuclei" in str(strategy.get("tools", "")).lower():
            LOG.info("Running Nuclei scan...")
            nuclei_res = await self.scanner.run_nuclei(target)
            results["findings"].append({"tool": "nuclei", "result": nuclei_res})

        # 4. Final Analysis
        analysis_prompt = f"""Analyze the following scan findings for {target}:
{json.dumps(results["findings"])}

Identify potential vulnerabilities, assign severity (CRITICAL, HIGH, MEDIUM, LOW), and suggest remediation.
Format your response as a professional bug bounty report in JSON."""

        final_report = await self.brain.reason(analysis_prompt, json_mode=True)
        results["report"] = final_report

        return results

if __name__ == "__main__":
    async def main():
        hunter = ApexHunter(allowed_targets=["scanme.nmap.org"])
        # result = await hunter.hunt("scanme.nmap.org")
        # print(json.dumps(result, indent=2))

    asyncio.run(main())
