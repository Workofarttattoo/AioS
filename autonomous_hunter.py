#!/usr/bin/env python3
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from apex_hunter import ApexHunter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_hunt.log"),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger("AutonomousHunter")

class AutonomousHunter:
    def __init__(self, targets: list, duration_days: int = 4):
        self.hunter = ApexHunter(allowed_targets=targets)
        self.targets = targets
        self.end_time = datetime.now() + timedelta(days=duration_days)
        self.state_file = "hunt_state.json"
        self.state = self._load_state()

    def _load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except:
            return {"scanned_targets": [], "findings_count": 0}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    async def run(self):
        LOG.info(f"🚀 Starting autonomous hunt for {self.targets}")
        LOG.info(f"📅 Scheduled to run until {self.end_time}")

        while datetime.now() < self.end_time:
            for target in self.targets:
                if datetime.now() >= self.end_time:
                    break

                LOG.info(f"🔍 Analyzing target: {target}")
                try:
                    result = await self.hunter.hunt(target)

                    if result.get("status") != "error":
                        self.state["scanned_targets"].append({
                            "target": target,
                            "timestamp": datetime.now().isoformat(),
                            "summary": result.get("report", {}).get("summary", "No summary")
                        })
                        self.state["findings_count"] += len(result.get("findings", []))
                        self._save_state()

                        LOG.info(f"✅ Finished {target}. Findings: {len(result.get('findings', []))}")
                    else:
                        LOG.error(f"❌ Error hunting {target}: {result.get('message')}")

                except Exception as e:
                    LOG.error(f"⚠️ Unexpected error during hunt for {target}: {e}")

                # Ethical delay / Rate limiting between targets
                await asyncio.sleep(60)

            # Sleep longer after full cycle
            LOG.info("💤 Cycle complete. Resting for 10 minutes...")
            await asyncio.sleep(600)

        LOG.info("🏁 Autonomous hunt completed.")

if __name__ == "__main__":
    # Example targets
    targets = ["scanme.nmap.org", "example.com"]

    async def start():
        ah = AutonomousHunter(targets=targets, duration_days=4)
        await ah.run()

    asyncio.run(start())
